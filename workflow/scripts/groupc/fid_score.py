"""[GPU-light] Score the #95 decode-fidelity conditions with deterministic instruments only.

Per decode, against its SOURCE paper (title + abstract):

  jaccard   content-word token Jaccard                     (the M1-style lexical instrument)
  rougeL    ROUGE-L F1 on content words
  kw_recall recall of the source's 15 highest-TF-IDF terms
  sbert     SBERT cosine to the source text                (semantic, still no LLM in the loop)
  rank / hit@1 / MRR
            rank of the TRUE source among all N sampled papers by SBERT cosine.  This is the
            discriminative test: a fluent decode that could have come from any paper scores at
            chance (rank ~ N/2), a faithful one puts its own paper first.

Out: fid_scores.json + fid_rows.parquet + figs/decode_fidelity.tex (table)
     + figs/decode_fidelity_examples.tex (verbatim Qwen3-4B decodes for App. D)
     + figs/decode_fidelity.pdf (rank distribution per condition)
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, "workflow/plot")

STOP = set("""a an the and or but if while of in on at to for from by with without into over under
this that these those is are was were be been being it its as we our they their he she his her not
no nor so than then there here which who whom whose what when where how all any both each few more
most other some such only own same too very can will just should now also using used use based
between during above below up down out off further once about against because both during through
""".split())

sample = pd.read_parquet(snakemake.input.sample)            # noqa: F821
sample["paper_id"] = sample.paper_id.astype(np.int64)
src_text = dict(zip(sample.paper_id, (sample.title.astype(str) + ". " + sample.abstract.astype(str))))
pids = sample.paper_id.tolist()


def toks(s: str) -> list:
    return [w for w in re.findall(r"[a-z][a-z0-9\-]+", str(s).lower()) if w not in STOP and len(w) > 2]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def rouge_l(a: list, b: list) -> float:
    """ROUGE-L F1 over token lists (LCS)."""
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(a), lcs / len(b)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# top TF-IDF terms per source paper (df computed over the sample)
src_toks = {p: toks(src_text[p]) for p in pids}
df_count = Counter()
for p in pids:
    df_count.update(set(src_toks[p]))
N = len(pids)
kw = {}
for p in pids:
    tf = Counter(src_toks[p])
    scored = sorted(tf, key=lambda w: -(tf[w] * np.log(N / (1 + df_count[w]))))
    kw[p] = [w for w in scored[:15]]

from sentence_transformers import SentenceTransformer  # noqa: E402

sb = SentenceTransformer(snakemake.params.sbert_model)      # noqa: F821
SRC = sb.encode([src_text[p] for p in pids], batch_size=64, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False)
prow = {p: i for i, p in enumerate(pids)}

rows = []
for path in snakemake.input.decodes:                        # noqa: F821
    blob = json.load(open(path))
    cond = blob["condition"]
    dec = blob["decodes"]
    texts = [dec[str(p)]["decode"] if str(p) in dec else "" for p in pids]
    DEC = sb.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
                    show_progress_bar=False)
    S = DEC @ SRC.T                       # [rows(decode), sources]
    for i, p in enumerate(pids):
        d = texts[i]
        dt = toks(d)
        st = src_toks[p]
        own = S[i, prow[p]]
        rank = int((S[i] > own).sum() + 1)
        rows.append({
            "cond": cond, "paper_id": int(p), "decode": d, "n_words": len(d.split()),
            "jaccard": jaccard(set(dt), set(st)),
            "rougeL": rouge_l(dt[:400], st[:400]),
            "kw_recall": float(np.mean([w in set(dt) for w in kw[p]])) if kw[p] else np.nan,
            "sbert": float(own), "rank": rank, "hit1": int(rank == 1),
            "mrr": 1.0 / rank,
            "sbert_best_other": float(np.sort(S[i])[-2] if len(pids) > 1 else np.nan),
        })
    print(f"[fid-score] {cond}: n={len(pids)}", flush=True)

R = pd.DataFrame(rows)
METRICS = ["jaccard", "rougeL", "kw_recall", "sbert", "hit1", "mrr", "n_words"]
summary = {}
for cond, g in R.groupby("cond"):
    summary[cond] = {m: float(g[m].mean()) for m in METRICS}
    summary[cond]["median_rank"] = float(g["rank"].median())
    summary[cond]["n"] = int(len(g))
    # bootstrap CI on the two headline metrics
    rng = np.random.default_rng(0)
    for m in ("sbert", "hit1"):
        bs = [g[m].values[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)]
        summary[cond][f"{m}_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
summary["_chance"] = {"hit1": 1.0 / len(pids), "median_rank": (len(pids) + 1) / 2, "n": len(pids)}

os.makedirs(os.path.dirname(snakemake.output.scores), exist_ok=True)   # noqa: F821
with open(snakemake.output.scores, "w") as fh:                         # noqa: F821
    json.dump(summary, fh, indent=1)
R.to_parquet(snakemake.output.rows, index=False)                       # noqa: F821
print(json.dumps(summary, indent=1), flush=True)

# ── table ─────────────────────────────────────────────────────────────────────────────
NICE = {"d2l": "Doc2LoRA decode (no text in prompt)", "incontext": "in-context reader (ceiling)",
        "mismatch": "mismatched adapter (floor)", "prior": "no adapter, no text (floor)"}
ORDER = ["incontext", "d2l", "mismatch", "prior"]
with open(snakemake.output.table, "w") as fh:                          # noqa: F821
    fh.write("% auto-generated by workflow/scripts/groupc/fid_score.py (issue #95) -- do not edit\n")
    fh.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
    fh.write("condition & Jaccard & ROUGE-L & keyword recall & SBERT cos. & source ranked 1st "
             "& median rank \\\\\n\\midrule\n")
    for c in ORDER:
        if c not in summary:
            continue
        s = summary[c]
        fh.write(f"{NICE[c]} & {s['jaccard']:.3f} & {s['rougeL']:.3f} & {s['kw_recall']:.3f} & "
                 f"{s['sbert']:.3f} & {100 * s['hit1']:.0f}\\% & {s['median_rank']:.0f} \\\\\n")
    ch = summary["_chance"]
    fh.write(f"\\midrule\nchance & --- & --- & --- & --- & {100 * ch['hit1']:.1f}\\% & "
             f"{ch['median_rank']:.0f} \\\\\n\\bottomrule\n\\end{{tabular}}\n")

# ── verbatim examples for App. D (the manuscript has no decoded scientific paper) ──────
n_ex = int(snakemake.params.n_examples)                                # noqa: F821
d2l_rows = R[R.cond == "d2l"].sort_values("sbert", ascending=False)
pick = pd.concat([d2l_rows.head(max(n_ex - 1, 1)), d2l_rows.tail(1)])   # best few + the worst


def tex_escape(s: str) -> str:
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"),
                 ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"),
                 ("^", "\\textasciicircum{}")):
        s = s.replace(a, b)
    return s


with open(snakemake.output.examples, "w") as fh:                        # noqa: F821
    fh.write("% auto-generated by workflow/scripts/groupc/fid_score.py (issue #95) -- do not edit\n")
    for _, r in pick.iterrows():
        row = sample[sample.paper_id == r.paper_id].iloc[0]
        fh.write("\\paragraph{" + tex_escape(str(row.title)) + "}\n")
        fh.write("\\emph{Source abstract (not shown to the model).} " +
                 tex_escape(str(row.abstract)[:900]) + "\n\n")
        fh.write("\\emph{Decoded from the embedding alone (Qwen3-4B, greedy; SBERT cos.\\ " +
                 f"{r.sbert:.2f}, source ranked {int(r['rank'])} of {len(pids)}).}} " +
                 tex_escape(r.decode) + "\n\n")

# ── figure: rank distribution per condition ───────────────────────────────────────────
import matplotlib.pyplot as plt              # noqa: E402
from _style import finalize, setup_style     # noqa: E402

setup_style(font_scale=1.1)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
xs = np.arange(1, len(pids) + 1)
for c in ORDER:
    g = R[R.cond == c]
    if not len(g):
        continue
    cdf = [(g["rank"] <= k).mean() for k in xs]
    ax.plot(xs, cdf, lw=2, label=NICE[c])
ax.plot(xs, xs / len(pids), color="0.6", ls="--", lw=1.2, label="chance")
ax.set_xscale("log")
ax.set_xlabel("rank of the source paper (log scale)")
ax.set_ylabel("cumulative fraction")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=8, loc="lower right")
finalize(fig, snakemake.output.fig)                                    # noqa: F821
print(f"[fid-score] wrote {snakemake.output.table}, {snakemake.output.examples}, "      # noqa: F821
      f"{snakemake.output.fig}", flush=True)                                            # noqa: F821
