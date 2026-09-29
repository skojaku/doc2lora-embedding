"""[GPU-light] Score the #101 incoherent-cluster control.

Two instruments, one deterministic and one judged, both blind to the arm:

  (1) label-member agreement (no LLM): SBERT cosine between the decoded label and the centroid of
      its own members' abstracts, plus the margin against every other cluster's member centroid and
      the rank of the own-cluster centroid.  If a label reports the cluster's shared content, its own
      members should be its nearest cluster; if it only reports the centroid's norm, they should not.

  (2) blinded judge panel (the roster already used for the cluster-label metrics): given 12 sampled
      member titles and a candidate label, does the set belong to the named field, and is the label
      specific, broad, or a generic non-answer?  The judge never sees which arm a row came from.

Out: data/groupc/incoherent/incoh_scores.json and incoh_rows.parquet. The table and the figure
are drawn from those two files by incoh_report.py (rule incoh_report), so they rebuild on a CPU
from the archived scores without SBERT, the APS text, or a judge call.
"""
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "data/labels")

LABELS = json.load(open(snakemake.input.labels))            # noqa: F821
CLUSTERS = json.load(open(snakemake.input.clusters))        # noqa: F821
NODES_EVAL = json.load(open(snakemake.params.nodes_eval))   # noqa: F821  ground-truth PACS labels
N_TITLES = int(snakemake.params.n_titles)                   # noqa: F821
N_MEMBERS_SBERT = int(snakemake.params.n_members_sbert)     # noqa: F821
SEED = int(snakemake.params.seed)                           # noqa: F821
USE_JUDGES = bool(snakemake.params.use_judges)              # noqa: F821

gt = {str(n["code"]): n.get("gt_label", "") for n in NODES_EVAL}
members = {c["node"]: c["members"] for c in CLUSTERS["real"]}
members.update({c["cid"]: c["members"] for c in CLUSTERS["control"]})

rows = []
for cond in ("real", "control_native", "control_matched"):
    for cid, rec in LABELS[cond].items():
        rows.append({"cond": cond, "cluster": cid, "label": rec["label"], "norm": rec["norm"],
                     "match_node": rec.get("match_node", cid),
                     "gt_label": gt.get(rec.get("match_node", cid), "")})
R = pd.DataFrame(rows)
print(f"[incoh-score] {len(R)} rows: " + ", ".join(f"{k}={v}" for k, v in R.cond.value_counts().items()),
      flush=True)

# ── (1) deterministic label-member agreement via SBERT ─────────────────────────────────
txt = pd.read_parquet(snakemake.input.paper_text)            # noqa: F821
idc = "aps_paper_id" if "aps_paper_id" in txt.columns else "paper_id"
id2title = dict(zip(txt[idc].astype(int), txt["title"].astype(str)))
id2text = dict(zip(txt[idc].astype(int), txt["text"].astype(str)))

from sentence_transformers import SentenceTransformer  # noqa: E402

sb = SentenceTransformer(snakemake.params.sbert_model)       # noqa: F821
rng = np.random.default_rng(SEED)

cl_ids = list(dict.fromkeys(R.cluster.tolist()))
cent = {}
for cid in cl_ids:
    mem = members[cid]
    sel = mem if len(mem) <= N_MEMBERS_SBERT else rng.choice(mem, N_MEMBERS_SBERT, replace=False)
    docs = [id2text[int(p)] for p in sel if int(p) in id2text]
    E = sb.encode(docs, batch_size=128, convert_to_numpy=True, normalize_embeddings=True,
                  show_progress_bar=False)
    v = E.mean(0)
    cent[cid] = v / (np.linalg.norm(v) + 1e-9)
C = np.stack([cent[c] for c in cl_ids])
cidx = {c: i for i, c in enumerate(cl_ids)}

lab_vecs = sb.encode(R.label.tolist(), batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
S = lab_vecs @ C.T                                   # [rows, clusters]
own = np.array([S[i, cidx[c]] for i, c in enumerate(R.cluster)])
others = S.copy()
for i, c in enumerate(R.cluster):
    others[i, cidx[c]] = -np.inf
R["cos_own"] = own
R["cos_best_other"] = others.max(1)
R["margin"] = R.cos_own - R.cos_best_other
R["rank_own"] = [(S[i] > S[i, cidx[c]]).sum() + 1 for i, c in enumerate(R.cluster)]
R["hit_at_1"] = (R.rank_own == 1).astype(int)

# ── (2) blinded judge panel ────────────────────────────────────────────────────────────
JUDGE_SYS = (
    "You are grading whether a short label correctly names the research area of a set of physics "
    "papers. Answer ONLY with a JSON object: "
    '{"belongs": "yes|partial|no", "altitude": "specific|broad|generic", "reason": "<12 words"}. '
    '"belongs": does the label name a research area that the listed papers actually share? '
    '"altitude": is the label a specific research topic, a broad subfield, or a generic '
    'non-answer such as "science" or "physics research"?'
)

judge_summary = {}
if USE_JUDGES:
    # JUDGES_V2, not the legacy roster: figs/incoherent_control.tex as committed was
    # produced by the current panel, and a generator that disagrees with the table it
    # writes is how App. E.1 came to sit on a stale panel in the first place.
    from label_eval_judges import JUDGES_V2 as JUDGES, judge_json, run_jobs  # noqa: E402

    rnd = random.Random(SEED)
    jobs = []
    for i, r in R.iterrows():
        mem = members[r.cluster]
        sel = mem if len(mem) <= N_TITLES else rnd.sample(list(mem), N_TITLES)
        titles = [id2title.get(int(p), "") for p in sel]
        titles = [t for t in titles if t.strip()][:N_TITLES]
        user = ("Candidate label: " + r.label + "\n\nPapers:\n" +
                "\n".join(f"- {t}" for t in titles))
        for jname, slug in JUDGES.items():
            jobs.append({"row": int(i), "judge": jname, "slug": slug, "user": user})
    print(f"[incoh-score] {len(jobs)} judge calls ({len(JUDGES)} judges x {len(R)} rows)", flush=True)
    res = run_jobs(jobs, lambda it: judge_json(it["slug"], JUDGE_SYS, it["user"]), max_workers=8)

    per_row = {i: {"belongs": [], "altitude": []} for i in R.index}
    for it, out in zip(jobs, res):
        if not isinstance(out, dict) or "_error" in out:
            continue
        b = str(out.get("belongs", "")).lower().strip()
        al = str(out.get("altitude", "")).lower().strip()
        if b in ("yes", "partial", "no"):
            per_row[it["row"]]["belongs"].append(b)
        if al in ("specific", "broad", "generic"):
            per_row[it["row"]]["altitude"].append(al)

    def majority(xs, order):
        if not xs:
            return ""
        return max(order, key=lambda o: (xs.count(o), -order.index(o)))

    R["belongs"] = [majority(per_row[i]["belongs"], ["yes", "partial", "no"]) for i in R.index]
    R["altitude"] = [majority(per_row[i]["altitude"], ["specific", "broad", "generic"]) for i in R.index]
    R["belongs_yes_frac"] = [
        (per_row[i]["belongs"].count("yes") / len(per_row[i]["belongs"])) if per_row[i]["belongs"] else np.nan
        for i in R.index]
    R["n_judges"] = [len(per_row[i]["belongs"]) for i in R.index]
    judge_summary = {
        cond: {
            "belongs_yes": float((g.belongs == "yes").mean()),
            "belongs_partial": float((g.belongs == "partial").mean()),
            "belongs_no": float((g.belongs == "no").mean()),
            "belongs_yes_frac_mean": float(g.belongs_yes_frac.mean()),
            "altitude_specific": float((g.altitude == "specific").mean()),
            "altitude_broad": float((g.altitude == "broad").mean()),
            "altitude_generic": float((g.altitude == "generic").mean()),
        } for cond, g in R.groupby("cond")
    }

# ── summary ───────────────────────────────────────────────────────────────────────────
summary = {}
for cond, g in R.groupby("cond"):
    summary[cond] = {
        "n": int(len(g)),
        "cos_own_mean": float(g.cos_own.mean()), "cos_own_sd": float(g.cos_own.std(ddof=1)),
        "margin_mean": float(g.margin.mean()), "hit_at_1": float(g.hit_at_1.mean()),
        "norm_mean": float(g.norm.mean()),
    }
    if cond in judge_summary:
        summary[cond]["judge"] = judge_summary[cond]
os.makedirs(os.path.dirname(snakemake.output.scores), exist_ok=True)   # noqa: F821
with open(snakemake.output.scores, "w") as fh:                          # noqa: F821
    json.dump({"summary": summary, "rows": R.to_dict(orient="records")}, fh, indent=1)
R.to_parquet(snakemake.output.rows, index=False)                        # noqa: F821
print(json.dumps(summary, indent=1), flush=True)
print(f"[incoh-score] wrote {snakemake.output.scores}, {snakemake.output.rows}", flush=True)   # noqa: F821
