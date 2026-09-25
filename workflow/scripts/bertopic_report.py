"""Read the BERTopic arms out of the cluster-label evaluation (issue #24).

The reported metric of the evaluation (label_eval_metric1.py) is a LEXICAL one:
the rapidfuzz token-set ratio against the true PACS label. That is the right
metric for methods that try to *name* the cluster, but it is harsh on a keyword
list: "heisenberg, spin, ising" and "General theory and models of magnetic
ordering" describe the same node and share no token. So this report adds two
things the headline table cannot show, for every method at once:

  1. a SEMANTIC score -- SBERT (all-mpnet-base-v2) cosine between the produced
     label and the true PACS label, i.e. "is this about the right thing" with
     the wording requirement dropped. Scored identically for every method, so
     the comparison is like-for-like. Its floor is not 0: two random physics
     phrases already sit well above zero, so the report also scores a SHUFFLED
     control (each method's labels permuted across nodes) as that method's own
     chance level, and reports the lift over it.
  2. a per-level split (field / division / subdivision), since a keyword list is
     expected to fare better on a narrow node than on a broad one.

When ``bertopic_judge.json`` is present (bertopic_judge.py, the metric-4 judge
panel run with the BERTopic arms in the field) its calibration, head-to-head and
win-rate tables are folded in as well, so the lexical, semantic and judged reads
of the same 28 nodes sit in one file.

It also scores the ``raw`` markup arm (bertopic_labels.py run with ``raw``) when
present, which measures how much of BERTopic's score is text preprocessing --
~31% of APS abstracts carry LaTeX / MathML that c-TF-IDF otherwise ranks as
topic words -- rather than the method itself.

Reads:  label_eval_nodes.json, label_eval_metric1.json, bertopic*_labels.json
Writes: bertopic_report.json + BERTOPIC.md
Run:    python bertopic_report.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("labels")           # where this chain writes

SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
N_BOOT = 1000
KINDS = ("field", "div", "sub")

# methods scored from label_eval_nodes.json (the registered arms)
METHODS = ["doc2lora", "incontext", "icae", "keyllm", "vec2text",
           "bertopic", "bertopic3", "bertopic_kbi", "bertopic_kbi3"]
LABEL = {"doc2lora": "Doc2LoRA", "incontext": "in-context (same Qwen, text)",
         "icae": "ICAE (raw decode)", "keyllm": "KeyLLM (raw keywords)",
         "vec2text": "vec2text (raw inversion)",
         "bertopic": "BERTopic (c-TF-IDF, top 10)",
         "bertopic3": "BERTopic (c-TF-IDF, top 3)",
         "bertopic_kbi": "BERTopic + KeyBERTInspired (top 10)",
         "bertopic_kbi3": "BERTopic + KeyBERTInspired (top 3)",
         "bertopic_raw": "BERTopic, markup kept (c-TF-IDF, top 10)",
         "bertopic_raw3": "BERTopic, markup kept (c-TF-IDF, top 3)"}

# extra arms read straight from their label files (not registered in the eval)
EXTRA_FILES = {"bertopic_raw": "bertopic_raw_labels.json",
               "bertopic_raw3": "bertopic_raw3_labels.json"}


def boot_se(x, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    return float(x[idx].mean(axis=1).std())


def fuzzy(pred, ref):
    from rapidfuzz import fuzz
    if not str(pred or "").strip() or not str(ref or "").strip():
        return 0.0
    return fuzz.token_set_ratio(str(pred), str(ref)) / 100.0


def main():
    nodes = json.loads((DATA / "label_eval_nodes.json").read_text())
    gt = [nd["gt_label"] for nd in nodes]
    kinds = [nd["kind"] for nd in nodes]

    decoded = {m: [nd["decoded"].get(m) or "" for nd in nodes] for m in METHODS}
    for m, f in EXTRA_FILES.items():
        p = HERE / f
        if p.exists():
            d = json.loads(p.read_text())
            decoded[m] = [d.get(nd["code"], "") for nd in nodes]
    methods = [m for m in list(METHODS) + list(EXTRA_FILES) if m in decoded]

    # --- semantic score: SBERT cosine(label, true PACS label) -----------------
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(SBERT_MODEL)

    def embed(texts):
        v = st.encode(list(texts), normalize_embeddings=True,
                      batch_size=64, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    gt_v = embed(gt)
    # shuffled control: the same labels attached to the wrong nodes, which is
    # this method's own chance level (a physics keyword list is never orthogonal
    # to a physics category name).
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(nodes))
    while np.any(perm == np.arange(len(nodes))):     # derangement: no node keeps its own label
        perm = rng.permutation(len(nodes))

    rows, per_method_cos, per_method_fz = {}, {}, {}
    for m in methods:
        pred_v = embed(decoded[m])
        cos = (pred_v * gt_v).sum(1)
        per_method_cos[m], per_method_fz[m] = cos, None
        cos_shuf = (pred_v[perm] * gt_v).sum(1)
        fz = np.array([fuzzy(p, g) for p, g in zip(decoded[m], gt)])
        per_method_fz[m] = fz
        words = np.array([len(str(s).split()) for s in decoded[m]])
        rows[m] = {
            "words": float(words.mean()),
            "fuzzy": float(fz.mean()), "fuzzy_se": boot_se(fz),
            "cosine": float(cos.mean()), "cosine_se": boot_se(cos),
            "cosine_shuffled": float(cos_shuf.mean()),
            "cosine_lift": float(cos.mean() - cos_shuf.mean()),
            "by_kind": {k: {"fuzzy": float(fz[[i for i, kk in enumerate(kinds) if kk == k]].mean()),
                            "cosine": float(cos[[i for i, kk in enumerate(kinds) if kk == k]].mean())}
                        for k in KINDS},
        }

    # head-to-head vs Doc2LoRA, per node, on the wording-free (cosine) score:
    # the mean hides that a keyword list can be the better description of a
    # narrow node while being far worse on a broad one.
    ref = per_method_cos["doc2lora"]
    head2head = {m: {"win": int((per_method_cos[m] > ref + 0.02).sum()),
                     "tie": int((np.abs(per_method_cos[m] - ref) <= 0.02).sum()),
                     "loss": int((per_method_cos[m] < ref - 0.02).sum())}
                 for m in methods if m != "doc2lora"}

    gt_self = float((gt_v * gt_v).sum(1).mean())
    out = {"n_nodes": len(nodes), "sbert_model": SBERT_MODEL,
           "control": {"fuzzy": 1.0, "cosine": gt_self,
                       "words": float(np.mean([len(s.split()) for s in gt]))},
           "methods": rows,
           "head2head_vs_doc2lora_cosine": head2head,
           "per_node": [{"code": nd["code"], "kind": nd["kind"], "gt_label": nd["gt_label"],
                         **{m: {"label": decoded[m][i],
                                "cosine": round(float(per_method_cos[m][i]), 4),
                                "fuzzy": round(float(per_method_fz[m][i]), 4)}
                            for m in methods}}
                        for i, nd in enumerate(nodes)]}
    (DATA / "bertopic_report.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # --- markdown ------------------------------------------------------------
    L = ["# BERTopic on the PACS cluster-labeling task\n",
         f"- {len(nodes)} PACS nodes (4 fields, 8 divisions, 16 subdivisions), "
         "every method scored on its native output.",
         "- BERTopic runs in **manual (predefined-cluster) mode**: the PACS node is "
         "handed in as `y`, UMAP/HDBSCAN are replaced by `BaseDimensionalityReduction` / "
         "`BaseCluster`, so what is under test is BERTopic's topic *representation* "
         "(c-TF-IDF, optionally re-ranked by KeyBERTInspired) on exactly the clusters "
         "every other method labels.",
         "- **Fuzzy** = the evaluation's reported lexical metric (rapidfuzz token-set "
         "ratio vs the true PACS label). **Cosine** = SBERT all-mpnet similarity to the "
         "same target, which ignores wording. **Lift** = cosine minus the same method's "
         "labels shuffled across nodes (its own chance level) — the honest read of "
         "cosine, since any two physics phrases already score high.",
         "- ± = bootstrap sd of the mean over nodes (1,000 resamples).\n",
         "| Method | Words | Fuzzy (±) | Cosine (±) | Shuffled | Lift |",
         "|---|---|---|---|---|---|",
         f"| _Ground truth (PACS label) — control_ | {out['control']['words']:.1f} "
         f"| 1.000 | {gt_self:.3f} | — | — |"]
    for m in methods:
        r = rows[m]
        L.append(f"| {LABEL[m]} | {r['words']:.1f} | {r['fuzzy']:.3f} (±{r['fuzzy_se']:.3f}) "
                 f"| {r['cosine']:.3f} (±{r['cosine_se']:.3f}) | {r['cosine_shuffled']:.3f} "
                 f"| {r['cosine_lift']:+.3f} |")

    L.append("\n## By hierarchy level (cosine / fuzzy)\n")
    L.append("| Method | field | division | subdivision |")
    L.append("|---|---|---|---|")
    for m in methods:
        b = rows[m]["by_kind"]
        L.append(f"| {LABEL[m]} | " + " | ".join(
            f"{b[k]['cosine']:.3f} / {b[k]['fuzzy']:.3f}" for k in KINDS) + " |")

    L.append("\n## Head-to-head vs Doc2LoRA (per node, cosine, \u00b10.02 = tie)\n")
    L.append("| Method | wins | ties | losses |")
    L.append("|---|---|---|---|")
    for m, h in head2head.items():
        L.append(f"| {LABEL[m]} | {h['win']} | {h['tie']} | {h['loss']} |")

    judge_path = DATA / "bertopic_judge.json"
    if judge_path.exists():
        jd = json.loads(judge_path.read_text())
        L.append(f"\n## Judge panel (metric-4 design, {len(jd['judges'])} judges, "
                 "both orders must agree)\n")
        L.append("Seats: " + ", ".join(sorted(jd["judges"])) + ".")
        L.append("Field: " + ", ".join(jd["field"]) +
                 ". Head-to-head rates depend only on the two arms compared, so they "
                 "are comparable with `label_eval_metric4.json`'s as long as the seat "
                 "roster matches; the win-rate column is scored against this field and "
                 "is never comparable across fields.\n")
        c = jd["control"]
        L.append(f"Calibration: true label vs itself ties {c['gt_vs_gt_tie_rate']:.3f} "
                 "(should be ~1.0); no arm ever beats the true label ("
                 + ", ".join(f"{m} {v['method_wins']:.3f}"
                             for m, v in c["gt_vs_method"].items())
                 + ", all ~0).\n")
        L.append("Ties WITH the true label — the arm's output judged as close to the "
                 "official name as that name itself, which is a result and not a "
                 "calibration failure: "
                 + ", ".join(f"{m} {v['tie']:.3f}" for m, v in c["gt_vs_method"].items())
                 + ".\n")
        L.append("| Arm | Win rate vs field (±) | vs Doc2LoRA | vs ICAE | vs in-context "
                 "| vs KeyLLM | vs vec2text |")
        L.append("|---|---|---|---|---|---|---|")
        opp = ["doc2lora", "icae", "incontext", "keyllm", "vec2text"]
        for m in jd["field"]:
            r = jd["methods"][m]
            cells = []
            for o in opp:
                if o == m:
                    cells.append("—")
                    continue
                d = jd["head_to_head"].get("|".join(sorted((m, o))))
                cells.append("—" if not d else f"{d['rate_' + m]:.3f}")
            L.append(f"| {LABEL.get(m, m)} | {r['win_rate']:.3f} (±{r['boot_sd']:.3f}) | "
                     + " | ".join(cells) + " |")
        if jd["n_errors"]:
            L.append(f"\n{jd['n_errors']} pair decisions errored out and are dropped.")

    L.append("\n## Per-node output\n")
    L.append("| Node | True PACS label | Doc2LoRA | BERTopic (top 10) |")
    L.append("|---|---|---|---|")
    for i, nd in enumerate(nodes):
        L.append(f"| {nd['code']} ({nd['kind']}) | {nd['gt_label']} "
                 f"| {decoded['doc2lora'][i]} | {decoded['bertopic'][i]} |")
    (DATA / "BERTOPIC.md").write_text("\n".join(L) + "\n")

    print("\n".join(L[:9 + len(methods)]))
    print(f"\nwrote {HERE / 'bertopic_report.json'} + BERTOPIC.md")


if __name__ == "__main__":
    main()
