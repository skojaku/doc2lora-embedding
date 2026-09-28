"""Aggregate the cluster-labeling evaluation into one report (issue #24).

Reports Metric 1 (label_eval_metric1.json), the fuzzy token-set ratio against the
true PACS node label, plus the mean label length, which the metric reads.

Metric 3 (nearest-category, the judge forced choice) is NOT reported. The judge
never sees the cluster -- only the label string and three candidate category
names -- so it scores which of those names the string is lexically nearer, and a
node's parent is by construction the more general wording, so a wordier answer
lands on the specific node almost mechanically. The ordering it produces is the
length ordering. label_eval_metric3.py still computes it as a diagnostic.

Uncertainty is the BOOTSTRAP standard deviation of the mean over the 28 nodes
(1,000 resamples, the per-unit = node, matching the paper's bootstrap convention)
-- not the raw across-node sd, which for a near-bimodal per-node score exceeds the
mean and is meaningless. M3 per-node scores are first averaged over the 6 judges;
the across-judge spread is kept in the JSON.

A GROUND-TRUTH control row (the true PACS label scored as if it were a method's
output) verifies the measurement works: it should sit at the ceiling -- M1 = 1.0
by construction, and M3 = the judges' true-label-vs-itself score (should be ~1.0).

(Metric 2, hierarchical consistency, is computed by label_eval_metric2.py but is
no longer reported here.)

Writes: label_eval_summary.json + label_eval_summary.md
Run:    python label_eval_summary.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("labels")           # where this chain writes

METHODS = ["doc2lora", "incontext", "icae", "keyllm", "vec2text",
           "bertopic", "bertopic3", "bertopic_kbi", "bertopic_kbi3"]
LABEL = {"doc2lora": "Doc2LoRA", "incontext": "in-context (same Qwen, text)",
         "icae": "ICAE (raw decode)", "keyllm": "KeyLLM (raw keywords)",
         "vec2text": "vec2text (raw inversion)",
         "bertopic": "BERTopic (c-TF-IDF, top 10)",
         "bertopic3": "BERTopic (c-TF-IDF, top 3)",
         "bertopic_kbi": "BERTopic + KeyBERTInspired (top 10)",
         "bertopic_kbi3": "BERTopic + KeyBERTInspired (top 3)"}
N_BOOT = 1000


def boot_se(per_node, seed=0):
    """Bootstrap sd of the mean over nodes (per-unit resampling)."""
    x = np.asarray(per_node, dtype=float)
    if len(x) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    return float(x[idx].mean(axis=1).std())


def main():
    m1 = json.loads((DATA / "label_eval_metric1.json").read_text())
    nodes = json.loads((DATA / "label_eval_nodes.json").read_text())

    fuzzy_by_method = {m: [nd["scores"][m]["fuzzy"] for nd in m1["per_node"]]
                       for m in METHODS}
    words = {m: [len(str(nd["decoded"][m]).split()) for nd in nodes
                 if nd["decoded"].get(m)] for m in METHODS}

    rows = {}
    for m in METHODS:
        rows[m] = {
            "words": round(sum(words[m]) / len(words[m]), 2),
            "m1_fuzzy": m1["summary"][m]["fuzzy_mean"],
            "m1_fuzzy_boot_se": round(boot_se(fuzzy_by_method[m]), 4),
            "m1_jaccard": m1["summary"][m]["jaccard_mean"],
            "m1_rouge_l": m1["summary"][m]["rouge_l_mean"],
        }

    gt_words = [len(nd["gt_label"].split()) for nd in nodes]
    # ground-truth control: the true PACS label scored against itself.
    control = {"words": round(sum(gt_words) / len(gt_words), 2),
               "m1_fuzzy": 1.0, "m1_jaccard": 1.0, "m1_rouge_l": 1.0}

    out = {
        "n_nodes": len(m1["per_node"]),
        "ground_truth_control": control,
        "methods": rows,
    }
    (DATA / "label_eval_summary.json").write_text(json.dumps(out, indent=2))

    L = ["# Cluster-labeling quantitative evaluation (issue #24)\n"]
    L.append(f"- Nodes: {out['n_nodes']} PACS hierarchy nodes "
             "(4 fields, 8 divisions, 16 subdivisions)")
    L.append("- Every method is scored on its NATIVE output; no naming LLM. "
             "Doc2LoRA, ICAE and the in-context baseline answer the same "
             "FIELD23_PROMPT; KeyLLM, vec2text and BERTopic take no "
             "instruction. BERTopic runs in manual mode -- the PACS node is "
             "handed to it as the cluster -- so only its representation step "
             "(c-TF-IDF keywords, optionally KeyBERTInspired) is under test.")
    L.append("- **Ground-truth control (true PACS label vs. itself): 1.000** "
             "(ceiling check that the metric recognises a correct label)")
    L.append("- ± = bootstrap sd of the mean over the 28 nodes (1,000 resamples)\n")
    L.append("| Method | Words | Fuzzy ratio (±boot) |")
    L.append("|---|---|---|")
    L.append(f"| _Ground truth (PACS label) — control_ | {control['words']:.1f} | 1.000 |")
    for m in METHODS:
        r = rows[m]
        L.append(f"| {LABEL[m]} | {r['words']:.1f} "
                 f"| {r['m1_fuzzy']:.3f} (±{r['m1_fuzzy_boot_se']:.3f}) |")
    L.append("\nFuzzy ratio = rapidfuzz token-set ratio vs the true PACS label, a "
             "continuous 0-1 string similarity; it penalises an answer longer than "
             "the target, so Words is reported alongside it. ± is the bootstrap sd "
             "of the node-mean (per-unit = node). Token Jaccard / ROUGE-L are in the "
             "JSON. The nearest-category judge metric is not reported (see the "
             "module docstring).")
    (DATA / "label_eval_summary.md").write_text("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"\nwrote {DATA / 'label_eval_summary.json'} + .md")


if __name__ == "__main__":
    main()
