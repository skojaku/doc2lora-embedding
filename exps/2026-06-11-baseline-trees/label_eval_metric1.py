"""Metric 1 (lexical overlap) for cluster-labeling (issue #24).

Surface agreement between each method's decoded label and the ground-truth PACS
node label. Headline score is the fuzzy token-set ratio (rapidfuzz, 0-1): a
CONTINUOUS string similarity, so per-node scores are not near-bimodal the way
token Jaccard is (Jaccard is ~0/1 per node, giving a meaningless across-node sd).
Token Jaccard and ROUGE-L are still recorded for reference. Cheap, no LLM.

Reads:  label_eval_nodes.json
Writes: label_eval_metric1.json
Run:    python label_eval_metric1.py
"""
import json
import re
from pathlib import Path

from rapidfuzz import fuzz

HERE = Path(__file__).resolve().parent
METHODS = ["doc2lora", "vec2text", "keyllm", "icae", "incontext",
           "t2l_gte", "t2l_hidden", "t2l_dw"]   # T2L arms, #151

STOP = {"and", "of", "the", "in", "a", "for", "to", "on", "with", "or"}


def fuzzy(pred, ref):
    """rapidfuzz token_set_ratio in [0,1] (order-independent, credits subsets)."""
    if not str(pred or "").strip() or not str(ref or "").strip():
        return 0.0
    return fuzz.token_set_ratio(str(pred), str(ref)) / 100.0


def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+", str(s or "").lower()) if t not in STOP]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def lcs(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[n][m]


def rouge_l(pred, ref):
    p, r = toks(pred), toks(ref)
    if not p or not r:
        return 0.0
    l = lcs(p, r)
    prec, rec = l / len(p), l / len(r)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def main():
    nodes = json.loads((HERE / "label_eval_nodes.json").read_text())
    per_node = []
    agg = {m: {"fuzzy": [], "jaccard": [], "rouge_l": []} for m in METHODS}
    for nd in nodes:
        gt = nd["gt_label"]
        row = {"code": nd["code"], "kind": nd["kind"], "gt_label": gt, "scores": {}}
        for m in METHODS:
            dec = nd["decoded"].get(m)
            f = fuzzy(dec, gt)
            j = jaccard(toks(dec), toks(gt))
            rl = rouge_l(dec, gt)
            row["scores"][m] = {"decoded": dec, "fuzzy": round(f, 4),
                                "jaccard": round(j, 4), "rouge_l": round(rl, 4)}
            agg[m]["fuzzy"].append(f)
            agg[m]["jaccard"].append(j)
            agg[m]["rouge_l"].append(rl)
        per_node.append(row)

    summary = {m: {"fuzzy_mean": round(sum(v["fuzzy"]) / len(v["fuzzy"]), 4),
                   "jaccard_mean": round(sum(v["jaccard"]) / len(v["jaccard"]), 4),
                   "rouge_l_mean": round(sum(v["rouge_l"]) / len(v["rouge_l"]), 4),
                   "n": len(v["fuzzy"])}
               for m, v in agg.items()}
    out = {"metric": "lexical_overlap", "headline": "fuzzy_token_set_ratio",
           "summary": summary, "per_node": per_node}
    (HERE / "label_eval_metric1.json").write_text(json.dumps(out, indent=2))
    print("Metric 1 — lexical overlap (mean over 28 nodes):")
    print(f"  {'method':10s} {'fuzzy':>8s} {'jaccard':>8s} {'rouge_l':>8s}")
    for m in METHODS:
        s = summary[m]
        print(f"  {m:10s} {s['fuzzy_mean']:8.3f} {s['jaccard_mean']:8.3f} {s['rouge_l_mean']:8.3f}")
    print(f"\nwrote {HERE / 'label_eval_metric1.json'}")


if __name__ == "__main__":
    main()
