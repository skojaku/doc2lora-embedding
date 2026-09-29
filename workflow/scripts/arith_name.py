"""Name a field from each arm's continuation text, and score M1 (#152).

`arith_decode.py` produces two kinds of output per node:

  *_label  a short label, from the shared LABEL_PROMPT
  *_cont   free continuation text

The continuation texts are not labels, so they cannot be compared against a PACS label
directly. One off-panel LLM turns each into a short field name, with **the identical
prompt and model for both arms** -- the whole point of the `*_cont` pairing is that the
activation arm gets a route that works for it, and fairness then requires the gene arm
to take the same route.

`minimax/minimax-m3` is the namer. It is deliberately NOT in `JUDGES_V2`, the current
judge roster, so the naming step cannot contaminate the M3 panel (#146).

M1 is the rapidfuzz token-set ratio against the official PACS label, exactly as
`label_eval_metric1.py` computes it, with bootstrap s.d. over nodes.

  set -a; . ../../.env; set +a      # OPENROUTER_API_KEY
  python arith_name.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("actpatch")           # where this chain writes

ROOT = HERE.parents[1]
BT = ROOT / "data/labels"
sys.path.insert(0, str(BT))
from label_eval_judges import judge_json  # noqa: E402

NAMER = "minimax/minimax-m3"        # off the JUDGES_V2 panel, per #146
BOOT = 2000

SYSTEM = ("You name scientific fields. Given a passage, reply with the research field "
          "it belongs to as a concise noun phrase of two to five words. "
          'Reply as JSON: {"field": "<phrase>"}. Nothing else.')


def name_field(text):
    if not str(text or "").strip():
        return ""
    v = judge_json(NAMER, SYSTEM, f"Passage:\n{text}\n\nWhat research field is this?")
    return " ".join(str((v or {}).get("field", "")).split())


def m1(pred, ref):
    if not str(pred or "").strip() or not str(ref or "").strip():
        return 0.0
    return fuzz.token_set_ratio(str(pred), str(ref)) / 100.0


def boot(v, rng, n=BOOT):
    v = np.asarray(v, dtype=float)
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), float(v[rng.integers(0, len(v), (n, len(v)))].mean(1).std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "arith_m1.json"))
    a = ap.parse_args()

    gt = {r["code"]: r["gt_label"] for r in json.loads((BT / "label_eval_nodes.json").read_text())}
    gene = json.loads((DATA / "arith_nodes_gene.json").read_text())
    act = json.loads((DATA / "arith_nodes_act.json").read_text())
    codes = [c for c in gene if c in act and c in gt]
    print(f"[nodes] {len(codes)} paired with a ground-truth label", flush=True)

    rows, labels = {}, {k: {} for k in
                        ("gene_label", "act_label", "gene_cont_named", "act_cont_named")}
    for c in codes:
        named_g = name_field(gene[c]["gene_cont"])
        named_a = name_field(act[c]["act_cont"])
        rows[c] = {
            "gt": gt[c],
            "gene_label": gene[c]["gene_label"],
            "act_label": act[c]["act_label"],
            "gene_cont": gene[c]["gene_cont"],
            "act_cont": act[c]["act_cont"],
            "gene_cont_named": named_g,
            "act_cont_named": named_a,
        }
        for k in labels:
            labels[k][c] = rows[c][k]
        print(f"  [{c:<6}] gt={gt[c]!r}\n"
              f"     gene_label={rows[c]['gene_label']!r} | act_label={rows[c]['act_label'][:50]!r}\n"
              f"     gene_named={named_g!r} | act_named={named_a!r}", flush=True)

    rng = np.random.default_rng(0)
    summary = {}
    for k in labels:
        v = [m1(labels[k][c], gt[c]) for c in codes]
        mean, sd = boot(v, rng)
        summary[k] = {"m1": mean, "m1_sd": sd, "n": len(v)}
        print(f"\n[M1] {k:<16} {mean:.3f} +- {sd:.3f}")

    # {code: label} files for METHOD_FILES registration (M3 runs from those)
    for k, fname in (("gene_label", "arith_gene_label.json"),
                     ("act_label", "arith_act_label.json"),
                     ("gene_cont_named", "arith_gene_named.json"),
                     ("act_cont_named", "arith_act_named.json")):
        (HERE / fname).write_text(json.dumps(labels[k], indent=2))

    Path(a.out).write_text(json.dumps({"namer": NAMER, "summary": summary, "rows": rows},
                                      indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
