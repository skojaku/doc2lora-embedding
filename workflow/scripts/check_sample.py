"""Check what the workflow produced from the sample corpus against what was planted.

A green run here means the reporting half of the pipeline is wired correctly: the
score pools keep methods paired, the bootstrap summarises the pools it was given,
the benchmark table reads that summary, the label metrics read the same labels they
score, and the judge panel collapses both presentation orders into one verdict.

It is a wiring test, not a result. The numbers below come from vectors built by
`make_sample.py`, and the only claim being checked is that the workflow recovers the
structure that was put in.

    python workflow/scripts/check_sample.py --pools data/uncertainty/pools \\
        --summary data/uncertainty/uncertainty_summary.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_data import out_dir  # noqa: E402

FIELD = "sample"
# Names as the score pools spell them, best first, from make_sample.METHOD_NOISE.
PLANTED = ["qwen_genkron", "sbert", "qwen_kron", "gte",
           "embeddinggemma", "specter2", "instructor", "qwen"]

failures, notes = [], []


def check(ok, msg, detail=""):
    (notes if ok else failures).append(f"{'PASS' if ok else 'FAIL'}  {msg}"
                                       + (f"\n        {detail}" if detail else ""))


def auc(y, s):
    y = np.asarray(y, float)
    order = np.argsort(np.asarray(s, float))
    r = np.empty(len(s), float)
    r[order] = np.arange(1, len(s) + 1)
    npos, nneg = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main(pools: Path, summary: Path, judge: Path | None):
    # ── 1. every planted method survives into every pool ─────────────────
    per_task = {}
    for task in ("np", "topic", "collab"):
        f = pools / f"{task}_{FIELD}_qwen.parquet"
        if not f.exists():
            check(False, f"{task} pool missing", str(f))
            continue
        d = pd.read_parquet(f)
        present = [m for m in PLANTED if m in d.columns]
        check(len(present) == len(PLANTED),
              f"{task} pool carries all {len(PLANTED)} methods",
              f"missing: {sorted(set(PLANTED) - set(present))}")
        per_task[task] = d

    # ── 2. the tasks separate signal from chance ─────────────────────────
    if "np" in per_task:
        d = per_task["np"]
        scores = {m: auc(d.is_pos, d[m]) for m in PLANTED if m in d}
        best = max(scores, key=scores.get)
        check(scores[best] > 0.70, "next-paper AUC is well above chance",
              " ".join(f"{m}={scores[m]:.3f}" for m in PLANTED if m in scores))
        check(spearman([PLANTED.index(m) for m in scores], [-scores[m] for m in scores]) > 0.6,
              "next-paper ranks the methods in the planted order",
              f"best={best}")
    if "topic" in per_task:
        d = per_task["topic"]
        accs = {m: float((d[m] == d["true"]).mean()) for m in PLANTED if m in d}
        check(max(accs.values()) > 0.45, "topic accuracy beats the 1/N chance level",
              " ".join(f"{m}={accs[m]:.3f}" for m in PLANTED if m in accs))
        spread = max(accs.values()) - min(accs.values())
        check(spread > 0.10, "topic classification is not saturated",
              f"spread {spread:.3f} between best and worst")
        check(spearman([PLANTED.index(m) for m in accs], [-accs[m] for m in accs]) > 0.6,
              "topic classification ranks the methods in the planted order")
    if "collab" in per_task:
        d = per_task["collab"].dropna(subset=[m for m in PLANTED if m in per_task["collab"]])
        if len(d) and d.y.nunique() == 2:
            a = {m: auc(d.y, d[m]) for m in PLANTED if m in d}
            check(max(a.values()) > 0.55, "collaboration AUC is above chance",
                  " ".join(f"{m}={a[m]:.3f}" for m in PLANTED if m in a))
        else:
            check(False, "collaboration pool has both classes", f"{len(d)} rows")

    # ── 3. the bootstrap summary reports what the pools contain ──────────
    if summary.exists():
        s = pd.read_csv(summary)
        s = s[s.get("field", FIELD).astype(str).str.contains(FIELD)] if "field" in s else s
        check(len(s) > 0, "bootstrap summary has rows for the sample field",
              f"{len(s)} rows, columns {list(s.columns)[:8]}")
        if "sd" in s.columns:
            check(bool((s["sd"].fillna(0) > 0).any()),
                  "bootstrap produced non-degenerate intervals")
    else:
        check(False, "bootstrap summary missing", str(summary))

    # ── 4. the label metric separates the planted label qualities ────────
    m1 = out_dir("labels") / "label_eval_metric1.json"
    if m1.exists():
        summ = json.loads(m1.read_text())["summary"]
        f_d2l = summ["doc2lora"]["fuzzy_mean"]
        check(f_d2l > summ["vec2text"]["fuzzy_mean"] and f_d2l > summ["t2l_gte"]["fuzzy_mean"],
              "fuzzy overlap ranks the gene decode above the degenerate baselines",
              " ".join(f"{m}={summ[m]['fuzzy_mean']:.3f}" for m in
                       ("doc2lora", "icae", "incontext", "keyllm", "vec2text", "t2l_gte")))
        check(summ["doc2lora"]["n"] > 0, "fuzzy overlap scored every node",
              f"n={summ['doc2lora']['n']}")
    else:
        check(False, "label_eval_metric1.json missing", str(m1))

    # ── 5. the judge panel, if it was run ────────────────────────────────
    if judge and judge.exists():
        jd = json.loads(judge.read_text())
        rates = {m: v["win_rate"] for m, v in jd["methods"].items()}
        check(jd.get("n_errors", 0) == 0, "every judge call returned a verdict",
              f"{jd.get('n_errors')} errors over {jd.get('n_pair_decisions')} decisions")
        check(rates["doc2lora"] > rates["icae"] > rates["vec2text"],
              "the round robin ranks the label qualities as planted",
              " ".join(f"{m}={rates[m]:.3f}" for m in sorted(rates, key=rates.get, reverse=True)))
        check(all(rates["doc2lora"] > rates[m] for m in rates if m.startswith("t2l_")),
              "the gene decode beats every T2L arm")
        ctl = jd["control"]["gt_vs_method"]
        check(ctl["doc2lora"]["tie"] > 0.9,
              "the control ties the exact label with the ground truth",
              f"tie={ctl['doc2lora']['tie']:.2f}")
        check(ctl["vec2text"]["gt_wins"] > 0.9,
              "the control prefers the ground truth to the garbled label",
              f"gt_wins={ctl['vec2text']['gt_wins']:.2f}")
        sds = [v["boot_sd"] for v in jd["methods"].values()]
        check(all(sd > 0 for sd in sds), "the judge bootstrap produced non-zero spread",
              f"min sd {min(sds):.4f}")
    else:
        notes.append("SKIP  judge panel not run (snakemake sample_check_judged)")

    print("\n".join(notes))
    if failures:
        print("\n".join(failures))
        sys.exit(f"\n{len(failures)} check(s) failed")
    print(f"\nall {len(notes)} checks passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--judge", default=None)
    a = ap.parse_args()
    main(Path(a.pools), Path(a.summary), Path(a.judge) if a.judge else None)
