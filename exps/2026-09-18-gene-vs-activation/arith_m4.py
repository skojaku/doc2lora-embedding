"""Metric 4 for the M-C node labels: gene vs raw activations (#152).

Replaces the M3 arm, which was dropped from this experiment's conclusions after
iclr2026-af showed the nearest-category judge never sees the cluster and ranks by
verbosity, and that its option shuffle was seeded with `hash(node_code)` — process
dependent, so it silently returned a different score on every run.

Metric 4 transfers to M-C without redesign: its reference is the official PACS label and
its candidates are short method labels, which is exactly what M-C produces. Nothing new
is invented here. Everything that removes bias is IMPORTED from
`label_eval_metric4.py` rather than copied:

  SYS / build_user   the prompt. Changing one character invalidates every cache key.
  order_jobs         both orders of one pair, one job per judge
  ask_order          one (pair, order, judge) call
  collapse_orders    a win requires BOTH orders to name the same side; an
                     order-dependent answer is a tie. This is where position bias
                     is removed, so it must be imported, never reimplemented.

These arms deliberately do NOT go into `METHOD_FILES`: they belong to #152, and
registering them would put them in the paper's cluster-label table.

Calibration runs first and must pass before any score is read — each arm against the
true label (the arm must never win) and the true label against itself (must tie).

  set -a; . ../../.env; set +a
  CONTROL_ONLY=1 python arith_m4.py     # calibration only
  python arith_m4.py
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BT = ROOT / "exps/2026-06-11-baseline-trees"
sys.path.insert(0, str(BT))
from label_eval_judges import JUDGES_V2, run_jobs  # noqa: E402
from label_eval_metric4 import ask_order, collapse_orders  # noqa: E402

GT = "_gt"
ARMS = ["gene_label", "act_label"]
BOOT = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results/arith_m4.json"))
    a = ap.parse_args()

    rows = json.loads((HERE / "results/arith_m1.json").read_text())["rows"]
    judges = dict(JUDGES_V2)
    if os.environ.get("JUDGES_SUBSET"):
        keep = set(os.environ["JUDGES_SUBSET"].split(","))
        judges = {k: v for k, v in judges.items() if k in keep}
    control_only = bool(os.environ.get("CONTROL_ONLY"))

    method_pairs = list(itertools.combinations(ARMS, 2))
    control_pairs = [(GT, m) for m in ARMS] + [(GT, GT)]
    pairs = control_pairs if control_only else method_pairs + control_pairs

    def label_of(code, m):
        return rows[code]["gt"] if m == GT else rows[code].get(m)

    jobs = []
    for code, r in rows.items():
        for x, y in pairs:
            lx, ly = label_of(code, x), label_of(code, y)
            if not lx or not ly:
                continue
            for j in (dict(code=code, gt=r["gt"], x=x, y=y, lx=lx, ly=ly, order=o)
                      for o in (0, 1)):
                jobs += [{**j, "judge": jn, "slug": sl} for jn, sl in judges.items()]
    print(f"Metric 4 (#152): {len(rows)} nodes x {len(pairs)} pairs x 2 orders x "
          f"{len(judges)} judges = {len(jobs)} calls (cached)", flush=True)

    res = run_jobs(jobs, ask_order, max_workers=8, every=200)
    outcomes = collapse_orders(res)
    n_err = sum(1 for v in outcomes.values() if v is None)

    # ---- calibration -------------------------------------------------------
    print("\nCalibration (must hold before reading the scores):")
    print("  arm         gt wins      tie  METHOD WINS  (method wins must be ~0)")
    cal = {}
    for m in ARMS:
        v = [w for (j, c, x, y), w in outcomes.items()
             if w is not None and {x, y} == {GT, m}]
        if not v:
            continue
        cal[m] = {"gt": v.count(GT) / len(v), "tie": v.count("tie") / len(v),
                  "method": v.count(m) / len(v), "n": len(v)}
        print(f"  {m:<12}{cal[m]['gt']:>7.3f}{cal[m]['tie']:>9.3f}{cal[m]['method']:>13.3f}")
    self_v = [w for (j, c, x, y), w in outcomes.items()
              if w is not None and x == GT and y == GT]
    self_tie = self_v.count("tie") / len(self_v) if self_v else float("nan")
    print(f"  true label vs itself, tie rate: {self_tie:.3f} (should be ~1.0)")

    out = {"arms": ARMS, "panel": list(judges), "calibration": cal,
           "gt_self_tie": self_tie, "n_errored": n_err, "control_only": control_only}
    if control_only:
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"\n{n_err} errored pair-decisions\nwrote {a.out} (calibration only)")
        return

    # ---- head-to-head ------------------------------------------------------
    rng = np.random.default_rng(0)
    per_node = {a_: [] for a_ in ARMS}
    for code in rows:
        got = {a_: [] for a_ in ARMS}
        for (j, c, x, y), w in outcomes.items():
            if c != code or w is None or x not in ARMS or y not in ARMS:
                continue
            sx = 1.0 if w == x else (0.5 if w == "tie" else 0.0)
            got[x].append(sx)
            got[y].append(1.0 - sx)
        for a_ in ARMS:
            if got[a_]:
                per_node[a_].append(float(np.mean(got[a_])))

    print("\nMetric 4 -- head-to-head win rate vs the official label (ties = 0.5):")
    for a_ in ARMS:
        v = np.array(per_node[a_])
        sd = float(v[rng.integers(0, len(v), (BOOT, len(v)))].mean(1).std())
        out.setdefault("scores", {})[a_] = {"win_rate": float(v.mean()), "sd": sd,
                                            "n_nodes": len(v)}
        print(f"  {a_:<12} {v.mean():.3f}  (+-{sd:.3f})  n={len(v)}")

    g = np.array(per_node["gene_label"])
    ac = np.array(per_node["act_label"])
    n = min(len(g), len(ac))
    d0 = g[:n] - ac[:n]
    bs = d0[rng.integers(0, n, (5000, n))].mean(1)
    out["paired"] = {"diff": float(d0.mean()), "sd": float(bs.std()),
                     "p_le_0": float((bs <= 0).mean()), "n": n,
                     "win": int((d0 > 0).sum()), "loss": int((d0 < 0).sum()),
                     "tie": int((d0 == 0).sum())}
    p = out["paired"]
    print(f"  [paired] gene - act: {p['diff']:+.3f} +- {p['sd']:.3f}  "
          f"P(<=0)={p['p_le_0']:.3f}  win/loss/tie {p['win']}/{p['loss']}/{p['tie']}")
    print(f"\n{n_err} errored pair-decisions")

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
