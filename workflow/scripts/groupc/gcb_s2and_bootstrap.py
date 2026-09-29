"""[CPU] Bootstrap the S2AND half of the symmetric-supervision control (#145).

Reads the per-signature B^3 pools written by gcb_s2and_pool.py and resamples SIGNATURES, the same
unit the reported uncertainty run uses. B^3-F1 is a ratio of means, so precision and recall are
resampled together: F = 2 * mean(P) * mean(R) / (mean(P) + mean(R)).

Out: data/groupc/bench/gcb_s2and_summary.csv  [field(dataset), task, method, value, lo, hi, n]
"""
import glob
import os

import numpy as np
import pandas as pd

POOLS = snakemake.params.pool_dir                    # noqa: F821
NBOOT = int(snakemake.params.nboot)                  # noqa: F821
OUT = snakemake.output.csv                           # noqa: F821
rng = np.random.default_rng(0)

rows = []
for f in sorted(glob.glob(os.path.join(POOLS, "s2and_*.parquet"))):
    d = pd.read_parquet(f)
    ds = d.dataset.iloc[0]
    for method, g in d.groupby("method"):
        P = g.b3_p.values.astype(float)
        R = g.b3_r.values.astype(float)
        n = len(P)

        def f1(idx):
            mp, mr = P[idx].mean(), R[idx].mean()
            return 0.0 if (mp + mr) == 0 else 2 * mp * mr / (mp + mr)

        point = f1(np.arange(n))
        reps = np.array([f1(rng.integers(0, n, n)) for _ in range(NBOOT)])
        rows.append({"field": ds, "task": "name_disambig", "method": method, "value": point,
                     "lo": float(np.percentile(reps, 2.5)), "hi": float(np.percentile(reps, 97.5)),
                     "n": n})
        print(f"  {ds:12} {method:24} B3-F1 {point:.3f} "
              f"[{rows[-1]['lo']:.3f},{rows[-1]['hi']:.3f}]", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"[saved] {OUT}  ({len(rows)} rows)", flush=True)
