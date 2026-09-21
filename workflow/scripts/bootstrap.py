"""Bootstrap CIs from the per-unit score pools (data/uncertainty/pools).
Resamples the natural unit per task and recomputes the metric N_BOOT times per method.

Outputs:
  bootstrap_replicates.parquet  [benchmark, group, enc, method, metric, boot, value]   (long; for violins/CDFs)
  uncertainty_summary.csv       [benchmark, group, enc, method, metric, mean, std, ci2.5, ci97.5, n_units]

Usage: python bootstrap.py [N_BOOT=1000]
"""
import sys, glob, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

POOLS = "data/uncertainty/pools"
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rng = np.random.default_rng(0)
reps = []


def add(benchmark, group, enc, method, metric, vals):
    for b, v in enumerate(vals):
        reps.append((benchmark, group, enc, method, metric, b, v))


def auc(y, s):
    m = ~np.isnan(s)
    return roc_auc_score(y[m], s[m]) if (m.sum() > 5 and len(np.unique(y[m])) > 1) else np.nan


# ---- collab: bootstrap over candidate pairs ----
for f in sorted(glob.glob(f"{POOLS}/collab_*.parquet")):
    field, enc = os.path.basename(f)[:-8].split("_")[1:3]
    d = pd.read_parquet(f); y = d.y.values.astype(float)
    methods = [c for c in d.columns if c not in ("window", "a", "b", "y")]
    n = len(d)
    for m in methods:
        s = d[m].values.astype(float)
        vals = [auc(y[idx], s[idx]) for idx in (rng.integers(0, n, n) for _ in range(N_BOOT))]
        add("collab", field, enc, m, "AUC", vals)

# ---- next-paper: cluster bootstrap over query_idx ----
for f in sorted(glob.glob(f"{POOLS}/np_*.parquet")):
    field, enc = os.path.basename(f)[:-8].split("_")[1:3]
    d = pd.read_parquet(f); qids = d.query_idx.values
    uq = np.unique(qids); byq = {q: np.flatnonzero(qids == q) for q in uq}
    methods = [c for c in d.columns if c not in ("query_idx", "cand_pid", "is_pos")]
    yv = d.is_pos.values.astype(float)
    for m in methods:
        sv = d[m].values.astype(float); vals = []
        for _ in range(N_BOOT):
            samp = np.concatenate([byq[q] for q in rng.choice(uq, len(uq))])
            vals.append(auc(yv[samp], sv[samp]))
        add("next_paper", field, enc, m, "AUC", vals)

# ---- topic: bootstrap over test papers, macro-F1 ----
for f in sorted(glob.glob(f"{POOLS}/topic_*.parquet")):
    field, enc = os.path.basename(f)[:-8].split("_")[1:3]
    d = pd.read_parquet(f); true = d.true.values; n = len(d)
    methods = [c for c in d.columns if c not in ("paper_id", "true")]
    for m in methods:
        pred = d[m].values; vals = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            vals.append(f1_score(true[idx], pred[idx], average="macro"))
        add("topic", field, enc, m, "F1", vals)

# ---- S2AND: bootstrap over signatures, B3 F1 from mean(P),mean(R) ----
for f in sorted(glob.glob(f"{POOLS}/s2and_*.parquet")):
    d = pd.read_parquet(f); ds = d.dataset.iloc[0]; enc = d.enc.iloc[0]
    for m, g in d.groupby("method"):
        P = g.b3_p.values; R = g.b3_r.values; n = len(g); vals = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            p, r = P[idx].mean(), R[idx].mean()
            vals.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
        add("name_disambig", ds, enc, m, "B3_F1", vals)

R = pd.DataFrame(reps, columns=["benchmark", "group", "enc", "method", "metric", "boot", "value"])
R.to_parquet(f"data/uncertainty/bootstrap_replicates.parquet")
summ = (R.groupby(["benchmark", "group", "enc", "method", "metric"])["value"]
        .agg(mean="mean", std="std", ci2_5=lambda x: x.quantile(.025), ci97_5=lambda x: x.quantile(.975))
        .reset_index())
summ.to_csv("data/uncertainty/uncertainty_summary.csv", index=False)
print(f"[saved] {len(R):,} replicate rows; {len(summ)} (benchmark×group×enc×method×metric) summary rows")
print(summ.head(20).to_string(index=False))
