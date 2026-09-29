"""[CPU] PAIRED head-to-head between two methods across all 14 similarity tests (#145).

gcb_bootstrap.py resamples each method separately, so its intervals answer "how precise is this
number", not "is A above B". Two methods scored on the same units are correlated, and the
difference is estimated far more precisely than the two intervals suggest. This script resamples
the SAME units once per replicate and recomputes A - B on them, which is the quantity a reader
needs when two rows of the table look close.

Estimators mirror gcb_bootstrap exactly: collab = per-window AUC averaged, resampling candidate
pairs within each window; next-paper = pooled AUC with a cluster bootstrap over queries; topic =
macro-F1 over test papers; S2AND = B3-F1 built from mean(P), mean(R), resampling signatures.

Usage:
  python workflow/scripts/groupc/gcb_headtohead.py --a genkron --b icae_genkron \
      --a-s2and genkron --b-s2and icae_genkron --out data/groupc/bench/gcb_headtohead.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

POOLS = os.path.join("data", "groupc", "bench", "pools")
FIELDS = ["aps", "economics", "psychology"]
S2AND = ["zbmath", "qian", "arnetminer", "pubmed", "kisti"]


def auc(y, s):
    """Mann-Whitney AUC via ranks (ties averaged) -- same as gcb_bootstrap."""
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarize(name, field, task, pa, pb, da, db, reps):
    reps = np.asarray([r for r in reps if np.isfinite(r)])
    return {"field": field, "task": task, "a": da, "b": db, "a_value": pa, "b_value": pb,
            "diff": pa - pb, "diff_lo": float(np.percentile(reps, 2.5)),
            "diff_hi": float(np.percentile(reps, 97.5)),
            "p_a_better": float((reps > 0).mean()), "n": name}


def run(a, b, a_s2and, b_s2and, nboot, seed):
    rng = np.random.default_rng(seed)
    out = []

    for field in FIELDS:
        # ---- collab -------------------------------------------------------------------
        # Pooled over anchor windows, matching the manuscript and gcb_bootstrap.
        d = pd.read_parquet(f"{POOLS}/collab_{field}.parquet").dropna(subset=[a, b])
        y = d.y.values.astype(np.int8)
        s1 = d[a].values.astype(np.float64)
        s2 = d[b].values.astype(np.float64)
        pa, pb = auc(y, s1), auc(y, s2)
        n = len(d)
        reps = []
        for _ in range(nboot):
            k = rng.integers(0, n, n)                        # SAME rows for both methods
            reps.append(auc(y[k], s1[k]) - auc(y[k], s2[k]))
        out.append(summarize(n, field, "collab", pa, pb, a, b, reps))

        # ---- next paper ---------------------------------------------------------------
        d = pd.read_parquet(f"{POOLS}/np_{field}.parquet").dropna(subset=[a, b])
        y = d.is_pos.values.astype(np.int8)
        s1 = d[a].values.astype(np.float64)
        s2 = d[b].values.astype(np.float64)
        pa, pb = auc(y, s1), auc(y, s2)
        q = d.query_idx.values
        order = np.argsort(q, kind="stable")
        qs = q[order]
        bounds = np.searchsorted(qs, np.unique(qs), side="left")
        ends = np.append(bounds[1:], len(qs))
        groups = [order[i:j] for i, j in zip(bounds, ends)]
        reps = []
        for _ in range(nboot):
            pick = rng.integers(0, len(groups), len(groups))  # SAME queries for both
            idx = np.concatenate([groups[i] for i in pick])
            reps.append(auc(y[idx], s1[idx]) - auc(y[idx], s2[idx]))
        out.append(summarize(len(groups), field, "np", pa, pb, a, b, reps))

        # ---- topic --------------------------------------------------------------------
        d = pd.read_parquet(f"{POOLS}/topic_{field}.parquet")
        yt = d.true.values
        p1, p2 = d[a].values, d[b].values
        pa = float(f1_score(yt, p1, average="macro"))
        pb = float(f1_score(yt, p2, average="macro"))
        reps = []
        for _ in range(max(nboot // 4, 50)):
            k = rng.integers(0, len(yt), len(yt))             # SAME papers for both
            reps.append(f1_score(yt[k], p1[k], average="macro")
                        - f1_score(yt[k], p2[k], average="macro"))
        out.append(summarize(len(yt), field, "topic", pa, pb, a, b, reps))

    # ---- S2AND ------------------------------------------------------------------------
    for ds in S2AND:
        d = pd.read_parquet(f"{POOLS}/s2and_{ds}.parquet")
        ga = d[d.method == a_s2and].set_index("signature_id")
        gb = d[d.method == b_s2and].set_index("signature_id")
        sig = ga.index.intersection(gb.index)
        Pa, Ra = ga.loc[sig].b3_p.values, ga.loc[sig].b3_r.values
        Pb, Rb = gb.loc[sig].b3_p.values, gb.loc[sig].b3_r.values

        def f1(P, R, idx):
            mp, mr = P[idx].mean(), R[idx].mean()
            return 0.0 if (mp + mr) == 0 else 2 * mp * mr / (mp + mr)

        full = np.arange(len(sig))
        pa, pb = f1(Pa, Ra, full), f1(Pb, Rb, full)
        reps = []
        for _ in range(nboot):
            k = rng.integers(0, len(sig), len(sig))           # SAME signatures for both
            reps.append(f1(Pa, Ra, k) - f1(Pb, Rb, k))
        out.append(summarize(len(sig), ds, "name_disambig", pa, pb, a_s2and, b_s2and, reps))

    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="genkron", help="method column in the field pools")
    p.add_argument("--b", default="icae_genkron")
    p.add_argument("--a-s2and", default=None, help="method name in the S2AND pools (default: --a)")
    p.add_argument("--b-s2and", default=None)
    p.add_argument("--nboot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/groupc/bench/gcb_headtohead.csv")
    args = p.parse_args()
    df = run(args.a, args.b, args.a_s2and or args.a, args.b_s2and or args.b, args.nboot, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200):
        print(df.round(4).to_string(index=False))
    sig = df[(df.diff_lo > 0) | (df.diff_hi < 0)]
    print(f"\nmean diff over {len(df)} tests: {df['diff'].mean():+.4f}   "
          f"A ahead in {(df['diff'] > 0).sum()}/{len(df)}   "
          f"interval excludes 0 in {len(sig)}/{len(df)} "
          f"(A ahead in {(sig['diff'] > 0).sum()} of those)")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
