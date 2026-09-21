"""[CPU, instant] Compute AUC / AP / hits@k (and combined-model AP) from saved collaboration score
files (data/collab_scores_<field>_<negtype>.parquet). No re-running the simulation.

Usage: python collab_metrics.py --field aps                 # both easy + hard
       python collab_metrics.py --field aps --ks 10,100,500
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ENCS = ["qwen", "gemma", "mistral", "specter2", "instructor", "sbert"]
TOPO = ["AA", "CN", "PA"]
GENES = ["qwen", "gemma", "mistral"]
TEXTS = ["specter2", "instructor", "sbert"]


def hits_at(y, s, k):
    return float(y[np.argsort(-s)[:k]].mean())


def single(d, feat, ks):
    au, ap, hk = [], [], {k: [] for k in ks}
    for _, g in d.groupby("window"):
        gg = g.dropna(subset=[feat])
        if gg.y.nunique() < 2:
            continue
        au.append(roc_auc_score(gg.y, gg[feat])); ap.append(average_precision_score(gg.y, gg[feat]))
        for k in ks:
            hk[k].append(hits_at(gg.y.values, gg[feat].values, k))
    return (np.mean(au) if au else np.nan, np.mean(ap) if ap else np.nan,
            {k: (np.mean(v) if v else np.nan) for k, v in hk.items()})


def combined(d, cols, seeds=3):
    aps = []
    for _, g in d.groupby("window"):
        gg = g.dropna(subset=cols)
        if gg.y.nunique() < 2 or len(gg) < 50:
            continue
        X = gg[cols].values.astype(float); y = gg.y.values
        for si in range(seeds):
            rng = np.random.default_rng(si)
            idx = rng.permutation(len(y)); tr, te = idx[:len(y) // 2], idx[len(y) // 2:]
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            aps.append(average_precision_score(y[te], m.predict_proba(sc.transform(X[te]))[:, 1]))
    return np.mean(aps) if aps else np.nan


def report(path, ks):
    d = pd.read_parquet(path)
    field = d.field.iloc[0]; neg = d.negtype.iloc[0]
    present = [e for e in ENCS if e in d.columns and d[e].notna().any()]
    hg = [g for g in GENES if g in present]; ht = [t for t in TEXTS if t in present]
    print(f"\n===== {field} / {neg} negatives  (prev={d.y.mean():.3f}, {d.window.nunique()} windows) =====")
    print(f"  {'feature':12}{'AUC':>7}{'AP':>7}" + "".join(f"{('h@'+str(k)):>8}" for k in ks))
    for feat in present + ["AA"]:
        au, ap, hk = single(d, feat, ks)
        print(f"  {feat:12}{au:>7.3f}{ap:>7.3f}" + "".join(f"{hk[k]:>8.3f}" for k in ks))
    # combined (AP, CV)
    if hg or ht:
        cg = combined(d, TOPO + hg); ct = combined(d, TOPO + ht)
        c0 = combined(d, TOPO); ca = combined(d, TOPO + ht + hg)
        print(f"  [combined AP]  topo={c0:.3f}  topo+text={ct:.3f}  topo+gene={cg:.3f}  all={ca:.3f}"
              f"   | ΔAP(gene over topo+text)={ca - ct:+.3f}")


def main(field, ks):
    import os
    for neg in ["easy", "hard"]:
        p = f"data/collab_scores_{field}_{neg}.parquet"
        if os.path.exists(p):
            report(p, ks)
        else:
            print(f"\n[{field}/{neg}] no score file ({p})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--ks", default="10,100,500")
    a = ap.parse_args()
    main(a.field, [int(x) for x in a.ks.split(",")])
