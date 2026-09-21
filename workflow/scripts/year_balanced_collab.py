"""[CPU] Robustness check: YEAR-BALANCED author centroids vs the flat paper-mean.

Flat (default in save_collab_scores): author vec = unweighted mean of ALL past-window papers
  -> prolific years dominate proportionally.
Year-balanced (this script): author vec = mean over YEARS of (mean of that year's papers)
  -> each active year gets equal say; bursts down-weighted.

Reuses the EXACT candidate pairs / labels / windows from data/collab_scores_<field>_hard.parquet,
recomputing only the per-encoder cosine with the year-balanced centroid. Reports AUC/AP per encoder
and the delta vs the flat-mean columns already in the parquet.

Usage: python year_balanced_collab.py --field economics --windows 2008,2012,2016 --dy 3

NOT WIRED INTO A RULE: standalone diagnostic, kept for provenance. The reported
numbers come from the chains in workflow/rules/; see REPRODUCE.md.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, "workflow/scripts")
from bench_data import load

ENC = [("qwen", "qwen_norm_lora_emb.npz", "embeddings"),
       ("gemma", "gemma_norm_lora_emb.npz", "embeddings"),
       ("specter2", "baseline_specter2.npz", "vecs"),
       ("instructor", "baseline_instructor.npz", "vecs"),
       ("sbert", "sbert_allmpnet.npz", "vecs")]


def yb_vectors(au_year_papers, E, prow, authors):
    """author -> L2-normalized year-balanced centroid (mean over years of per-year means)."""
    out = {}
    for a in authors:
        yp = au_year_papers.get(a)
        if not yp:
            continue
        yvecs = []
        for _, paps in yp.items():
            rows = [prow[p] for p in paps if p in prow]
            if rows:
                yvecs.append(E[rows].mean(0))
        if yvecs:
            v = np.mean(yvecs, 0)
            n = np.linalg.norm(v)
            if n > 0:
                out[a] = (v / n).astype(np.float32)
    return out


def auc_ap(y, s):
    m = ~np.isnan(s)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return np.nan, np.nan
    return roc_auc_score(y[m], s[m]), average_precision_score(y[m], s[m])


def main(field, windows, dy=3):
    pt, edges, emb_dir = load(field)
    yr = dict(zip(pt["paper_id"].astype(int), pt["year"]))
    edges = edges.copy(); edges["year"] = edges["paper_id"].map(yr)
    edges = edges.dropna(subset=["year"]); edges["year"] = edges["year"].astype(int)

    dpar = pd.read_parquet(f"data/collab_scores_{field}_hard.parquet").reset_index(drop=True)

    # per-window author -> {year: (paper_ids,)}
    perwin = {}
    for t in windows:
        past = edges[edges["year"].between(t - dy, t)]
        ayp = {}
        for (a, y), g in past.groupby(["author_id", "year"]):
            ayp.setdefault(int(a), {})[int(y)] = tuple(int(x) for x in g["paper_id"])
        perwin[t] = ayp

    yb_cols = {}
    for name, fn, key in ENC:
        fp = os.path.join(emb_dir, fn)
        if not os.path.exists(fp) or name not in dpar.columns:
            continue
        z = np.load(fp, allow_pickle=True)
        E = z[key].astype(np.float32)
        if E.ndim > 2:
            E = E.reshape(E.shape[0], -1)
        prow = {int(p): i for i, p in enumerate(z["paper_ids"])}
        col = np.full(len(dpar), np.nan, np.float32)
        for t in windows:
            sub = dpar[dpar["window"] == t]
            auth = set(sub["a"]) | set(sub["b"])
            vecs = yb_vectors(perwin[t], E, prow, auth)
            ia = sub.index.values
            for k, a, b in zip(ia, sub["a"].values, sub["b"].values):
                va = vecs.get(int(a)); vb = vecs.get(int(b))
                if va is not None and vb is not None:
                    col[k] = float(va @ vb)
        yb_cols[name] = col
        del E, z
        print(f"  scored {name}", flush=True)

    # report flat (parquet column) vs year-balanced, per encoder, per metric
    print(f"\n[{field}] hard negatives — FLAT paper-mean  vs  YEAR-BALANCED  (AUC / AP)")
    print(f"  {'encoder':11}{'AUC_flat':>9}{'AUC_yb':>9}{'dAUC':>8}   {'AP_flat':>8}{'AP_yb':>8}{'dAP':>8}")
    rows = []
    for name in yb_cols:
        af = ap_f = ay = ap_y = []
        AUf, APf, AUy, APy = [], [], [], []
        for t in windows:
            sub = dpar[dpar["window"] == t]
            y = sub["y"].values
            a1, p1 = auc_ap(y, sub[name].values)               # flat (saved)
            a2, p2 = auc_ap(y, yb_cols[name][sub.index.values])  # year-balanced
            if a1 == a1:
                AUf.append(a1); APf.append(p1)
            if a2 == a2:
                AUy.append(a2); APy.append(p2)
        AUf, APf, AUy, APy = (np.mean(x) if x else np.nan for x in (AUf, APf, AUy, APy))
        print(f"  {name:11}{AUf:>9.3f}{AUy:>9.3f}{AUy-AUf:>+8.3f}   {APf:>8.3f}{APy:>8.3f}{APy-APf:>+8.3f}")
        rows.append(dict(field=field, encoder=name, AUC_flat=AUf, AUC_yb=AUy, AP_flat=APf, AP_yb=APy))
    pd.DataFrame(rows).to_csv(f"data/yearbalanced_collab_{field}.csv", index=False)
    print(f"[{field}] -> data/yearbalanced_collab_{field}.csv", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--dy", type=int, default=3)
    a = ap.parse_args()
    main(a.field, [int(x) for x in a.windows.split(",")], a.dy)
