"""[CPU] Build collaboration candidate pairs and SAVE per-pair similarity scores (all encoders +
topology + label) to a parquet, ONCE. Metrics (AUC/AP/hits@k, easy or hard) are then computed from
the file by collab_metrics.py — no need to re-load embeddings / re-run the simulation.

negtype:
  easy = positives (new future edges) vs RANDOM author pairs       (tests 'same topic?' — text wins)
  hard = candidates are distance-2 pairs (share a past collaborator); label = future edge
         (topic-CONTROLLED: among same-area researchers, who collaborates)

Output: data/collab_scores_<field>_<negtype>.parquet with columns
  field, negtype, window, a, b, y, AA, CN, PA, qwen, gemma, mistral, specter2, instructor, sbert

Usage: python save_collab_scores.py --field aps --negtype hard --windows 2000,2004,2008
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "workflow/scripts")
from bench_data import load, collab_net
from eval_collab2 import candidate_pool, author_cos, topo_feats, ENC


def easy_candidates(past_net, fut_net, active, rng, n=2000, ratio=10):
    aset = set(int(a) for a in active)
    fc = fut_net.tocoo()
    pos = [(int(r), int(c)) for r, c in zip(fc.row, fc.col)
           if r < c and r in aset and c in aset and past_net[r, c] == 0]
    rng.shuffle(pos); pos = pos[:n]
    act = np.array(sorted(aset)); neg = []
    nt = ratio * len(pos)
    while len(neg) < nt:
        a, b = int(rng.choice(act)), int(rng.choice(act))
        if a < b and fut_net[a, b] == 0 and past_net[a, b] == 0:
            neg.append((a, b))
    cand = pos + neg
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(int)
    return cand, y


def main(field, negtype, windows, dy=3, n=2000, ratio=10):
    negtypes = ["easy", "hard"] if negtype == "both" else [negtype]
    pt, edges, emb_dir = load(field)
    yr = dict(zip(pt["paper_id"].astype(int), pt["year"]))
    edges = edges.copy(); edges["year"] = edges["paper_id"].map(yr)
    edges = edges.dropna(subset=["year"]); edges["year"] = edges["year"].astype(int)
    n_pap = int(max(pt["paper_id"].max(), edges["paper_id"].max())) + 1
    n_auth = int(edges["author_id"].max()) + 1

    # build candidate sets for all (negtype, window); share au_papers/nets per window
    W = {nt: {} for nt in negtypes}
    for t in windows:
        rng = np.random.default_rng(t)
        past = edges[edges["year"].between(t - dy, t)]; fut = edges[edges["year"].between(t + 1, t + dy)]
        if len(past) < 1000 or len(fut) < 500:
            print(f"  t={t}: sparse, skip"); continue
        past_net = collab_net(past, n_auth, n_pap); fut_net = collab_net(fut, n_auth, n_pap)
        active = np.unique(past["author_id"].values).astype(int)
        deg = np.asarray(past_net.sum(1)).ravel()
        with np.errstate(divide="ignore"):
            invlog = np.where(deg > 1, 1.0 / np.log(deg), 0.0)
        au_papers = past.groupby("author_id")["paper_id"].apply(lambda s: tuple(int(x) for x in s)).to_dict()
        for nt in negtypes:
            cand, y = (candidate_pool(past_net, fut_net, active, np.random.default_rng(t))
                       if nt == "hard" else easy_candidates(past_net, fut_net, active, rng, n, ratio))
            if y.sum() < 20:
                print(f"  t={t}/{nt}: too few positives, skip"); continue
            topo = topo_feats(past_net, cand, deg, invlog)
            W[nt][t] = dict(cand=cand, y=y, topo=topo, au_papers=au_papers)
        print(f"  t={t}: " + " ".join(f"{nt}={len(W[nt].get(t,{}).get('cand',[]))}" for nt in negtypes), flush=True)

    # score each encoder ONCE, fill columns for every (negtype, window)
    colvals = {nt: {t: {} for t in W[nt]} for nt in negtypes}
    for name, fn, key in ENC:
        fp = os.path.join(emb_dir, fn)
        if not os.path.exists(fp):
            continue
        z = np.load(fp, allow_pickle=True)
        E = z[key].astype(np.float32)
        if E.ndim > 2:
            E = E.reshape(E.shape[0], -1)
        prow = {int(p): i for i, p in enumerate(z["paper_ids"])}
        for nt in negtypes:
            for t in W[nt]:
                s, _ = author_cos(W[nt][t]["cand"], W[nt][t]["au_papers"], E, prow)
                colvals[nt][t][name] = s
        del E, z
        print(f"  scored {name}", flush=True)

    for nt in negtypes:
        rows = []
        for t in W[nt]:
            m = W[nt][t]
            for i, (a, b) in enumerate(m["cand"]):
                r = dict(field=field, negtype=nt, window=t, a=a, b=b, y=int(m["y"][i]),
                         AA=float(m["topo"][i, 0]), CN=float(m["topo"][i, 1]), PA=float(m["topo"][i, 2]))
                for name in colvals[nt][t]:
                    r[name] = float(colvals[nt][t][name][i])
                rows.append(r)
        if not rows:
            continue
        out = f"data/collab_scores_{field}_{nt}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"[{field}/{nt}] saved {len(rows)} pairs -> {out}", flush=True)


if __name__ == "__main__":
    if "snakemake" in sys.modules:
        sm = snakemake  # noqa: F821
        main(sm.params["field"], "both",
             [int(x) for x in str(sm.params["windows"]).split(",")],
             int(sm.params.get("dy", 3)))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--field", required=True)
        ap.add_argument("--negtype", default="both", choices=["easy", "hard", "both"])
        ap.add_argument("--windows", required=True)
        ap.add_argument("--dy", type=int, default=3)
        a = ap.parse_args()
        main(a.field, a.negtype, [int(x) for x in a.windows.split(",")], a.dy)
