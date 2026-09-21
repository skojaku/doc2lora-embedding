"""[CPU] RIGOROUS collaboration link prediction (reviewer must/should-haves).

Fixes vs eval_collab.py: (a) topology-MATCHED negatives -- candidate set = distance-2 pairs (share >=1
past collaborator), so topology is a FAIR baseline; (b) NO future leak -- author vector = mean of PAST-window
papers only; (c) memory-safe -- one paper-embedding matrix loaded at a time.

Per field, averaged over MULTIPLE WINDOWS x SEEDS: single-feature AP/P@100 for topology(AA) / genes
(qwen,gemma,mistral) / text (specter2,instructor,sbert); COMBINED logistic-regression test-AP for
topo | topo+text | topo+gene | all -> does doc2lora content ADD over topology (and over text)?
Plus cross-subfield vs within-subfield breakdown (mechanism) when a topic label exists.

Usage: python eval_collab2.py --field psychology --windows 2008,2012,2016 --dy 3 --seeds 3
"""
import argparse
import collections
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "workflow/scripts")
from bench_data import load, collab_net

ENC = [("qwen", "qwen_norm_lora_emb.npz", "embeddings"),
       ("gemma", "gemma_norm_lora_emb.npz", "embeddings"),
       ("mistral", "mistral_norm_lora_emb.npz", "embeddings"),
       ("specter2", "baseline_specter2.npz", "vecs"),
       ("instructor", "baseline_instructor.npz", "vecs"),
       ("sbert", "sbert_allmpnet.npz", "vecs")]
GENES = ["qwen", "gemma", "mistral"]
TEXTS = ["specter2", "instructor", "sbert"]


def candidate_pool(past_net, fut_net, active, rng, max_cand=30000):
    aset = set(int(a) for a in active)
    seen = set(); cand = []
    order = active.copy(); rng.shuffle(order)
    for r in order:
        nbr = past_net.indices[past_net.indptr[r]:past_net.indptr[r + 1]]
        for mid in nbr:
            for c in past_net.indices[past_net.indptr[mid]:past_net.indptr[mid + 1]]:
                if c == r or c not in aset:
                    continue
                a, b = (int(r), int(c)) if r < c else (int(c), int(r))
                if (a, b) in seen or past_net[a, b] != 0:
                    continue
                seen.add((a, b)); cand.append((a, b))
        if len(cand) >= max_cand:
            break
    cand = cand[:max_cand]
    y = np.array([1 if fut_net[a, b] else 0 for a, b in cand], dtype=int)
    return cand, y


def topo_feats(past_net, cand, deg, invlog):
    aa = np.empty(len(cand)); cn = np.empty(len(cand)); pa = np.empty(len(cand))
    for k, (a, b) in enumerate(cand):
        na = past_net.indices[past_net.indptr[a]:past_net.indptr[a + 1]]
        nb = past_net.indices[past_net.indptr[b]:past_net.indptr[b + 1]]
        common = np.intersect1d(na, nb, assume_unique=True)
        cn[k] = len(common); aa[k] = invlog[common].sum(); pa[k] = deg[a] * deg[b]
    return np.column_stack([aa, cn, np.log1p(pa)])


def author_cos(cand, au_papers, paper_emb, prow, keep=None):
    """Cosine between PAST-window author centroids for each candidate pair; nan if uncovered.
    keep: optional set of paper_ids that count as 'embedded' (to simulate reduced coverage)."""
    au = sorted({a for p in cand for a in p})
    arow = {a: i for i, a in enumerate(au)}
    rows, cols = [], []
    for a in au:
        for p in au_papers.get(a, ()):
            if keep is not None and p not in keep:
                continue
            j = prow.get(p)
            if j is not None:
                rows.append(arow[a]); cols.append(j)
    a2p = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(au), paper_emb.shape[0]))
    rs = np.maximum(1, np.asarray(a2p.sum(1)).ravel())
    AE = (sp.diags(1.0 / rs) @ a2p) @ paper_emb
    nrm = np.linalg.norm(AE, axis=1, keepdims=True)
    cov_au = (np.asarray(a2p.sum(1)).ravel() > 0)
    AE = AE / (nrm + 1e-9)
    out = np.full(len(cand), np.nan, np.float32)
    cov = np.zeros(len(cand), bool)
    for k, (a, b) in enumerate(cand):
        ia, ib = arow[a], arow[b]
        if cov_au[ia] and cov_au[ib]:
            out[k] = float(AE[ia] @ AE[ib]); cov[k] = True
    return out, cov


def pk(y, s, k=100):
    return float(y[np.argsort(-s)[:k]].mean())


def main(field, windows, dy=3, seeds=3, out=None):
    out = out or f"data/collab2_{field}.csv"
    pt, edges, emb_dir = load(field)
    yr = dict(zip(pt["paper_id"].astype(int), pt["year"]))
    edges = edges.copy(); edges["year"] = edges["paper_id"].map(yr)
    edges = edges.dropna(subset=["year"]); edges["year"] = edges["year"].astype(int)
    n_pap = int(max(pt["paper_id"].max(), edges["paper_id"].max())) + 1
    n_auth = int(edges["author_id"].max()) + 1

    # ---- Phase 1: per window — nets, candidates, labels, topology, past author->papers ----
    W = {}
    for t in windows:
        rng = np.random.default_rng(t)
        past = edges[edges["year"].between(t - dy, t)]; fut = edges[edges["year"].between(t + 1, t + dy)]
        if len(past) < 1000 or len(fut) < 500:
            print(f"  t={t}: sparse, skip"); continue
        past_net = collab_net(past, n_auth, n_pap); fut_net = collab_net(fut, n_auth, n_pap)
        active = np.unique(past["author_id"].values).astype(int)
        cand, y = candidate_pool(past_net, fut_net, active, rng)
        if y.sum() < 20:
            print(f"  t={t}: too few positives ({y.sum()}), skip"); continue
        deg = np.asarray(past_net.sum(1)).ravel()
        with np.errstate(divide="ignore"):
            invlog = np.where(deg > 1, 1.0 / np.log(deg), 0.0)
        X_topo = topo_feats(past_net, cand, deg, invlog)
        au_papers = past.groupby("author_id")["paper_id"].apply(lambda s: tuple(int(x) for x in s)).to_dict()
        W[t] = dict(cand=cand, y=y, X_topo=X_topo, au_papers=au_papers)
        print(f"  t={t}: {len(cand)} candidates, {int(y.sum())} positives (prev={y.mean():.3f})", flush=True)
    if not W:
        print(f"[{field}] no usable windows"); return

    # ---- Phase 2: one encoder at a time — cosine feature per candidate (past-only centroids) ----
    feats = {t: {} for t in W}; cov = {t: np.ones(len(W[t]["cand"]), bool) for t in W}
    present = []
    for name, fn, key in ENC:
        fp = os.path.join(emb_dir, fn)
        if not os.path.exists(fp):
            continue
        present.append(name)
        z = np.load(fp, allow_pickle=True)
        E = z[key].astype(np.float32)
        if E.ndim > 2:
            E = E.reshape(E.shape[0], -1)
        prow = {int(p): i for i, p in enumerate(z["paper_ids"])}
        for t in W:
            s, c = author_cos(W[t]["cand"], W[t]["au_papers"], E, prow)
            feats[t][name] = s; cov[t] &= c
        del E, z
    have_g = [g for g in GENES if g in present]; have_t = [tt for tt in TEXTS if tt in present]
    print(f"[{field}] encoders={present}", flush=True)

    # ---- Phase 3: model + metrics over windows x seeds ----
    agg = collections.defaultdict(list)
    for t in W:
        m = cov[t]
        y = W[t]["y"][m]; Xtopo = W[t]["X_topo"][m]
        fe = {nm: feats[t][nm][m] for nm in present}
        if y.sum() < 10:
            continue
        for si in range(seeds):
            rng = np.random.default_rng(10 * t + si)
            idx = rng.permutation(len(y)); ntr = len(y) // 2; tr, te = idx[:ntr], idx[ntr:]
            for nm in have_g + have_t:
                agg[("AP", nm)].append(average_precision_score(y[te], fe[nm][te]))
                agg[("P100", nm)].append(pk(y[te], fe[nm][te]))
            agg[("AP", "AA")].append(average_precision_score(y[te], Xtopo[te, 0]))
            agg[("P100", "AA")].append(pk(y[te], Xtopo[te, 0]))

            def lr(cols):
                Xtr = np.column_stack(cols)
                sc = StandardScaler().fit(Xtr[tr])
                mdl = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr[tr]), y[tr])
                p = mdl.predict_proba(sc.transform(Xtr[te]))[:, 1]
                return average_precision_score(y[te], p), pk(y[te], p)
            tcols = [fe[tt] for tt in have_t]; gcols = [fe[g] for g in have_g]
            for lbl, cols in [("topo", [Xtopo]), ("topo+text", [Xtopo] + tcols),
                              ("topo+gene", [Xtopo] + gcols), ("topo+text+gene", [Xtopo] + tcols + gcols)]:
                ap, p100 = lr([c if c.ndim > 1 else c.reshape(-1, 1) for c in cols])
                agg[("AP", lbl)].append(ap); agg[("P100", lbl)].append(p100)
            agg[("prev", "p")].append(y.mean())

    pv = np.mean(agg[("prev", "p")])
    print(f"\n[{field}] RIGOROUS collab (distance-2 matched, no-leak, {len(W)}win x {seeds}seed, prev={pv:.3f})")
    print(f"  {'feature/model':18}{'AP mean±sd':>15}{'P@100':>9}")
    rows = []
    for nm in ["AA"] + have_t + have_g + ["topo", "topo+text", "topo+gene", "topo+text+gene"]:
        ap = np.array(agg[("AP", nm)]); p = np.array(agg[("P100", nm)])
        if len(ap) == 0:
            continue
        print(f"  {nm:18}{f'{ap.mean():.3f}±{ap.std(ddof=1) if len(ap)>1 else 0:.3f}':>15}{p.mean():>9.3f}")
        rows.append(dict(field=field, model=nm, AP_mean=ap.mean(),
                         AP_sd=ap.std(ddof=1) if len(ap) > 1 else 0, P100_mean=p.mean(), n=len(ap)))

    def M(k): a = np.array(agg[("AP", k)]); return a.mean() if len(a) else float("nan")
    print(f"\n  ΔAP(topo+gene − topo)        = {M('topo+gene') - M('topo'):+.3f}")
    print(f"  ΔAP(topo+text − topo)        = {M('topo+text') - M('topo'):+.3f}")
    print(f"  ΔAP(all − topo+text)         = {M('topo+text+gene') - M('topo+text'):+.3f}  <- gene adds over topo+text?")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[{field}] -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--windows", default="2008,2012,2016")
    ap.add_argument("--dy", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(a.field, [int(x) for x in a.windows.split(",")], a.dy, a.seeds)
