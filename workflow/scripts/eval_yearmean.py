"""[GPU] Year-mean "position" mobility test (small/quick).

For each author with >=2 active years, pick one consecutive-active-year boundary and form:
  q_pos = MEAN embedding of all year-t papers   (the author's "position" at t)
  k_pos = MEAN embedding of all year-(t+1) papers
POSITIVE = (q_pos, k_pos) for the same author. NEGATIVE = (q_pos_i, k_pos_j) for random i != j.
Score by cosine; hits@k averaged over positives. Uses EVERY paper per year (no random discard,
no cross-product oversampling). Tests whether yearly-mean position predicts next-year position,
and whether doc2lora genes do this better than text.

Usage: python eval_yearmean.py --field economics --n-authors 1000 --gpu 1

NOT WIRED INTO A RULE: standalone diagnostic, kept for provenance. The reported
numbers come from the chains in workflow/rules/; see REPRODUCE.md.
"""
import argparse
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "workflow/scripts")
from eval_next_paper_ondemand import embed_qwen, embed_st  # reuse on-demand embedders

PREP = "/data/projects/gravity-of-ideas/tmp-data/preprocessed"


def sample_author_years(field, n_authors, seed, min_chars=200):
    rng = np.random.default_rng(seed)
    if field == "aps":
        import scipy.sparse as sp
        ptt = pd.read_csv("/data/datasets/aps/preprocessed/paper_table.csv",
                          usecols=["paper_id", "frac_year"]).dropna()
        yr = dict(zip(ptt["paper_id"].astype(int), ptt["frac_year"].astype(float).astype(int)))
        tx = pd.read_parquet("data/aps/paper_text.parquet",
                             columns=["aps_paper_id", "title", "abstract"]).dropna(subset=["abstract"])
        tx = tx[tx["abstract"].astype(str).str.len() >= min_chars]
        text = {int(p): f"Title: {t}\nAbstract: {a}" for p, t, a in
                zip(tx["aps_paper_id"].astype(int), tx["title"].fillna(""), tx["abstract"])}
        has = set(text)
        d = np.load("/data/datasets/aps/preprocessed/paper_author_net.npz", allow_pickle=True)
        AP = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"])).T.tocsr()
        def papers_of(a):
            return [p for p in AP.indices[AP.indptr[a]:AP.indptr[a + 1]] if p in has]
        author_ids = np.arange(AP.shape[0])
    else:
        src = f"{PREP}/openalex-{field}"
        ptt = pd.read_csv(f"{src}/paper_table.csv", usecols=["paper_id", "frac_year", "title", "abstract"])
        ptt = ptt.dropna(subset=["abstract"])
        ptt = ptt[ptt["abstract"].astype(str).str.len() >= min_chars]
        yr = dict(zip(ptt["paper_id"].astype(int), ptt["frac_year"].astype(float).astype(int)))
        text = {int(p): f"Title: {t}\nAbstract: {a}" for p, t, a in
                zip(ptt["paper_id"].astype(int), ptt["title"].fillna(""), ptt["abstract"])}
        has = set(text)
        ap = pd.read_csv(f"{src}/author_paper_table.csv", usecols=["paper_id", "author_id"]).dropna().astype(np.int64)
        ap = ap[ap["paper_id"].isin(has)]
        grp = ap.groupby("author_id")["paper_id"].apply(list)
        grp = grp[grp.apply(len) >= 2]
        author_ids = grp.index.to_numpy()
        papers_of = lambda a: grp[a]
    rng.shuffle(author_ids)

    samples = []
    for a in author_ids:
        ps = papers_of(a)
        if len(ps) < 2:
            continue
        yrs = np.array([yr.get(p, -1) for p in ps])
        ok = yrs >= 0
        ps = [p for p, o in zip(ps, ok) if o]; yrs = yrs[ok]
        uy = np.unique(yrs)
        if len(uy) < 2:
            continue
        bi = rng.integers(0, len(uy) - 1)
        tps = [int(p) for p, y in zip(ps, yrs) if y == uy[bi]]
        tp1 = [int(p) for p, y in zip(ps, yrs) if y == uy[bi + 1]]
        samples.append((tps, tp1))
        if len(samples) >= n_authors:
            break
    return samples, text


def centroids(samples, emb):
    def mean_unit(ids):
        vs = [emb[i] for i in ids if i in emb]
        if not vs:
            return None
        m = np.mean(vs, axis=0)
        return m / (np.linalg.norm(m) + 1e-9)
    Q, K = [], []
    for tps, tp1 in samples:
        q, k = mean_unit(tps), mean_unit(tp1)
        if q is not None and k is not None:
            Q.append(q); K.append(k)
    return np.array(Q), np.array(K)


def hits(Q, K, seed, ks, n_neg=1000):
    """Rank each positive (q_i.k_i) against a fixed pool of n_neg random cross-pairs q_a.k_b (a!=b)."""
    pos = np.sum(Q * K, axis=1)
    rng = np.random.default_rng(seed)
    n = len(Q)
    a = rng.integers(0, n, size=n_neg); b = rng.integers(0, n, size=n_neg)
    bad = a == b; b[bad] = (b[bad] + 1) % n
    neg = np.sum(Q[a] * K[b], axis=1)
    neg_sorted = np.sort(neg)
    out = {}
    for k in ks:
        ranks = len(neg_sorted) - np.searchsorted(neg_sorted, pos, side="right") + 1
        out[k] = float((ranks <= k).mean())
    return out, n


def main(field, n_authors=1000, gpu=1, seed=0):
    samples, text = sample_author_years(field, n_authors, seed)
    uids = sorted({p for s in samples for grp in s for p in grp})
    print(f"[{field}] {len(samples)} author year-pairs, {len(uids)} unique papers", flush=True)
    utexts = [text[i] for i in uids]
    encs = {"qwen": embed_qwen(utexts, uids, gpu),
            "sbert": embed_st(utexts, uids, "sentence-transformers/all-mpnet-base-v2")}
    ks = [1, 5, 10, 50, 100, 500]
    print(f"\n{field}: year-mean position hits@k (rank vs 1000 random author-pair centroids)")
    print(f"{'model':8}" + "".join(f"{('@'+str(k)):>8}" for k in ks))
    for name, emb in encs.items():
        Q, K = centroids(samples, emb)
        h, n = hits(Q, K, seed, ks)
        print(f"{name:8}" + "".join(f"{h[k]:>8.3f}" for k in ks) + f"   (n={n})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--n-authors", type=int, default=1000)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.field, a.n_authors, a.gpu, a.seed)
