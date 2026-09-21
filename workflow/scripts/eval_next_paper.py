"""[CPU] Classification-free next-paper-prediction benchmark (apples-to-apples across domains).

For each author, order papers by active year; a (paper@t, paper@t+1) pair is a POSITIVE.
Negatives = (same query paper, a random paper the author did NOT write). Score each pair by
cosine similarity in an encoder's space; an encoder is good if true next-papers rank above
random ones (hits@k). No topic codes anywhere → directly comparable across APS / arXiv / OpenAlex.

Compares doc2lora genes (qwen/gemma/mistral) vs SBERT / SPECTER2 / INSTRUCTOR / text2vec on the
SAME pairs (restricted to papers that have embeddings).

Usage: python eval_next_paper.py --field arxiv_math --n-authors 20000 --n-neg 20000

NOT WIRED INTO A RULE: standalone diagnostic, kept for provenance. The reported
numbers come from the chains in workflow/rules/; see REPRODUCE.md.
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import sparse
from numba import njit

PREP = "/data/projects/gravity-of-ideas/tmp-data/preprocessed"


# ---------- per-field data locations ----------
def field_paths(field):
    if field == "aps":
        return dict(emb="data/aps/embeddings", idcol="aps_paper_id",
                    net="/data/datasets/aps/preprocessed/paper_author_net.npz",
                    ptable="/data/datasets/aps/preprocessed/paper_table.csv",
                    encoders={"qwen": "qwen_norm_lora_emb.npz", "gemma": "gemma_norm_lora_emb.npz",
                              "mistral": "mistral_norm_lora_emb.npz", "sbert": "sbert_allmpnet.npz"})
    if field.startswith("arxiv_"):
        b = f"data/fields/{field}"
        return dict(emb=f"{b}/embeddings", idcol="paper_id",
                    ap=f"{b}/author_paper_table.csv", ptable=f"{b}/paper_table.csv",
                    encoders={"qwen": "qwen_norm_lora_emb.npz", "gemma": "gemma_norm_lora_emb.npz",
                              "mistral": "mistral_norm_lora_emb.npz", "sbert": "sbert_allmpnet.npz",
                              "specter2": "baseline_specter2.npz", "instructor": "baseline_instructor.npz",
                              "text2vec": "baseline_text2vec.npz"})
    b = f"data/fields/{field}"  # openalex econ/psych/chem
    return dict(emb=f"{b}/embeddings", idcol="paper_id",
                ap=f"{PREP}/openalex-{field}/author_paper_table.csv",
                ptable=f"{PREP}/openalex-{field}/paper_table.csv",
                encoders={"qwen": "qwen_norm_lora_emb.npz", "gemma": "gemma_norm_lora_emb.npz",
                          "mistral": "mistral_norm_lora_emb.npz", "sbert": "sbert_allmpnet.npz",
                          "specter2": "baseline_specter2.npz", "instructor": "baseline_instructor.npz",
                          "text2vec": "baseline_text2vec.npz"})


def author2paper_from_net(net_file):
    d = np.load(net_file, allow_pickle=True)
    PA = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    return PA.T.tocsr()  # authors x papers


def author2paper_from_table(ap_file):
    t = pd.read_csv(ap_file, usecols=["paper_id", "author_id"]).dropna().astype(np.int64)
    r, c = t["author_id"].values, t["paper_id"].values
    return sparse.csr_matrix((np.ones(len(r), np.int8), (r, c)),
                             shape=(r.max() + 1, c.max() + 1))


@njit
def _yb(sy):
    if len(sy) == 0:
        return np.array([0], np.int64)
    ch = np.where(np.diff(sy) != 0)[0] + 1
    b = np.empty(len(ch) + 2, np.int64); b[0] = 0
    if len(ch) > 0:
        b[1:-1] = ch
    b[-1] = len(sy); return b


@njit
def _pairs_for_author(a, indptr, indices, years, has_emb):
    s, e = indptr[a], indptr[a + 1]
    ap = indices[s:e]
    ap = ap[has_emb[ap]]                    # only papers with embeddings
    if len(ap) < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    si = np.argsort(years[ap]); sp = ap[si]; sy = years[sp]
    b = _yb(sy); ny = len(b) - 1
    if ny <= 1:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    q, k = [], []
    for yi in range(1, ny):
        prev = sp[b[yi - 1]:b[yi]]; cur = sp[b[yi]:b[yi + 1]]
        for p in prev:
            for c in cur:
                q.append(p); k.append(c)
    return np.array(q, np.int64), np.array(k, np.int64)


def gen_pairs(a2p, years, has_emb, n_authors, seed, max_pairs=40000):
    rng = np.random.default_rng(seed)
    cnt = np.diff(a2p.indptr)
    elig = np.where(cnt > 1)[0]
    samp = rng.choice(elig, size=min(n_authors, len(elig)), replace=False)
    Q, K, A = [], [], []
    for a in samp:
        q, k = _pairs_for_author(a, a2p.indptr, a2p.indices, years, has_emb)
        if len(q):
            Q.append(q); K.append(k); A.append(np.full(len(q), a, np.int64))
    Q = np.concatenate(Q); K = np.concatenate(K); A = np.concatenate(A)
    if len(Q) > max_pairs:
        idx = rng.choice(len(Q), max_pairs, replace=False)
        Q, K, A = Q[idx], K[idx], A[idx]
    return Q, K, A


def count_top_k_hits(pos, neg, k):
    neg = np.sort(neg)
    higher = len(neg) - np.searchsorted(neg, pos, side="right")
    return float((higher + 1 <= k).mean())


class EDB:
    def __init__(self, emb, ids):
        emb = emb.reshape(emb.shape[0], -1).astype(np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        self.emb = emb
        self.row = {int(p): i for i, p in enumerate(ids)}
    def has(self, pid):
        return pid in self.row
    def score(self, qids, kids):
        ri = np.fromiter((self.row.get(int(q), -1) for q in qids), np.int64, len(qids))
        rj = np.fromiter((self.row.get(int(k), -1) for k in kids), np.int64, len(kids))
        valid = (ri >= 0) & (rj >= 0)
        out = np.full(len(qids), np.nan, np.float32)
        out[valid] = np.sum(self.emb[ri[valid]] * self.emb[rj[valid]], axis=1)
        return out


def main(field, n_authors=20000, n_neg=20000, seed=0, out=None, cov_cap=0):
    out = out or f"data/next_paper_{field}.csv"
    P = field_paths(field)
    # years indexed by paper_id
    pt = pd.read_csv(P["ptable"], usecols=["paper_id", "frac_year"]).dropna()
    nmax = int(pt["paper_id"].max()) + 1
    years = np.full(nmax, 1 << 30, np.int64)
    years[pt["paper_id"].astype(int).values] = pt["frac_year"].astype(float).astype(int).values
    a2p = author2paper_from_net(P["net"]) if "net" in P else author2paper_from_table(P["ap"])
    if a2p.shape[1] < nmax:  # pad columns
        a2p = sparse.hstack([a2p, sparse.csr_matrix((a2p.shape[0], nmax - a2p.shape[1]))]).tocsr()

    # embedding-coverage mask (use qwen genes as the reference coverage set; intersect all later)
    enc = {}
    for name, fn in P["encoders"].items():
        fp = os.path.join(P["emb"], fn)
        if not os.path.exists(fp):
            print(f"  [skip] {name} ({fn} missing)"); continue
        d = np.load(fp, allow_pickle=True)
        ek = "embeddings" if "embeddings" in d.files else "vecs"
        enc[name] = EDB(d[ek], d["paper_ids"].astype(int))
    print(f"[{field}] encoders: {list(enc)}")
    # coverage = papers embedded by ALL encoders (fair comparison on identical pairs)
    cov = np.zeros(nmax, bool)
    common_ids = None
    for e in enc.values():
        ids = set(e.row)
        common_ids = ids if common_ids is None else (common_ids & ids)
    for p in common_ids:
        if p < nmax:
            cov[p] = True
    if cov_cap and cov.sum() > cov_cap:   # downsample coverage for fair cross-field comparison
        on = np.where(cov)[0]
        keep = np.random.default_rng(seed + 7).choice(on, size=cov_cap, replace=False)
        cov = np.zeros(nmax, bool); cov[keep] = True
        print(f"[{field}] coverage downsampled to {cov_cap:,}")
    print(f"[{field}] papers embedded by all {len(enc)} encoders: {cov.sum():,}")

    Q, K, A = gen_pairs(a2p, years, cov, n_authors, seed)
    print(f"[{field}] positive pairs: {len(Q):,}")
    # negatives: same queries, random embedded paper not by the author
    rng = np.random.default_rng(seed + 1)
    emb_papers = np.where(cov)[0]
    qn = Q[rng.integers(0, len(Q), size=n_neg)]
    kn = rng.choice(emb_papers, size=n_neg, replace=True)

    rows = []
    ks = sorted(set(int(x) for x in np.logspace(0, np.log10(max(2, n_neg // 2)), 18)))
    for name, e in enc.items():
        ps = e.score(Q, K); ns = e.score(qn, kn)
        ps = ps[~np.isnan(ps)]; ns = ns[~np.isnan(ns)]
        for k in ks:
            rows.append({"field": field, "model": name, "k": k,
                         "hit": count_top_k_hits(ps, ns, k)})
        h1 = count_top_k_hits(ps, ns, 1); h10 = count_top_k_hits(ps, ns, 10)
        print(f"  {name:10} hits@1={h1:.3f} hits@10={h10:.3f}  (n_pos={len(ps)})")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[{field}] -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--n-authors", type=int, default=20000)
    ap.add_argument("--n-neg", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cov-cap", type=int, default=0, help="downsample coverage to N papers (fairness)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.field, a.n_authors, a.n_neg, a.seed, a.out, a.cov_cap)
