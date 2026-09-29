"""[GPU] Dump per-signature B^3 (precision, recall) for each method on an S2AND dataset, so name-
disambiguation uncertainty can be bootstrapped (over signatures, or over name-blocks).

Methods: gene, gene_kron (same-author adapter REG=10), gene_genkron (general OpenAlex adapter), specter,
sbert, instructor. Same test blocks / tuned thresholds as the reported eval.

Usage: CUDA_VISIBLE_DEVICES=0 python score_pool_s2and.py qian qwen
Output: data/uncertainty/pools/s2and_<ds>_<enc>.parquet  [block, signature_id, method, b3_p, b3_r]
"""
import sys, os
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, "data/kron"); sys.path.insert(0, "data/s2and")
from kron import KronAdapter, info_nce
from s2lib import b3, b3_per_sig

DS = sys.argv[1]; ENC = sys.argv[2]
base = "data/s2and"; dev = "cuda" if torch.cuda.is_available() else "cpu"
from bench_data import out_dir   # bench_data.py sits next to this file
OUT = str(out_dir("uncertainty") / "pools"); os.makedirs(OUT, exist_ok=True)
st = pd.read_parquet(f"{base}/proc/{DS}/sig_table.parquet"); st["paper_id"] = st.paper_id.astype(str)


def load_npz(path, key):
    z = np.load(path, allow_pickle=True)
    ids = np.array([str(p) for p in z["paper_ids"]]); V = z[key].astype(np.float32)
    return {p: V[i] for i, p in enumerate(ids)}, V.shape[1]


# Every method below is part of the reported matrix, so a missing npz must stop the run rather than
# quietly yield a pool with fewer methods. ALLOW_MISSING_METHODS=1 restores best-effort behaviour for
# exploratory runs.
WANT = [("gene", f"genes_{ENC}.npz", "embeddings"), ("specter", "specter.npz", "vecs"),
        ("sbert", "sbert.npz", "vecs"), ("instructor", "instructor.npz", "vecs"),
        ("embeddinggemma", "embeddinggemma.npz", "vecs"), ("gte", "gte.npz", "vecs"),
        ("gene_genkron", f"genes_{ENC}_genkron.npz", "embeddings")]
missing = [(n, f"{base}/proc/{DS}/{fn}") for n, fn, _ in WANT
           if not os.path.exists(f"{base}/proc/{DS}/{fn}")]
if missing:
    msg = (f"[{DS}/{ENC}] {len(missing)} required embedding(s) missing under {base}/proc/{DS}:\n"
           + "\n".join(f"    {n:16s} {p}" for n, p in missing)
           + "\n  Scoring would silently drop these methods from the pool. Build them first, "
             "or set ALLOW_MISSING_METHODS=1 to proceed anyway (exploratory runs only).")
    if os.environ.get("ALLOW_MISSING_METHODS") != "1":
        sys.exit(msg)
    print(f"WARNING: {msg}", flush=True)
EMB = {n: load_npz(f"{base}/proc/{DS}/{fn}", k) for n, fn, k in WANT
       if os.path.exists(f"{base}/proc/{DS}/{fn}")}

have = set.intersection(*[set(d) for d, _ in EMB.values()])
st = st[st.paper_id.isin(have)].reset_index(drop=True)
blocks = sorted(st.block.unique()); rng = np.random.default_rng(42); rng.shuffle(blocks)
ntr = int(0.6 * len(blocks)); train_bl = set(blocks[:ntr]); test_bl = set(blocks[ntr:])
tr = st[st.block.isin(train_bl)]; te = st[st.block.isin(test_bl)]


def mat(df, emb):
    d, _ = emb
    return np.vstack([d[p] for p in df.paper_id.values]).astype(np.float32)


def cluster(X, thr):
    if len(X) == 1:
        return np.array([0])
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    D = np.clip(1.0 - Xn @ Xn.T, 0.0, 2.0); np.fill_diagonal(D, 0.0)
    return AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                   distance_threshold=thr).fit_predict(D)


def clusters(df, emb, thr):
    true_c, pred_c = {}, {}
    for bl, g in df.groupby("block"):
        lab = cluster(mat(g, emb), thr); sids = g.signature_id.values
        for s, c in zip(sids, g.cluster_id.values):
            true_c.setdefault(c, []).append(s)
        for s, c in zip(sids, lab):
            pred_c.setdefault(f"{bl}__{c}", []).append(s)
    return true_c, pred_c


def tune(df, emb):
    best = (-1, 0.5)
    for thr in np.linspace(0.05, 0.95, 19):
        tc, pc = clusters(df, emb, thr); f = b3(tc, pc)[2]
        if f > best[0]:
            best = (f, thr)
    return best[1]


def train_genekron():
    d, D = EMB["gene"]; L = D // 512
    pos = []
    for _, g in tr.groupby("cluster_id"):
        ps = g.paper_id.values
        if len(ps) >= 2:
            for i in range(len(ps)):
                pos.append((ps[i], ps[(i + 1) % len(ps)]))
    if len(pos) < 50:
        return None
    A = np.vstack([d[a] for a, _ in pos]); B = np.vstack([d[b] for _, b in pos])
    m = KronAdapter(L=L, d=512).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    At = torch.tensor(A, device=dev); Bt = torch.tensor(B, device=dev); r2 = np.random.default_rng(0)
    bs = min(256, len(pos))
    for _ in range(2000):
        sel = r2.integers(0, len(pos), bs)
        za = m(At[sel].view(bs, L, 512)); zp = m(Bt[sel].view(bs, L, 512))
        idp = m.s.pow(2).mean() + m.alpha.pow(2).mean() + m.P.pow(2).mean()
        loss = info_nce(za, zp, 0.05) + 10.0 * idp
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    out = {}
    with torch.no_grad():
        ks = list(d)
        for i in range(0, len(ks), 4096):
            ch = ks[i:i + 4096]
            Z = m(torch.tensor(np.vstack([d[k] for k in ch]), device=dev).view(len(ch), L, 512)).cpu().numpy()
            for k, z in zip(ch, Z):
                out[k] = z
    return out, D


gk2 = train_genekron()
if gk2 is not None:
    EMB["gene_kron"] = gk2

sig2blk = dict(zip(st.signature_id, st.block))
rows = []
for name in ["gene", "gene_kron", "gene_genkron", "specter", "sbert", "instructor", "embeddinggemma", "gte"]:
    if name not in EMB:
        continue
    thr = tune(tr, EMB[name])
    tc, pc = clusters(te, EMB[name], thr)
    P, R, F, per = b3_per_sig(tc, pc)
    for sid, (p, r) in per.items():
        rows.append({"dataset": DS, "enc": ENC, "method": name, "block": sig2blk.get(sid),
                     "signature_id": sid, "b3_p": p, "b3_r": r})
    print(f"  {name:13} thr={thr:.2f} B3 F1={F:.3f} (n_sig={len(per)})", flush=True)
pd.DataFrame(rows).to_parquet(f"{OUT}/s2and_{DS}_{ENC}.parquet")
print(f"[DONE] {DS}/{ENC} -> {OUT}/s2and_{DS}_{ENC}.parquet", flush=True)
