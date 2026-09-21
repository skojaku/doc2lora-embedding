"""[GPU] Author-name-disambiguation on an S2AND sub-dataset, embedding-isolation protocol.

Per signature embedding = its paper's vector. Blocks (normalized names) split train/test.
For each embedding {gene, gene_kron, specter, (sbert/instructor if present)}:
  - tune a cosine-distance threshold on TRAIN blocks (agglomerative avg-linkage) to max B^3 F1,
  - cluster TEST blocks at that threshold, report pooled B^3 P/R/F1.
gene_kron = gene passed through a Kronecker adapter trained with SAME-AUTHOR in-batch InfoNCE on TRAIN.

Usage: CUDA_VISIBLE_DEVICES=0 python and_eval.py zbmath
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, "data/kron")
sys.path.insert(0, "data/s2and")
from kron import KronAdapter, info_nce
from s2lib import b3

DS = sys.argv[1] if len(sys.argv) > 1 else "zbmath"
ENC = sys.argv[2] if len(sys.argv) > 2 else "gemma"
base = "data/s2and"; dev = "cuda"
proc = os.environ.get("BENCH_S2_PROC", f"{base}/proc")   # read sliced subset from data/bench/s2and
st = pd.read_parquet(f"{proc}/{DS}/sig_table.parquet")
st["paper_id"] = st.paper_id.astype(str)


def load_npz(path, key):
    z = np.load(path, allow_pickle=True)
    ids = np.array([str(p) for p in z["paper_ids"]])
    V = z[key].astype(np.float32)
    return {p: V[i] for i, p in enumerate(ids)}, V.shape[1]


EMB = {}
EMB["gene"] = load_npz(f"{proc}/{DS}/genes_{ENC}.npz", "embeddings")
EMB["specter"] = load_npz(f"{proc}/{DS}/specter.npz", "vecs")
for opt in ["sbert", "instructor", "embeddinggemma", "icae"]:
    p = f"{proc}/{DS}/{opt}.npz"
    if os.path.exists(p):
        EMB[opt] = load_npz(p, "vecs")
gkp = f"{proc}/{DS}/genes_{ENC}_genkron.npz"   # GENERAL OpenAlex adapter applied to S2AND genes
if os.path.exists(gkp):
    EMB["gene_genkron"] = load_npz(gkp, "embeddings")
igk = f"{proc}/{DS}/icae_genkron.npz"          # invertible per-token ICAE adapter applied to ICAE embeddings
if os.path.exists(igk):
    EMB["icae_genkron"] = load_npz(igk, "vecs")

# keep signatures whose paper has ALL embeddings (fair, identical signature set)
have = set.intersection(*[set(d.keys()) for d, _ in EMB.values()])
st = st[st.paper_id.isin(have)].reset_index(drop=True)
print(f"[{DS}] {len(st)} signatures on {st.paper_id.nunique()} papers (all-embedding coverage); "
      f"blocks={st.block.nunique()} clusters={st.cluster_id.nunique()}", flush=True)

# block-level train/test split
blocks = sorted(st.block.unique())
rng = np.random.default_rng(42); rng.shuffle(blocks)
ntr = int(0.6 * len(blocks)); train_bl = set(blocks[:ntr]); test_bl = set(blocks[ntr:])
tr = st[st.block.isin(train_bl)]; te = st[st.block.isin(test_bl)]
print(f"  train blocks={len(train_bl)} ({len(tr)} sigs) | test blocks={len(test_bl)} ({len(te)} sigs)", flush=True)


def mat_for(df, emb):
    d, _ = emb
    return np.vstack([d[p] for p in df.paper_id.values]).astype(np.float32)


def cluster_block(X, thr):
    if len(X) == 1:
        return np.array([0])
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)   # zero rows stay ~zero
    D = np.clip(1.0 - Xn @ Xn.T, 0.0, 2.0)                       # cosine distance (zero vec -> dist 1)
    np.fill_diagonal(D, 0.0)
    return AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                   distance_threshold=thr).fit_predict(D)


def eval_blocks(df, emb, thr):
    true_c, pred_c = {}, {}
    for bl, g in df.groupby("block"):
        X = mat_for(g, emb); lab = cluster_block(X, thr)
        sids = g.signature_id.values
        for s, c in zip(sids, g.cluster_id.values):
            true_c.setdefault(c, []).append(s)
        for s, c in zip(sids, lab):
            pred_c.setdefault(f"{bl}__{c}", []).append(s)
    return b3(true_c, pred_c)


def tune_thr(df, emb):
    best = (-1, 0.5)
    for thr in np.linspace(0.05, 0.95, 19):
        _, _, f = eval_blocks(df, emb, thr)
        if f > best[0]:
            best = (f, thr)
    return best[1]


def train_kron(tr_df, emb, latent):
    """Same-author in-batch InfoNCE on TRAIN signatures, on ANY embedding.
    latent = factor size: 512 for genes (L=D//512); =D for flat text (L=1)."""
    d, D = emb; L = D // latent
    pos = []
    for cid, g in tr_df.groupby("cluster_id"):
        ps = g.paper_id.values
        if len(ps) >= 2:
            for i in range(len(ps)):
                pos.append((ps[i], ps[(i + 1) % len(ps)]))
    if len(pos) < 50:
        return None
    A = np.vstack([d[a] for a, _ in pos]).astype(np.float32)
    B = np.vstack([d[b] for _, b in pos]).astype(np.float32)
    steps = int(os.environ.get("KRON_STEPS", "2000"))
    reg = float(os.environ.get("KRON_REG", "10"))      # identity-prior strength (keeps map near isometry; 10 cures overfit)
    model = KronAdapter(L=L, d=latent).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    At = torch.tensor(A, device=dev); Bt = torch.tensor(B, device=dev)
    rng2 = np.random.default_rng(0); bs = min(256, len(pos))
    for step in range(steps):
        sel = rng2.integers(0, len(pos), bs)
        za = model(At[sel].view(bs, L, latent)); zp = model(Bt[sel].view(bs, L, latent))
        idprior = model.s.pow(2).mean() + model.alpha.pow(2).mean() + model.P.pow(2).mean()
        loss = info_nce(za, zp, 0.05) + reg * idprior   # pull s,alpha->0 (Λ,A->I) and P->0 (Q->I)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, L, latent


def apply_kron(model_L, emb):
    model, L, latent = model_L; d, D = emb
    out = {}
    with torch.no_grad():
        ks = list(d.keys())
        for i in range(0, len(ks), 4096):
            chunk = ks[i:i + 4096]
            X = torch.tensor(np.vstack([d[k] for k in chunk]), device=dev).view(len(chunk), L, latent)
            Z = model(X).cpu().numpy()
            for k, z in zip(chunk, Z):
                out[k] = z
    return out, D


# fairness control: train the SAME same-author adapter on EVERY embedding (genes + text baselines)
t0 = time.time()
for bname in ["gene", "specter", "sbert", "instructor", "embeddinggemma", "icae"]:
    if bname not in EMB:
        continue
    latent = 512 if bname == "gene" else EMB[bname][1]   # genes: [L,512]; flat text: [1,D]
    mk = train_kron(tr, EMB[bname], latent)
    if mk is not None:
        EMB[f"{bname}_kron"] = apply_kron(mk, EMB[bname])
print(f"  trained same-author adapters for {[k for k in EMB if k.endswith('_kron')]} ({time.time()-t0:.0f}s)", flush=True)

rows = []
order = ["gene", "gene_kron", "gene_genkron", "specter", "specter_kron", "sbert", "sbert_kron",
         "instructor", "instructor_kron", "embeddinggemma", "embeddinggemma_kron",
         "icae", "icae_kron", "icae_genkron"]
for name in order:
    if name not in EMB:
        continue
    thr = tune_thr(tr, EMB[name])
    P, R, F = eval_blocks(te, EMB[name], thr)
    rows.append(dict(model=name, thr=round(thr, 3), B3_P=round(P, 3), B3_R=round(R, 3), B3_F1=round(F, 3)))
    print(f"  {name:11} thr={thr:.2f}  B3  P={P:.3f} R={R:.3f} F1={F:.3f}", flush=True)

df = pd.DataFrame(rows)
print(f"\n===== S2AND author disambiguation B^3 ({DS}, gene={ENC}, test blocks) =====")
print(df.to_string(index=False))
suf = os.environ.get("OUT_SUFFIX", "")
df.to_csv(f"{base}/results_{DS}_{ENC}{suf}.csv", index=False)
print(f"[saved] {base}/results_{DS}_{ENC}{suf}.csv")
