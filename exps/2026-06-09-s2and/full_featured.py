"""[GPU] FULL-FEATURED S2AND-style AND: metadata features + a swappable EMBEDDING-cosine feature,
pairwise classifier -> per-block agglomerative clustering -> B^3. Measures each embedding's ADDED value
(ΔB3 vs metadata-only) and compares gene/gene_kron vs specter as the embedding feature.

Faithful subset of S2AND's featurizer (name / coauthor / venue / journal / year / affiliation) + emb cosine.
Usage: CUDA_VISIBLE_DEVICES=0 python full_featured.py zbmath gemma
"""
import sys, os, json, re, itertools, time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.cluster import AgglomerativeClustering
import torch

sys.path.insert(0, "exps/2026-06-09-kron-adapter"); sys.path.insert(0, "exps/2026-06-09-s2and")
from kron import KronAdapter, info_nce
from s2lib import b3

DS = sys.argv[1] if len(sys.argv) > 1 else "zbmath"
ENC = sys.argv[2] if len(sys.argv) > 2 else "gemma"
base = "exps/2026-06-09-s2and"; dev = "cuda"
norm = lambda x: re.sub(r"[^a-z0-9]", "", (x or "").lower())


# ---------- load raw + build per-signature records ----------
sig = json.load(open(f"{base}/data/{DS}/{DS}_signatures.json"))
pap = json.load(open(f"{base}/data/{DS}/{DS}_papers.json"))
clu = json.load(open(f"{base}/data/{DS}/{DS}_clusters.json"))
sig2clu = {str(s): cid for cid, c in clu.items() for s in c["signature_ids"]}

rec = {}
for sid, s in sig.items():
    sid = str(sid)
    if sid not in sig2clu:
        continue
    ai = s["author_info"]; pid = str(s["paper_id"]); p = pap.get(pid) or pap.get(s["paper_id"]) or {}
    authors = p.get("authors") or []
    pos = ai.get("position", 0)
    aname = lambda a: a.get("author_name", "") if isinstance(a, dict) else str(a)
    apos = lambda a, i: a.get("position", i) if isinstance(a, dict) else i
    coau = {norm(aname(a)) for i, a in enumerate(authors) if apos(a, i) != pos and norm(aname(a))}
    rec[sid] = dict(pid=pid, block=ai["block"], clu=sig2clu[sid],
                    first=norm(ai.get("first")), mid=norm(ai.get("middle")),
                    coau=coau, venue=norm(p.get("venue")), jour=norm(p.get("journal_name")),
                    year=p.get("year") or 0,
                    affil={norm(a) for a in (ai.get("affiliations") or []) if norm(a)})

st = pd.DataFrame([{"sid": k, "block": v["block"], "clu": v["clu"], "pid": v["pid"]} for k, v in rec.items()])
# blocks train/test split (same as embedding-isolation: seed 42, 60/40)
blocks = sorted(st.block.unique()); rng = np.random.default_rng(42); rng.shuffle(blocks)
ntr = int(0.6 * len(blocks)); train_bl = set(blocks[:ntr]); test_bl = set(blocks[ntr:])
print(f"[{DS}/{ENC}] {len(rec)} sigs, {len(blocks)} blocks ({len(train_bl)} train/{len(test_bl)} test)", flush=True)


# ---------- embeddings (specter / gene / gene_kron) ----------
def load_npz(path, key):
    z = np.load(path, allow_pickle=True)
    ids = np.array([str(p) for p in z["paper_ids"]]); V = z[key].astype(np.float32)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return {p: V[i] for i, p in enumerate(ids)}, V.shape[1]


SP, _ = load_npz(f"{base}/proc/{DS}/specter.npz", "vecs")
GN, GD = load_npz(f"{base}/proc/{DS}/genes_{ENC}.npz", "embeddings")


def train_gene_kron():
    """same-author adapter on TRAIN blocks (reg=10), return paper->kron-vec dict."""
    tr = st[st.block.isin(train_bl)]
    pos = []
    for _, g in tr.groupby("clu"):
        ps = g.pid.values
        if len(ps) >= 2:
            for i in range(len(ps)):
                pos.append((ps[i], ps[(i + 1) % len(ps)]))
    if len(pos) < 50:
        return None
    L = GD // 512
    A = np.vstack([GN[a] for a, _ in pos]); B = np.vstack([GN[b] for _, b in pos])
    m = KronAdapter(L=L, d=512).to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    At = torch.tensor(A, device=dev); Bt = torch.tensor(B, device=dev); rng2 = np.random.default_rng(0)
    bs = min(256, len(pos))
    for step in range(2000):
        sel = rng2.integers(0, len(pos), bs)
        za = m(At[sel].view(bs, L, 512)); zp = m(Bt[sel].view(bs, L, 512))
        idp = m.s.pow(2).mean() + m.alpha.pow(2).mean() + m.P.pow(2).mean()
        loss = info_nce(za, zp, 0.05) + 10.0 * idp
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    out = {}
    with torch.no_grad():
        ks = list(GN.keys())
        for i in range(0, len(ks), 4096):
            ch = ks[i:i + 4096]
            Z = m(torch.tensor(np.vstack([GN[k] for k in ch]), device=dev).view(len(ch), L, 512)).cpu().numpy()
            Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
            for k, z in zip(ch, Z):
                out[k] = z
    return out


GK = train_gene_kron()
EMB = {"specter": SP, "gene": GN, "gene_kron": GK}


# ---------- pairwise features ----------
def pair_feats(a, b, embdict=None):
    ra, rb = rec[a], rec[b]
    f = [
        float(ra["first"] == rb["first"] and ra["first"] != ""),
        float(ra["first"][:1] == rb["first"][:1] and ra["first"] != ""),
        float(ra["mid"] == rb["mid"] and ra["mid"] != ""),
        float(ra["mid"][:1] == rb["mid"][:1]) if ra["mid"] and rb["mid"] else 0.5,  # compat if one missing
        len(ra["coau"] & rb["coau"]),
        (len(ra["coau"] & rb["coau"]) / len(ra["coau"] | rb["coau"])) if (ra["coau"] | rb["coau"]) else 0.0,
        float(ra["venue"] == rb["venue"] and ra["venue"] != ""),
        float(ra["jour"] == rb["jour"] and ra["jour"] != ""),
        abs(ra["year"] - rb["year"]) if ra["year"] and rb["year"] else 50,
        (len(ra["affil"] & rb["affil"]) / len(ra["affil"] | rb["affil"])) if (ra["affil"] | rb["affil"]) else 0.0,
    ]
    if embdict is not None:
        va, vb = embdict.get(ra["pid"]), embdict.get(rb["pid"])
        f.append(float(va @ vb) if (va is not None and vb is not None) else 0.0)
    return f


def block_pairs(df):
    out = []
    for bl, g in df.groupby("block"):
        sids = g.sid.values; clus = g.clu.values
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                out.append((bl, sids[i], sids[j], int(clus[i] == clus[j])))
    return out


tr_pairs = block_pairs(st[st.block.isin(train_bl)])
te_df = st[st.block.isin(test_bl)]
rng3 = np.random.default_rng(0)
if len(tr_pairs) > 300000:
    idx = rng3.choice(len(tr_pairs), 300000, replace=False); tr_pairs = [tr_pairs[i] for i in idx]
print(f"  train pairs={len(tr_pairs)} (prev={np.mean([p[3] for p in tr_pairs]):.3f})", flush=True)


def cluster_test(prob_of_pair):
    """agglomerate each test block on 1-prob; threshold tuned on... fixed 0.5 distance (prob>=0.5 link)."""
    true_c, pred_c = {}, {}
    for bl, g in te_df.groupby("block"):
        sids = list(g.sid.values)
        for s, c in zip(sids, g.clu.values):
            true_c.setdefault(c, []).append(s)
        n = len(sids)
        if n == 1:
            pred_c.setdefault(f"{bl}_0", []).append(sids[0]); continue
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 - prob_of_pair[(bl, sids[i], sids[j])]
                D[i, j] = D[j, i] = d
        lab = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                      distance_threshold=0.5).fit_predict(D)
        for s, c in zip(sids, lab):
            pred_c.setdefault(f"{bl}_{c}", []).append(s)
    return b3(true_c, pred_c)


# ---------- run each embedding variant (+ metadata-only) ----------
rows = []
te_pairs = block_pairs(te_df)
for variant in [None, "specter", "gene", "gene_kron"]:
    emb = EMB.get(variant) if variant else None
    if variant and emb is None:
        continue
    Xtr = np.array([pair_feats(a, b, emb) for _, a, b, _ in tr_pairs], np.float32)
    ytr = np.array([y for *_, y in tr_pairs])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                         max_depth=4, random_state=0).fit(Xtr, ytr)
    Xte = np.array([pair_feats(a, b, emb) for _, a, b, _ in te_pairs], np.float32)
    prob = clf.predict_proba(Xte)[:, 1]
    pmap = {(bl, a, b): pr for (bl, a, b, _), pr in zip(te_pairs, prob)}
    P, R, F = cluster_test(pmap)
    name = variant or "metadata_only"
    rows.append(dict(features=name, B3_P=round(P, 3), B3_R=round(R, 3), B3_F1=round(F, 3)))
    print(f"  {name:14} B3 P={P:.3f} R={R:.3f} F1={F:.3f}", flush=True)

df = pd.DataFrame(rows)
md = df[df.features == "metadata_only"].B3_F1.values[0]
df["delta_vs_metadata"] = (df.B3_F1 - md).round(3)
print(f"\n===== FULL-FEATURED S2AND-style B^3 ({DS}, gene={ENC}) =====")
print(df.to_string(index=False))
df.to_csv(f"{base}/full_{DS}_{ENC}.csv", index=False)
print(f"[saved] {base}/full_{DS}_{ENC}.csv")
