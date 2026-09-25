"""[CPU] Per-signature B^3 pools for the S2AND half of the symmetric-supervision control (#145).

`workflow/scripts/score_pool_s2and.py` scores the reported methods: the raw gene, the
gene with the manuscript's g_theta, and the text baselines off the shelf. This script scores the
SAME units with the same protocol, and adds one adapted column per text baseline, so every space
in the table has had the identical citation-trained bijective transform applied to it.

Protocol, copied from the reported run so the two are comparable:
  * coverage = papers every method carries (adapted files share ids with their source, so adding
    them cannot move the coverage intersection),
  * blocks shuffled with seed 42, 60% train / 40% test,
  * agglomerative average-linkage on cosine distance, threshold tuned per method on the train
    blocks (each method gets its own tuned threshold, as in the reported run),
  * B^3 precision/recall dumped per test signature, so the bootstrap unit is the signature.

Usage:
  python workflow/scripts/groupc/gcb_s2and_pool.py --ds qian --enc qwen \
      --out data/groupc/bench/pools/s2and_qian.parquet
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, "data/s2and")
from s2lib import b3, b3_per_sig  # noqa: E402

S2 = "data/s2and"
GCB = os.path.join("data", "groupc", "bench", "s2and")

# name -> (path template, npz key).  {ds} / {enc} are filled per call.
#   raw      = what the reported table scores
#   _kron_gc = the same space with this control's transform applied
RAW = [("gene", f"{S2}/proc/{{ds}}/genes_{{enc}}.npz"),
       ("genkron", f"{S2}/proc/{{ds}}/genes_{{enc}}_genkron.npz"),
       ("specter", f"{S2}/proc/{{ds}}/specter.npz"),
       ("sbert", f"{S2}/proc/{{ds}}/sbert.npz"),
       ("instructor", f"{S2}/proc/{{ds}}/instructor.npz"),
       ("embeddinggemma", f"{S2}/proc/{{ds}}/embeddinggemma.npz"),
       ("gte", f"{S2}/proc/{{ds}}/gte.npz"),
       ("icae", f"{S2}/proc/{{ds}}/icae.npz")]
ADAPTED = [("gene_kron_gc", f"{GCB}/{{ds}}/gene_kron_gc.npz"),
           ("specter_kron_gc", f"{GCB}/{{ds}}/specter_kron_gc.npz"),
           ("sbert_kron_gc", f"{GCB}/{{ds}}/sbert_kron_gc.npz"),
           ("instructor_kron_gc", f"{GCB}/{{ds}}/instructor_kron_gc.npz"),
           ("embeddinggemma_kron_gc", f"{GCB}/{{ds}}/embeddinggemma_kron_gc.npz"),
           ("gte_kron_gc", f"{GCB}/{{ds}}/gte_kron_gc.npz"),
           # ICAE's transform was trained by the ICAE benchmark chain, not here: it factorises over
           # 128 memory tokens x 4,096 channels (16.8M parameters) and does not fit gcb_train's flat
           # path. Same supervision, much larger map -- state both when reporting it.
           ("icae_genkron", f"{S2}/proc/{{ds}}/icae_genkron.npz")]


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    key = "embeddings" if "embeddings" in z.files else "vecs"
    V = np.asarray(z[key], dtype=np.float32)
    if V.ndim > 2:
        V = V.reshape(V.shape[0], -1)
    ids = np.array([str(p) for p in z["paper_ids"]])
    return {p: V[i] for i, p in enumerate(ids)}


def cluster(X, thr):
    if len(X) == 1:
        return np.array([0])
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    D = np.clip(1.0 - Xn @ Xn.T, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    return AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                   distance_threshold=thr).fit_predict(D)


def clusters(df, emb, thr):
    true_c, pred_c = {}, {}
    for bl, g in df.groupby("block"):
        X = np.vstack([emb[p] for p in g.paper_id.values]).astype(np.float32)
        lab = cluster(X, thr)
        for s, c in zip(g.signature_id.values, g.cluster_id.values):
            true_c.setdefault(c, []).append(s)
        for s, c in zip(g.signature_id.values, lab):
            pred_c.setdefault(f"{bl}__{c}", []).append(s)
    return true_c, pred_c


def tune(df, emb):
    best = (-1.0, 0.5)
    for thr in np.linspace(0.05, 0.95, 19):
        f = b3(*clusters(df, emb, thr))[2]
        if f > best[0]:
            best = (f, thr)
    return best[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ds", required=True)
    p.add_argument("--enc", default="qwen")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    st = pd.read_parquet(f"{S2}/proc/{a.ds}/sig_table.parquet")
    st["paper_id"] = st.paper_id.astype(str)

    want = [(n, t.format(ds=a.ds, enc=a.enc)) for n, t in RAW + ADAPTED]
    missing = [(n, pth) for n, pth in want if not os.path.exists(pth)]
    if missing:
        raise SystemExit(f"[{a.ds}] {len(missing)} embedding(s) missing:\n"
                         + "\n".join(f"    {n:24s} {pth}" for n, pth in missing))
    EMB = {n: load_npz(pth) for n, pth in want}

    have = set.intersection(*[set(d) for d in EMB.values()])
    st = st[st.paper_id.isin(have)].reset_index(drop=True)
    blocks = sorted(st.block.unique())
    rng = np.random.default_rng(42)
    rng.shuffle(blocks)
    ntr = int(0.6 * len(blocks))
    tr = st[st.block.isin(set(blocks[:ntr]))]
    te = st[st.block.isin(set(blocks[ntr:]))]
    print(f"[gcb-s2and {a.ds}] {len(EMB)} methods, {len(have):,} papers, "
          f"{len(tr):,} train / {len(te):,} test signatures", flush=True)

    sig2blk = dict(zip(st.signature_id, st.block))
    rows = []
    for name in EMB:
        thr = tune(tr, EMB[name])
        P, R, F, per = b3_per_sig(*clusters(te, EMB[name], thr))
        for sid, (pp, rr) in per.items():
            rows.append({"dataset": a.ds, "enc": a.enc, "method": name, "block": sig2blk.get(sid),
                         "signature_id": sid, "b3_p": pp, "b3_r": rr})
        print(f"  {name:24} thr={thr:.2f}  B3 F1={F:.3f}  (n_sig={len(per)})", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_parquet(a.out)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
