"""[CPU] Storage / index cost for #71: bytes per paper, corpus index size, and PQ recall loss.

Reads the cached APS embeddings, so nothing is re-encoded.  For each representation:
  bytes/paper (fp16), projected index size at `corpus_size` papers,
  and a product-quantised variant (faiss IndexPQ, 8 bits per subvector) with recall@10 measured
  against exact cosine search on held-out queries.

Out: eff_index.json
"""
import json
import os

import faiss
import numpy as np

REPS = dict(snakemake.params.reps)              # noqa: F821  {label: (file, key)}
EMB_DIR = snakemake.params.emb_dir              # noqa: F821
N_TRAIN = int(snakemake.params.n_train)         # noqa: F821
N_BASE = int(snakemake.params.n_base)           # noqa: F821
N_QUERY = int(snakemake.params.n_query)         # noqa: F821
CORPUS = int(snakemake.params.corpus_size)      # noqa: F821
PQ_BYTES = list(snakemake.params.pq_bytes)      # noqa: F821  target bytes/vector
SEED = int(snakemake.params.seed)               # noqa: F821
K = 10

out = {}
rng = np.random.default_rng(SEED)
for label, (fn, key) in REPS.items():
    path = os.path.join(EMB_DIR, fn)
    if not os.path.exists(path):
        print(f"[eff-index] skip {label}: {path} absent", flush=True)
        continue
    z = np.load(path, allow_pickle=True)
    E = z[key]
    n, d = E.shape[0], int(np.prod(E.shape[1:]))
    take = np.sort(rng.choice(n, size=min(N_TRAIN + N_BASE + N_QUERY, n), replace=False))
    X = np.asarray(E[take], dtype=np.float32).reshape(len(take), d)
    del E, z
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    tr = X[:N_TRAIN]
    base = X[N_TRAIN:N_TRAIN + N_BASE]
    qs = X[N_TRAIN + N_BASE:N_TRAIN + N_BASE + N_QUERY]
    rec = {"dim": d, "n_corpus": int(n), "bytes_per_paper_fp16": 2 * d,
           "index_gb_fp16": 2 * d * CORPUS / 1e9, "pq": []}

    flat = faiss.IndexFlatIP(d)
    flat.add(base)
    _, gold = flat.search(qs, K)

    for target in PQ_BYTES:
        m = int(target)                      # 8 bits per subvector -> m bytes per vector
        while m > 0 and d % m:
            m -= 1
        if m < 1:
            continue
        try:
            idx = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT)
            idx.train(tr)
            idx.add(base)
            _, got = idx.search(qs, K)
        except Exception as e:  # noqa: BLE001
            print(f"[eff-index] {label} PQ m={m} failed: {e}", flush=True)
            continue
        r = float(np.mean([len(set(g) & set(h)) / K for g, h in zip(gold, got)]))
        rec["pq"].append({"m_bytes": m, "sub_dim": d // m, "recall_at_10": r,
                          "index_gb": m * CORPUS / 1e9,
                          "compression": (2 * d) / m})
        print(f"  {label}: PQ m={m} ({d // m} dims/sub) recall@{K}={r:.3f} "
              f"({m} B/vec vs {2 * d} B)", flush=True)
    out[label] = rec
    print(f"[eff-index] {label}: d={d} fp16 {2 * d} B/paper -> "
          f"{rec['index_gb_fp16']:.1f} GB at {CORPUS:,} papers", flush=True)

os.makedirs(os.path.dirname(snakemake.output.json), exist_ok=True)     # noqa: F821
with open(snakemake.output.json, "w") as fh:                           # noqa: F821
    json.dump({"corpus_size": CORPUS, "k": K, "n_base": N_BASE, "n_query": N_QUERY,
               "reps": out}, fh, indent=1)
print(f"[eff-index] wrote {snakemake.output.json}", flush=True)         # noqa: F821
