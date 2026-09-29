"""Reusable retrieval primitives for the precomputed APS/field benchmark embeddings.

The bench `.npz` files all store a 2-D embedding matrix plus a row-aligned
``paper_ids`` vector, but under different array keys (``embeddings`` for the
doc2lora gene files, ``vecs`` for the text baselines). These helpers hide that and
give every experiment one batched, memory-efficient cosine path instead of an
ad-hoc gemv loop (which re-reads the whole matrix per query and is bandwidth-bound).
"""
import numpy as np

EMB_KEYS = ("embeddings", "vecs")


def load_bench_emb(npz_path, normalize=True, dtype=np.float32):
    """Load a bench embedding file -> (emb, paper_ids).

    Auto-detects the array key, casts to ``dtype``, and (default) L2-normalizes the
    rows so a plain dot product is a cosine similarity.
    """
    z = np.load(npz_path, allow_pickle=True)
    key = next((k for k in EMB_KEYS if k in z.files), None)
    if key is None:
        raise KeyError(f"{npz_path}: no embedding array among {EMB_KEYS}; has {z.files}")
    emb = z[key].astype(dtype)
    pids = np.asarray([int(x) for x in z["paper_ids"]])
    if normalize:
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb, pids


def unit(vec):
    """L2-normalize a single vector (or each row of a 2-D array)."""
    vec = np.asarray(vec, dtype=np.float32)
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-12)


def cosine_scores(emb, queries):
    """Score every query against every row in ONE matmul -> (n_rows, n_queries).

    ``emb`` should already be row-normalized (see :func:`load_bench_emb`) and
    ``queries`` unit vectors. Batching all queries into a single matmul reads the
    big matrix once total instead of once per query.
    """
    Q = np.ascontiguousarray(np.asarray(queries, dtype=np.float32).T)  # (d, n_q)
    return emb @ Q


def topk_indices(sims, k):
    """Indices of the top-``k`` scores in descending order (argpartition + sort)."""
    n = sims.shape[0]
    k = min(k, n)
    part = np.argpartition(-sims, k - 1)[:k]
    return part[np.argsort(-sims[part])]


def precision_at_ks(sims, is_target, ks):
    """Precision@k of a ranking for a boolean target mask, for each k in ``ks``."""
    order = topk_indices(sims, max(ks))
    return {k: float(is_target[order[:k]].mean()) for k in ks}
