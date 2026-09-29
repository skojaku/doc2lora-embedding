"""Idea arithmetic in the pre-head (`norm_lora_emb`) space.

The doc2lora head ``f`` is a single linear map, so the LoRA adapter is a linear
image of the pre-head embedding: ``LoRA = f(e)``. This makes the embedding space
the natural place to do *idea arithmetic* — averaging, interpolating, and
extrapolating documents — because

    f(sum_i w_i e_i) = sum_i w_i f(e_i),

i.e. a weighted combination of embeddings is *exactly* the same weighted
combination of the adapters they generate. After combining embeddings we run the
result through ``hypernet.head`` and apply it to the base model, so generation is
conditioned on the synthetic "idea."

The captured embedding has shape ``[n_layers, n_modules, r, latent]`` (e.g.
``[32, 1, 8, 512]`` for Mistral-7B), where ``r`` is the per-layer rank — the
"8 query vectors." Each ``[..., latent]`` vector is L2-normalised.

Two ways to combine, exposed via the ``mode`` argument:

- ``"full"``  (option i): combine the *full* ``[L, M, r, latent]`` tensors
  element-wise, preserving all ``r`` rank vectors, then feed to the head.
- ``"pooled"`` (option ii): mean-pool over the ``r`` axis to ``[L, M, 1, latent]``,
  combine the pooled vectors, then broadcast the result back to ``r`` identical
  copies before feeding to the head.

By default the combined per-vector embeddings are re-normalised to unit length
over the latent axis (``renorm=True``), matching the distribution the head was
trained on; the linear-image identity above holds up to that per-vector rescale.
"""

import torch

from ctx_to_lora.modeling.lora_merger import combine_lora
from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers

from doc2lora_legacy.embed import extract_norm_lora_emb_batch


def extract_norm_lora_emb(model, ctx_tokenizer, text, max_length=2048):
    """Extract the pre-head embedding for a single document.

    Returns:
        torch.Tensor of shape [n_layers, n_modules, r, latent].
    """
    return extract_norm_lora_emb_batch(
        model, ctx_tokenizer, [text], max_length=max_length, batch_size=1
    )[0]


def _renorm(emb, eps=1e-8):
    """L2-normalise over the latent (last) axis."""
    return emb / (torch.norm(emb, dim=-1, keepdim=True) + eps)


def _pool_r(emb):
    """Mean-pool over the rank axis -> [n_layers, n_modules, 1, latent]."""
    return emb.mean(dim=-2, keepdim=True)


def mix_embeddings(embs, weights=None, mode="full", renorm=True):
    """Weighted combination of pre-head embeddings.

    Args:
        embs: list of tensors, each [n_layers, n_modules, r, latent].
        weights: list of floats (one per emb). Defaults to a uniform mean.
        mode: "full" (combine all r rank vectors) or "pooled" (pool over r,
            combine, then broadcast back to r copies).
        renorm: if True, L2-normalise each per-vector result over the latent axis.

    Returns:
        torch.Tensor of shape [n_layers, n_modules, r, latent].
    """
    if weights is None:
        weights = [1.0 / len(embs)] * len(embs)
    if len(weights) != len(embs):
        raise ValueError("weights and embs must have the same length")

    ref = embs[0]
    device, dtype = ref.device, ref.dtype
    embs = [e.to(device=device, dtype=torch.float32) for e in embs]

    if mode == "pooled":
        embs = [_pool_r(e) for e in embs]  # [L, M, 1, latent]

    combined = sum(w * e for w, e in zip(weights, embs))

    if mode == "pooled":
        r = ref.shape[-2]
        combined = combined.expand(-1, -1, r, -1).contiguous()
    elif mode != "full":
        raise ValueError(f"unknown mode: {mode!r}")

    if renorm:
        combined = _renorm(combined)
    return combined.to(dtype=dtype)


def mean_embeddings(embs, mode="full", renorm=True):
    """Uniform mean of a list of pre-head embeddings."""
    return mix_embeddings(embs, weights=None, mode=mode, renorm=renorm)


def interpolate_embeddings(emb_a, emb_b, alpha, mode="full", renorm=True):
    """Linear interpolation/extrapolation between two embeddings.

    Returns ``(1 - alpha) * emb_a + alpha * emb_b``. ``alpha`` in [0, 1]
    interpolates; ``alpha`` outside [0, 1] extrapolates beyond the segment.
    """
    return mix_embeddings(
        [emb_a, emb_b], weights=[1.0 - alpha, alpha], mode=mode, renorm=renorm
    )


def splice_query_vectors(emb_a, emb_b, k_from_a, renorm=False):
    """Fuse two ideas by mixing their per-layer rank ("query") vectors.

    Builds a new embedding whose first ``k_from_a`` of the ``r`` rank vectors come
    from ``emb_a`` and whose remaining ``r - k_from_a`` come from ``emb_b`` (per
    layer/module), keeping the total at ``r``. This is a different fusion operator
    from :func:`mix_embeddings`: instead of blending the *values* of the vectors,
    it partitions the ``r`` slots between the two documents, so each rank vector
    stays an intact, unit-norm vector from one source.

    Args:
        emb_a, emb_b: tensors of shape [n_layers, n_modules, r, latent].
        k_from_a: number of rank vectors taken from ``emb_a`` (0..r). The rest
            come from ``emb_b``.
        renorm: extracted vectors are already unit-norm, so this is off by
            default; kept for symmetry with the other operators.

    Returns:
        torch.Tensor of shape [n_layers, n_modules, r, latent].
    """
    r = emb_a.shape[-2]
    if not (0 <= k_from_a <= r):
        raise ValueError(f"k_from_a must be in [0, {r}], got {k_from_a}")
    a = emb_a[..., :k_from_a, :]
    b = emb_b.to(device=emb_a.device, dtype=emb_a.dtype)[..., k_from_a:, :]
    out = torch.cat([a, b], dim=-2)
    if renorm:
        out = _renorm(out)
    return out


def even_split_counts(n_docs, total=8):
    """Distribute ``total`` rank slots among ``n_docs`` as evenly as possible,
    with at least one slot per document.

    E.g. (3, 8) -> [3, 3, 2]; (5, 8) -> [2, 2, 2, 1, 1]; (8, 8) -> [1]*8.
    """
    if not (1 <= n_docs <= total):
        raise ValueError(f"n_docs must be in [1, {total}], got {n_docs}")
    base, rem = divmod(total, n_docs)
    return [base + 1 if i < rem else base for i in range(n_docs)]


def mix_query_vectors(embs, counts=None, seed=None, renorm=False):
    """Synthesize one embedding by stacking rank ("query") vectors drawn from
    several documents — the multi-document generalisation of
    :func:`splice_query_vectors`.

    The ``r`` rank slots of the output are partitioned among the input documents:
    document ``i`` contributes ``counts[i]`` of its own rank vectors, chosen at
    random (without replacement) from its ``r`` slots. With ``counts=None`` the
    slots are split as evenly as possible with at least one per document
    (:func:`even_split_counts`), so ``len(embs)`` documents up to ``r`` can be
    fused — at ``len(embs) == r`` each document contributes exactly one vector.

    Args:
        embs: list of N tensors [n_layers, n_modules, r, latent], 1 <= N <= r.
        counts: optional per-document slot counts (must sum to r). Defaults to an
            even split.
        seed: RNG seed controlling which rank indices are sampled per document.
            Order does not matter to the adapter, so only the multiset of chosen
            vectors is relevant; the seed varies that multiset.
        renorm: extracted vectors are already unit-norm; off by default.

    Returns:
        torch.Tensor of shape [n_layers, n_modules, r, latent], plus the list of
        per-document sampled rank indices (for logging/reproducibility).
    """
    ref = embs[0]
    r = ref.shape[-2]
    n = len(embs)
    if counts is None:
        counts = even_split_counts(n, r)
    if len(counts) != n:
        raise ValueError("counts must have one entry per document")
    if sum(counts) != r:
        raise ValueError(f"counts must sum to r={r}, got {sum(counts)}")

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))

    slots, picks = [], []
    for emb, c in zip(embs, counts):
        idx = torch.randperm(r, generator=gen)[:c]
        picks.append(idx.tolist())
        emb = emb.to(device=ref.device, dtype=ref.dtype)
        slots.append(emb[..., idx.to(emb.device), :])
    out = torch.cat(slots, dim=-2)  # [n_layers, n_modules, r, latent]
    if renorm:
        out = _renorm(out)
    return out, picks


def _kmeans(x, k, iters=100, seed=None):
    """Minimal Lloyd k-means on rows of ``x`` ([N, D]). Returns (centers, assign)."""
    n = x.shape[0]
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    centers = x[torch.randperm(n, generator=gen)[:k].to(x.device)].clone()
    assign = torch.zeros(n, dtype=torch.long, device=x.device)
    for _ in range(iters):
        assign = torch.cdist(x, centers).argmin(dim=1)
        new = centers.clone()
        for c in range(k):
            m = assign == c
            if m.any():
                new[c] = x[m].mean(dim=0)
        if torch.allclose(new, centers):
            centers = new
            break
        centers = new
    return centers, assign


def cluster_query_vectors(embs, k=8, seed=None, renorm=True, per_layer=True):
    """Aggregate *more than* ``k`` documents into ``k`` representative rank vectors.

    With ``M`` documents there are ``M*r`` rank ("query") vectors but only ``r``
    slots. Heuristic: since query vectors of similar ideas are similar, **k-means
    them into ``k`` representatives**. This summarises an arbitrary number of
    documents into a single ``r``-rank adapter.

    Two clustering scopes:

    - ``per_layer=True`` (default, recommended): for **each layer (and module)
      independently**, pool the ``M*r`` latent vectors at that layer and k-means
      them into ``k`` centroids. The head consumes each layer's rank slots
      independently, so clustering in the per-layer ``latent``-dim space keeps the
      representatives on the manifold the head expects.
    - ``per_layer=False``: a single global k-means over the cross-layer components
      (each a flattened ``[n_layers*latent]`` vector). Tends to give unbalanced
      clusters and off-manifold slot tuples; kept for comparison.

    Args:
        embs: list of M tensors [n_layers, n_modules, r, latent] (M may exceed r).
        k: number of representative rank slots.
        seed: RNG seed for k-means initialisation.
        renorm: re-normalise each centroid to unit length over the latent axis
            (the head was trained on unit-norm vectors).
        per_layer: cluster per layer (True) or globally over cross-layer
            components (False).

    Returns:
        (torch.Tensor [n_layers, n_modules, k, latent], cluster_sizes list).
        For ``per_layer=True`` the sizes are summed over layers/modules.
    """
    ref = embs[0]
    n_layers, n_modules, r, latent = ref.shape
    stacked = torch.stack(
        [e.to(device=ref.device, dtype=torch.float32) for e in embs], dim=0
    )  # [M, n_layers, n_modules, r, latent]
    m_docs = stacked.shape[0]

    if per_layer:
        out = torch.empty(n_layers, n_modules, k, latent, device=ref.device)
        sizes = [0] * k
        for l in range(n_layers):
            for mod in range(n_modules):
                pts = stacked[:, l, mod, :, :].reshape(m_docs * r, latent)
                centers, assign = _kmeans(pts, k, seed=seed)
                out[l, mod] = centers
                for c in range(k):
                    sizes[c] += int((assign == c).sum())
    else:
        comps = stacked.permute(0, 3, 1, 2, 4).reshape(m_docs * r, -1)  # [M*r, L*M_mod*D]
        centers, assign = _kmeans(comps, k, seed=seed)
        sizes = [int((assign == c).sum()) for c in range(k)]
        out = centers.reshape(k, n_layers, n_modules, latent).permute(1, 2, 0, 3).contiguous()

    if renorm:
        out = _renorm(out)
    return out.to(ref.dtype), sizes


def stack_doc_pooled_vectors(embs, k=8, seed=None, renorm=True):
    """Aggregate documents by stacking one mean-pooled unit query vector per doc.

    Per document: mean-pool its ``r`` query vectors into a single vector (per
    layer/module) and L2-normalise it — i.e. the per-doc "pooled" representation.
    Then fill the ``k`` rank slots with these per-document vectors:

    - ``M == k``: one document per slot.
    - ``M < k``: round-robin broadcast the documents to fill ``k`` slots.
    - ``M > k``: k-means the ``M`` per-document vectors (whole cross-layer
      representation) into ``k`` representatives.

    Unlike :func:`cluster_query_vectors`, the clustering unit here is the
    *document* (one pooled vector each), not the individual rank vectors, and it
    preserves each slot's cross-layer coherence. Pooled vectors are the same kind
    the head sees under ``pooled`` value-mixing, so they stay on-distribution.

    Returns:
        (torch.Tensor [n_layers, n_modules, k, latent], usage/cluster sizes).
    """
    ref = embs[0]
    n_layers, n_modules, r, latent = ref.shape
    pooled = []
    for e in embs:
        e = e.to(device=ref.device, dtype=torch.float32)
        v = e.mean(dim=-2)  # [n_layers, n_modules, latent], mean over rank
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        pooled.append(v)
    P = torch.stack(pooled, dim=0)  # [M, n_layers, n_modules, latent]
    m_docs = P.shape[0]

    if m_docs == k:
        slots, sizes = P, [1] * k
    elif m_docs < k:
        idx = [i % m_docs for i in range(k)]
        slots = P[idx]
        sizes = [idx.count(i) for i in range(m_docs)]
    else:
        flat = P.reshape(m_docs, -1)
        centers, assign = _kmeans(flat, k, seed=seed)
        slots = centers.reshape(k, n_layers, n_modules, latent)
        sizes = [int((assign == c).sum()) for c in range(k)]

    out = slots.permute(1, 2, 0, 3).contiguous()  # [n_layers, n_modules, k, latent]
    if renorm:
        out = _renorm(out)
    return out.to(ref.dtype), sizes


def internalize_from_norm_lora_emb(model, norm_lora_emb):
    """Inject a pre-head embedding into the model as LoRA weights.

    Runs ``norm_lora_emb`` through ``hypernet.head`` (and the head bias via
    ``combine_lora``), then applies the resulting LoRA to the base model. After
    this call the model is conditioned on the (possibly synthetic) embedding and
    ready for ``model.generate``.

    Args:
        model: ModulatedPretrainedModel.
        norm_lora_emb: tensor of shape [n_layers, n_modules, r, latent] or
            [1, n_layers, n_modules, r, latent].

    Note:
        Call ``model.reset()`` before this to clear any previous document.
    """
    if not isinstance(norm_lora_emb, torch.Tensor):
        norm_lora_emb = torch.from_numpy(norm_lora_emb)
    norm_lora_emb = norm_lora_emb.to(model.device)
    if norm_lora_emb.dim() == 4:
        norm_lora_emb = norm_lora_emb.unsqueeze(0)  # add batch dim -> [1, L, M, r, latent]

    model.patch_lora_forward()
    hypernet = model.hypernet

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        flat_loras = hypernet.head(norm_lora_emb)
        lora_dict = hypernet._to_lora_dict(flat_loras)

    n_ctx_chunks = torch.tensor((1,), device=model.device)
    generated_loras = combine_lora(
        lora_dict,
        n_ctx_chunks,
        lora_bias=hypernet.get_head_bias() if hypernet.config.use_bias else None,
    )
    n_queries = torch.ones(1, dtype=torch.int32, device=model.device)
    apply_lora_to_layers(
        model.base_model, hypernet.layer_indices, generated_loras, n_queries, None
    )
    model.generated_loras = lora_dict
