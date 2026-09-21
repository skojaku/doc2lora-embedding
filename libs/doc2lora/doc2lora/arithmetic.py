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

from doc2lora.device import autocast_context
from doc2lora.embed import extract_norm_lora_emb_batch

# NOTE: ``ctx_to_lora`` (the backbone) is imported lazily inside
# ``internalize_from_norm_lora_emb`` so that ``import doc2lora`` and the pure
# tensor ops below (mix / mean / interpolate / renorm) work in a clean
# environment without the heavy, non-PyPI backbone installed.


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
    from ctx_to_lora.modeling.lora_merger import combine_lora
    from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers

    if not isinstance(norm_lora_emb, torch.Tensor):
        norm_lora_emb = torch.from_numpy(norm_lora_emb)
    norm_lora_emb = norm_lora_emb.to(model.device)
    if norm_lora_emb.dim() == 4:
        norm_lora_emb = norm_lora_emb.unsqueeze(0)  # add batch dim -> [1, L, M, r, latent]

    model.patch_lora_forward()
    hypernet = model.hypernet

    with torch.no_grad(), autocast_context(model.device):
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
