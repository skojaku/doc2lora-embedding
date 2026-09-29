"""Cluster summarization / labeling from pooled adapters.

Recipe (validated on APS/PACS, 2026-06): given the mean-pooled ``norm_lora_emb``
adapters of a cluster's member documents (e.g. the on-disk benchmark adapters, shape
``[n_layers * latent]`` per doc), build a centroid, expand it across the ``r`` rank
slots, internalize, and decode a constrained label.

Key empirical findings baked into the defaults:

* **Do NOT renormalize the expanded slots.** The pooled adapter is a mean over the
  ``r`` unit slots, so its natural per-layer magnitude is < 1. Broadcasting it to
  ``r`` identical slots and L2-renormalizing each to unit makes them sum to an
  ``r``x over-magnitude rank-1 adapter that is off-distribution and decodes to
  garbage. With ``renorm=False`` (default) the pooled centroid decodes cleanly and
  matches/beats the full-rank perceiver path. So decodability is governed by the
  adapter's effective *magnitude*, not by having ``r`` distinct slots.
* **Centroid operator.** ``mean`` is the baseline. ``intersection`` (consistency-
  gated: ``mean * |mean| / (|mean| + std)``) suppresses dimensions the members
  disagree on and gives sharper, more canonical labels on BROAD clusters
  (field/division level) and is robust to "no shared topic" refusals; plain
  ``mean`` is better at narrow leaves. ``union`` (per-dim max magnitude) is
  off-distribution and decodes poorly -- provided for completeness.
* **Label prompt.** A single constrained instruction ("name the field/topic in N
  words") works best; chain-of-thought and open-ended "describe" over-specify or
  confabulate. The prompt sets the abstraction altitude (ask for "field" on broad
  clusters, "topic" on leaves); the adapter does not self-broaden with tree height.
* **The norm_lora_emb head path is magnitude-sensitive** (this module), so
  ``scale`` in ``pooled_to_norm_lora_emb`` / ``decode_cluster`` changes the
  effective conditioning strength.
"""
import numpy as np
import torch

from .arithmetic import internalize_from_norm_lora_emb
from .generate import generate_text

DEFAULT_LABEL_PROMPT = (
    "In 3 to 6 words, name the single scientific subfield or research topic that all "
    "of these documents share. Reply with only the topic name, nothing else."
)
FIELD_LABEL_PROMPT = (
    "In 2 to 4 words, name the broad scientific field or discipline that all of these "
    "documents belong to, not a specific topic or result. Reply with only the field name."
)


def _as_layers_latent(adapters, n_layers=None):
    """Coerce adapters to a float tensor of shape [..., n_layers, latent]."""
    if isinstance(adapters, np.ndarray):
        adapters = torch.from_numpy(adapters)
    adapters = adapters.float()
    if adapters.dim() >= 2 and adapters.shape[-1] != 1 and (n_layers is None) and adapters.dim() == 3:
        return adapters  # already [N, L, latent]
    if adapters.dim() == 2 and n_layers is not None:          # [N, L*latent] -> [N, L, latent]
        return adapters.view(adapters.shape[0], n_layers, -1)
    if adapters.dim() == 1 and n_layers is not None:          # [L*latent] -> [L, latent]
        return adapters.view(n_layers, -1)
    return adapters


def cluster_centroid(adapters, op="mean", n_layers=None, eps=1e-6):
    """Aggregate member adapters into a single [n_layers, latent] centroid.

    Args:
        adapters: member adapters, ``[N, n_layers, latent]`` or ``[N, n_layers*latent]``
            (then pass ``n_layers``). Tensor or ndarray.
        op: ``"mean"`` | ``"intersection"`` | ``"union"``.
        n_layers: required if ``adapters`` is flat ``[N, n_layers*latent]``.
    Returns:
        torch.Tensor ``[n_layers, latent]``.
    """
    g = _as_layers_latent(adapters, n_layers)
    if g.dim() != 3:
        raise ValueError(f"expected member adapters [N, L, latent], got {tuple(g.shape)}")
    mean = g.mean(0)
    if op == "mean":
        return mean
    if op == "intersection":
        std = g.std(0)
        return mean * (mean.abs() / (mean.abs() + std + eps))
    if op == "union":
        idx = g.abs().argmax(0, keepdim=True)
        return torch.gather(g, 0, idx).squeeze(0)
    raise ValueError(f"unknown op {op!r} (mean|intersection|union)")


def pooled_to_norm_lora_emb(pooled, n_slots=8, scale=1.0, renorm=False, n_layers=None):
    """Expand a pooled centroid ``[n_layers, latent]`` to a ``norm_lora_emb``
    tensor ``[n_layers, 1, n_slots, latent]`` by broadcasting across the rank axis.

    Args:
        pooled: ``[n_layers, latent]`` (or flat ``[n_layers*latent]`` + ``n_layers``).
        n_slots: rank ``r`` of the adapter (8 for the released checkpoints).
        scale: multiply the magnitude (see module docstring; 1.0 keeps natural scale).
        renorm: if True, L2-normalize each slot over the latent axis. Default False
            (renorm decodes to garbage for broadcast slots -- kept for ablation).
    Returns:
        torch.Tensor ``[n_layers, 1, n_slots, latent]`` ready for
        :func:`internalize_from_norm_lora_emb`.
    """
    p = _as_layers_latent(pooled, n_layers).float() * scale
    if p.dim() != 2:
        raise ValueError(f"expected pooled [n_layers, latent], got {tuple(p.shape)}")
    if renorm:
        p = p / (p.norm(dim=-1, keepdim=True) + 1e-8)
    return p.unsqueeze(1).unsqueeze(1).expand(p.shape[0], 1, n_slots, p.shape[1]).contiguous()


def decode_cluster(model, tokenizer, pooled, prompt=DEFAULT_LABEL_PROMPT, *,
                   n_slots=8, scale=1.0, renorm=False, n_layers=None,
                   max_new_tokens=24, **gen_kwargs):
    """Internalize a pooled centroid and decode a label/summary.

    ``pooled`` is a ``[n_layers, latent]`` centroid (e.g. from
    :func:`cluster_centroid`). Returns the generated string.
    """
    nle = pooled_to_norm_lora_emb(pooled, n_slots=n_slots, scale=scale,
                                  renorm=renorm, n_layers=n_layers)
    model.reset()
    internalize_from_norm_lora_emb(model, nle.to(model.device))
    return generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens, **gen_kwargs)


def summarize_cluster(model, tokenizer, adapters, *, op="mean", prompt=DEFAULT_LABEL_PROMPT,
                      n_slots=8, scale=1.0, renorm=False, n_layers=None,
                      max_new_tokens=24, **gen_kwargs):
    """One-call cluster label: aggregate member ``adapters`` (op) then decode.

    The ``prompt`` wording sets the abstraction altitude of the label (ask for
    "field" on broad clusters, "topic" on leaves); the adapter does not
    self-broaden with tree height. For BROAD clusters prefer
    ``op="intersection"`` + ``FIELD_LABEL_PROMPT``; for narrow leaves
    ``op="mean"`` + ``DEFAULT_LABEL_PROMPT``.
    """
    c = cluster_centroid(adapters, op=op, n_layers=n_layers)
    return decode_cluster(model, tokenizer, c, prompt, n_slots=n_slots, scale=scale,
                          renorm=renorm, max_new_tokens=max_new_tokens, **gen_kwargs)
