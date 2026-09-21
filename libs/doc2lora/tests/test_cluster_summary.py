"""CPU unit tests for cluster-centroid ops and pooled expansion (no checkpoint)."""

import pytest
import torch

from doc2lora.cluster_summary import cluster_centroid, pooled_to_norm_lora_emb


def _members(n=5, L=3, latent=4, seed=0):
    return torch.randn(n, L, latent, generator=torch.Generator().manual_seed(seed))


def test_centroid_mean_matches_torch_mean():
    g = _members()
    out = cluster_centroid(g, op="mean")
    assert out.shape == g.shape[1:]
    assert torch.allclose(out, g.mean(0), atol=1e-6)


def test_centroid_intersection_shrinks_toward_zero():
    """intersection = mean * |mean|/(|mean|+std) — magnitude ≤ |mean| everywhere."""
    g = _members()
    inter = cluster_centroid(g, op="intersection")
    mean = g.mean(0)
    assert inter.shape == mean.shape
    assert torch.all(inter.abs() <= mean.abs() + 1e-6)


def test_centroid_intersection_equals_mean_when_no_variance():
    # identical members -> std 0 -> intersection == mean
    base = torch.randn(3, 4)
    g = base.unsqueeze(0).repeat(5, 1, 1)
    inter = cluster_centroid(g, op="intersection")
    assert torch.allclose(inter, base, atol=1e-5)


def test_centroid_union_picks_max_magnitude_per_dim():
    g = torch.tensor([
        [[1.0, -5.0]],
        [[3.0, 2.0]],
    ])  # [N=2, L=1, latent=2]
    union = cluster_centroid(g, op="union")
    assert torch.allclose(union, torch.tensor([[3.0, -5.0]]))


def test_centroid_flat_input_with_n_layers():
    g_flat = torch.randn(5, 3 * 4)  # [N, L*latent]
    out = cluster_centroid(g_flat, op="mean", n_layers=3)
    assert out.shape == (3, 4)


def test_centroid_unknown_op_raises():
    with pytest.raises(ValueError):
        cluster_centroid(_members(), op="bogus")


def test_pooled_to_norm_lora_emb_broadcast_shape():
    pooled = torch.randn(6, 8)  # [L, latent]
    out = pooled_to_norm_lora_emb(pooled, n_slots=8)
    assert out.shape == (6, 1, 8, 8)
    # slots are broadcast copies
    for k in range(1, 8):
        assert torch.allclose(out[:, 0, k, :], out[:, 0, 0, :], atol=1e-6)


def test_pooled_renorm_false_preserves_magnitude():
    pooled = torch.randn(4, 5)
    out = pooled_to_norm_lora_emb(pooled, n_slots=8, renorm=False)
    # default renorm=False: slot norm equals the (non-unit) pooled norm
    slot_norm = out[:, 0, 0, :].norm(dim=-1)
    assert not torch.allclose(slot_norm, torch.ones_like(slot_norm))
    assert torch.allclose(slot_norm, pooled.norm(dim=-1), atol=1e-5)


def test_pooled_renorm_true_unit_norm():
    pooled = torch.randn(4, 5)
    out = pooled_to_norm_lora_emb(pooled, n_slots=8, renorm=True)
    slot_norm = out[:, 0, 0, :].norm(dim=-1)
    assert torch.allclose(slot_norm, torch.ones_like(slot_norm), atol=1e-5)
