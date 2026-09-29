"""CPU unit tests for adapter arithmetic (no checkpoint / no backbone needed).

These pin the numerical invariants the paper leans on (§3.3): the head is
linear, so a weighted combination of embeddings is the same weighted
combination of the LoRA weights they generate. We test the library-side half of
that — that ``mix_embeddings`` is exactly the weighted sum and commutes with a
random linear map — plus the renorm / weight-validation contracts.
"""

import pytest
import torch

from doc2lora.arithmetic import (
    _renorm,
    interpolate_embeddings,
    mean_embeddings,
    mix_embeddings,
)


def _rand_embs(n=3, shape=(3, 1, 4, 8), seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*shape, generator=g) for _ in range(n)]


def test_mix_is_weighted_sum_without_renorm():
    embs = _rand_embs()
    w = [0.2, 0.5, 0.3]
    combined = mix_embeddings(embs, weights=w, mode="full", renorm=False)
    expected = sum(wi * e for wi, e in zip(w, embs))
    assert torch.allclose(combined, expected, atol=1e-6)


def test_mix_is_linear_equivariant():
    """f(Σ wᵢ eᵢ) = Σ wᵢ f(eᵢ) for a linear head f (acting on the latent axis)."""
    embs = _rand_embs()
    w = [0.2, 0.5, 0.3]
    latent = embs[0].shape[-1]
    A = torch.randn(latent, latent, generator=torch.Generator().manual_seed(1))

    combined = mix_embeddings(embs, weights=w, mode="full", renorm=False)
    lhs = combined @ A                                    # f(Σ wᵢ eᵢ)
    rhs = sum(wi * (e @ A) for wi, e in zip(w, embs))     # Σ wᵢ f(eᵢ)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_mix_default_weights_are_uniform_mean():
    embs = _rand_embs()
    combined = mix_embeddings(embs, weights=None, renorm=False)
    expected = sum(embs) / len(embs)
    assert torch.allclose(combined, expected, atol=1e-6)


def test_mean_embeddings_matches_uniform_mix():
    embs = _rand_embs()
    a = mean_embeddings(embs, renorm=False)
    b = mix_embeddings(embs, weights=[1 / 3] * 3, renorm=False)
    assert torch.allclose(a, b, atol=1e-6)


def test_mix_weight_length_mismatch_raises():
    embs = _rand_embs()
    with pytest.raises(ValueError):
        mix_embeddings(embs, weights=[0.5, 0.5])


def test_mix_unknown_mode_raises():
    embs = _rand_embs()
    with pytest.raises(ValueError):
        mix_embeddings(embs, mode="bogus")


def test_renorm_gives_unit_latent_norm():
    embs = _rand_embs()
    combined = mix_embeddings(embs, mode="full", renorm=True)
    norms = combined.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_renorm_helper_unit_norm():
    x = torch.randn(5, 7)
    out = _renorm(x)
    assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_pooled_mode_broadcasts_over_rank():
    embs = _rand_embs(shape=(3, 1, 4, 8))
    combined = mix_embeddings(embs, mode="pooled", renorm=False)
    assert combined.shape == embs[0].shape
    # every rank slot is identical after pooled combine
    r = combined.shape[-2]
    for k in range(1, r):
        assert torch.allclose(combined[..., k, :], combined[..., 0, :], atol=1e-6)


@pytest.mark.parametrize("alpha,expected_idx", [(0.0, 0), (1.0, 1)])
def test_interpolate_endpoints(alpha, expected_idx):
    a, b = _rand_embs(n=2)
    out = interpolate_embeddings(a, b, alpha=alpha, renorm=False)
    assert torch.allclose(out, (a, b)[expected_idx], atol=1e-6)
