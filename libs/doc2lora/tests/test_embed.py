"""CPU unit tests for pooling and batching (no checkpoint needed)."""

import numpy as np
import torch

from doc2lora.embed import _make_batches, mean_pool_and_flatten


def test_mean_pool_and_flatten_shape_and_dtype():
    L, M, r, latent = 6, 1, 8, 16
    adapter = torch.randn(L, M, r, latent)
    out = mean_pool_and_flatten(adapter)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float16
    assert out.shape == (L * latent,)


def test_mean_pool_and_flatten_matches_manual_pool():
    adapter = torch.randn(4, 1, 8, 5)
    out = mean_pool_and_flatten(adapter)
    expected = adapter.mean(dim=(1, 2)).flatten().to(torch.float16).numpy()
    assert np.allclose(out, expected, atol=1e-3)


def test_mean_pool_accepts_numpy():
    adapter = np.random.RandomState(0).randn(3, 1, 8, 4).astype(np.float32)
    out = mean_pool_and_flatten(adapter)
    assert out.shape == (3 * 4,)


def test_make_batches_fixed_size_covers_all_indices_once():
    lengths = [5, 1, 9, 3, 7, 2]
    batches = _make_batches(lengths, batch_size=2)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths)))
    assert all(len(b) <= 2 for b in batches)


def test_make_batches_is_length_sorted():
    lengths = [5, 1, 9, 3, 7, 2]
    batches = _make_batches(lengths, batch_size=2)
    order = [i for b in batches for i in b]
    assert [lengths[i] for i in order] == sorted(lengths)


def test_make_batches_token_budget_respected():
    lengths = [2, 2, 2, 10]
    budget = 8
    batches = _make_batches(lengths, max_batch_tokens=budget)
    for b in batches:
        longest = max(lengths[i] for i in b)
        # a batch either fits the budget or is a single over-budget doc
        assert longest * len(b) <= budget or len(b) == 1
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths)))
