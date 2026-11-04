#!/usr/bin/env python
"""The sync-free routing histogram replacing ``torch.bincount``, whose read-back costs a sync per MoE layer per
forward. Miscounts mis-slice experts silently; DeepEP hands back int32 ids."""

from __future__ import annotations

import pytest
import torch

from halo.kernels.histogram import accumulate_bincount, sync_free_bincount


def test_counts_match_bincount():
    indices = torch.tensor([3, 0, 0, 7, 3, 3])
    counts = sync_free_bincount(indices, 8, dtype=torch.long)
    assert torch.equal(counts, torch.bincount(indices, minlength=8))


def test_a_multi_dimensional_selection_is_flattened():
    """Routing selections arrive as ``[T, top_k]``."""
    topk_indices = torch.tensor([[0, 1], [1, 2], [1, 1]])
    counts = sync_free_bincount(topk_indices, 4, dtype=torch.long)
    assert counts.tolist() == [1, 4, 1, 0]
    assert int(counts.sum()) == topk_indices.numel()


def test_empty_selection_yields_a_zero_histogram():
    counts = sync_free_bincount(torch.empty(0, dtype=torch.long), 3, dtype=torch.long)
    assert counts.tolist() == [0, 0, 0]


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64, torch.int16])
def test_narrow_index_dtypes_are_accepted(index_dtype):
    """DeepEP returns int32 expert ids at ``ep_size > 1``; scatter_add_ demands int64."""
    indices = torch.tensor([2, 2, 0], dtype=index_dtype)
    assert sync_free_bincount(indices, 3, dtype=torch.long).tolist() == [1, 0, 2]


@pytest.mark.parametrize("dtype", [torch.float32, torch.long])
def test_the_counter_dtype_is_the_callers(dtype):
    counts = sync_free_bincount(torch.tensor([1, 1]), 2, dtype=dtype)
    assert counts.dtype is dtype


def test_size_is_the_counter_width_not_the_max_index():
    """No host read-back of ``indices.max()``."""
    assert sync_free_bincount(torch.tensor([0]), 16, dtype=torch.long).numel() == 16


def test_accumulate_creates_on_the_first_call_and_adds_afterwards():
    counter = accumulate_bincount(None, torch.tensor([0, 1]), 3)
    assert counter.tolist() == [1.0, 1.0, 0.0]
    assert counter.dtype is torch.float32

    same = accumulate_bincount(counter, torch.tensor([1, 2]), 3)
    assert same is counter, "the accumulation must be in place — a fresh tensor per step leaks"
    assert counter.tolist() == [1.0, 2.0, 1.0]


def test_weights_count_an_occurrence_that_many_times():
    """What a token shared by several examples needs."""
    indices = torch.tensor([[0, 2], [2, 1]])
    weights = torch.tensor([[3, 3], [1, 1]])
    assert sync_free_bincount(indices, 4, weights=weights).tolist() == [3.0, 1.0, 4.0, 0.0]
    counter = accumulate_bincount(torch.ones(4), indices, 4, weights=weights)
    assert counter.tolist() == [4.0, 2.0, 5.0, 1.0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
