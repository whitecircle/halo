#!/usr/bin/env python
"""CPU tests for the sync-free routing histogram shared by the EP layers and the MoE metrics callback.

``src.kernels.histogram`` replaces ``torch.bincount`` on the routing hot path, where bincount's
device→host read-back of the max index costs a full sync per MoE layer per forward. Three
properties are load-bearing and each has a way of failing silently:

* the counts must equal ``bincount``'s — a miscounting histogram feeds grouped-GEMM the wrong
  offsets, which mis-slices experts rather than raising;
* ``int32`` indices must be accepted — DeepEP hands back int32 local expert ids at ``ep_size > 1``,
  and ``scatter_add_`` requires int64, so a dropped cast breaks every distributed EP run while
  ``ep_size == 1`` (int64) stays green;
* the counter dtype is the CALLER's choice — fp32 for the metric accumulators, int64 for the
  grouped-GEMM offsets, where inexact adds would drift the boundaries.

Run: ``python tests/cpu/kernels/test_histogram.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch

from src.kernels.histogram import accumulate_bincount, sync_free_bincount


def test_counts_match_bincount():
    indices = torch.tensor([3, 0, 0, 7, 3, 3])
    counts = sync_free_bincount(indices, 8, dtype=torch.long)
    assert torch.equal(counts, torch.bincount(indices, minlength=8))


def test_a_multi_dimensional_selection_is_flattened():
    """Routing selections arrive as ``[T, top_k]``; every (token, slot) pair counts once."""
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
    """The whole point of the pre-sized buffer: no host read-back of ``indices.max()``."""
    assert sync_free_bincount(torch.tensor([0]), 16, dtype=torch.long).numel() == 16


def test_accumulate_creates_on_the_first_call_and_adds_afterwards():
    counter = accumulate_bincount(None, torch.tensor([0, 1]), 3)
    assert counter.tolist() == [1.0, 1.0, 0.0]
    assert counter.dtype is torch.float32

    same = accumulate_bincount(counter, torch.tensor([1, 2]), 3)
    assert same is counter, "the accumulation must be in place — a fresh tensor per step leaks"
    assert counter.tolist() == [1.0, 2.0, 1.0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
