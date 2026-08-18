#!/usr/bin/env python
"""CPU tests for the EP expert-axis all-gather transport.

``EPMoELayerBase._all_gather_cat`` is the collective behind every gathered save, shard merge and vLLM
weight sync. On the expert axis (``dim=0``) it must receive into ONE preallocated output: an
``all_gather`` shard list plus a ``cat`` holds the whole gathered tensor TWICE on every rank of the
group, and the expert axis is where a fine-grained MoE keeps essentially all of its parameters (tens
of GB per layer at 397B/ep64). Expert-TP gathers split the intermediate dim, which has no
single-buffer form, so those keep the list + concat.

Asserted here: the two paths agree value-for-value, and the dim-0 path allocates exactly one
full-size buffer — a revert to shard-list + ``cat`` shows up as extra allocations, not just as slower
code.

Run: pytest tests/cpu/parallelism/test_ep_all_gather_cat.py
"""

import sys

import pytest
import torch

from src.distributed.expert_parallel import expert_gather
from src.distributed.expert_parallel.base_layer import EPMoELayerBase

WORLD = 4
GROUP = "fake-ep-group"


def _peer_shard(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """What peer ``rank`` contributes. Distinct per rank, so a wrong offset or a dropped peer changes
    the value rather than only the shape."""
    return tensor + 100.0 * rank


def _expected(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.cat([_peer_shard(tensor, rank) for rank in range(WORLD)], dim=dim)


@pytest.fixture
def collectives(monkeypatch):
    """Emulate a ``WORLD``-rank group for both all-gather spellings, counting each."""
    counts = {"into_tensor": 0, "list": 0, "empty": 0, "empty_like": 0, "cat": 0, "widest_alloc": 0}

    def fake_all_gather_into_tensor(output, tensor, group=None):
        assert group == GROUP
        counts["into_tensor"] += 1
        rows = tensor.shape[0]
        for rank in range(WORLD):
            output[rank * rows : (rank + 1) * rows].copy_(_peer_shard(tensor, rank))

    def fake_all_gather(shards, tensor, group=None):
        assert group == GROUP
        counts["list"] += 1
        for rank, shard in enumerate(shards):
            shard.copy_(_peer_shard(tensor, rank))

    real_empty, real_empty_like, real_cat = torch.empty, torch.empty_like, torch.cat

    def counting_empty(*args, **kwargs):
        counts["empty"] += 1
        out = real_empty(*args, **kwargs)
        counts["widest_alloc"] = max(counts["widest_alloc"], out.numel())
        return out

    def counting_empty_like(*args, **kwargs):
        counts["empty_like"] += 1
        out = real_empty_like(*args, **kwargs)
        counts["widest_alloc"] = max(counts["widest_alloc"], out.numel())
        return out

    def counting_cat(*args, **kwargs):
        counts["cat"] += 1
        out = real_cat(*args, **kwargs)
        counts["widest_alloc"] = max(counts["widest_alloc"], out.numel())
        return out

    monkeypatch.setattr(expert_gather.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    monkeypatch.setattr(expert_gather.dist, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch, "empty", counting_empty)
    monkeypatch.setattr(torch, "empty_like", counting_empty_like)
    monkeypatch.setattr(torch, "cat", counting_cat)
    return counts


def test_expert_axis_gather_matches_the_shard_list_and_concat(collectives):
    """Same values as ``all_gather`` + ``cat`` — the layout the merge and every loader expect."""
    local = torch.randn(2, 3, 5)
    gathered = EPMoELayerBase._all_gather_cat(local, 0, GROUP, WORLD)
    assert gathered.shape == (2 * WORLD, 3, 5)
    torch.testing.assert_close(gathered, _expected(local, 0))


def test_expert_axis_gather_allocates_one_full_buffer(collectives):
    """The peak is one gathered tensor, not a shard list plus its concatenation."""
    local = torch.randn(2, 3, 5)
    EPMoELayerBase._all_gather_cat(local, 0, GROUP, WORLD)
    assert collectives["into_tensor"] == 1
    assert collectives["list"] == 0
    assert collectives["empty"] == 1, "one preallocated receive buffer"
    assert collectives["empty_like"] == 0, "a per-peer shard list doubles the transient"
    assert collectives["cat"] == 0, "concatenating the shards is the second full copy"
    assert collectives["widest_alloc"] == local.numel() * WORLD


def test_intermediate_axis_gather_keeps_the_concat_path(collectives):
    """Expert-TP splits the intermediate dim, where a single receive buffer would interleave the
    peers' columns; the list + concat stays, and stays correct."""
    local = torch.randn(2, 3, 5)
    gathered = EPMoELayerBase._all_gather_cat(local, 2, GROUP, WORLD)
    during = dict(collectives)  # the reference below concatenates too, through the same counters
    assert gathered.shape == (2, 3, 5 * WORLD)
    torch.testing.assert_close(gathered, _expected(local, 2))
    assert during["into_tensor"] == 0
    assert during["list"] == 1
    assert during["cat"] == 1


@pytest.mark.parametrize("dim", (0, 2))
def test_single_rank_group_is_identity(collectives, dim):
    """No group to gather across: neither collective may be entered (an ep1 save must not issue one)."""
    local = torch.randn(2, 3, 5)
    assert EPMoELayerBase._all_gather_cat(local, dim, GROUP, 1) is local
    assert collectives["into_tensor"] == 0
    assert collectives["list"] == 0


def test_non_contiguous_input_is_gathered_by_value(collectives):
    """The ETP branch hands in transposed views; the receive buffer must carry their VALUES, not the
    strides of whatever storage they aliased."""
    local = torch.randn(3, 2, 5).transpose(0, 1)
    assert not local.is_contiguous()
    gathered = EPMoELayerBase._all_gather_cat(local, 0, GROUP, WORLD)
    torch.testing.assert_close(gathered, _expected(local.contiguous(), 0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
