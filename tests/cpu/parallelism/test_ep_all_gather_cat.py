#!/usr/bin/env python
"""``all_gather_cat`` allocates once, ahead of the collective, on every axis. On the expert axis a shard list plus ``cat``
holds the tensor twice, tens of GB per layer at 397B/ep64."""

import pytest
import torch

from halo.runtime import collectives as runtime_collectives
from halo.runtime.collectives import all_gather_cat

WORLD = 4
GROUP = "fake-ep-group"


def _peer_shard(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """Distinct per rank, so a wrong offset or dropped peer changes the value."""
    return tensor + 100.0 * rank


def _expected(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.cat([_peer_shard(tensor, rank) for rank in range(WORLD)], dim=dim)


@pytest.fixture
def collectives(monkeypatch):
    """Emulates a ``WORLD``-rank group for both spellings, counting each."""
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

    monkeypatch.setattr(runtime_collectives.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    monkeypatch.setattr(runtime_collectives.dist, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch, "empty", counting_empty)
    monkeypatch.setattr(torch, "empty_like", counting_empty_like)
    monkeypatch.setattr(torch, "cat", counting_cat)
    return counts


def test_expert_axis_gather_matches_the_shard_list_and_concat(collectives):
    """The layout the merge and every loader expect."""
    local = torch.randn(2, 3, 5)
    gathered = all_gather_cat(local, 0, GROUP, WORLD)
    assert gathered.shape == (2 * WORLD, 3, 5)
    torch.testing.assert_close(gathered, _expected(local, 0))


def test_expert_axis_gather_allocates_one_full_buffer(collectives):
    """The peak is one gathered tensor."""
    local = torch.randn(2, 3, 5)
    all_gather_cat(local, 0, GROUP, WORLD)
    assert collectives["into_tensor"] == 1
    assert collectives["list"] == 0
    assert collectives["empty"] == 1, "one preallocated receive buffer"
    assert collectives["empty_like"] == 0, "a per-peer shard list doubles the transient"
    assert collectives["cat"] == 0, "concatenating the shards is the second full copy"
    assert collectives["widest_alloc"] == local.numel() * WORLD


def test_intermediate_axis_gather_allocates_once_before_the_collective(collectives):
    """A later allocation would fail with the collective behind it, and a caller retrying would pair with the wrong
    tensor; the peers' columns are written through views of one buffer, as one receive would interleave them."""
    local = torch.randn(2, 3, 5)
    gathered = all_gather_cat(local, 2, GROUP, WORLD)
    during = dict(collectives)  # The reference below concatenates too, through the same counters.
    assert gathered.shape == (2, 3, 5 * WORLD)
    torch.testing.assert_close(gathered, _expected(local, 2))
    assert during["into_tensor"] == 0
    assert during["list"] == 1
    assert during["empty"] == 1 and during["empty_like"] == 0
    assert during["cat"] == 0


@pytest.mark.parametrize("dim", (0, 2))
def test_single_rank_group_is_identity(collectives, dim):
    """An ep1 save must not issue a collective."""
    local = torch.randn(2, 3, 5)
    assert all_gather_cat(local, dim, GROUP, 1) is local
    assert collectives["into_tensor"] == 0
    assert collectives["list"] == 0


def test_non_contiguous_input_is_gathered_by_value(collectives):
    """The ETP branch hands in transposed views; the buffer carries values, not aliased strides."""
    local = torch.randn(3, 2, 5).transpose(0, 1)
    assert not local.is_contiguous()
    gathered = all_gather_cat(local, 0, GROUP, WORLD)
    torch.testing.assert_close(gathered, _expected(local.contiguous(), 0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
