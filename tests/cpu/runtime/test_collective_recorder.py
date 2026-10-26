#!/usr/bin/env python
"""The collective recorder the loader trace golden runs on: it catches a collective under any name a
module holds it by, refuses one it has no model of, and models what a rank receives from its peers.

Run: python tests/cpu/runtime/test_collective_recorder.py
"""

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed import distributed_c10d as c10d
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from src.runtime.collectives import broadcast_from_rank0
from tests.common.collective_recorder import recording_rank


def test_a_dtensor_scatter_is_recorded_through_its_own_binding():
    """DTensor's collective helpers import ``scatter`` from ``distributed_c10d`` by name."""
    with recording_rank(0, 2, 2) as trace:
        distribute_tensor(torch.ones(4, 4), init_device_mesh("cpu", (2,)), [Shard(0)])
    assert trace.events == ["init_device_mesh cpu (2,) None", "scatter group_src=0 float32[2, 4] [0, 1]"]


@pytest.mark.parametrize(
    "issue",
    [
        lambda: dist.all_to_all_single(torch.ones(2), torch.ones(2)),
        lambda: dist.all_reduce_coalesced([torch.ones(2)]),
        lambda: c10d.reduce_scatter_tensor(torch.ones(1), torch.ones(2)),
        lambda: funcol.all_reduce(torch.ones(2), "sum", dist.group.WORLD),
    ],
    ids=["all_to_all_single", "all_reduce_coalesced", "reduce_scatter_tensor", "functional all_reduce"],
)
def test_a_collective_with_no_model_raises(issue):
    with recording_rank(0, 2, 2), pytest.raises(NotImplementedError, match="has no model of"):
        issue()


def test_a_rank_receives_the_object_broadcasts_rank_0_made():
    with recording_rank(0, 2, 2) as rank0:
        assert broadcast_from_rank0("rank 0's verdict") == "rank 0's verdict"
    with recording_rank(1, 2, 2, rank0=rank0):
        assert broadcast_from_rank0("rank 1's own") == "rank 0's verdict"
    with (
        recording_rank(1, 2, 2, rank0=rank0),
        pytest.raises(RuntimeError, match="more often than rank 0 did"),
    ):
        broadcast_from_rank0("first")
        broadcast_from_rank0("one rank 0 never sent")


def test_a_sum_counts_every_rank_and_a_maximum_none_twice():
    with recording_rank(0, 4, 2) as trace:
        total, maximum = torch.ones(1), torch.ones(1)
        dist.all_reduce(total)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    assert (total.item(), maximum.item()) == (4.0, 1.0)
    assert trace.events == ["all_reduce SUM float32[1] world", "all_reduce MAX float32[1] world"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
