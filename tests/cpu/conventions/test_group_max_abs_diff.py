#!/usr/bin/env python
"""``tests.common.distributed.group_max_abs_diff`` reads replica identity the way its GPU callers need, on two real
gloo ranks: a difference on one rank reads on every rank, and a NaN reads NaN, which a ``== 0.0`` verdict fails."""

import math

import pytest
import torch
import torch.distributed as dist

from tests.common.distributed import group_max_abs_diff, max_abs_diff_across
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
SHIFT = 0.25


def _worker(rank: int) -> None:
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert group_max_abs_diff(base) == 0.0, "identical replicas must read 0.0"

    shifted = base.clone()
    if rank == 1:
        shifted[1, 2] += SHIFT
    assert group_max_abs_diff(shifted, dist.group.WORLD) == SHIFT, "every rank reads the peer's difference"

    poisoned = base.clone()
    if rank == 1:
        poisoned[0, 0] = math.nan
    assert math.isnan(group_max_abs_diff(poisoned)), "a NaN on one rank must not read as agreement"

    assert group_max_abs_diff(torch.empty(0, 3)) == 0.0, "an empty FSDP2 shard has nothing to differ"
    assert max_abs_diff_across([torch.empty(0, 3), shifted]) == SHIFT, "an empty shard hides no other difference"

    assert max_abs_diff_across([base, shifted]) == SHIFT, "the largest difference of the tensors"
    assert math.isnan(max_abs_diff_across([shifted, poisoned])), "a NaN after a larger difference still reads NaN"
    assert math.isnan(max_abs_diff_across([])), "no tensors compared must not read as agreement"


def test_group_max_abs_diff_reads_identity_difference_and_nan():
    run_gloo_ranks(_worker, WORLD_SIZE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
