#!/usr/bin/env python
"""The Ulysses exchange refuses a head count or sequence length cp_size does not divide, on every rank before any
collective. 3 heads of dim 4 on 2 ranks flatten evenly, cutting a head in two with no error."""

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from halo.distributed.context_parallel.all_to_all import heads_to_sequence, sequence_to_heads
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
BATCH, LOCAL_SEQ, HEAD_DIM = 1, 4, 4


def _qkv(q_heads: int, kv_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.randn(BATCH, LOCAL_SEQ, heads, HEAD_DIM) for heads in (q_heads, kv_heads, kv_heads))


def _worker(rank: int) -> None:
    group = dist.group.WORLD
    refusals = (
        lambda: sequence_to_heads(*_qkv(3, 2), group),  # Q heads.
        lambda: sequence_to_heads(*_qkv(4, 3), group),  # K/V heads, Q's divisible.
        lambda: heads_to_sequence(torch.randn(BATCH, 5, 2, HEAD_DIM), group),
    )
    no_collective = AssertionError(f"rank {rank}: an indivisible exchange reached the collective")
    with patch.object(dist, "all_to_all_single", side_effect=no_collective):
        for exchange in refusals:
            with pytest.raises(ValueError, match="not divisible by context_parallel_size=2"):
                exchange()


def test_an_indivisible_exchange_raises_on_every_rank_before_the_collective():
    run_gloo_ranks(_worker, WORLD_SIZE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
