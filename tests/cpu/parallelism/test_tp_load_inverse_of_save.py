#!/usr/bin/env python
"""The TP resume load places what transformers' TP load placed, for every plan style. ``packed_colwise`` is
``_StridedShard``, so a contiguous load hands ranks gate-only or up-only rows."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

from tests.common.distributed import fake_process_group_mesh

TP_SIZE = 4
# Every dim a multiple of TP_SIZE * split_factor, so each style shards without padding.
LINEAR_SHAPE = (64, 32)
EXPERT_SHAPE = (4, 64, 32)
STYLES = ("colwise", "rowwise", "packed_colwise", "packed_rowwise")


def _shard_with_transformers(style_name: str, full: torch.Tensor, mesh):
    holder = nn.Module()
    holder.weight = nn.Parameter(full.clone())
    ALL_PARALLEL_STYLES[style_name].shard_param(holder, "weight", mesh)
    return holder.weight


@pytest.mark.parametrize("shape", [LINEAR_SHAPE, EXPERT_SHAPE], ids=["linear", "grouped_gemm_experts"])
@pytest.mark.parametrize("style_name", STYLES)
def test_load_reproduces_the_shard_transformers_placed(style_name, shape):
    full = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
    for rank in range(TP_SIZE):
        with fake_process_group_mesh(rank, TP_SIZE) as mesh:
            placed = _shard_with_transformers(style_name, full, mesh)
            loaded = distribute_tensor(full, mesh, placed.placements, src_data_rank=None)
            assert loaded.placements == placed.placements
            assert torch.equal(loaded.to_local(), placed.to_local()), f"{style_name} rank {rank}"


def test_the_tied_embedding_style_loads_back_into_its_own_shard():
    """The module class decides the dim, so the oracle must be a real ``nn.Embedding``."""
    full = torch.arange(torch.Size(LINEAR_SHAPE).numel(), dtype=torch.float32).reshape(LINEAR_SHAPE)
    for rank in range(TP_SIZE):
        with fake_process_group_mesh(rank, TP_SIZE) as mesh:
            embedding = nn.Embedding(*LINEAR_SHAPE)
            embedding.weight = nn.Parameter(full.clone())
            ALL_PARALLEL_STYLES["embedding_rowwise"].shard_param(embedding, "weight", mesh)
            placed = embedding.weight
            assert placed.placements == (Shard(0),), "premise: the vocab dim is what shards"
            loaded = distribute_tensor(full, mesh, placed.placements, src_data_rank=None)
            assert torch.equal(loaded.to_local(), placed.to_local()), f"embedding_rowwise rank {rank}"


def test_the_packed_styles_really_are_strided():
    """Anti-vacuity: a plain chunk would not reproduce the shard."""
    full = torch.arange(torch.Size(LINEAR_SHAPE).numel(), dtype=torch.float32).reshape(LINEAR_SHAPE)
    for rank in range(TP_SIZE):
        with fake_process_group_mesh(rank, TP_SIZE) as mesh:
            placed = _shard_with_transformers("packed_colwise", full, mesh)
            assert isinstance(placed.placements[0], _StridedShard)
            contiguous = full.chunk(TP_SIZE, dim=0)[rank]
            assert contiguous.shape == placed.to_local().shape
            assert not torch.equal(contiguous, placed.to_local())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
