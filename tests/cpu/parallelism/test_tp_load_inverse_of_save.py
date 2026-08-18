#!/usr/bin/env python
"""The TP resume load must place exactly what transformers' TP load placed, for every plan style.

``CheckpointLoader._load_tp`` reads each checkpoint tensor whole and
``distribute_tensor(value, mesh, param.placements, src_data_rank=None)``s it into the live DTensor.
The checkpoint holds ``full_tensor()`` of the same DTensor, so the load is correct exactly when
``distribute_tensor`` under the live placements reproduces the shard ``shard_param`` created — a
property that is obvious for a contiguous ``Shard`` and NOT for the packed styles: a
``packed_colwise`` weight is ``_StridedShard(dim, split_factor)``, rank r holding the r-th chunk of
EACH packed block. A load that chunked it contiguously would hand the low ranks gate-only rows and
the high ranks up-only rows, with matching shapes and no error.

The oracle is transformers' own shard machinery run per rank on a fake process group
(``distribute_tensor(..., src_data_rank=None)`` slices locally, no collectives), so a single CPU
process can obtain every rank's shard from the real placement code and compare.

    python tests/cpu/parallelism/test_tp_load_inverse_of_save.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

from tests.common.distributed import fake_process_group_mesh

TP_SIZE = 4
# A dense projection weight and a grouped-GEMM expert stack; every dim a multiple of
# TP_SIZE * split_factor so each style shards without padding.
LINEAR_SHAPE = (64, 32)
EXPERT_SHAPE = (4, 64, 32)
STYLES = ("colwise", "rowwise", "packed_colwise", "packed_rowwise")


def _shard_with_transformers(style_name: str, full: torch.Tensor, mesh):
    """The DTensor transformers' load would leave on this rank for ``full`` under ``style_name``."""
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
    """``embedding_rowwise`` is on the resume path for every TIED dense TP model: its injected entry
    is what shards the embedding/head pair, so the reload must invert it too.

    The module CLASS decides the dim — this style shards an ``nn.Embedding`` on the vocab
    (``Shard(0)``) and a Linear on the input dim — so the oracle has to be a real ``nn.Embedding``,
    not the bare holder the other styles use.
    """
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
    """Anti-vacuity for the packed rows above: a plain chunk would NOT reproduce the shard, so the
    equality is not an identity that any placement satisfies."""
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
