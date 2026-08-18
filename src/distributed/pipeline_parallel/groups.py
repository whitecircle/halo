"""Pipeline process groups — the per-chain and per-stage groups PP runs on, and the chain's
last-stage broadcasts.

A **chain** is the set of ranks holding the same intra-stage position across every stage; it carries
all of PP's point-to-point activation traffic and nothing else. Members are ``stage_world_size``
apart, so on the intended placement each sits in a different NVLink domain and the chain is exactly
the traffic that crosses EFA/IB. A **stage group** is one contiguous rank block — the FSDP
data-parallel width under PP+FSDP and the scope for any collective over ranks holding the same
layers (loss-token counts, stage-local metrics, per-stage gathers).

Group creation is collective: every rank calls ``dist.new_group`` for EVERY group in the same fixed
order and keeps only its own, matching the invariant ``EPConfig._create_process_groups`` follows.

Only the last stage computes a loss or a prediction; the broadcasts below are how those reach the
rest of the chain, so every rank logs the same value and HF's rank-uniform eval path has something
to gather.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import current_device, get_nccl_timeout

logger = logging.getLogger(__name__)


def _create_groups_fixed_order(rank_lists: list[list[int]], mine: int, what: str) -> dist.ProcessGroup:
    """Create a process group for every rank list, in order; return the ``mine``-th.

    Every rank must call this with the identical ``rank_lists`` — ``dist.new_group`` is collective,
    and a rank skipping or reordering a creation deadlocks the job.
    """
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("Pipeline parallelism requires an initialized process group (launch with torchrun).")
    my_group = None
    for index, ranks in enumerate(rank_lists):
        group = dist.new_group(ranks, timeout=get_nccl_timeout())
        if index == mine:
            my_group = group
    if my_group is None:
        raise RuntimeError(f"Rank {dist.get_rank()} was not assigned a pipeline {what} (index {mine}).")
    return my_group


def _stage_rank_blocks(config: ParallelismConfig) -> list[list[int]]:
    """Each stage's contiguous rank block, in stage order — the rank layout both groups slice.

    Row ``s`` is stage ``s``'s ranks; column ``p`` is the chain of intra-stage position ``p``, which
    is why the chain lists below are this matrix transposed rather than a second rank formula.
    """
    return [
        list(range(stage * config.stage_world_size, (stage + 1) * config.stage_world_size))
        for stage in range(config.pp_size)
    ]


def create_pipeline_group(config: ParallelismConfig) -> dist.ProcessGroup | None:
    """Create every pipeline chain's process group; return the one containing this rank.

    Returns ``None`` when PP is disabled.
    """
    if config.pp_size <= 1:
        return None
    chains = [list(chain) for chain in zip(*_stage_rank_blocks(config), strict=True)]
    group = _create_groups_fixed_order(chains, config.stage_local_rank, "chain")
    logger.info(
        "Pipeline chain for rank %d: %s (stage %d/%d)",
        config.global_rank,
        chains[config.stage_local_rank],
        config.pp_rank,
        config.pp_size,
    )
    return group


def create_stage_group(config: ParallelismConfig) -> dist.ProcessGroup | None:
    """Create every stage's process group; return the one containing this rank.

    Returns ``None`` when PP is disabled.
    """
    if config.pp_size <= 1:
        return None
    return _create_groups_fixed_order(_stage_rank_blocks(config), config.pp_rank, "stage group")


def broadcast_scalar_from_last_stage(tensor: torch.Tensor | None, src: int, group: dist.ProcessGroup) -> torch.Tensor:
    """Broadcast the last stage's scalar down the chain so every rank holds the same value.

    Non-last stages pass ``None`` and receive into a scalar fp32 buffer, so the source is cast to
    that shape and dtype here: NCCL validates neither across ranks, and an adapter whose loss is
    bf16 would land as garbage on every stage that did not compute it. The shape is known here, so
    this stays a single collective on the per-step path; :func:`broadcast_tensor_from_last_stage`
    pays a metadata hop for tensors whose shape is not.
    """
    tensor = tensor.detach().reshape(()).float() if tensor is not None else torch.zeros((), device=current_device())
    dist.broadcast(tensor, src=src, group=group)
    return tensor


def broadcast_tensor_from_last_stage(tensor: torch.Tensor | None, src: int, group: dist.ProcessGroup) -> torch.Tensor:
    """Broadcast a last-stage tensor down the chain, shape and dtype carried with it.

    Unlike the scalar broadcast the receiving ranks cannot know the shape: it is whatever the last
    stage produced. One object hop carries the metadata, so no caller has to declare a shape it does
    not own — worth it off the per-step path only.
    """
    meta = [(tuple(tensor.shape), tensor.dtype) if tensor is not None else None]
    dist.broadcast_object_list(meta, src=src, group=group)
    described = meta[0]
    if described is None:
        raise RuntimeError(
            "The last pipeline stage produced no tensor to broadcast. Every stage reaches this "
            "collective, so the chain would hang here rather than fail on the stage at fault."
        )
    shape, dtype = described
    if tensor is None:
        tensor = torch.empty(shape, dtype=dtype, device=current_device())
    tensor = tensor.contiguous()
    dist.broadcast(tensor, src=src, group=group)
    return tensor
