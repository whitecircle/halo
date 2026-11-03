"""Pipeline process groups and the chain's last-stage broadcasts. A chain holds the ranks at one intra-stage position
across stages and carries P2P activations; a stage group is one contiguous rank block, the FSDP scope."""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from halo.distributed.parallelism_config import ParallelismConfig
from halo.hardware import current_device
from halo.runtime.collectives import build_groups
from halo.runtime.ranks import has_process_group

logger = logging.getLogger(__name__)


def build_pipeline_group(config: ParallelismConfig) -> dist.ProcessGroup | None:
    """COLLECTIVE: create every pipeline chain's process group; this rank's, or ``None`` off PP."""
    if config.pp_size <= 1:
        return None
    if not has_process_group():
        raise RuntimeError("Pipeline parallelism requires an initialized process group (launch with torchrun).")
    chains = [list(chain) for chain in zip(*config.pp_stage_rank_blocks(), strict=True)]
    group = build_groups(chains, mine=config.stage_local_rank, what="pipeline chain")
    logger.info(
        "Pipeline chain for rank %d: %s (stage %d/%d)",
        config.global_rank,
        chains[config.stage_local_rank],
        config.pp_rank,
        config.pp_size,
    )
    return group


def build_stage_group(config: ParallelismConfig) -> dist.ProcessGroup | None:
    """COLLECTIVE: create every stage's process group; this rank's, or ``None`` off PP."""
    if config.pp_size <= 1:
        return None
    if not has_process_group():
        raise RuntimeError("Pipeline parallelism requires an initialized process group (launch with torchrun).")
    return build_groups(config.pp_stage_rank_blocks(), mine=config.pp_rank, what="pipeline stage group")


def broadcast_scalar_from_last_stage(tensor: torch.Tensor | None, src: int, group: dist.ProcessGroup) -> torch.Tensor:
    """COLLECTIVE over the chain: the last stage's scalar on every rank. Non-last stages receive into a scalar fp32
    buffer, so the source is cast to match here (NCCL validates neither shape nor dtype)."""
    tensor = tensor.detach().reshape(()).float() if tensor is not None else torch.zeros((), device=current_device())
    dist.broadcast(tensor, src=src, group=group)
    return tensor


def broadcast_tensor_from_last_stage(tensor: torch.Tensor | None, src: int, group: dist.ProcessGroup) -> torch.Tensor:
    """COLLECTIVE over the chain: a last-stage tensor on every rank, its shape and dtype carried by
    one object hop first (off the per-step path only)."""
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
