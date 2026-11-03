"""Rank getters and node math; the ``launcher_*`` getters and node math work before ``init_process_group``."""

import logging
import os
import re

import torch
import torch.distributed as dist

from halo.log import warn_once

logger = logging.getLogger(__name__)

# The fallback resolves alike on every call, so warn once per process.
_local_world_size_fallback_warned: set[str] = set()


def has_process_group() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    return dist.get_rank() if has_process_group() else 0


def get_global_world_size() -> int:
    return dist.get_world_size() if has_process_group() else 1


def launcher_global_rank() -> int:
    """Global rank from the live group, else ``RANK`` / ``SLURM_PROCID``. ``RANK`` wins: under ``srun torchrun``
    ``SLURM_PROCID`` counts srun tasks (one per node)."""
    if has_process_group():
        return dist.get_rank()
    return int(os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or 0)


def launcher_global_world_size() -> int:
    """World size from the live group, else ``WORLD_SIZE`` / ``SLURM_NTASKS``. ``SLURM_NTASKS`` counts only with
    ``SLURM_PROCID`` (bare ``srun``): ``sbatch`` and ``salloc`` export it to a world-1 ``python`` launch."""
    if has_process_group():
        return dist.get_world_size()
    slurm_world = os.environ.get("SLURM_NTASKS") if "SLURM_PROCID" in os.environ else None
    return int(os.environ.get("WORLD_SIZE") or slurm_world or 1)


def get_local_world_size() -> int:
    """``LOCAL_WORLD_SIZE`` / ``SLURM_NTASKS_PER_NODE``, else the visible CUDA device count, else the global world size
    (warned once, since it collapses the run to one node)."""
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])

    # SLURM may report a compact form ("8(x2)", "4,8"); take the leading int.
    slurm_ntpn = os.environ.get("SLURM_NTASKS_PER_NODE")
    if slurm_ntpn:
        m = re.match(r"\d+", slurm_ntpn.strip())
        if m:
            return int(m.group())

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 0:
            return count

    # Same last resort before and after init_process_group, so the answer cannot change mid-run.
    world = launcher_global_world_size()
    if world > 1:
        warn_once(
            logger,
            _local_world_size_fallback_warned,
            "local_world_size",
            "Could not determine local world size from LOCAL_WORLD_SIZE / "
            "SLURM_NTASKS_PER_NODE / CUDA device count; falling back to global "
            f"world_size={world} (assuming single node). On a real multi-node job "
            "this builds WRONG node-local EP/CP groups — set LOCAL_WORLD_SIZE "
            "explicitly or pass gpus_per_node to ParallelismConfig.",
        )
    return world


def _valid_local_world_size() -> int | None:
    """:func:`get_local_world_size` when positive, else None after a warning, so a bad value cannot silently put every
    rank on node 0 / ``cuda:0``."""
    local_world_size = get_local_world_size()
    if local_world_size > 0:
        return local_world_size
    logger.warning(
        "Local world size resolved to %d, which is not a positive process count — node-local rank "
        "math degrades to a single node with this rank at index 0. Set LOCAL_WORLD_SIZE explicitly "
        "(or pass gpus_per_node to ParallelismConfig).",
        local_world_size,
    )
    return None


def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])

    local_world_size = _valid_local_world_size()
    return 0 if local_world_size is None else launcher_global_rank() % local_world_size


def get_num_nodes() -> int:
    local_world_size = _valid_local_world_size()
    return 1 if local_world_size is None else max(1, launcher_global_world_size() // local_world_size)


def get_node_rank() -> int:
    local_world_size = _valid_local_world_size()
    return 0 if local_world_size is None else launcher_global_rank() // local_world_size


def is_multi_rank_run() -> bool:
    return has_process_group() and dist.get_world_size() > 1


def is_global_main_process() -> bool:
    return get_global_rank() == 0


def is_local_main_process() -> bool:
    return get_local_rank() == 0


def rank_tag() -> str:
    rank = get_global_rank()
    width = max(2, len(str(max(0, get_global_world_size() - 1))))
    return f"rank{rank:0{width}d}"
