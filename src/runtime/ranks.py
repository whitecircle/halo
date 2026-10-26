"""Rank and world-size getters, launcher-derived node math, main-process predicates and the rank tag.

The ``launcher_*`` getters and the node math are valid before ``init_process_group``.
"""

import logging
import os
import re

import torch
import torch.distributed as dist

from src.log import warn_once

logger = logging.getLogger(__name__)

# Process-wide: the fallback resolves the same way on every call, so one line says it.
_lws_fallback_announced: set[str] = set()


def has_process_group() -> bool:
    """Whether a default process group is up, so ``dist.get_rank()`` and collectives can run."""
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    """Global rank of this process, or 0 if distributed is not initialized."""
    return dist.get_rank() if has_process_group() else 0


def get_global_world_size() -> int:
    """Total process count across all nodes, or 1 if not initialized."""
    return dist.get_world_size() if has_process_group() else 1


def launcher_global_rank() -> int:
    """Global rank from the live process group, else the launcher's ``RANK`` / ``SLURM_PROCID``.

    ``RANK`` wins: under ``srun torchrun`` both are set and ``SLURM_PROCID`` counts srun tasks (one per
    node), not ranks.
    """
    if has_process_group():
        return dist.get_rank()
    return int(os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or 0)


def launcher_global_world_size() -> int:
    """World size from the live process group, else ``WORLD_SIZE`` / ``SLURM_NTASKS`` (1 when none).

    ``SLURM_NTASKS`` counts only alongside ``SLURM_PROCID`` (a bare ``srun`` launch): ``sbatch`` and
    ``salloc`` export it to a plain ``python`` launch that really is world 1.
    """
    if has_process_group():
        return dist.get_world_size()
    slurm_world = os.environ.get("SLURM_NTASKS") if "SLURM_PROCID" in os.environ else None
    return int(os.environ.get("WORLD_SIZE") or slurm_world or 1)


def get_local_world_size() -> int:
    """Processes on this node: ``LOCAL_WORLD_SIZE`` / ``SLURM_NTASKS_PER_NODE``, else the visible CUDA
    device count, else the global world size (warned once: it collapses the run to one node).
    """
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])

    # Bare srun: SLURM may report a compact/list form ("8(x2)", "4,8"), so parse leading int.
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
            _lws_fallback_announced,
            "local_world_size",
            "Could not determine local world size from LOCAL_WORLD_SIZE / "
            "SLURM_NTASKS_PER_NODE / CUDA device count; falling back to global "
            f"world_size={world} (assuming single node). On a real multi-node job "
            "this builds WRONG node-local EP/CP groups — set LOCAL_WORLD_SIZE "
            "explicitly or pass gpus_per_node to ParallelismConfig.",
        )
    return world


def _valid_local_world_size() -> int | None:
    """:func:`get_local_world_size` when positive, else None after a warning: a nonsensical value must
    not silently collapse every rank onto node 0 / ``cuda:0``."""
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
    """Local rank within this node: ``LOCAL_RANK`` / ``SLURM_LOCALID``, else the global rank modulo
    this node's process count."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])

    local_world_size = _valid_local_world_size()
    return 0 if local_world_size is None else launcher_global_rank() % local_world_size


def get_num_nodes() -> int:
    """Number of nodes, or 1 if not distributed."""
    local_world_size = _valid_local_world_size()
    return 1 if local_world_size is None else max(1, launcher_global_world_size() // local_world_size)


def get_node_rank() -> int:
    """Rank of the current node, or 0 if not distributed."""
    local_world_size = _valid_local_world_size()
    return 0 if local_world_size is None else launcher_global_rank() // local_world_size


def is_multi_rank_run() -> bool:
    """Whether a collective would reach a peer: an initialized group of more than one rank."""
    return has_process_group() and dist.get_world_size() > 1


def is_global_main_process() -> bool:
    """Whether this is the global main process (rank 0)."""
    return get_global_rank() == 0


def is_local_main_process() -> bool:
    """Whether this is the main process on its node (local rank 0)."""
    return get_local_rank() == 0


def rank_tag() -> str:
    """``rankNN`` tag for the current global rank, zero-padded to world-size width."""
    rank = get_global_rank()
    width = max(2, len(str(max(0, get_global_world_size() - 1))))
    return f"rank{rank:0{width}d}"
