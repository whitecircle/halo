"""Job-wide settings every rank must agree on, and the COLLECTIVE check that refuses a divergence."""

import os
from datetime import timedelta

from halo.checkpoint.writing import resolve_checkpoint_write_threads, validate_checkpoint_write_threads
from halo.distributed.expert_parallel.deepep.capacity import GIN_MAX_TOKENS_PER_RANK
from halo.distributed.expert_parallel.deepep.settings import (
    DEEPEP_GPU_TIMEOUT_S,
    DEEPEP_NUM_QPS,
    DEEPEP_NUM_SMS,
    EP_CAPACITY_DEDUP,
)
from halo.distributed.grad_reduce import GRAD_BUCKET_MB
from halo.env import fs_probe_timeout, nccl_timeout, store_timeout
from halo.models.attention.selection import FLASH_ATTENTION_DETERMINISTIC_ENV, flash_attention_deterministic
from halo.runtime.consensus import reject_divergent_settings
from halo.runtime.ranks import has_process_group

# Fixed at import or written identically on every rank (``full_determinism``), so one join per process is enough.
_env_uniformity_verified = False


def rank_uniform_settings() -> dict[str, object]:
    """Settings every rank of a collective path must agree on, as resolved values, so a launcher exporting a default on
    the head node only still passes. ``EP_DISABLE_GIN`` is DeepEP's raw string; unset, the elastic backend derives it
    from topology identically on every rank of a group."""
    return {
        "HALO_EP_CAPACITY_DEDUP": EP_CAPACITY_DEDUP,
        "HALO_DEEPEP_GPU_TIMEOUT_SECONDS": DEEPEP_GPU_TIMEOUT_S,
        "HALO_DEEPEP_NUM_SMS": DEEPEP_NUM_SMS,
        "HALO_DEEPEP_NUM_QPS": DEEPEP_NUM_QPS,
        "HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK": GIN_MAX_TOKENS_PER_RANK,
        # In the operator's unit, so a divergence message names a value they can grep for.
        "HALO_GRAD_BUCKET_MB": GRAD_BUCKET_MB,
        "HALO_CHECKPOINT_WRITE_THREADS": resolve_checkpoint_write_threads(),
        "DIST_NCCL_TIMEOUT_MINUTES": nccl_timeout() // timedelta(minutes=1),
        "DIST_STORE_TIMEOUT_HOURS": store_timeout() // timedelta(hours=1),
        "DIST_FS_PROBE_TIMEOUT_SECONDS": fs_probe_timeout() // timedelta(seconds=1),
        "EP_DISABLE_GIN": os.environ.get("EP_DISABLE_GIN"),
        FLASH_ATTENTION_DETERMINISTIC_ENV: flash_attention_deterministic(),
    }


def agree_rank_uniform_env() -> None:
    """COLLECTIVE over the world, once per process: raise when ranks disagree on a setting, since a divergence hangs
    (with ``HALO_EP_CAPACITY_DEDUP`` off a rank runs the capacity all-reduce on every MoE layer, its peers once).
    Latches only once a process group exists. Also refuses a negative ``HALO_CHECKPOINT_WRITE_THREADS``."""
    global _env_uniformity_verified
    if _env_uniformity_verified:
        return
    reject_divergent_settings(
        rank_uniform_settings(),
        "Rank-uniform toolkit environment",
        "Every rank of the job must agree on these — the check is world-wide, because "
        "HALO_EP_CAPACITY_DEDUP changes how many collectives a rank runs, HALO_GRAD_BUCKET_MB sets "
        "the chunk boundaries of the bucketed gradient reductions (EP, TP and QLoRA sweeps alike), "
        "HALO_CHECKPOINT_WRITE_THREADS whether a rank refuses a save its peers have entered, "
        "DIST_NCCL_TIMEOUT_MINUTES / DIST_STORE_TIMEOUT_HOURS decide which rank gives up on a join "
        "first, DIST_FS_PROBE_TIMEOUT_SECONDS how long every rank looks for the filesystem probe's file, "
        "FLASH_ATTENTION_DETERMINISTIC which attention backend a head_dim-256 model loads (FA4 or its "
        "fallback), and the rest are wire parameters shared by both ends of the all-to-all.",
    )
    # Agreed by now, so a refused count raises on every rank before a save's first gather.
    validate_checkpoint_write_threads(resolve_checkpoint_write_threads())
    _env_uniformity_verified = has_process_group()
