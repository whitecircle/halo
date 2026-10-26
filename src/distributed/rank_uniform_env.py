"""Job-wide settings every rank must agree on, and the COLLECTIVE check that refuses a divergence."""

import os
from datetime import timedelta

import torch.distributed as dist

from src.distributed.expert_parallel.deepep.capacity import GIN_MAX_TOKENS_PER_RANK
from src.distributed.expert_parallel.deepep.settings import (
    DEEPEP_GPU_TIMEOUT_S,
    DEEPEP_NUM_QPS,
    DEEPEP_NUM_SMS,
    EP_CAPACITY_DEDUP,
)
from src.distributed.grad_reduce import GRAD_BUCKET_MB
from src.env import nccl_timeout, store_timeout
from src.runtime.consensus import reject_divergent_settings

# The settings are fixed at import, so one join per process is enough.
_env_uniformity_verified = False


def rank_uniform_settings() -> dict[str, object]:
    """The settings every rank of a collective path must agree on, as resolved values.

    Resolved rather than raw: :mod:`src.env` reads an absent variable and one set to its default
    identically, so raw strings would reject a launcher that exports a default on the head node only.
    ``EP_DISABLE_GIN`` is DeepEP's own and read there as a raw string; unset, the elastic backend
    derives it from topology identically on every rank of a group.
    """
    return {
        "HALO_EP_CAPACITY_DEDUP": EP_CAPACITY_DEDUP,
        "HALO_DEEPEP_GPU_TIMEOUT_SECONDS": DEEPEP_GPU_TIMEOUT_S,
        "HALO_DEEPEP_NUM_SMS": DEEPEP_NUM_SMS,
        "HALO_DEEPEP_NUM_QPS": DEEPEP_NUM_QPS,
        "HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK": GIN_MAX_TOKENS_PER_RANK,
        # In the unit the operator sets, so a divergence message names a value they can grep for.
        "HALO_GRAD_BUCKET_MB": GRAD_BUCKET_MB,
        "DIST_NCCL_TIMEOUT_MINUTES": nccl_timeout() // timedelta(minutes=1),
        "DIST_STORE_TIMEOUT_HOURS": store_timeout() // timedelta(hours=1),
        "EP_DISABLE_GIN": os.environ.get("EP_DISABLE_GIN"),
    }


def agree_rank_uniform_env() -> None:
    """COLLECTIVE over the world. Raise when ranks disagree on a setting in :func:`rank_uniform_settings`.

    A divergence is a hang, not a slow run: a rank with ``HALO_EP_CAPACITY_DEDUP`` off runs the
    capacity all-reduce on every MoE layer while its peers run it once. Runs at most once per process,
    and latches only once a process group exists: before ``init_process_group`` the join compares nothing.
    """
    global _env_uniformity_verified
    if _env_uniformity_verified:
        return
    reject_divergent_settings(
        rank_uniform_settings(),
        "Rank-uniform toolkit environment",
        "Every rank of the job must agree on these — the check is world-wide, because "
        "HALO_EP_CAPACITY_DEDUP changes how many collectives a rank runs, HALO_GRAD_BUCKET_MB sets "
        "the chunk boundaries of the bucketed gradient reductions (EP, TP and QLoRA sweeps alike), "
        "DIST_NCCL_TIMEOUT_MINUTES / DIST_STORE_TIMEOUT_HOURS decide which rank gives up on a join "
        "first, and the rest are wire parameters shared by both ends of the all-to-all.",
    )
    _env_uniformity_verified = dist.is_available() and dist.is_initialized()
