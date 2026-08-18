"""Build a :class:`ParallelismConfig` (or a single-process :class:`EPConfig`) for an arbitrary
simulated topology on a CPU-only box.

``world_size`` / ``gpus_per_node`` / ``nvlink_domain_size`` are real constructor fields that the
config consults the dist primitives for only when they are left at ``0`` — so a test states its
topology by passing them, never by patching. The one primitive that cannot be passed is
``get_global_rank``: ``ParallelismConfig.global_rank`` is ``init=False``.
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.parallelism_config import ParallelismConfig

_MOD = "src.distributed.parallelism_config"


@contextlib.contextmanager
def _mocked_rank(rank: int):
    """Present ``rank`` as this process's global rank (and main-process status) while inside."""
    with (
        patch(f"{_MOD}.get_global_rank", return_value=rank),
        patch(f"{_MOD}.is_global_main_process", return_value=rank == 0),
    ):
        yield


def make_parallelism_config(
    *,
    world_size: int = 8,
    gpus_per_node: int = 8,
    nvlink_domain_size: int | None = None,
    rank: int = 0,
    **kwargs,
) -> ParallelismConfig:
    """A config for the given simulated topology, as rank ``rank``.

    ``nvlink_domain_size`` defaults to ``gpus_per_node`` rather than to the field default, so a
    test never inherits the host's ``NVLINK_DOMAIN_SIZE`` (an NVL72 box would resolve 72 and change
    every domain-locality verdict). Pass ``nvlink_domain_size=0`` to exercise that resolution.
    """
    with _mocked_rank(rank):
        return ParallelismConfig(
            world_size=world_size,
            gpus_per_node=gpus_per_node,
            nvlink_domain_size=gpus_per_node if nvlink_domain_size is None else nvlink_domain_size,
            **kwargs,
        )


def create_config(**kwargs):
    """A ParallelismConfig on the simulated topology ``world_size``/``gpus_per_node``, as ``rank``.

    Takes no ``local_rank``: production is never handed one (every group derives from the GLOBAL
    rank and the NVLink-domain coordinate), so swallowing one here would let a test assert a
    locality claim the config could not have read.
    """
    return make_parallelism_config(**kwargs)


def single_process_ep_config(num_experts: int, **overrides) -> EPConfig:
    """An :class:`EPConfig` owning all ``num_experts`` experts on one rank, no process group.

    The shape every CPU wrapper test needs: ``ep_size=1`` on a world of 1 distributes nothing, so an
    EP layer built on it runs the same routing and expert compute a real rank runs while the
    comparison stays in-process. ``use_grouped_gemm=False`` keeps the eager expert loop — the
    grouped-GEMM kernel needs SM90+ — and the expert assignment is finalized here because the layers
    read ``expert_start_idx``/``expert_end_idx`` at construction.
    """
    config = EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False, **overrides)
    config.finalize_expert_assignment(num_experts)
    return config
