"""Simulated job topologies on a CPU-only box: a :class:`ParallelismConfig` or single-process
:class:`EPConfig` for any shape, and a ``torch.distributed`` whose process groups compare as their
member tuples. A config takes its topology as constructor fields; only the global rank, an
``init=False`` field, is patched in.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from unittest.mock import patch

import torch.distributed as dist

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


@contextlib.contextmanager
def simulated_world(world_size: int, gpus_per_node: int, rank: int = 0):
    """Present a ``world_size``-rank job on ``gpus_per_node``-GPU nodes, as ``rank``, to a config that
    resolves its topology from the dist primitives (``parallelism_config_from_args`` passes none)."""
    with (
        patch(f"{_MOD}.get_global_world_size", return_value=world_size),
        patch(f"{_MOD}.get_local_world_size", return_value=gpus_per_node),
        _mocked_rank(rank),
    ):
        yield


class StubGroup(tuple):
    """A process group of :func:`stubbed_process_groups`: its member ranks, compared and hashed as the
    tuple of them, and the keyword arguments ``new_group`` built it with. Each call returns a distinct
    object, so ``is`` tells two groups of the same members apart."""

    kwargs: dict

    def __new__(cls, ranks: Iterable[int], kwargs: dict) -> StubGroup:
        group = super().__new__(cls, ranks)
        group.kwargs = kwargs
        return group


@contextlib.contextmanager
def stubbed_process_groups(world_size: int, rank: int, calls: list[StubGroup] | None = None) -> Iterator[None]:
    """``torch.distributed`` as global rank ``rank`` of a live ``world_size``-rank job, where
    ``new_group`` returns a :class:`StubGroup`, which compares equal to its member tuple and is read back
    by ``get_process_group_ranks``. Every group built is appended to ``calls`` when one is given."""

    def new_group(ranks=None, **kwargs):
        group = StubGroup(ranks, kwargs)
        if calls is not None:
            calls.append(group)
        return group

    with (
        patch.object(dist, "is_initialized", new=lambda: True),
        patch.object(dist, "get_rank", new=lambda group=None: rank),
        patch.object(dist, "get_world_size", new=lambda group=None: world_size),
        patch.object(dist, "new_group", new=new_group),
        patch.object(dist, "get_process_group_ranks", new=list),
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

    ``nvlink_domain_size`` defaults to ``gpus_per_node`` rather than to the field default, so a test
    never inherits the host's ``NVLINK_DOMAIN_SIZE`` (an NVL72 box would resolve 72 and change every
    domain-locality verdict). Pass ``nvlink_domain_size=0`` to exercise that resolution.
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

    Takes no ``local_rank``: production is never handed one (every group derives from the global rank
    and the NVLink-domain coordinate), so accepting one here would let a test assert a locality claim
    the config could not have read.
    """
    return make_parallelism_config(**kwargs)


def single_process_ep_config(num_experts: int | None = None, **overrides) -> EPConfig:
    """An :class:`EPConfig` owning every expert on one rank, no process group.

    The shape every CPU wrapper test needs: ``ep_size=1`` on a world of 1 distributes nothing, so an
    EP layer built on it runs the same routing and expert compute a real rank runs while the
    comparison stays in-process. ``use_grouped_gemm=False`` keeps the eager expert loop (the
    grouped-GEMM kernel needs SM90+; a GPU caller overrides it). A layer built directly reads
    ``expert_start_idx``/``expert_end_idx`` at construction, so ``num_experts`` finalizes the expert
    assignment here; :func:`patch_moe_model_for_ep` finalizes it off the model when it is left unset.
    """
    config = EPConfig(**{"ep_size": 1, "world_size": 1, "gpus_per_node": 1, "use_grouped_gemm": False, **overrides})
    if num_experts is not None:
        config.finalize_expert_assignment(num_experts)
    return config
