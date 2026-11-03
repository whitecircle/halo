"""Simulated job topologies on a CPU-only box; a config takes its topology as fields, and only the ``init=False``
global rank is patched."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from unittest.mock import patch

import torch.distributed as dist

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.runtime import collectives
from halo.runtime.ranks import get_global_rank

_MOD = "halo.distributed.parallelism_config"
_RULES_MOD = "halo.distributed.parallelism_rules"


@contextlib.contextmanager
def _mocked_rank(rank: int):
    with (
        patch(f"{_MOD}.get_global_rank", return_value=rank),
        patch(f"{_MOD}.is_global_main_process", return_value=rank == 0),
        patch(f"{_RULES_MOD}.is_global_main_process", return_value=rank == 0),
    ):
        yield


@contextlib.contextmanager
def simulated_world(world_size: int, gpus_per_node: int, rank: int = 0):
    """Present a ``world_size``-rank job as ``rank`` to a config resolving its topology from dist primitives."""
    with (
        patch(f"{_MOD}.get_global_world_size", return_value=world_size),
        patch(f"{_MOD}.get_local_world_size", return_value=gpus_per_node),
        _mocked_rank(rank),
    ):
        yield


class StubGroup(tuple):
    """A process group stand-in: hashes and compares as its member tuple; ``is`` tells two groups of the same members
    apart."""

    kwargs: dict

    def __new__(cls, ranks: Iterable[int], kwargs: dict) -> StubGroup:
        group = super().__new__(cls, ranks)
        group.kwargs = kwargs
        return group


@contextlib.contextmanager
def stubbed_process_groups(
    world_size: int, rank: int, calls: list[StubGroup] | None = None, *, splits: list[tuple] | None = None
) -> Iterator[None]:
    """``torch.distributed`` as ``rank`` of a live job, where ``new_group`` returns a :class:`StubGroup` and appends it
    to ``calls``. With ``splits``, the default group counts as device-bound NCCL and ``split_group`` returns the split
    holding ``rank``, or None."""

    def new_group(ranks=None, **kwargs):
        group = StubGroup(ranks, kwargs)
        if calls is not None:
            calls.append(group)
        return group

    def split_group(parent_pg=None, split_ranks=None, **kwargs):
        splits.append((kwargs.get("group_desc"), [list(ranks) for ranks in split_ranks]))
        return next((StubGroup(sorted(ranks), kwargs) for ranks in split_ranks if rank in ranks), None)

    with (
        patch.object(dist, "is_initialized", new=lambda: True),
        patch.object(dist, "get_rank", new=lambda group=None: rank),
        patch.object(dist, "get_world_size", new=lambda group=None: world_size),
        # The split verdict alone: a real default group keeps answering its collectives.
        patch.object(collectives, "_splits_default_group", new=lambda: splits is not None),
        patch.object(dist, "new_group", new=new_group),
        patch.object(dist, "split_group", new=split_group),
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
    """A config for the simulated topology, as ``rank``. ``nvlink_domain_size`` defaults to ``gpus_per_node`` so a test
    never inherits the host's."""
    with _mocked_rank(rank):
        return ParallelismConfig(
            world_size=world_size,
            gpus_per_node=gpus_per_node,
            nvlink_domain_size=gpus_per_node if nvlink_domain_size is None else nvlink_domain_size,
            **kwargs,
        )


def built_ep_config(config: ParallelismConfig) -> EPConfig:
    with stubbed_process_groups(config.world_size, config.global_rank):
        return config.ep_config()


def create_config(**kwargs):
    """A ParallelismConfig on the simulated topology. Takes no ``local_rank``: production is never handed one."""
    return make_parallelism_config(**kwargs)


def stubbed_rank_ep_config(world_size: int, ep_size: int, rank: int, num_experts: int, **overrides) -> EPConfig:
    """Rank ``rank``'s :class:`EPConfig` at ``ep_size``-way dispatch and the quotient expert TP, over stubbed groups,
    experts FSDP-ignored."""
    with stubbed_process_groups(world_size, rank):
        config = EPConfig(
            ep_size=ep_size,
            expert_tp_size=world_size // ep_size,
            stage_world_size=world_size,
            nvlink_domain_size=world_size,
            experts_fsdp_managed=False,
            **overrides,
        )
        config.finalize_expert_assignment(num_experts)
    return config


def single_process_ep_config(num_experts: int | None = None, **overrides) -> EPConfig:
    """An :class:`EPConfig` owning every expert on one rank, no process group. ``use_grouped_gemm=False`` keeps the
    eager loop, since the kernel needs SM90+. ``num_experts`` finalizes the assignment for a layer built directly."""
    config = EPConfig(
        **{
            "ep_size": 1,
            "stage_world_size": 1,
            "stage_base_rank": get_global_rank(),
            "nvlink_domain_size": 1,
            "use_grouped_gemm": False,
            "experts_fsdp_managed": True,
            **overrides,
        }
    )
    if num_experts is not None:
        config.finalize_expert_assignment(num_experts)
    return config
