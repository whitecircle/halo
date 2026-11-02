"""Hand-built collectives over the default group: the backend's tensor device, barriers, broadcasts,
fixed-order group creation and the rank-0 gather and combine of per-rank Python objects."""

from collections.abc import Callable
from typing import Any, TypeVar

import torch
import torch.distributed as dist

from halo.env import nccl_timeout
from halo.hardware import current_device
from halo.runtime.ranks import has_process_group, is_multi_rank_run

# NCCL has no 16-bit-integer type; such tensors ride the wire as a lossless uint8 bit view.
_NCCL_UNSUPPORTED_DTYPES = (torch.int16, torch.uint16)

_Combined = TypeVar("_Combined")


def collective_device(group: dist.ProcessGroup | None = None) -> torch.device:
    """Where a hand-built collective tensor must live for ``group``'s backend (the default group when
    None): CPU for gloo even on a GPU host, where :func:`current_device` answers ``cuda:N``."""
    backend = dist.get_backend(group) if has_process_group() else None
    if backend is not None and backend.lower() == dist.Backend.NCCL and torch.cuda.is_available():
        return current_device()
    return torch.device("cpu")


def barrier(group: dist.ProcessGroup | None = None):
    """Synchronize the ranks of ``group`` (the default group when None); a no-op without one."""
    if has_process_group():
        dist.barrier(group=group)


def nccl_safe_broadcast(tensor: torch.Tensor, src: int, group: dist.ProcessGroup | None = None) -> None:
    """In-place ``dist.broadcast`` that sends NCCL-unsupported dtypes as a uint8 view of the same
    storage, an exact round trip. A non-contiguous such tensor fails in ``Tensor.view``."""
    wire = tensor.view(torch.uint8) if tensor.dtype in _NCCL_UNSUPPORTED_DTYPES else tensor
    dist.broadcast(wire, src=src, group=group)


def broadcast_from_rank0(value):
    """Rank 0's ``value`` on every rank (COLLECTIVE); ``value`` itself without a multi-rank group."""
    if is_multi_rank_run():
        box = [value]
        dist.broadcast_object_list(box, src=0)
        return box[0]
    return value


def _splits_default_group() -> bool:
    """Whether new groups can come from one split of the default group's communicator: an NCCL default group
    initialized on its bound device, the check ``DeviceMesh`` makes before ``split_group``."""
    return (
        dist.get_backend() == dist.Backend.NCCL
        and getattr(dist.distributed_c10d._get_default_group(), "bound_device_id", None) is not None
    )


def build_groups(rank_lists: list[list[int]], *, mine: int, what: str) -> dist.ProcessGroup:
    """Create a process group for every rank list and return the ``mine``-th, which must hold this rank.
    COLLECTIVE.

    Every rank passes the identical, disjoint ``rank_lists``. Over an NCCL default group bound to its device the
    call is one communicator split (``split_group``): a ``new_group`` per list splits the world communicator once
    per list, each split blocking every rank of the job at startup.

    Raises:
        ValueError: the rank lists overlap.
        RuntimeError: this rank is not in the ``mine``-th list.
    """
    members = [rank for ranks in rank_lists for rank in ranks]
    if len(members) != len(set(members)):
        raise ValueError(f"The {what} rank lists overlap, so no split of the world holds them: {rank_lists}")
    rank = dist.get_rank()
    if _splits_default_group():
        my_group = dist.split_group(split_ranks=rank_lists, timeout=nccl_timeout(), group_desc=what)
    else:
        my_group = None
        # One rank skipping or reordering a creation deadlocks the job.
        for index, ranks in enumerate(rank_lists):
            group = dist.new_group(ranks, timeout=nccl_timeout())
            if index == mine:
                my_group = group
    if my_group is None or not 0 <= mine < len(rank_lists) or rank not in rank_lists[mine]:
        raise RuntimeError(f"Rank {rank} was not assigned a {what} (index {mine} of {len(rank_lists)}).")
    return my_group


def gather_on_rank0(payload: Any) -> list[Any] | None:
    """COLLECTIVE: every rank's ``payload`` in rank order on rank 0, None on the others; ``[payload]`` without a
    process group."""
    if not has_process_group():
        return [payload]
    # Non-None on rank 0 alone, as ``gather_object`` requires.
    gathered: list[Any] | None = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(payload, gathered, dst=0)
    return gathered


def combine_across_ranks(payload: Any, combine: Callable[[list[Any]], _Combined], *, what: str) -> _Combined:
    """COLLECTIVE: ``combine`` every rank's ``payload`` on rank 0 and return the result on every rank.

    Rank 0 gathers, combines and broadcasts the one result; an all-gather would leave every rank holding
    one payload per rank of something only the result needs. The combine's failure travels in the result's
    place, so it raises on every rank. Without a process group, ``combine([payload])``.
    """
    if not has_process_group():
        return combine([payload])

    gathered = gather_on_rank0(payload)

    # Any exception, not just the one the combine raises on purpose: one escaping rank 0 here would leave
    # every peer in the next collective until the watchdog fires.
    outcome: list[Any] = [None]
    if gathered is not None:
        try:
            outcome[0] = combine(gathered)
        except Exception as failure:
            # Re-boxed because the failure travels by pickle and only its message is load-bearing.
            outcome[0] = (
                failure
                if type(failure) is RuntimeError
                else RuntimeError(f"merging {what} failed on rank 0: {type(failure).__name__}: {failure}")
            )
    dist.broadcast_object_list(outcome, src=0)

    combined = outcome[0]
    if isinstance(combined, RuntimeError):
        raise combined
    if combined is None:
        raise RuntimeError(f"the merge of {what} never reached this rank: rank 0 broadcast no verdict.")
    return combined


def all_gather_cat(tensor: torch.Tensor, dim: int, group, world_size: int) -> torch.Tensor:
    """COLLECTIVE over ``group``: every rank's equal-shaped ``tensor`` concatenated along ``dim`` in rank order;
    ``tensor`` itself when ``world_size <= 1``.

    ``dim == 0`` receives into one preallocated output: a shard list plus a ``cat`` would hold the gathered tensor
    twice, and an expert-axis gather carries most of a fine-grained MoE's parameters.
    """
    if world_size <= 1:
        return tensor
    tensor = tensor.contiguous()
    if dim == 0:
        gathered = torch.empty(
            (tensor.shape[0] * world_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
        )
        dist.all_gather_into_tensor(gathered, tensor, group=group)
        return gathered
    shards = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(shards, tensor, group=group)
    return torch.cat(shards, dim=dim)
