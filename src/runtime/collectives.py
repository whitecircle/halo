"""Hand-built collectives over the default group: the backend's tensor device, barriers, broadcasts,
fixed-order group creation and the rank-0 combine of per-rank Python objects."""

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

import torch
import torch.distributed as dist

from src.env import nccl_timeout
from src.hardware import current_device
from src.runtime.ranks import has_process_group, is_multi_rank_run

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


@contextmanager
def barrier_on_exit(group: dist.ProcessGroup | None = None):
    """Run a save-rank-only write, then barrier even if the write raised, so a writer's I/O failure
    propagates instead of hanging its peers in the barrier. COLLECTIVE."""
    try:
        yield
    finally:
        barrier(group)


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


def build_groups(rank_lists: list[list[int]], *, mine: int, what: str) -> dist.ProcessGroup:
    """Create a process group for every rank list, in order, and return the ``mine``-th. COLLECTIVE.

    Every rank passes the identical ``rank_lists``: one skipping or reordering a creation deadlocks the job.
    """
    my_group = None
    for index, ranks in enumerate(rank_lists):
        group = dist.new_group(ranks, timeout=nccl_timeout())
        if index == mine:
            my_group = group
    if my_group is None:
        raise RuntimeError(f"Rank {dist.get_rank()} was not assigned a {what} (index {mine}).")
    return my_group


def combine_across_ranks(payload: Any, combine: Callable[[list[Any]], _Combined], *, what: str) -> _Combined:
    """COLLECTIVE: ``combine`` every rank's ``payload`` on rank 0 and return the result on every rank.

    Rank 0 gathers, combines and broadcasts the one result; an all-gather would leave every rank holding
    one payload per rank of something only the result needs. The combine's failure travels in the result's
    place, so it raises on every rank. Without a process group, ``combine([payload])``.
    """
    if not has_process_group():
        return combine([payload])

    # Non-None on rank 0 alone, as ``gather_object`` requires.
    gathered: list[Any] | None = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(payload, gathered, dst=0)

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
