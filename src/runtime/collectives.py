"""Hand-built collectives over the default group: the backend's tensor device, barriers, broadcasts
and fixed-order group creation."""

from contextlib import contextmanager

import torch
import torch.distributed as dist

from src.env import nccl_timeout
from src.hardware import current_device
from src.runtime.ranks import has_process_group, is_multi_rank_run

# NCCL has no 16-bit-integer type; such tensors ride the wire as a lossless uint8 bit view.
_NCCL_UNSUPPORTED_DTYPES = (torch.int16, torch.uint16)


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
