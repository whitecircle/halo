"""Hand-built collectives over the default group."""

import functools
import itertools
import pickle
from collections.abc import Callable
from typing import Any, TypeVar

import torch
import torch.distributed as dist

from halo.env import nccl_timeout
from halo.hardware import current_device
from halo.runtime.ranks import has_process_group, is_multi_rank_run

# NCCL has no 16-bit-integer type; such tensors ride the wire as a lossless uint8 bit view.
_NCCL_UNSUPPORTED_DTYPES = (torch.int16, torch.uint16)
# Per-round bytes :func:`gather_on_rank0` holds on each rank's collective device; a larger gather takes more rounds.
_GATHER_WINDOW_BYTES = 64 * 1024 * 1024

_Combined = TypeVar("_Combined")


def collective_device(group: dist.ProcessGroup | None = None) -> torch.device:
    """Device for a hand-built collective tensor on ``group``'s backend; CPU for gloo even on a GPU host."""
    backend = dist.get_backend(group) if has_process_group() else None
    if backend is not None and backend.lower() == dist.Backend.NCCL and torch.cuda.is_available():
        return current_device()
    return torch.device("cpu")


@functools.cache
def _side_stream(device_index: int) -> torch.cuda.Stream:
    """One stream per device for the process, since the caching allocator keeps a pool per stream."""
    return torch.cuda.Stream(device=device_index)


def collective_side_stream() -> torch.cuda.Stream | None:
    """The stream a collective runs on beside the caller's, so later work on the caller's stream does not wait for
    it; None without CUDA."""
    return _side_stream(torch.cuda.current_device()) if torch.cuda.is_available() else None


def barrier(group: dist.ProcessGroup | None = None):
    if has_process_group():
        dist.barrier(group=group)


def nccl_safe_broadcast(tensor: torch.Tensor, src: int, group: dist.ProcessGroup | None = None) -> None:
    """In-place ``dist.broadcast``; NCCL-unsupported dtypes go as a uint8 view of the same storage, so a non-contiguous
    one fails in ``Tensor.view``."""
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
    """Whether the default group is NCCL on a bound device, the check ``DeviceMesh`` makes before ``split_group``."""
    return (
        dist.get_backend() == dist.Backend.NCCL
        and getattr(dist.distributed_c10d._get_default_group(), "bound_device_id", None) is not None
    )


def build_groups(rank_lists: list[list[int]], *, mine: int, what: str) -> dist.ProcessGroup:
    """COLLECTIVE: create a group per rank list and return the ``mine``-th. Every rank passes identical, disjoint
    ``rank_lists``. On a bound NCCL default group this is one ``split_group``; a ``new_group`` per list splits the
    world communicator once per list, each split blocking every rank of the job at startup."""
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
    """COLLECTIVE: every rank's ``payload`` in rank order on rank 0, None on the others. SUM-reduces the pickles laid
    end to end over the communicator's existing ring; ``gather_object`` under NCCL connects every peer to rank 0 point
    to point, holding about 0.3 GiB per NVLink peer for the communicator's lifetime."""
    if not has_process_group():
        return [payload]
    rank, world = dist.get_rank(), dist.get_world_size()
    device = collective_device()
    mine = torch.frombuffer(bytearray(pickle.dumps(payload)), dtype=torch.uint8)
    sizes = torch.empty(world, dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(sizes, torch.tensor([mine.numel()], dtype=torch.int64, device=device))
    offsets = list(itertools.accumulate(sizes.tolist(), initial=0))
    begin, end, total = offsets[rank], offsets[rank + 1], offsets[-1]
    received = memoryview(bytearray(total)) if rank == 0 else None
    window = torch.empty(min(total, _GATHER_WINDOW_BYTES), dtype=torch.uint8, device=device)
    for start in range(0, total, _GATHER_WINDOW_BYTES):
        span = window[: min(_GATHER_WINDOW_BYTES, total - start)].zero_()
        low, high = max(begin, start), min(end, start + span.numel())
        if low < high:
            span[low - start : high - start].copy_(mine[low - begin : high - begin])
        dist.reduce(span, dst=0)
        if received is not None:
            received[start : start + span.numel()] = span.cpu().numpy()
    if received is None:
        return None
    return [pickle.loads(received[offsets[peer] : offsets[peer + 1]]) for peer in range(world)]


def group_holds_rank(group: dist.ProcessGroup | None, rank: int) -> bool:
    """Whether global ``rank`` is a member of ``group`` (``None`` is the world)."""
    return group is None or rank in dist.get_process_group_ranks(group)


def combine_across_ranks(payload: Any, combine: Callable[[list[Any]], _Combined], *, what: str) -> _Combined:
    """COLLECTIVE: ``combine`` every rank's ``payload`` on rank 0 and return the result on every rank; its failure
    raises on every rank. An all-gather would leave every rank holding one payload per rank."""
    if not has_process_group():
        return combine([payload])

    gathered = gather_on_rank0(payload)

    # Any exception: one escaping rank 0 here leaves every peer in the next collective until the watchdog fires.
    outcome: list[Any] = [None]
    if gathered is not None:
        try:
            outcome[0] = combine(gathered)
        except Exception as failure:
            # Re-boxed since it travels by pickle and only its message matters.
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
    """COLLECTIVE over ``group``: every rank's equal-shaped ``tensor`` concatenated along ``dim`` in rank order. Its one
    allocation comes before the collective, so a raise out of it means this rank never entered, and a caller can free
    memory and call again in step with its peers. ``dim == 0`` receives in place, since shards plus a ``cat`` would hold
    twice an expert-axis gather, which carries most of a fine-grained MoE's parameters."""
    if world_size <= 1:
        return tensor
    tensor = tensor.contiguous()
    shape = list(tensor.shape)
    shape[dim] *= world_size
    gathered = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    if dim == 0:
        dist.all_gather_into_tensor(gathered, tensor, group=group)
    else:
        dist.all_gather(list(gathered.split(tensor.shape[dim], dim=dim)), tensor, group=group)
    return gathered


def gather_cat_to_rank(
    tensor: torch.Tensor, group: dist.ProcessGroup | None, world_size: int, *, dst: int, receive: bool = True
) -> torch.Tensor | None:
    """COLLECTIVE over ``group``, which must hold global rank ``dst``: every rank's equal-shaped ``tensor`` concatenated
    along dim 0 in group-rank order, on ``dst`` only (None elsewhere, and on ``dst`` without ``receive``). Like
    :func:`gather_on_rank0` it SUM-reduces windows of the bytes over the communicator's existing ring, each rank
    contributing zeros outside its own span, so the result is bit exact and no rank but ``dst`` holds more than a window.
    ``dst`` allocates its result before the collective, so a raise out of it means it never entered and a caller can
    free memory and call again with ``receive=False`` in step with its peers, which discards the result."""
    if world_size <= 1:
        return tensor
    tensor = tensor.contiguous()
    position = dist.get_rank(group)
    sends = tensor.view(-1).view(torch.uint8)
    total = sends.numel() * world_size
    begin, end = position * sends.numel(), (position + 1) * sends.numel()
    result = None
    if receive and dist.get_rank() == dst:
        result = torch.empty(
            (tensor.shape[0] * world_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
        )
        wire = result.view(-1).view(torch.uint8)
    else:
        scratch = torch.empty(min(total, _GATHER_WINDOW_BYTES), dtype=torch.uint8, device=tensor.device)
    for start in range(0, total, _GATHER_WINDOW_BYTES):
        stop = min(start + _GATHER_WINDOW_BYTES, total)
        window = (wire[start:stop] if result is not None else scratch[: stop - start]).zero_()
        low, high = max(begin, start), min(end, stop)
        if low < high:
            window[low - start : high - start].copy_(sends[low - begin : high - begin])
        dist.reduce(window, dst=dst, group=group)
    return result
