"""DTensor-agnostic tensor access; full-tensor reads and writes are mesh COLLECTIVES."""

from collections.abc import Iterable

import torch
from torch.distributed.tensor import DTensor, distribute_tensor


def materialize_dtensor(data: torch.Tensor | None) -> torch.Tensor | None:
    """A DTensor's full tensor, autograd-aware (COLLECTIVE over its mesh); others pass through."""
    if isinstance(data, DTensor):
        return data.full_tensor()
    return data


def copy_full_tensor(target: torch.Tensor, value: torch.Tensor, *, src_data_rank: int | None = 0) -> None:
    """Copy whole ``value`` into live ``target``. A DTensor broadcasts mesh rank 0's copy (COLLECTIVE, same key order
    on every rank); ``src_data_rank=None`` shards each rank's own whole copy, collective-free."""
    data = target.data if isinstance(target, torch.nn.Parameter) else target
    value = value.to(data.dtype)
    if isinstance(data, DTensor):
        value = distribute_tensor(value, data.device_mesh, data.placements, src_data_rank=src_data_rank)
    data.copy_(value)


def to_local(tensor: torch.Tensor) -> torch.Tensor:
    """This rank's shard of a DTensor, no collective; plain tensors pass through."""
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


def local_numel(param: torch.Tensor) -> int:
    """Elements on this rank; ``DTensor.numel()`` is global, an FSDP-ignored EP expert tensor already per-rank."""
    return to_local(param).numel()


def full_cpu_tensor(param_data: torch.Tensor) -> torch.Tensor:
    """The full tensor on CPU, detached: a DTensor through ``full_tensor()`` (COLLECTIVE over its mesh)."""
    if isinstance(param_data, DTensor):
        return param_data.full_tensor().cpu()
    return param_data.detach().cpu()


def gather_full_tensors(items: Iterable[tuple[str, torch.Tensor]], *, retain: bool) -> dict[str, torch.Tensor]:
    """COLLECTIVE: gather every item on every rank, keeping host copies only where ``retain``. Gating the gather itself
    on ``retain`` would hang a gathered save."""
    resolved: dict[str, torch.Tensor] = {}
    for name, tensor in items:
        if retain:
            resolved[name] = full_cpu_tensor(tensor)
        else:
            materialize_dtensor(tensor)
        # Free it before ``items`` yields the next, or a derived item (a LoRA fold) lives beside its successor.
        del tensor
    return resolved
