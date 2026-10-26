"""DTensor-agnostic tensor access: the full tensor (a mesh COLLECTIVE), the local shard, whole-tensor
writes into a live parameter, and the retain-gated gather a gathered save runs on every rank."""

from collections.abc import Iterable

import torch
from torch.distributed.tensor import DTensor, distribute_tensor


def materialize_dtensor(data: torch.Tensor | None) -> torch.Tensor | None:
    """A DTensor's full tensor, autograd-aware (COLLECTIVE over its mesh); ``None`` and plain tensors
    pass through."""
    if isinstance(data, DTensor):
        return data.full_tensor()
    return data


def copy_full_tensor(target: torch.Tensor, value: torch.Tensor) -> None:
    """Copy the whole tensor ``value`` into the live parameter or buffer ``target``, at its dtype.

    A DTensor target goes through ``distribute_tensor``, which broadcasts mesh rank 0's copy: a mesh
    COLLECTIVE every rank issues in the same key order.
    """
    data = target.data if isinstance(target, torch.nn.Parameter) else target
    value = value.to(data.dtype)
    if isinstance(data, DTensor):
        value = distribute_tensor(value, data.device_mesh, data.placements)
    data.copy_(value)


def to_local(tensor: torch.Tensor) -> torch.Tensor:
    """This rank's shard of a DTensor, no collective; plain tensors pass through."""
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    return tensor


def local_numel(param: torch.Tensor) -> int:
    """Element count held by this rank; ``DTensor.numel()`` reports the global size, while an
    FSDP-ignored EP expert tensor is already per-rank."""
    return to_local(param).numel()


def full_cpu_tensor(param_data: torch.Tensor) -> torch.Tensor:
    """The full tensor on CPU, detached: a DTensor through ``full_tensor()`` (COLLECTIVE over its mesh)."""
    if isinstance(param_data, DTensor):
        return param_data.full_tensor().cpu()
    return param_data.detach().cpu()


def resolve_retained(items: Iterable[tuple[str, torch.Tensor]], *, retain: bool) -> dict[str, torch.Tensor]:
    """Gather every item on every rank; return the host copies only where ``retain``. COLLECTIVE.

    Gating the gather itself on ``retain`` would hang a gathered save; non-writers join through
    :func:`materialize_dtensor`, skipping the host copy.
    """
    resolved: dict[str, torch.Tensor] = {}
    for name, tensor in items:
        if retain:
            resolved[name] = full_cpu_tensor(tensor)
        else:
            materialize_dtensor(tensor)
    return resolved
