"""The two registries of plain (non-DTensor) parameters TP leaves partial on each rank. Hand-sliced ones (GptOss sinks)
hold this rank's slice, so every consumer shipping the model must all-gather them. Per-head norm ones are replicas
whose gradient covers this rank's heads, so the trainer SUMs them over the TP group once per optimizer step."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn

from halo.distributed.mesh import model_tp_mesh
from halo.runtime.dtensor import materialize_dtensor

_HAND_SLICED_ATTR = "_halo_tp_hand_sliced"
_PER_HEAD_NORM_PARAMS_ATTR = "_halo_tp_per_head_norm_params"


def register_hand_sliced(model: nn.Module, suffix: str, shard_dim: int) -> None:
    """Record that every parameter named ``*suffix`` holds this rank's slice along ``shard_dim``."""
    entries = hand_sliced_dims(model)
    if entries.get(suffix, shard_dim) != shard_dim:
        raise ValueError(
            f"{suffix!r} is already registered as hand-sliced along dim {entries[suffix]}, not {shard_dim}"
        )
    entries[suffix] = shard_dim
    setattr(model, _HAND_SLICED_ATTR, entries)


def hand_sliced_dims(model: nn.Module) -> dict[str, int]:
    """``{suffix: shard_dim}`` of the hand-sliced parameters, in registration order."""
    return dict(getattr(model, _HAND_SLICED_ATTR, None) or {})


def hand_sliced_suffixes(model: nn.Module) -> tuple[str, ...]:
    return tuple(hand_sliced_dims(model))


def iter_hand_sliced_full(model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """COLLECTIVE over the TP group, drained in order on every rank: ``(name, full GPU tensor)`` of each
    hand-sliced parameter, which ``full_tensor()`` never reconstructs."""
    dims = hand_sliced_dims(model)
    tp_mesh = model_tp_mesh(model)
    if not dims or tp_mesh is None:
        return

    tp_size = tp_mesh.size()
    if tp_size <= 1:
        return

    # Over the TP group: on EP+TP a world gather would concatenate other groups' head-shards.
    tp_group = tp_mesh.get_group()

    for name, param in model.named_parameters():
        # Registration order, never a set's: str hashing is randomized per process, so a set would order the
        # suffixes differently on each rank and pair mismatched shard dims in the all_gather below.
        for suffix, shard_dim in dims.items():
            if name.endswith(suffix):
                # DP-sharded under FSDP2: full_tensor over the dp mesh first (collective).
                tensor = materialize_dtensor(param.data)
                if tensor.device.type == "cpu":
                    tensor = tensor.to(f"cuda:{torch.cuda.current_device()}")
                gathered = [torch.empty_like(tensor) for _ in range(tp_size)]
                dist.all_gather(gathered, tensor.contiguous(), group=tp_group)
                yield name, torch.cat(gathered, dim=shard_dim)
                break


def register_per_head_norm_params(model: nn.Module, modules: list[nn.Module]) -> None:
    """Add the parameters of ``modules`` to those the trainer SUMs over the TP group once per step (a name
    missing would be AVG-reduced: ``1/tp_size`` of its gradient). Names, not ids: FSDP2 replaces the
    ``Parameter`` objects after this runs."""
    if not modules:
        return
    owned = {id(p) for module in modules for p in module.parameters()}
    registered = {name for name, p in model.named_parameters() if id(p) in owned}
    registered.update(per_head_norm_param_names(model))
    setattr(model, _PER_HEAD_NORM_PARAMS_ATTR, sorted(registered))


def per_head_norm_param_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(getattr(model, _PER_HEAD_NORM_PARAMS_ATTR, None) or ())
