"""TP state-dict helpers: the hand-sliced params' all-gather and the plan's sharding lookup.

Both TP mechanisms (HF's ``tp_plan`` styles and the toolkit's attention-only ``parallelize_module``)
place sharded params as DTensors on the TP mesh, which the gathered-save walk reconstructs. Only the
params sliced by hand (GptOss sinks) need the gather implemented here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor
from transformers.distributed.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    TensorParallelLayer,
    _get_parameter_tp_plan,
)

from src.checkpoint.format import save_dtype_caster
from src.distributed.mesh import get_tp_submesh, has_tp_dim
from src.distributed.runtime import is_global_main_process, materialize_dtensor

logger = logging.getLogger(__name__)


def get_tp_mesh(model: nn.Module):
    """The TP device mesh a model was parallelized on, or ``None``.

    The ``None``-on-miss variant of :func:`~src.distributed.mesh.get_tp_submesh`: a model with no
    mesh, or one :func:`~src.distributed.mesh.has_tp_dim` rejects, returns ``None`` rather than
    falling back to a DP-only mesh.
    """
    device_mesh = getattr(model, "_device_mesh", None)
    if device_mesh is None or not has_tp_dim(device_mesh):
        return None
    return get_tp_submesh(device_mesh)


def input_embeddings_tp_sharded(model: nn.Module) -> bool:
    """Whether ``model``'s input embedding is placed as an HF-native TP shard.

    A vocabulary grow rebuilds that module and ``resize_token_embeddings`` cannot re-shard a DTensor,
    so the sharding-agnostic tokenizer setup takes this as a predicate instead.
    """
    return isinstance(model.get_input_embeddings().weight, DTensor)


def tp_sharded_non_dtensor_suffixes(model: nn.Module) -> tuple[str, ...]:
    """Suffixes of params TP-sharded by plain tensor slicing (``model._tp_sharded_non_dtensor``).

    The name-level companion of :func:`iter_tp_sharded_non_dtensor_full`, for consumers that exclude
    or special-case those params by name (the EP save's dense pass, the vLLM dense send).
    """
    return tuple(suffix for suffix, _dim in getattr(model, "_tp_sharded_non_dtensor", None) or ())


def iter_tp_sharded_non_dtensor_full(model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, full GPU tensor)`` for every param TP-sharded by plain tensor slicing.

    Some params (e.g. GptOss attention sinks) carry no TP plan entry and are not DTensors: they are
    sliced by hand and recorded in ``model._tp_sharded_non_dtensor`` as ``(suffix, shard_dim)``.
    Neither the plan-driven gather nor ``full_tensor()`` reconstructs them, so every consumer shipping
    a TP model elsewhere (checkpoint save, vLLM weight sync) must drain this iterator. Collective:
    every TP-group rank must iterate it to exhaustion, in order.
    """
    sharded_info = getattr(model, "_tp_sharded_non_dtensor", None)
    tp_mesh = get_tp_mesh(model)
    if not sharded_info or tp_mesh is None:
        return

    tp_size = tp_mesh.size()
    if tp_size <= 1:
        return

    # Over the TP group rather than world: on EP+TP a world gather would concatenate other groups'
    # head-shards.
    tp_group = tp_mesh.get_group()

    for name, param in model.named_parameters():
        # Iterate the registered list rather than a set of it: str hashing is randomized per process,
        # so a set would order the suffixes differently on each rank and pair mismatched shard_dims
        # in the all_gather below. The registrar dedupes and keeps insertion order.
        for suffix, shard_dim in sharded_info:
            if name.endswith(suffix):
                # DP-sharded under FSDP2: full_tensor over the dp mesh first (collective).
                tensor = materialize_dtensor(param.data)
                if tensor.device.type == "cpu":
                    tensor = tensor.to(f"cuda:{torch.cuda.current_device()}")
                gathered = [torch.empty_like(tensor) for _ in range(tp_size)]
                dist.all_gather(gathered, tensor.contiguous(), group=tp_group)
                yield name, torch.cat(gathered, dim=shard_dim)
                break


def gather_tp_sharded_non_dtensor_params(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    retain: bool = True,
) -> None:
    """Write ``model``'s manually TP-sharded params into ``state_dict`` as full CPU tensors.

    All ranks in the TP group must call this (the gathers are collective); ``retain=False`` joins
    every collective but writes nothing (save-rank gating without N× host copies).

    Cast through the model-derived :func:`save_dtype_caster`, as the callers do for the params they
    resolve themselves: a blanket cast would push a norm / balancing / fp32-pinned tensor that happens
    to be TP-sharded plain down to bf16 while its gathered twin kept the trained dtype.
    """
    cast = save_dtype_caster(model)
    gathered_count = 0
    for name, full_tensor in iter_tp_sharded_non_dtensor_full(model):
        if retain:
            state_dict[name] = cast(name, full_tensor.cpu())
        gathered_count += 1

    if gathered_count > 0 and is_global_main_process():
        logger.info(f"All-gathered {gathered_count} TP-sharded non-DTensor params")


def tp_plan_shards_params(key: str, tp_plan: dict[str, str]) -> bool:
    """Whether HF's TP plan puts ``key`` under a style that shards parameters.

    False for a key the plan does not cover, and for the activation-transform styles
    (``moe_tp_experts``, ``mla_kv_a_proj``, ``sequence_parallel``, ``ep_router``, ...), which install
    forward/backward transforms only and leave their parameters whole on every rank. Decided by
    whether the style class overrides the base no-op ``shard_param``, not by a mirrored style-to-dim
    table: transformers derives placements inside ``shard_param``, conditional on tensor rank
    (``Shard(meta.ndim - 2)``) and module type, so a restating table would drift.

    The plan lookup uses transformers' own ``_get_parameter_tp_plan``, the function
    ``apply_tensor_parallelism`` uses to decide what to shard. A re-implementation would drift: HF's
    wildcarding replaces digits only as whole path components, so a name with an embedded digit
    (``fc1``, ``dense_h_to_4h``) is still sharded.
    """
    style_name = _get_parameter_tp_plan(key, tp_plan)
    if style_name is None:
        return False
    style = ALL_PARALLEL_STYLES.get(style_name)
    if style is None:
        # transformers rejects such a plan at apply time (``_validate_tp_plan_styles``); answering
        # "replicated" here would report a load that cannot run as one that shards nothing.
        raise ValueError(
            f"TP plan style {style_name!r} (for parameter {key!r}) is not registered in transformers' "
            f"ALL_PARALLEL_STYLES ({sorted(ALL_PARALLEL_STYLES.keys())}); nothing can be derived for it."
        )
    return type(style).shard_param is not TensorParallelLayer.shard_param


def reject_plan_sharded_plain_params(model: nn.Module, tp_plan: dict[str, str]) -> None:
    """Reject a load whose applied TP plan claims a param sharded but materialized a plain tensor.

    The zero-shard guard reads the plan's style classes only, never the loaded tensors. A param the
    plan shards that comes back plain is a rank-local slice the trainer cannot distinguish from a
    replica: it lands in the replicated average bucket and the TP group averages disjoint slices.
    Params the toolkit slices by hand are recorded in ``_tp_sharded_non_dtensor`` and exempt.
    """
    hand_sliced = tp_sharded_non_dtensor_suffixes(model)
    offenders = [
        name
        for name, param in model.named_parameters()
        if not isinstance(param, DTensor) and not name.endswith(hand_sliced) and tp_plan_shards_params(name, tp_plan)
    ]
    if offenders:
        raise ValueError(
            f"The applied TP plan shards {len(offenders)} parameter(s) that materialized as plain "
            f"tensors (e.g. {offenders[:3]}): each rank holds a bare slice that the gradient sync "
            f"would average against other ranks' DIFFERENT slices. This is a transformers "
            f"tensor-parallel materialization regression — pin a transformers version whose "
            f"tp_plan='auto' places DTensors, or drop tensor_parallel_size for this model."
        )
