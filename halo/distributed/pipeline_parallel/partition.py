"""The pipeline layer partition: where the stage boundaries go, and the gate every boundary passes."""

from __future__ import annotations

import torch.nn as nn

from halo.distributed.pipeline_parallel.families import resolve_pp_spec
from halo.distributed.pipeline_parallel.model_rules import require_backbone
from halo.models.attention.layout import resolve_head_dim, resolve_num_key_value_heads
from halo.models.config_access import get_config_field, text_config
from halo.models.module_tree import backbone_with_layers, decoder_layers
from halo.models.moe.config_fields import has_routed_experts, resolve_router_topk


def layer_types_shift_invariant(layer_types, shift: int) -> bool:
    """Whether re-basing a decoder-layer list by ``shift`` leaves every layer's ``layer_types`` entry unchanged."""
    return not shift or not layer_types or layer_types[shift:] == layer_types[:-shift]


def validate_partition(model: nn.Module, partition: list[tuple[int, int]]) -> None:
    """Raise if any stage boundary would silently change the model. Run on every rank, so an invalid split fails them
    together. Several families select each layer's mask as ``config.layer_types[i]`` by list position, which slicing
    re-bases: an offset must be a whole number of periods unless the spec is ``layer_types_rebase_safe``."""
    spec = resolve_pp_spec(backbone_with_layers(model))
    if not spec.layer_types_rebase_safe:
        # Read through the same accessor layer_types_period uses, so a manual pipeline_split is
        # gated on exactly the field the automatic partition snaps to.
        layer_types = getattr(text_config(getattr(model, "config", None)), "layer_types", None)
        for stage, (lo, _) in enumerate(partition):
            if not layer_types_shift_invariant(layer_types, lo):
                raise ValueError(
                    f"{type(model).__name__} selects its attention mask by position in the decoder-layer "
                    f"list (config.layer_types), and pipeline stage {stage} starts at layer {lo}, which is "
                    f"not a whole number of periods of that pattern {tuple(layer_types[:4])}... . Choose a "
                    f"pipeline_parallel_size or pipeline_split whose boundaries land on the pattern period."
                )
    if spec.reject_severed_boundaries is not None:
        spec.reject_severed_boundaries(model, partition)


def _spread(total: int, parts: int) -> list[int]:
    """``total`` in ``parts`` near-equal counts, the remainder middle-out: stage 0 carries the
    embedding and the last stage the head, so the ends stay light."""
    sizes = [total // parts] * parts
    for i in sorted(range(parts), key=lambda i: (-min(i, parts - 1 - i), i))[: total % parts]:
        sizes[i] += 1
    return sizes


def _balanced_sizes(num_layers: int, pp_size: int, head_layer_equivalents: float, period: int) -> list[int]:
    """Per-stage layer counts; the last stage's budget shrinks by the head's layer-equivalent cost. With ``period > 1``
    every boundary is a whole number of periods, the ragged tail on the last stage."""
    uniform = _spread(num_layers, pp_size)
    if pp_size == 1 or (head_layer_equivalents <= 0.0 and period <= 1):
        return uniform
    units_total, tail = divmod(num_layers, period)
    if units_total < pp_size:
        return uniform  # period-sized units cannot give every stage a layer

    # In period-sized units: the last stage's budget is the per-stage share of (layers + head) minus
    # the head and the ragged tail it already carries.
    head_units = head_layer_equivalents / period
    tail_units = tail / period
    target = (units_total + tail_units + head_units) / pp_size
    last = round(target - head_units - tail_units)
    # At least one unit per front stage; a last stage carrying the tail may hold zero whole units.
    last = max(0 if tail else 1, min(units_total - (pp_size - 1), last))
    front = _spread(units_total - last, pp_size - 1)
    return [units * period for units in front] + [last * period + tail]


def compute_layer_partition(
    num_layers: int,
    pp_size: int,
    *,
    head_layer_equivalents: float = 0.0,
    boundary_period: int = 1,
    pp_split: list[int] | None = None,
) -> list[tuple[int, int]]:
    """Split ``num_layers`` into ``pp_size`` contiguous ``[lo, hi)`` ranges. ``pp_split`` overrides; otherwise the last
    stage, carrying ``lm_head`` + CE, gets ``head_layer_equivalents`` fewer layers in ``boundary_period`` units."""
    if num_layers < pp_size:
        raise ValueError(
            f"Cannot split {num_layers} decoder layers across {pp_size} pipeline stages: every stage "
            f"needs at least one layer. Reduce pipeline_parallel_size to at most {num_layers}."
        )
    if pp_split is not None:
        if len(pp_split) != pp_size:
            raise ValueError(f"pipeline_split has {len(pp_split)} entries but pipeline_parallel_size={pp_size}.")
        if min(pp_split) < 1:
            raise ValueError(f"pipeline_split entries must be >= 1 decoder layer, got {pp_split}.")
        if sum(pp_split) != num_layers:
            raise ValueError(f"pipeline_split sums to {sum(pp_split)} but the model has {num_layers} decoder layers.")
        sizes = list(pp_split)
    else:
        sizes = _balanced_sizes(num_layers, pp_size, head_layer_equivalents, max(1, boundary_period))

    partition, lo = [], 0
    for size in sizes:
        partition.append((lo, lo + size))
        lo += size
    return partition


def head_cost_layer_equivalents(config) -> float:
    """The ``lm_head`` matmul's cost in decoder-layer equivalents, from parameter counts; 0.0 (uniform split) when the
    config lacks the fields. Ignoring the attention quadratic term and embedding lookup only softens the correction."""
    text = text_config(config)
    hidden = getattr(text, "hidden_size", None)
    vocab = getattr(text, "vocab_size", None)
    # Per-layer-aware reads: on a per-layer-heterogeneous family (step3p7's 64 full / 96 sliding
    # heads) the bare attribute raises, and this is a cost ceiling, so ``max`` reduces.
    heads = get_config_field(config, "num_attention_heads", per_layer_reduce=max)
    if not hidden or not vocab or not heads:
        return 0.0
    head_dim = resolve_head_dim(config)
    kv_heads = resolve_num_key_value_heads(config)
    attn = 2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim
    intermediate = get_config_field(config, "moe_intermediate_size", per_layer_reduce=max) or get_config_field(
        config, "intermediate_size", per_layer_reduce=max
    )
    if not intermediate:
        return 0.0
    # Gated on the model being MoE: the ``top_k`` spelling doubles as a legacy generation parameter a
    # dense config may carry.
    ffn_mult = (resolve_router_topk(config) or 1) if has_routed_experts(config) else 1
    mlp = ffn_mult * 3 * hidden * intermediate
    return (hidden * vocab) / (attn + mlp)


def layer_types_period(config) -> int:
    """Smallest shift that leaves ``layer_types`` unchanged (1 when uniform or absent). Shift-invariance, so a ragged
    list (step3p7's 45 layers = 11 ``full,s,s,s`` periods plus one) still snaps."""
    layer_types = getattr(text_config(config), "layer_types", None)
    if not layer_types or len(set(layer_types)) == 1:
        return 1
    n = len(layer_types)
    return next((period for period in range(1, n) if layer_types_shift_invariant(layer_types, period)), n)


def resolve_layer_partition(
    model: nn.Module, pp_size: int, pp_split: list[int] | None = None
) -> list[tuple[int, int]]:
    """The partition every PP consumer shares: manual ``pp_split``, else the head-weighted default."""
    backbone = require_backbone(model)
    config = getattr(model, "config", None)
    # A rebase-safe family needs no period snapping, which for aperiodic layer_types would disable the
    # head-weighted rebalance (GLM-5's 45-layer L,L,L,D pattern has no divisor period).
    period = 1 if resolve_pp_spec(backbone).layer_types_rebase_safe else layer_types_period(config)
    return compute_layer_partition(
        len(decoder_layers(backbone)),
        pp_size,
        head_layer_equivalents=head_cost_layer_equivalents(config),
        boundary_period=period,
        pp_split=pp_split,
    )
