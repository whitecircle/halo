"""Selective TP for attention layers; MoE FFNs are skipped (EP owns them).

Applies :class:`ColwiseParallel` / :class:`RowwiseParallel` to attention Q/K/V/O projections.
Attention sinks (e.g. GptOss) are sliced to the TP-local head count and kept as plain tensors
(not DTensors) since the forward concatenates them with already-sharded logits.
"""

from __future__ import annotations

import logging
from collections import Counter

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from src.distributed.grad_reduce import SumGradAcrossGroup
from src.distributed.mesh import get_tp_submesh
from src.distributed.tensor_parallel.module_types import TP_SHARDABLE_ATTENTION_CLASSES
from src.models.loading.config_levels import get_config_field
from src.models.structure import DECODER_LAYER_LIST_ATTRS, backbone_with_layers, decoder_layers

logger = logging.getLogger(__name__)

# transformers' replicated-norm gradient handling, retargeted below: the plan style a config asks
# for, and the class that installs the per-backward all-reduce for it.
_HF_REPLICATED_GRAD_STYLE = "replicated_with_grad_allreduce"
_HF_REPLICATED_GRAD_CLASS = "ReplicatedWithGradAllReduce"

# Config fields that mark an MLA attention (compressed KV): it shards by query head with the KV
# compression replicated, so the KV-head rule below does not apply to it.
_MLA_CONFIG_FIELDS = ("kv_lora_rank", "qk_rope_head_dim")

# Attention projection -> its TP style, in application order. Colwise splits an output landing on the
# head dim, including the MLA expansions (q_b_proj / kv_b_proj) while the compressions stay
# replicated; the output projection reduces back over it. A family exposing none of these shards
# nothing, which the zero-shard guard below turns into a raise.
_ATTENTION_TP_STYLES: dict[str, type] = {
    "q_proj": ColwiseParallel,
    "k_proj": ColwiseParallel,
    "v_proj": ColwiseParallel,
    "q_b_proj": ColwiseParallel,
    "kv_b_proj": ColwiseParallel,
    "o_proj": RowwiseParallel,
    "out_proj": RowwiseParallel,
}


def _declared_head_counts(text_config, field: str) -> tuple[int, ...]:
    """Every distinct declared value of ``field``: one entry for a homogeneous config, each per-layer
    value for a heterogeneous one (step3p7's 64 full / 96 sliding heads), empty when unset.

    A bare attribute read raises transformers' ``AmbiguousGlobalPerLayerAttributeError`` on a
    heterogeneous family, and TP must split every layer's heads evenly, so the gate checks each
    declared value rather than one reduced number.
    """
    value = get_config_field(text_config, field, per_layer_reduce=frozenset)
    if value is None:
        return ()
    if isinstance(value, frozenset):
        return tuple(sorted(int(v) for v in value))
    return (int(value),)


def validate_tp_head_divisibility(text_config, tp_size: int, *, uses_mla: bool | None = None) -> None:
    """Reject a head count TP cannot split evenly.

    Both loaders route here: the toolkit's selective-TP path with the module tree in hand, and
    :meth:`ParallelismConfig.validate_against_model_config` before any weight is read. Without the
    second, dense TP (HF-native ``tp_plan="auto"`` validates no head count of its own) gets a per-rank
    shard that is not a multiple of ``head_dim`` and fails on the first forward's reshape, after the
    whole checkpoint has been pulled and placed on every rank.

    ``uses_mla`` defaults to a config probe; the module-tree caller passes what it observed.
    """
    if tp_size <= 1 or text_config is None:
        return
    for n_heads in _declared_head_counts(text_config, "num_attention_heads"):
        if n_heads % tp_size != 0:
            raise ValueError(
                f"Tensor parallelism requires num_attention_heads ({n_heads}) divisible by tp_size "
                f"({tp_size}); an uneven query/output split silently corrupts attention."
            )
    if uses_mla is None:
        uses_mla = any(getattr(text_config, field, None) for field in _MLA_CONFIG_FIELDS)
    if uses_mla:
        return
    for n_kv in _declared_head_counts(text_config, "num_key_value_heads"):
        if n_kv % tp_size != 0:
            raise ValueError(
                f"Tensor parallelism requires num_key_value_heads ({n_kv}) divisible by tp_size "
                f"({tp_size}); GQA with fewer KV heads than TP ranks would split individual KV heads "
                f"(numerically wrong). Reduce tp_size or use EP for this model."
            )


def shard_sinks_param(
    model: nn.Module,
    attn: nn.Module,
    attn_name: str,
    tp_mesh: DeviceMesh,
    log_first: bool = False,
) -> None:
    """Shard attention sinks as a plain tensor and register for save-time gathering.

    Sinks stay plain tensors: the forward concatenates them with locally-sharded logits, so a
    DTensor's global shape would mismatch. Sliced to this rank's head range and recorded on
    ``model._tp_sharded_non_dtensor`` for save-time all-gather.

    ``attn_name`` is the attribute the family holds its attention module under (``self_attn``,
    ``attention``, ``attn``), so the registered suffix matches the parameter's real FQN. A fixed
    spelling would match nothing on the other families, leaving every rank's partial head slice in the
    checkpoint and out of the TP gradient reduction.
    """
    tp_size = tp_mesh.size()
    tp_rank = tp_mesh.get_local_rank()
    total_heads = attn.sinks.shape[0]
    local_heads = total_heads // tp_size
    start = tp_rank * local_heads
    end = start + local_heads

    if total_heads % tp_size:
        raise ValueError(
            f"Cannot shard {total_heads} attention sinks across tp_size={tp_size}: an indivisible "
            f"count would silently drop the tail heads."
        )

    with torch.no_grad():
        local_sinks = attn.sinks.data[start:end].clone()
    attn.sinks = nn.Parameter(local_sinks, requires_grad=attn.sinks.requires_grad)

    if not hasattr(model, "_tp_sharded_non_dtensor"):
        model._tp_sharded_non_dtensor = []
    # One (suffix, shard_dim) entry covers every layer's sinks, so register it once.
    entry = (f"{attn_name}.sinks", 0)
    if entry not in model._tp_sharded_non_dtensor:
        model._tp_sharded_non_dtensor.append(entry)

    if log_first:
        logger.info(f"Sharded attention sinks: {total_heads} -> {local_heads} (tp_rank={tp_rank}, tp_size={tp_size})")


def _parallelize_or_raise(module: nn.Module, tp_mesh: DeviceMesh, plan: dict, what: str) -> None:
    """Apply a TP ``plan`` to ``module``, raising ``RuntimeError`` on failure.

    A skipped shard leaves a full-size weight while the DTensor mesh assumes sharded dims, which
    surfaces later as a shape mismatch.
    """
    try:
        parallelize_module(module, tp_mesh, plan)
    except Exception as e:
        raise RuntimeError(f"TP sharding failed for {what}: {e}") from e

    # Plan keys resolve against child modules; a parameter-keyed plan leaves weights full-size.
    if not any(isinstance(p, DTensor) for p in module.parameters(recurse=True)):
        logger.warning(
            f"TP plan for {what} was a no-op: no parameter became a DTensor. Plan keys must name "
            f"CHILD MODULES, not parameters (a parameter-keyed plan like {{'weight': ...}} on a leaf "
            f"module is silently skipped by torch). This module stays REPLICATED — correct numerics, "
            f"but the promised memory saving does not apply."
        )


def _find_attention(layer: nn.Module) -> tuple[nn.Module | None, str | None]:
    """Return ``(attn_module, attr_name)`` for the first of ``self_attn``/``attention``/``attn``
    present on ``layer``, else ``(None, None)``."""
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name), name
    return None, None


def _per_head_norm_modules(attn: nn.Module, plan: dict) -> list[nn.Module]:
    """Attention children whose gradient covers only this rank's heads.

    A normalization applied after a colwise projection (Qwen3/Qwen3.5 ``q_norm``/``k_norm``, LFM2
    ``q_layernorm``/``k_layernorm``) is a replicated ``(head_dim,)`` parameter shared across heads.
    ``ColwiseParallel`` defaults ``use_local_output=True``, so the projection returns a plain
    per-rank-heads tensor and the DTensor graph ends there: each rank's gradient for that norm covers
    its own heads only, and the true gradient is the sum over the group.

    Discovered structurally (any unplanned direct child with trainable parameters) rather than by
    name. MLA families are exempt: their ``q_a``/``kv_a`` norms sit before the colwise expansion,
    where DTensor's ``Replicate`` backward already all-reduces.
    """
    if "q_b_proj" in plan or "kv_b_proj" in plan:  # MLA: norms precede the expansion
        return []
    return [
        child
        for child_name, child in attn.named_children()
        if child_name not in plan and any(p.requires_grad for p in child.parameters(recurse=True))
    ]


def register_mla_rope_grad_reduction(attn: nn.Module, tp_mesh: DeviceMesh, config) -> None:
    """Sum the MLA ``kv_a_proj_with_mqa`` rope gradient over the TP group, in backward.

    That projection emits ``[kv_lora_rank | qk_rope_head_dim]``. The ``kv_lora_rank`` half feeds
    ``kv_b_proj`` (colwise), whose ``Replicate`` input redistribute all-reduces its gradient, so those
    rows are already complete. The rope half bypasses ``kv_b_proj``: it is expanded to this rank's
    local head count and concatenated into ``key_states``, never crossing a DTensor boundary, so each
    rank's gradient for those rows covers only its own heads and the true gradient is the sum over the
    group. The weight is a plain replica, so the trainer's replicated bucket averages it and would
    train the rope rows on 1/tp_size of their gradient while the lora rows stay correct.

    Reducing at the module output (transformers' own ``mla_kv_a_proj`` remedy) keeps the weight a
    plain replica and makes its gradient complete on every rank, leaving that average idempotent.
    """
    text_config = config.get_text_config() if config is not None else None
    rope_dim = getattr(text_config, "qk_rope_head_dim", None)
    if rope_dim is None:
        raise ValueError(
            f"{type(attn).__name__} exposes kv_a_proj_with_mqa (MLA) but its config declares no "
            f"qk_rope_head_dim, so the rope half of that projection cannot be located. Without it "
            f"the rope rows would be AVG-reduced over the TP group and train on 1/tp_size of their "
            f"gradient. Add qk_rope_head_dim to the model config, or drop this family from "
            f"TP_SHARDABLE_ATTENTION_CLASSES."
        )

    def reduce_rope_grad(_module, _args, output):
        k_pass, k_rot = output.split([output.shape[-1] - rope_dim, rope_dim], dim=-1)
        return torch.cat([k_pass, SumGradAcrossGroup.apply(k_rot, tp_mesh.get_group())], dim=-1)

    attn.kv_a_proj_with_mqa.register_forward_hook(reduce_rope_grad)


def _register_per_head_norm_params(model: nn.Module, modules: list[nn.Module]) -> None:
    """Record the per-head norm parameters on ``model`` for the step-time TP gradient sum.

    Parameter names rather than ids: FSDP2 replaces managed ``Parameter`` objects after this runs.
    transformers' ``ReplicatedWithGradAllReduce`` is not used, because its
    ``register_full_backward_hook`` re-reduces whatever sits in ``.grad`` on every backward, which
    under gradient accumulation multiplies each earlier micro-batch's contribution by ``tp_size``
    again per micro-step. The trainer reduces these once per optimizer step instead.

    Accumulates across writers (the HF-native ``tp_plan`` retarget and the toolkit's selective TP both
    register here); a name missing from the set is averaged as a plain replica instead of summed,
    leaving it on ``1/tp_size`` of its true gradient.
    """
    if not modules:
        return
    owned = {id(p) for module in modules for p in module.parameters()}
    registered = {name for name, p in model.named_parameters() if id(p) in owned}
    registered.update(getattr(model, "_tp_per_head_norm_params", None) or ())
    model._tp_per_head_norm_params = sorted(registered)


def retarget_hf_replicated_grad_hooks(model: nn.Module) -> int:
    """Move transformers' per-backward norm all-reduce onto the trainer's step-time SUM.

    ``tp_plan="auto"`` installs :class:`ReplicatedWithGradAllReduce`, whose ``full_backward_hook``
    all-reduces ``param.grad`` in place on every backward, so under gradient accumulation each earlier
    micro-batch's contribution is multiplied by ``tp_size`` again per micro-step (GA=8, tp=2 scales the
    first micro-batch by 128). Those hooks are stripped and their parameters registered the way the
    toolkit TP path does, so the reduction happens once per optimizer step.

    Returns the number of modules retargeted. Raises when the plan asks for the layer and no hook was
    found, since doing nothing would leave the norms on 1/tp_size of their gradient.
    """
    plan = getattr(model, "_tp_plan", None) or {}
    plan_wants_it = any(style == _HF_REPLICATED_GRAD_STYLE for style in plan.values())

    modules: list[nn.Module] = []
    for module in model.modules():
        hooks = getattr(module, "_backward_hooks", None)
        if not hooks:
            continue
        stale = [
            handle
            for handle, fn in hooks.items()
            if getattr(fn, "__qualname__", "").startswith(f"{_HF_REPLICATED_GRAD_CLASS}.")
        ]
        if not stale:
            continue
        for handle in stale:
            del hooks[handle]
        modules.append(module)

    if plan_wants_it and not modules:
        raise RuntimeError(
            f"The TP plan requests {_HF_REPLICATED_GRAD_STYLE!r} but no "
            f"{_HF_REPLICATED_GRAD_CLASS} backward hook was found to retarget — transformers has "
            "changed how it installs that reduction. Without the retarget the per-head attention "
            "norms are re-reduced on every backward, which compounds under gradient accumulation."
        )

    _register_per_head_norm_params(model, modules)
    if modules:
        logger.info(f"Retargeted {len(modules)} per-backward norm all-reduces to the step-time TP SUM")
    return len(modules)


def apply_tp_to_attention_only(
    model: nn.Module,
    device_mesh: DeviceMesh,
) -> int:
    """Apply TP to attention layers only (MoE FFNs are never visited — EP owns them).

    Enables EP+TP: TP shards the attention path while EP handles MoE experts.
    Accepts 1-D (DP=1) or 2-D (DP>1) meshes — the ``"tp"`` sub-mesh is
    auto-extracted. Returns the number of modules parallelized.
    """
    tp_mesh = get_tp_submesh(device_mesh)
    patched = 0
    per_head_norms: list[nn.Module] = []

    # embed_tokens/lm_head stay replicated: DTensor-sharding them faults cuBLAS under EP+TP.
    underlying = backbone_with_layers(model)
    layers = decoder_layers(underlying) if underlying is not None else None
    if layers is None:
        # Raise rather than skip: returning 0 would leave every weight replicated while the (dp, tp)
        # mesh assumes sharding, the same state the patched==0 check below prevents.
        raise ValueError(
            f"Tensor parallelism found no decoder-layer list on {type(model).__name__}: "
            f"backbone_with_layers could not reach a module holding {DECODER_LAYER_LIST_ATTRS}. "
            f"TP cannot shard this layout — train it without tensor parallelism, or teach the "
            f"backbone descent in src/distributed/runtime.py about it."
        )

    tp_size = tp_mesh.size()
    cfg = getattr(model, "config", None)
    # Module-tree MLA detection where available; the pre-load gate uses the helper's config fallback.
    first_attn = next((a for a in (_find_attention(layer)[0] for layer in layers) if a is not None), None)
    uses_mla = first_attn is not None and (hasattr(first_attn, "kv_b_proj") or hasattr(first_attn, "q_b_proj"))
    validate_tp_head_divisibility(cfg.get_text_config() if cfg is not None else None, tp_size, uses_mla=uses_mla)

    unsharded: Counter[str] = Counter()

    for layer_idx, layer in enumerate(layers):
        attn, attn_name = _find_attention(layer)
        if attn is None:
            # Hybrid families (Qwen3.5/3.6 GatedDeltaNet, LFM2 short-conv) interleave layers with no
            # attention module at all; they carry parameters that TP cannot shard.
            unsharded[f"{type(layer).__name__} (no attention submodule)"] += 1
            continue

        class_name = attn.__class__.__name__
        if class_name not in TP_SHARDABLE_ATTENTION_CLASSES:
            unsharded[class_name] += 1
            continue

        # ``is not None`` rather than ``hasattr``: transformers keeps an unused MLA Q-branch attribute
        # present but None, and a plan entry naming it would shard nothing.
        plan = {name: style() for name, style in _ATTENTION_TP_STYLES.items() if getattr(attn, name, None) is not None}

        if plan:
            _parallelize_or_raise(attn, tp_mesh, plan, f"layer {layer_idx} {attn_name} ({class_name})")
            patched += 1
            logger.debug(f"Parallelized layer {layer_idx} {attn_name} ({class_name})")
            per_head_norms.extend(_per_head_norm_modules(attn, plan))
            # MLA only: the rope rows of this projection are per-rank-partial (see the helper).
            if tp_size > 1 and getattr(attn, "kv_a_proj_with_mqa", None) is not None:
                register_mla_rope_grad_reduction(attn, tp_mesh, cfg)

        if hasattr(attn, "sinks") and isinstance(attn.sinks, nn.Parameter):
            shard_sinks_param(model, attn, attn_name, tp_mesh, log_first=(layer_idx == 0))

    # Zero shardable layers under tp_size>1 leaves the model replicated while the (dp, tp) mesh
    # assumes sharding: a memory blowup and a wrong DP gradient denominator.
    if tp_size > 1 and patched == 0:
        model_type = getattr(cfg, "model_type", "unknown")
        attn_classes = sorted({type(a).__name__ for a in (_find_attention(layer)[0] for layer in layers) if a})
        raise ValueError(
            f"Tensor parallelism (tp_size={tp_size}) sharded ZERO attention layers on this model "
            f"(model_type={model_type!r}, attention classes {attn_classes}): none are in "
            f"TP_SHARDABLE_ATTENTION_CLASSES, so every weight would stay replicated while the TP mesh "
            f"assumes sharding. This model family does not support TP — use EP (and/or ETP for "
            f"experts) instead, or add validated entries to "
            f"src/distributed/tensor_parallel/module_types.py."
        )

    # Partial sharding is numerically correct: an unsharded layer's weights are plain replicas that
    # `_sync_tp_replicated_grads` averages over the TP group. Only the memory expectation breaks, since
    # every TP rank keeps a full copy of those layers. Warned once rather than rejected, which would
    # drop Qwen3.5/3.6 and LFM2, whose attention layers TP shards correctly.
    if tp_size > 1 and unsharded:
        detail = ", ".join(f"{name} x{count}" for name, count in sorted(unsharded.items()))
        logger.warning(
            f"Selective TP sharded {patched} of {len(layers)} decoder layers (tp_size={tp_size}). "
            f"Left REPLICATED on every TP rank: {detail}. Their gradients are AVG-reduced over the TP "
            f"group, so training is numerically correct — but their weights, gradients and optimizer "
            f"state are NOT divided by tp_size, so budget memory for full copies of those layers."
        )

    _register_per_head_norm_params(model, per_head_norms)

    logger.info(
        f"✓ Applied selective TP to {patched} attention modules; "
        f"{len(per_head_norms)} per-head norms registered for the step-time gradient SUM"
    )
    return patched
