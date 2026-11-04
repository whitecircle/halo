"""Selective TP: colwise / rowwise sharding of the attention projections only, for the MoE families whose experts
EP owns (and pure TP on a MoE, whose experts stay replicated); the projections compute on their local shards."""

from __future__ import annotations

import logging
from collections import Counter

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from halo.distributed.grad_reduce import SumGradAcrossGroup
from halo.distributed.mesh import tp_submesh
from halo.distributed.tensor_parallel.local_linears import shard_linear
from halo.distributed.tensor_parallel.plain_params import register_hand_sliced, register_per_head_norm_params
from halo.models.config_access import get_config_field
from halo.models.module_tree import DECODER_LAYER_LIST_ATTRS, backbone_with_layers, decoder_layers

logger = logging.getLogger(__name__)

# The MoE attention classes the plan below is validated on; transformers >= 5 dispatches backends through the
# attention interface, so one entry covers a family. A dense model takes HF's own ``tp_plan`` instead.
TP_SHARDABLE_ATTENTION_CLASSES = frozenset(
    {
        "Cohere2MoeAttention",
        "GptOssAttention",
        "Qwen3MoeAttention",
        "Glm4MoeLiteAttention",
        "Lfm2MoeAttention",
        "Qwen3_5MoeAttention",
        # Mistral4 (mistral3 VLM backbone), MLA: only q_b_proj / kv_b_proj are colwise-shardable.
        "Mistral4Attention",
    }
)

# Attention projection -> its TP style, in application order. Colwise splits an output that lands on the head
# dim (the MLA expansions q_b_proj / kv_b_proj included, while its compressions stay replicated); the output
# projection reduces back over it.
_ATTENTION_TP_STYLES: dict[str, str] = {
    "q_proj": "colwise",
    "k_proj": "colwise",
    "v_proj": "colwise",
    "q_b_proj": "colwise",
    "kv_b_proj": "colwise",
    "o_proj": "rowwise",
    "out_proj": "rowwise",
}


def _find_attention(layer: nn.Module) -> tuple[nn.Module | None, str | None]:
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name), name
    return None, None


def _is_mla_attention(attn: nn.Module) -> bool:
    """Compressed-KV attention, sharded by query head. ``is not None``, since transformers keeps an unused MLA Q-branch
    attribute present but None."""
    return getattr(attn, "q_b_proj", None) is not None or getattr(attn, "kv_b_proj", None) is not None


def _parallelize_or_raise(attn: nn.Module, tp_mesh: DeviceMesh, plan: dict[str, str], what: str) -> None:
    try:
        for name, style_name in plan.items():
            shard_linear(getattr(attn, name), style_name, tp_mesh, name=name)
    except Exception as e:
        raise RuntimeError(f"TP sharding failed for {what}: {e}") from e


def _per_head_norm_modules(attn: nn.Module, plan: dict) -> list[nn.Module]:
    """The unplanned trainable children of ``attn``: norms after a colwise projection (Qwen3 ``q_norm``, LFM2
    ``q_layernorm``), whose replicated gradient covers this rank's heads only. MLA's norms sit before the colwise
    expansion, whose input gradient is SUMmed."""
    if _is_mla_attention(attn):
        return []
    return [
        child
        for child_name, child in attn.named_children()
        if child_name not in plan and any(p.requires_grad for p in child.parameters(recurse=True))
    ]


def _attention_tp_plan(attn: nn.Module) -> dict[str, str]:
    """The planned projections ``attn`` holds, each with its TP style. ``is not None``: a plan entry naming a
    present-but-None attribute would shard nothing."""
    return {name: style for name, style in _ATTENTION_TP_STYLES.items() if getattr(attn, name, None) is not None}


def _is_tp_shardable(attn: nn.Module | None) -> bool:
    return (
        attn is not None and type(attn).__name__ in TP_SHARDABLE_ATTENTION_CLASSES and bool(_attention_tp_plan(attn))
    )


def _tp_decoder_layers(model: nn.Module, cfg, tp_size: int) -> nn.ModuleList:
    """The decoder layers TP shards, heads checked to divide ``tp_size``. embed_tokens/lm_head stay replicated, since
    DTensor-sharding them faults cuBLAS under EP+TP."""
    underlying = backbone_with_layers(model)
    layers = decoder_layers(underlying) if underlying is not None else None
    if layers is None:
        raise ValueError(
            f"Tensor parallelism found no decoder-layer list on {type(model).__name__}: "
            f"backbone_with_layers could not reach a module holding {DECODER_LAYER_LIST_ATTRS}. "
            f"TP cannot shard this layout — train it without tensor parallelism, or teach "
            f"backbone_with_layers to reach its decoder layers."
        )
    first_attn = next((a for a in (_find_attention(layer)[0] for layer in layers) if a is not None), None)
    uses_mla = first_attn is not None and _is_mla_attention(first_attn)
    validate_tp_head_divisibility(cfg.get_text_config() if cfg is not None else None, tp_size, uses_mla=uses_mla)
    return layers


def _reject_zero_shardable(layers: nn.ModuleList, cfg, tp_size: int) -> None:
    """Refuse a TP run whose plan shards no decoder layer's attention: every weight would stay replicated while
    the (dp, tp) mesh assumes sharding, a silent memory blowup and a wrong DP gradient denominator."""
    if tp_size <= 1 or any(_is_tp_shardable(_find_attention(layer)[0]) for layer in layers):
        return
    model_type = getattr(cfg, "model_type", "unknown")
    attn_classes = sorted({type(a).__name__ for a in (_find_attention(layer)[0] for layer in layers) if a})
    raise ValueError(
        f"Tensor parallelism (tensor_parallel_size={tp_size}) shards ZERO attention layers on "
        f"this model (model_type={model_type!r}, attention classes {attn_classes}): none are in "
        f"TP_SHARDABLE_ATTENTION_CLASSES, so every weight would stay replicated while the TP mesh "
        f"assumes sharding. This model family does not support TP — use EP (and/or ETP for "
        f"experts) instead, or add validated entries to TP_SHARDABLE_ATTENTION_CLASSES."
    )


def reject_unshardable_tp_model(model: nn.Module, tp_size: int) -> None:
    """Refuse a model :func:`apply_tp_to_attention_only` cannot shard. Reads the module tree alone, so a meta-device
    shell answers before any weight loads."""
    cfg = getattr(model, "config", None)
    _reject_zero_shardable(_tp_decoder_layers(model, cfg, tp_size), cfg, tp_size)


def apply_tp_to_attention_only(model: nn.Module, device_mesh: DeviceMesh) -> int:
    """Shard the attention modules of ``model`` over the ``tp`` dim of ``device_mesh`` (1-D, or 2-D with DP);
    returns how many were parallelized. MoE blocks are never visited."""
    tp_mesh = tp_submesh(device_mesh)
    tp_size = tp_mesh.size()
    cfg = getattr(model, "config", None)
    layers = _tp_decoder_layers(model, cfg, tp_size)
    _reject_zero_shardable(layers, cfg, tp_size)
    patched = 0
    per_head_norms: list[nn.Module] = []
    unsharded: Counter[str] = Counter()

    for layer_idx, layer in enumerate(layers):
        attn, attn_name = _find_attention(layer)
        if attn is None:
            # Hybrid families (Qwen3.5/3.6 GatedDeltaNet, LFM2 short-conv) interleave attention-free layers.
            unsharded[f"{type(layer).__name__} (no attention submodule)"] += 1
            continue

        class_name = attn.__class__.__name__
        if class_name not in TP_SHARDABLE_ATTENTION_CLASSES:
            unsharded[class_name] += 1
            continue

        plan = _attention_tp_plan(attn)
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

    # Partial sharding trains correctly (the replicas' gradients are AVG-reduced) but saves less memory than
    # 1/tp_size; a warning, not a raise, so the hybrid families keep TP on their attention layers.
    if tp_size > 1 and unsharded:
        detail = ", ".join(f"{name} x{count}" for name, count in sorted(unsharded.items()))
        logger.warning(
            f"Selective TP sharded {patched} of {len(layers)} decoder layers "
            f"(tensor_parallel_size={tp_size}). Left REPLICATED on every TP rank: {detail}. Their "
            f"gradients are AVG-reduced over the TP group, so training is numerically correct — but "
            f"their weights, gradients and optimizer state are NOT divided by tensor_parallel_size, "
            f"so budget memory for full copies of those layers."
        )

    register_per_head_norm_params(model, per_head_norms)

    logger.info(
        f"✓ Applied selective TP to {patched} attention modules; "
        f"{len(per_head_norms)} per-head norms registered for the step-time gradient SUM"
    )
    return patched


def _declared_head_counts(text_config, field: str) -> tuple[int, ...]:
    """Every distinct declared value of ``field`` (each per-layer value of a heterogeneous config, step3p7's 64
    full / 96 sliding heads), empty when unset: TP must split every layer's heads evenly."""
    value = get_config_field(text_config, field, per_layer_reduce=frozenset)
    if value is None:
        return ()
    if isinstance(value, frozenset):
        return tuple(sorted(int(v) for v in value))
    return (int(value),)


def validate_tp_head_divisibility(text_config, tp_size: int, *, uses_mla: bool | None = None) -> None:
    """Reject a head count TP cannot split evenly. Also run before any weight is read, since HF-native
    ``tp_plan="auto"`` checks none and dies on the first forward's reshape."""
    if tp_size <= 1 or text_config is None:
        return
    for n_heads in _declared_head_counts(text_config, "num_attention_heads"):
        if n_heads % tp_size != 0:
            raise ValueError(
                f"Tensor parallelism requires num_attention_heads ({n_heads}) divisible by "
                f"tensor_parallel_size ({tp_size}); an uneven query/output split silently corrupts attention."
            )
    if uses_mla is None:
        uses_mla = any(getattr(text_config, field, None) for field in ("kv_lora_rank", "qk_rope_head_dim"))
    # MLA shards by query head with the KV compression replicated, so the KV-head rule does not apply.
    if uses_mla:
        return
    for n_kv in _declared_head_counts(text_config, "num_key_value_heads"):
        if n_kv % tp_size != 0:
            raise ValueError(
                f"Tensor parallelism requires num_key_value_heads ({n_kv}) divisible by "
                f"tensor_parallel_size ({tp_size}); GQA with fewer KV heads than TP ranks would "
                f"split individual KV heads (numerically wrong). Reduce tensor_parallel_size or use "
                f"EP for this model."
            )


def register_mla_rope_grad_reduction(attn: nn.Module, tp_mesh: DeviceMesh, config) -> None:
    """SUM the gradient of the rope half of MLA's ``kv_a_proj_with_mqa`` output over the TP group, in backward. That
    half is broadcast to this rank's heads without passing a TP projection, so its gradient covers those heads only;
    reducing at the output keeps the weight a plain replica, so the trainer's AVG is idempotent."""
    text_config = config.get_text_config() if config is not None else None
    rope_dim = getattr(text_config, "qk_rope_head_dim", None)
    if rope_dim is None:
        raise ValueError(
            f"{type(attn).__name__} exposes kv_a_proj_with_mqa (MLA) but its config declares no "
            f"qk_rope_head_dim, so the rope half of that projection cannot be located. Without it "
            f"the rope rows would be AVG-reduced over the TP group and train on 1/tensor_parallel_size "
            f"of their gradient. Add qk_rope_head_dim to the model config, or drop this family from "
            f"TP_SHARDABLE_ATTENTION_CLASSES."
        )

    def reduce_rope_grad(_module, _args, output):
        k_pass, k_rot = output.split([output.shape[-1] - rope_dim, rope_dim], dim=-1)
        return torch.cat([k_pass, SumGradAcrossGroup.apply(k_rot, tp_mesh.get_group())], dim=-1)

    attn.kv_a_proj_with_mqa.register_forward_hook(reduce_rope_grad)


def shard_sinks_param(
    model: nn.Module, attn: nn.Module, attn_name: str, tp_mesh: DeviceMesh, log_first: bool = False
) -> None:
    """Slice ``attn.sinks`` to this rank's heads as a plain tensor (the forward concatenates it with local
    logits) and register it as hand-sliced under ``{attn_name}.sinks``, the parameter's real name."""
    tp_size = tp_mesh.size()
    tp_rank = tp_mesh.get_local_rank()
    total_heads = attn.sinks.shape[0]
    local_heads = total_heads // tp_size
    start = tp_rank * local_heads
    end = start + local_heads

    if total_heads % tp_size:
        raise ValueError(
            f"Cannot shard {total_heads} attention sinks across tensor_parallel_size={tp_size}: an "
            f"indivisible count would silently drop the tail heads."
        )

    with torch.no_grad():
        local_sinks = attn.sinks.data[start:end].clone()
    attn.sinks = nn.Parameter(local_sinks, requires_grad=attn.sinks.requires_grad)

    register_hand_sliced(model, f"{attn_name}.sinks", 0)

    if log_first:
        logger.info(f"Sharded attention sinks: {total_heads} -> {local_heads} (tp_rank={tp_rank}, tp_size={tp_size})")
