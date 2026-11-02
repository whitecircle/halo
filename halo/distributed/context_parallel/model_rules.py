"""Which models Ulysses CP can run: a wrapped attention class, a flash attention label, head counts ``cp_size``
divides, and no layer that carries state along the sequence axis."""

from __future__ import annotations

import logging

import torch.nn as nn

from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.families import ULYSSES_ATTENTION_MAP
from halo.models.attention.selection import VARLEN_ATTN_IMPLEMENTATIONS, effective_attn_implementation
from halo.models.config_access import text_config

logger = logging.getLogger(__name__)

# The labels CP accepts: the flash builds whose kernels its attention calls.
SUPPORTED_ATTN_IMPLEMENTATIONS = VARLEN_ATTN_IMPLEMENTATIONS


def _sequence_axis_detail(mechanism: str, remedy: str) -> str:
    """A refusal for a layer carrying state along the sequence axis, which the Ulysses path never exchanges
    across chunk boundaries."""
    return (
        f"{mechanism} over the sequence axis, which the Ulysses CP path does not halo-exchange across "
        f"ranks — a CP shard severs it at every chunk boundary. {remedy}"
    )


_GATED_DELTANET_DETAIL = _sequence_axis_detail(
    "these layers run a Conv1d and a recurrent gated-delta-rule scan",
    "Workarounds: train at a sequence length that fits a single rank without CP, or set "
    '``layer_types=["full_attention"] * num_hidden_layers`` (only viable for custom from-scratch '
    "training — pretrained Qwen3.5/3.6 checkpoints have GatedDeltaNet weights that would be discarded).",
)
_DEEPSEEK_V4_DETAIL = _sequence_axis_detail(
    "this DeepSeek-V4 compressor/indexer pools non-overlapping token windows",
    "DeepSeek-V4 does not support CP — use EP (and expert LoRA) instead.",
)
# Every family declaring the ``linear_attention`` layer type runs a causal convolution and a recurrent scan;
# the config-only refusal for a model whose modules are not instantiated yet.
_LINEAR_ATTENTION_DETAIL = _sequence_axis_detail(
    "a ``linear_attention`` layer runs a causal convolution and a recurrent scan",
    "Train the family without Context Parallelism, and shard the experts with EP instead.",
)
_UNSUPPORTED_SEQUENCE_AXIS_LAYERS: dict[str, str] = {
    "ZayaCCAProjection": _sequence_axis_detail(
        "Zaya's CCA projection convolves, and shifts its delayed-value stream one position,",
        "Train Zaya without Context Parallelism (agent-docs/models/zaya.md).",
    ),
    "Qwen3_5MoeGatedDeltaNet": _GATED_DELTANET_DETAIL,
    "Qwen3_5GatedDeltaNet": _GATED_DELTANET_DETAIL,
    "DeepseekV4CSACompressor": _DEEPSEEK_V4_DETAIL,
    "DeepseekV4HCACompressor": _DEEPSEEK_V4_DETAIL,
    "DeepseekV4Indexer": _DEEPSEEK_V4_DETAIL,
    "InklingShortConvolution": _sequence_axis_detail(
        "this depthwise causal Conv1d runs (Inkling applies it to the projected K/V inside attention and "
        "around each decoder sublayer)",
        "Inkling also positions via an additive relative-logits bias that flash_attn_func cannot take, so "
        "no Ulysses wrapper is possible. Inkling does not support CP — use EP (optionally + ETP) instead.",
    ),
    "BailingMoeV2LinearAttention": _sequence_axis_detail(
        "``Ring-mini-linear-2.0`` runs Lightning Attention-2 (an ``fla`` gated-linear recurrence) in most of its layers",
        "Its file also names the remaining full-attention classes exactly as Ling 2.0 does, so without "
        "this rejection CP would wrap those few and leave the recurrent ones scanning each rank's chunk "
        "in isolation. Use CP on ``inclusionAI/Ling-mini-2.0`` (softmax GQA throughout), or train this "
        "checkpoint with EP instead.",
    ),
    "Glm5NextTextLinearAttention": _sequence_axis_detail(
        "GLM-5's KDA linear attention runs a causal Conv1d (kernel 4) and a recurrent delta-rule scan",
        "GLM-5 (glm5_next) does not support CP — train at a sequence length that fits a single rank, and "
        "shard the experts with EP instead.",
    ),
}


def reject_unsupported_ulysses_model(model: nn.Module, cp_size: int) -> None:
    """Raise :class:`CPConfigError` unless Ulysses CP can run ``model``: no layer carrying state along the
    sequence axis, a wrapped attention class, a flash label where the family's own forward is what the label
    describes, and Q/KV head counts ``cp_size`` divides."""
    config = model.config
    model_type = getattr(config, "model_type", "unknown")
    effective_config = text_config(config)

    # Before the supported-attention scan, so a linear-attention model gets the precise message.
    for module in model.modules():
        class_name = module.__class__.__name__
        detail = _UNSUPPORTED_SEQUENCE_AXIS_LAYERS.get(class_name)
        if detail is not None:
            raise CPConfigError(f"Ulysses CP cannot run on a model that instantiates {class_name!r}: {detail}")

    # The config-only signal, for modules not instantiated yet (meta-init).
    layer_types = getattr(effective_config, "layer_types", None)
    if isinstance(layer_types, (list, tuple)):
        num_linear = sum(layer_type == "linear_attention" for layer_type in layer_types)
        if num_linear:
            raise CPConfigError(
                f"Ulysses CP cannot run on {model_type!r} because {num_linear} of {len(layer_types)} "
                f"layers are ``linear_attention``: {_LINEAR_ATTENTION_DETAIL}"
            )

    attention_module = next((m for m in model.modules() if m.__class__.__name__ in ULYSSES_ATTENTION_MAP), None)
    if attention_module is None:
        raise CPConfigError(
            f"Ulysses CP only supports specific attention architectures. "
            f"Supported: {tuple(ULYSSES_ATTENTION_MAP)}. "
            f"Model type '{model_type}' does not appear to use a supported attention module."
        )
    attention_class = attention_module.__class__.__name__
    wrapper_cls = ULYSSES_ATTENTION_MAP[attention_class]

    # A composite (VLM) wrapper records the backend on the text sub-config and only sometimes mirrors
    # it up, so reading the top level alone would misreport a model that is running flash attention.
    attn_impl = effective_attn_implementation(config)
    if wrapper_cls.REQUIRES_FLASH_ATTN_LABEL and attn_impl not in SUPPORTED_ATTN_IMPLEMENTATIONS:
        raise CPConfigError(
            f"Ulysses CP requires Flash Attention. "
            f"Got attn_implementation='{attn_impl}', expected one of {SUPPORTED_ATTN_IMPLEMENTATIONS}. "
            f"Set attn_implementation to 'flash_attention_2'/'flash_attention_3' (Hopper) or "
            f"'flash_attention_4' (Blackwell), or leave it as the auto-detected default."
        )

    num_q_heads = getattr(effective_config, "num_attention_heads", None)
    if num_q_heads is None:
        raise CPConfigError(
            "Model config must have num_attention_heads for Ulysses CP. "
            "For VLM models, check text_config.num_attention_heads."
        )

    # Asked of the wrapper against the unpatched module: MLA's K/V come out of ``kv_b_proj``, not
    # ``num_key_value_heads``.
    try:
        num_kv_heads = wrapper_cls._resolve_num_kv_heads(attention_module)
    except AttributeError as e:
        raise CPConfigError(
            f"Cannot resolve the KV head count for {attention_class} under Ulysses CP: {e}. "
            f"For VLM models, check text_config.num_key_value_heads."
        ) from e

    for label, num_heads in (("Q", num_q_heads), ("KV", num_kv_heads)):
        if num_heads % cp_size != 0:
            raise CPConfigError(
                f"{label} heads ({num_heads}) must be divisible by context_parallel_size ({cp_size}). Choose a "
                f"context_parallel_size that divides {num_heads} evenly."
            )

    logger.info(
        f"✓ Model validated for Ulysses CP: "
        f"attn={attn_impl}, class={attention_class}, "
        f"Q_heads={num_q_heads}→{num_q_heads // cp_size}, "
        f"KV_heads={num_kv_heads}→{num_kv_heads // cp_size}"
    )
