"""Compatibility validation for Ulysses sequence parallelism.

:func:`validate_model_for_ulysses` checks the model — Flash Attention, a known attention class,
head counts divisible by ``cp_size``. :func:`validate_trainer_args_for_cp` checks the trainer
settings whose loss/metric path assumes full-sequence logits, which a CP rank never holds.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from src.distributed.context_parallel.base_layer import HUB_FLASH_ATTN2_KERNEL
from src.distributed.context_parallel.layers.registry import CP_SUPPORTED_ATTENTION_CLASSES, WRAPPER_CLASS_MAP
from src.models.loading.config_levels import text_config
from src.models.patches.attention import effective_attn_implementation

logger = logging.getLogger(__name__)


SUPPORTED_ATTN_IMPLEMENTATIONS = (
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention_4",
    HUB_FLASH_ATTN2_KERNEL,
    "kernels-community/flash-attn3",
)


# Shared rejection text: a layer carrying state along the sequence axis cannot be sharded by it,
# since the Ulysses path halo-exchanges nothing across chunk boundaries. Each family supplies its own
# (mechanism, remedy) pair.
def _sequence_axis_detail(mechanism: str, remedy: str) -> str:
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

_BAILING_LINEAR_DETAIL = _sequence_axis_detail(
    "``Ring-mini-linear-2.0`` runs Lightning Attention-2 (an ``fla`` gated-linear recurrence) in most of its layers",
    "Its file also names the remaining full-attention classes exactly as Ling 2.0 does, so without "
    "this rejection CP would wrap those few and leave the recurrent ones scanning each rank's chunk "
    "in isolation. Use CP on ``inclusionAI/Ling-mini-2.0`` (softmax GQA throughout), or train this "
    "checkpoint with EP instead.",
)

_INKLING_DETAIL = _sequence_axis_detail(
    "this depthwise causal Conv1d runs (Inkling applies it to the projected K/V inside attention and "
    "around each decoder sublayer)",
    "Inkling also positions via an additive relative-logits bias that flash_attn_func cannot take, so "
    "no Ulysses wrapper is possible. Inkling does not support CP — use EP (optionally + ETP) instead.",
)

_ZAYA_DETAIL = _sequence_axis_detail(
    "Zaya's CCA projection convolves, and shifts its delayed-value stream one position,",
    "Train Zaya without Context Parallelism (agent-docs/models/zaya.md).",
)

_GLM5_NEXT_LINEAR_DETAIL = _sequence_axis_detail(
    "GLM-5's KDA linear attention runs a causal Conv1d (kernel 4) and a recurrent delta-rule scan",
    "GLM-5 (glm5_next) does not support CP — train at a sequence length that fits a single rank, and "
    "shard the experts with EP instead.",
)

# Config-only rejection for the ``linear_attention`` layer type: every family declaring it runs a
# causal convolution and a recurrent scan. Once the modules exist, the instantiated-tree scan below
# names the class instead.
_LINEAR_ATTENTION_DETAIL = _sequence_axis_detail(
    "a ``linear_attention`` layer runs a causal convolution and a recurrent scan",
    "Train the family without Context Parallelism, and shard the experts with EP instead.",
)

# Layers crossing sequence chunks (Conv1d, recurrent scans, window pooling), mapped to the rejection
# detail each raises with.
_UNSUPPORTED_SEQUENCE_AXIS_LAYERS: dict[str, str] = {
    "ZayaCCAProjection": _ZAYA_DETAIL,
    "Qwen3_5MoeGatedDeltaNet": _GATED_DELTANET_DETAIL,
    "Qwen3_5GatedDeltaNet": _GATED_DELTANET_DETAIL,
    "DeepseekV4CSACompressor": _DEEPSEEK_V4_DETAIL,
    "DeepseekV4HCACompressor": _DEEPSEEK_V4_DETAIL,
    "DeepseekV4Indexer": _DEEPSEEK_V4_DETAIL,
    "InklingShortConvolution": _INKLING_DETAIL,
    "BailingMoeV2LinearAttention": _BAILING_LINEAR_DETAIL,
    "Glm5NextTextLinearAttention": _GLM5_NEXT_LINEAR_DETAIL,
}


class UlyssesConfigError(ValueError):
    """Raised when Ulysses CP is misconfigured for the given model."""


def validate_model_for_ulysses(model: nn.Module, cp_size: int) -> None:
    """Validate a model for Ulysses CP: a wrapped attention class, Flash Attention where the family's
    own attention path is what runs, and Q/KV head counts divisible by ``cp_size``.

    Raises:
        UlyssesConfigError: If the model is not compatible.
    """
    config = model.config
    model_type = getattr(config, "model_type", "unknown")
    effective_config = text_config(config)

    # Check before the supported-attention scan so linear-attention configs get a precise message.
    for module in model.modules():
        class_name = module.__class__.__name__
        detail = _UNSUPPORTED_SEQUENCE_AXIS_LAYERS.get(class_name)
        if detail is not None:
            raise UlyssesConfigError(f"Ulysses CP cannot run on a model that instantiates {class_name!r}: {detail}")

    # Config-only signal, for the case where modules aren't instantiated yet (meta-init).
    layer_types = getattr(effective_config, "layer_types", None)
    if isinstance(layer_types, (list, tuple)):
        linear_indices = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
        if linear_indices:
            raise UlyssesConfigError(
                f"Ulysses CP cannot run on {model_type!r} because {len(linear_indices)} of {len(layer_types)} "
                f"layers are ``linear_attention``: {_LINEAR_ATTENTION_DETAIL}"
            )

    wrapper_cls = None
    attention_class = None
    attention_module = None
    for module in model.modules():
        class_name = module.__class__.__name__
        wrapper_cls = WRAPPER_CLASS_MAP.get(class_name)
        if wrapper_cls is not None:
            attention_class = class_name
            attention_module = module
            break

    if wrapper_cls is None:
        raise UlyssesConfigError(
            f"Ulysses CP only supports specific attention architectures. "
            f"Supported: {CP_SUPPORTED_ATTENTION_CLASSES}. "
            f"Model type '{model_type}' does not appear to use a supported attention module."
        )

    # A composite (VLM) wrapper records the backend on the text sub-config and only sometimes mirrors
    # it up, so reading the top level alone would misreport a model that is running flash attention.
    attn_impl = effective_attn_implementation(config)
    # Asked of the wrapper, not of the label alone: CP calls flash itself, so the label matters only
    # for families whose own attention forward is a real fallback path (see the wrapper flag).
    if wrapper_cls.REQUIRES_FLASH_ATTN_LABEL and attn_impl not in SUPPORTED_ATTN_IMPLEMENTATIONS:
        raise UlyssesConfigError(
            f"Ulysses CP requires Flash Attention. "
            f"Got attn_implementation='{attn_impl}', expected one of {SUPPORTED_ATTN_IMPLEMENTATIONS}. "
            f"Set attn_implementation to 'flash_attention_2'/'flash_attention_3' (Hopper) or "
            f"'flash_attention_4' (Blackwell), or leave it as the auto-detected default."
        )

    num_q_heads = getattr(effective_config, "num_attention_heads", None)
    if num_q_heads is None:
        raise UlyssesConfigError(
            "Model config must have num_attention_heads for Ulysses CP. "
            "For VLM models, check text_config.num_attention_heads."
        )

    # Asked of the wrapper class against the unpatched module, not of the config: MLA expands its
    # compressed KV through ``kv_b_proj`` with one head per query head, so ``num_key_value_heads``
    # describes a tensor this path never builds.
    try:
        num_kv_heads = wrapper_cls._resolve_num_kv_heads(attention_module)
    except AttributeError as e:
        raise UlyssesConfigError(
            f"Cannot resolve the KV head count for {attention_class} under Ulysses CP: {e}. "
            f"For VLM models, check text_config.num_key_value_heads."
        ) from e

    if num_q_heads % cp_size != 0:
        raise UlyssesConfigError(
            f"Q heads ({num_q_heads}) must be divisible by CP size ({cp_size}). "
            f"Choose a CP size that divides {num_q_heads} evenly."
        )
    if num_kv_heads % cp_size != 0:
        raise UlyssesConfigError(
            f"KV heads ({num_kv_heads}) must be divisible by CP size ({cp_size}). "
            f"Choose a CP size that divides {num_kv_heads} evenly."
        )

    logger.info(
        f"✓ Model validated for Ulysses CP: "
        f"attn={attn_impl}, class={attention_class}, "
        f"Q_heads={num_q_heads}→{num_q_heads // cp_size}, "
        f"KV_heads={num_kv_heads}→{num_kv_heads // cp_size}"
    )


def validate_trainer_args_for_cp(
    training_args,
    compute_metrics=None,
    preprocess_logits_for_metrics=None,
) -> None:
    """Reject trainer settings whose loss/metric path needs full-sequence logits.

    A CP rank's forward returns logits for its own chunk only, and
    :class:`~src.distributed.context_parallel.wrapper.UlyssesCPModelWrapper` computes the
    boundary-aware loss itself. Anything that re-derives the loss or a metric from
    ``(logits, labels)`` outside the wrapper pairs a chunk of logits with the full-length labels.

    Raises:
        UlyssesConfigError: If a rejected setting is active.
    """
    if getattr(training_args, "label_smoothing_factor", 0.0):
        raise UlyssesConfigError(
            "label_smoothing_factor > 0 is not supported with Context Parallelism: the Trainer pops "
            "`labels` before the model call and smooths the loss itself, so the CP wrapper returns "
            "no loss and the smoother compares FULL-length labels against this rank's "
            "sequence-chunk logits. Set label_smoothing_factor=0, or train without CP."
        )

    if getattr(training_args, "loss_type", None) == "dft":
        raise UlyssesConfigError(
            "loss_type='dft' is not supported with Context Parallelism: TRL implements it either by "
            "installing a compute_loss_func (the Trainer then pops `labels`, leaving the CP wrapper "
            "with no loss and the func with full-length labels) or, under Liger, by a "
            "`use_token_scaling` forward flag the CP wrapper's own loss ignores. Use "
            "loss_type='nll', or train without CP."
        )

    # Not gated on eval_strategy: a trainer built with eval_strategy='no' can still reach the
    # misaligned metric path through a direct evaluate()/predict() call.
    if compute_metrics is not None or preprocess_logits_for_metrics is not None:
        raise UlyssesConfigError(
            "Evaluation under Context Parallelism is loss-only: compute_metrics / "
            "preprocess_logits_for_metrics receive this rank's sequence-chunk logits gathered "
            "against the full-length labels, so the two are misaligned. Drop them (eval_loss still "
            "reports), or evaluate from a saved checkpoint without CP."
        )
