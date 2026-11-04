"""Liger-backed cross-entropy scoped to ``transformers.loss.loss_utils``, the fused-loss forward every
toolkit causal-LM spec installs, and the fused head loss a caller scoring hidden states itself runs."""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.transformers import LigerCrossEntropyLoss
from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss, unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerMoeCausalLMOutputWithPast
from transformers.loss import loss_utils

from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models.config_access import get_config_field
from halo.models.head_transform import HeadTransform, resolve_head_transform, resolve_head_transform_spec
from halo.models.unwrapping import base_transformers_model

# Captured before any patch so the CPU / unsupported-kwarg fallback below cannot recurse.
_TORCH_CROSS_ENTROPY = F.cross_entropy


@functools.lru_cache(maxsize=1)
def _default_ce():
    return LigerCrossEntropyLoss(ignore_index=-100, reduction="mean")


def liger_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Drop-in ``F.cross_entropy`` backed by Liger; CPU tensors and deprecated ``size_average``/``reduce`` stay on
    torch, as Liger is Triton/CUDA-only."""
    if not input.is_cuda or size_average is not None or reduce is not None:
        return _TORCH_CROSS_ENTROPY(
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    if weight is not None or (ignore_index, reduction, label_smoothing) != (-100, "mean", 0.0):
        return LigerCrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )(input, target)
    return _default_ce()(input, target)


class _AttributeOverride:
    """Read-through view of a module with attributes replaced; writes reach the wrapped module, so a later writer
    through ``torch.nn`` (upstream Liger's appliers) mutates the real one."""

    def __init__(self, wrapped, **overrides):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name: str):
        overrides = self.__dict__["_overrides"]
        return overrides[name] if name in overrides else getattr(self.__dict__["_wrapped"], name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self.__dict__["_wrapped"], name, value)


# ``loss_utils`` reaches cross-entropy only as ``nn.functional.cross_entropy``, so overriding its ``nn`` global
# redirects that call alone. Assigning ``torch.nn.functional.cross_entropy`` would swap the kernel process-wide,
# including CP's fp32 loss and any 3-D or soft-target call, which this replacement cannot serve.
_LOSS_UTILS_NN = _AttributeOverride(
    nn, functional=_AttributeOverride(nn.functional, cross_entropy=liger_cross_entropy)
)


def patch_loss_utils_cross_entropy() -> None:
    """Route ``transformers.loss.loss_utils``' cross-entropy through Liger. Idempotent."""
    loss_utils.nn = _LOSS_UTILS_NN


def _resolve_fused_scoring(model: nn.Module, skip_logits: bool | None, has_labels: bool) -> tuple[bool, HeadTransform]:
    """Whether a forward scores through the fused loss (unset, a training forward with labels does), and the family's
    head transform; checked before the backbone runs."""
    if skip_logits and not has_labels:
        raise ValueError("skip_logits is True, but labels and shift_labels are None")
    if skip_logits is None:
        skip_logits = model.training and has_labels
    transform = resolve_head_transform_spec(type(model)).transform(model.config)
    if skip_logits and (transform.softcap is not None or transform.vocab_size is not None):
        raise ValueError(
            f"Liger fused_linear_cross_entropy cannot reproduce the head of {type(model).__name__} "
            f"({transform}): the fused loss contracts the hidden states with lm_head.weight directly, "
            f"so a softcap or a vocabulary cut would be dropped. Set fused_linear_cross_entropy: false "
            f"in liger_kernel_config."
        )
    return skip_logits, transform


def _resolve_router_logits_flag(model: nn.Module, explicit: bool | None, router_aux_loss_in_head: bool) -> bool | None:
    """The ``output_router_logits`` to forward as the family's head does: an explicit value as given, True where only
    the config asks, else ``None`` (the backbone resolves an omitted flag from the config)."""
    requested = explicit if explicit is not None else get_config_field(model.config, "output_router_logits", False)
    if requested and router_aux_loss_in_head:
        raise ValueError(
            f"Liger fused_linear_cross_entropy on {type(model).__name__} requires "
            f"output_router_logits=False: the fused loss replaces the lm_head projection the "
            f"family's own head runs before assembling a router aux loss, so nothing would add "
            f"that term to the objective. Use moe_balancing: bias_update (or none), or set "
            f"fused_linear_cross_entropy: false in liger_kernel_config."
        )
    if explicit is not None:
        return explicit
    return True if requested else None


def _fused_causal_lm_loss(
    model: nn.Module, transform, hidden_states: torch.Tensor, labels, shift_labels, kwargs: dict
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The fused loss and token accuracy over the kept hidden states, the logit scale folded onto them."""
    if getattr(model.lm_head, "bias", None) is not None:
        raise ValueError(
            f"Liger fused_linear_cross_entropy computes logits from lm_head.weight alone, but "
            f"{type(model).__name__} carries an lm_head bias — the fused loss would silently "
            f"differ from the unfused one. Set fused_linear_cross_entropy: false for this "
            f"checkpoint."
        )
    # LigerForCausalLMLoss applies the shift, the `num_items_in_batch` normalisation matching ForCausalLMLoss (a mean
    # would normalise per-rank, scaling the loss by world_size), and the token accuracy TRL reads under
    # use_liger_kernel.
    fused_hidden_states = transform.scale_hidden(hidden_states)
    if transform.logit_scale is not None:
        fused_hidden_states = fused_hidden_states * transform.logit_scale
    result = LigerForCausalLMLoss(
        hidden_states=fused_hidden_states,
        lm_head_weight=model.lm_head.weight,
        labels=labels,
        shift_labels=shift_labels,
        # Off the head, as a composite wrapper's `config.hidden_size` is the wrapper's.
        hidden_size=model.lm_head.weight.shape[-1],
        **kwargs,
    )
    loss, _, token_accuracy, _ = unpack_cross_entropy_result(result)
    return loss, token_accuracy


def build_lce_forward(router_aux_loss_in_head: bool = False, fp32_logits: bool = False) -> Callable:
    """A fused-loss ``forward`` applying the family's head transform. A logit scale folds onto the hidden states
    (``(s·h) @ Wᵀ == s·(h @ Wᵀ)``), bit-exact in bf16 for power-of-two scales; a softcap or vocabulary cut is refused.
    ``router_aux_loss_in_head`` refuses ``output_router_logits``, since nothing would add that term to the
    objective. ``fp32_logits`` casts the logits of a forward that returns them, as the family's own head does."""

    def lce_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        skip_logits: bool | None = None,
        **kwargs,
    ) -> LigerMoeCausalLMOutputWithPast:
        shift_labels = kwargs.pop("shift_labels", None)
        # From kwargs: `honors_router_logits_flag` looks for this parameter on the forward, and declaring it would
        # report that a family whose head lacks it honors the flag, moving its router balancing off `bias_update`.
        explicit_router_logits = kwargs.pop("output_router_logits", None)
        skip_logits, transform = _resolve_fused_scoring(
            self, skip_logits, labels is not None or shift_labels is not None
        )
        router_logits_flag = _resolve_router_logits_flag(self, explicit_router_logits, router_aux_loss_in_head)
        if router_logits_flag is not None:
            kwargs["output_router_logits"] = router_logits_flag

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        kept_hidden_states = outputs.last_hidden_state[:, slice_indices, :]
        logits = loss = token_accuracy = None
        if skip_logits:
            loss, token_accuracy = _fused_causal_lm_loss(
                self, transform, kept_hidden_states, labels, shift_labels, kwargs
            )
        else:
            logits = transform.project(self.lm_head, kept_hidden_states)
            if fp32_logits:
                logits = logits.float()
            if labels is not None or shift_labels is not None:
                loss = self.loss_function(
                    logits=logits, labels=labels, shift_labels=shift_labels, vocab_size=logits.shape[-1], **kwargs
                )

        # Liger's own output class: TRL reads token_accuracy off it whenever use_liger_kernel is on.
        return LigerMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=None,
            token_accuracy=token_accuracy,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=getattr(outputs, "router_logits", None),
        )

    return lce_forward


class FusedHeadLossUnavailable(ValueError):
    """Liger's fused linear cross-entropy cannot reproduce a model's head path; the message says why."""


def fused_head_ce_sum(
    head: nn.Linear, transform: HeadTransform, hidden: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The fp32 cross-entropy summed over the rows of ``hidden`` under ``transform.project(head, hidden)``, and each
    supervised row's argmax (``-1`` when ignored), through Liger's fused kernel. No ``[N, V]`` logits plane exists: the
    kernel projects a chunk at a time and writes gradients in the forward. A logit scale needs a bias-free head; the
    softcap runs in the kernel."""
    weight, bias = transform.kept_rows(head.weight, head.bias)
    fused_hidden = transform.scale_hidden(hidden)
    if transform.logit_scale is not None:
        fused_hidden = fused_hidden * transform.logit_scale
    result = liger_fused_linear_cross_entropy(
        fused_hidden,
        weight,
        targets,
        bias,
        ignore_index=LABEL_IGNORE_INDEX,
        reduction="sum",
        softcap=transform.softcap,
        accum_dtype=torch.float32,
        return_predicted_tokens=True,
    )
    return result.loss, result.predicted_tokens


def fused_head_loss(model: nn.Module) -> Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """:func:`fused_head_ce_sum` bound to ``model``'s head. Raises :class:`FusedHeadLossUnavailable` on a PEFT tuner or
    a low-precision module on the output embedding, an unverifiable head path, or a logit scale over a biased head."""
    base = base_transformers_model(model)
    head = base.get_output_embeddings()
    if not isinstance(head, nn.Linear) or type(head).forward is not nn.Linear.forward:
        raise FusedHeadLossUnavailable(
            f"{type(base).__name__}'s output embedding is a {type(head).__name__}, not a plain nn.Linear"
        )
    if any(getattr(module, "base_layer", None) is head for module in base.modules()):
        raise FusedHeadLossUnavailable(
            f"a PEFT tuner wraps {type(base).__name__}'s output embedding, whose delta the kernel would drop"
        )
    try:
        transform = resolve_head_transform(base)
    except (TypeError, ValueError) as error:
        raise FusedHeadLossUnavailable(str(error)) from error
    if transform.logit_scale is not None and head.bias is not None:
        raise FusedHeadLossUnavailable(
            f"{type(base).__name__}'s logit scale multiplies a biased head, so it cannot fold onto the hidden states"
        )
    return functools.partial(fused_head_ce_sum, head, transform)
