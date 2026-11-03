"""The last pipeline stage's losses. ``torch.distributed.pipelining`` calls the loss once per microbatch and sums
nothing, so every loss returns a sum the runner divides by the whole step's normalizer."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from halo.kernels.logprobs import chunked_token_sum, cross_entropy_sum, logit_chunk_rows, shift_labels_left
from halo.models.head_transform import HeadTransform

# ``(last-stage output, target) -> summed loss``; the target is the labels, or a
# ``{"labels": ..., **extras}`` dict when the step ships extra targets.
TokenLossFn = Callable[[torch.Tensor, torch.Tensor | dict], torch.Tensor]
# ``(head, head_transform, hidden_states, labels) -> summed loss``, run inside the last stage's forward.
FusedHeadLossFn = Callable[[nn.Module, HeadTransform, torch.Tensor, torch.Tensor], torch.Tensor]


def _head_ce_sum_chunk(
    head: nn.Module, head_transform: HeadTransform, chunk_hidden: torch.Tensor, chunk_labels: torch.Tensor
) -> torch.Tensor:
    """One token chunk's CE through the family's head path: the unit the checkpoint recomputes."""
    return cross_entropy_sum(head_transform.project(head, chunk_hidden), chunk_labels)


def causal_lm_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted causal-LM cross-entropy summed over tokens, chunked, at least fp32."""
    return chunked_token_sum(
        cross_entropy_sum,
        logits.reshape(-1, logits.size(-1)),
        shift_labels_left(labels).reshape(-1),
        logit_chunk_rows(logits.size(-1)),
    )


def fused_causal_lm_token_loss(
    head: nn.Module, head_transform: HeadTransform, hidden_states: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """:func:`causal_lm_token_loss` with the head folded in, so no ``[mb, S, V]`` logits plane is built. Head and CE
    run per token chunk under a non-reentrant checkpoint, sized from the head's full width. The loss and activation
    gradients match the unfused path exactly; the head weight's gradient is reassociated per chunk (rounding noise in
    fp32, ~1e-2 relative in bf16)."""
    vocab_size = getattr(head, "out_features", None)
    if vocab_size is None:
        raise TypeError(
            f"The fused last-stage loss sizes its token chunks from the head's output width, and "
            f"{type(head).__name__} exposes no `out_features`. Only a linear task head can be fused; "
            f"set HALO_PP_FUSED_HEAD_LOSS=0 to keep the unfused logits path."
        )
    return chunked_token_sum(
        functools.partial(_head_ce_sum_chunk, head, head_transform),
        hidden_states.reshape(-1, hidden_states.size(-1)),
        shift_labels_left(labels).reshape(-1),
        logit_chunk_rows(vocab_size),
    )


@dataclass(frozen=True)
class TokenLoss:
    """The last stage returns its output and the schedule scores each microbatch with ``fn``."""

    fn: TokenLossFn = causal_lm_token_loss


@dataclass(frozen=True)
class FusedHeadLoss:
    """The last stage computes the causal-LM CE inside its forward
    (:func:`fused_causal_lm_token_loss`) and returns the summed loss instead of logits."""
