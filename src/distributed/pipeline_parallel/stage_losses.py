"""The last pipeline stage's losses: the loss value a runner is built with, and the causal-LM CE.

``torch.distributed.pipelining`` calls the loss once per microbatch and sums nothing itself, so every
loss here returns a sum, which the runner divides by the whole step's normalizer.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kernels.logprobs import (
    LABEL_IGNORE_INDEX,
    at_least_fp32,
    checkpointed_chunks,
    logit_chunk_rows,
    shift_labels_left,
)
from src.models.head_transform import HeadTransform

# ``(last-stage output, target) -> summed loss``; the target is the labels, or a
# ``{"labels": ..., **extras}`` dict when the step ships extra targets.
TokenLossFn = Callable[[torch.Tensor, torch.Tensor | dict], torch.Tensor]
# ``(head, head_transform, hidden_states, labels) -> summed loss``, run inside the last stage's forward.
FusedHeadLossFn = Callable[[nn.Module, HeadTransform, torch.Tensor, torch.Tensor], torch.Tensor]


def _ce_sum_chunk(chunk_logits: torch.Tensor, chunk_labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(at_least_fp32(chunk_logits), chunk_labels, ignore_index=LABEL_IGNORE_INDEX, reduction="sum")


def _head_ce_sum_chunk(
    head: nn.Module, head_transform: HeadTransform, chunk_hidden: torch.Tensor, chunk_labels: torch.Tensor
) -> torch.Tensor:
    """One token chunk projected through the family's head path, its CE summed: the unit the
    checkpoint recomputes."""
    return _ce_sum_chunk(head_transform.project(head, chunk_hidden), chunk_labels)


def _chunked_token_sum(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rows: torch.Tensor,
    flat_labels: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    """``sum(fn(rows[chunk], flat_labels[chunk]))`` over checkpointed token chunks, accumulated in at
    least fp32 (an fp64 chunk result widens the sum). A single chunk needs no checkpoint."""
    total = rows.new_zeros((), dtype=torch.float32)
    if flat_labels.numel() <= chunk_rows:
        return total + fn(rows, flat_labels)
    for value in checkpointed_chunks(fn, rows, flat_labels, chunk_rows):
        total = total + value
    return total


def causal_lm_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted causal-LM cross-entropy summed over tokens, chunked, at least fp32."""
    return _chunked_token_sum(
        _ce_sum_chunk,
        logits.reshape(-1, logits.size(-1)),
        shift_labels_left(labels).reshape(-1),
        logit_chunk_rows(logits.size(-1)),
    )


def fused_causal_lm_token_loss(
    head: nn.Module, head_transform: HeadTransform, hidden_states: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """:func:`causal_lm_token_loss` with the head folded in, so no ``[mb, S, V]`` logits plane is built.

    Head projection and CE run together over one token chunk at a time under a non-reentrant
    checkpoint, so neither the logits nor their gradient exist whole. The loss and the activation
    gradients match the unfused path exactly; the head weight's gradient is reassociated into one
    partial GEMM per chunk (rounding noise in fp32, ~1e-2 relative in bf16). Chunks are sized from the
    head's full width, which a vocabulary truncation only narrows.
    """
    vocab_size = getattr(head, "out_features", None)
    if vocab_size is None:
        raise TypeError(
            f"The fused last-stage loss sizes its token chunks from the head's output width, and "
            f"{type(head).__name__} exposes no `out_features`. Only a linear task head can be fused; "
            f"set HALO_PP_FUSED_HEAD_LOSS=0 to keep the unfused logits path."
        )
    return _chunked_token_sum(
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
