"""A trainer's pipeline-loss contract and the batch shaping it relies on.

The last stage's loss runs once per microbatch, so an objective must sum to the full-batch loss under a
whole-batch normalizer and survive the ``tensor_split(dim=0)`` microbatch split (pairs interleaved).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from src.distributed.pipeline_parallel.stage_losses import (
    FusedHeadLoss,
    TokenLoss,
    TokenLossFn,
    causal_lm_token_loss,
)
from src.kernels.logprobs import LABEL_IGNORE_INDEX


@dataclass(frozen=True)
class PipelineLossAdapter:
    """How a trainer's objective becomes a last-stage pipeline loss, declared per trainer class.

    Attributes:
        token_loss_fn: ``(output, target) -> summed loss``; target = labels, or ``{"labels", **extras}``.
        batch_transform: collator-space transform of the prepared inputs before the step.
        step_state_fn: ``(inputs) -> None``, the per-step constants a closure loss reads (eval: real rows).
        normalizer: ``(inputs) -> step normalizer``; ``None`` = the DP-global loss-token count / dp.
        extra_target_keys: batch keys shipped to the loss as per-example side tensors.
        pad_spec: pad values for seq-shaped extra keys the fixed-shape collator must also pad.
        eval_normalizer: ``(inputs) -> eval denominator`` over the real rows; ``None`` = this rank's count.
        pin_runner_columns: pin the runner's keys as HF's signature columns; ``False`` keeps the trainer's.
        extra_signature_columns: dataset columns added to the pinned set for the collator.
        predictions_fn: ``(output, inputs) -> predictions``, reduced on the last stage before the broadcast.
        eval_labels_fn: ``(inputs) -> labels`` paired with those predictions; ``None`` = ``inputs["labels"]``.
        metrics_fn: ``() -> {name: 0-dim tensor}`` per step, names from config: only values cross the chain.

    ``predictions_fn`` and ``eval_labels_fn`` emit a fixed number of entries per example, the unit the
    mixin trims the frozen shape's filler rows in.
    """

    token_loss_fn: TokenLossFn
    batch_transform: Callable[[dict], dict] | None = None
    step_state_fn: Callable[[dict], None] | None = None
    normalizer: Callable[[dict], torch.Tensor | float] | None = None
    extra_target_keys: tuple[str, ...] = ()
    pad_spec: Mapping[str, int] | None = None
    eval_normalizer: Callable[[dict], torch.Tensor | float] | None = None
    pin_runner_columns: bool = True
    extra_signature_columns: tuple[str, ...] = ()
    predictions_fn: Callable[[torch.Tensor, dict], torch.Tensor] | None = None
    eval_labels_fn: Callable[[dict], torch.Tensor] | None = None
    metrics_fn: Callable[[], dict[str, torch.Tensor]] | None = None

    def stage_loss(self, *, fuse_head: bool) -> TokenLoss | FusedHeadLoss:
        """The loss the runner is built with.

        Only the causal-LM CE with raw-logit predictions fuses into the last stage's forward: it
        decomposes exactly over token chunks, while every other objective reads the assembled output.
        """
        if fuse_head and self.token_loss_fn is causal_lm_token_loss and self.predictions_fn is None:
            return FusedHeadLoss()
        return TokenLoss(self.token_loss_fn)


def interleave_pairs(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    """Lay a preference batch out as ``[c0, r0, c1, r1, ...]``, so an even-sized ``tensor_split``
    microbatch never separates a pair (a ``[chosen ⧺ rejected]`` concatenation would)."""
    if chosen.shape != rejected.shape:
        raise ValueError(f"chosen {tuple(chosen.shape)} and rejected {tuple(rejected.shape)} must match")
    stacked = torch.stack((chosen, rejected), dim=1)
    return stacked.reshape(2 * chosen.size(0), *chosen.shape[1:])


def split_pairs(interleaved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`interleave_pairs`: ``(chosen, rejected)`` by strided indexing, valid on the
    whole batch or on any microbatch of it."""
    if interleaved.size(0) % 2 != 0:
        raise ValueError(
            f"A paired batch must have an even leading dimension, got {interleaved.size(0)}. "
            f"Under a pipeline this means per_device_train_batch_size / pipeline_microbatches must "
            f"be even, so no pair is split across microbatches."
        )
    return interleaved[0::2], interleaved[1::2]


def rows_with_labels(labels: torch.Tensor) -> torch.Tensor:
    """Bool ``[rows]`` mask, ``False`` for the all-ignore filler rows the PP eval appends.

    Pair and row losses are nonzero at zero log-probs (``-logsigmoid(0)``, ``σ(0)``), so they multiply
    their per-row terms by it before summing.
    """
    return (labels != LABEL_IGNORE_INDEX).any(dim=-1)


def row_count_normalizer(inputs: Mapping[str, torch.Tensor]) -> float:
    """Chain-local row count for unpaired per-row objectives (KTO), whose TRL losses are a
    per-replica mean over rows."""
    return float(inputs["input_ids"].size(0))


def pair_count_normalizer(inputs: Mapping[str, torch.Tensor]) -> float:
    """Chain-local pair count for interleaved preference batches (DPO, Bradley-Terry reward)."""
    return row_count_normalizer(inputs) / 2
