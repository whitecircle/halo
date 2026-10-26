"""A trainer's pipeline-loss contract and the batch shaping it relies on.

The last stage's loss runs once per microbatch, so an objective must sum to the full-batch loss under a
whole-batch normalizer and survive the ``tensor_split(dim=0)`` microbatch split (pairs interleaved).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import torch

from halo.distributed.pipeline_parallel.stage_losses import (
    FusedHeadLoss,
    TokenLoss,
    TokenLossFn,
    causal_lm_token_loss,
)
from halo.kernels.logprobs import LABEL_IGNORE_INDEX


@dataclass(frozen=True)
class PipelineLossAdapter:
    """How a trainer's objective becomes a last-stage pipeline loss, declared per trainer class.

    Attributes:
        token_loss_fn: ``(output, target) -> summed loss``; target = labels, or ``{"labels", **extras}``.
        batch_transform / step_state_fn: the prepared inputs' transform, and the per-step constants a
            closure loss reads (in eval, of the real rows).
        normalizer / eval_normalizer: the step / eval denominator; ``None`` = the DP-global loss-token
            count / dp, and this rank's count over the real rows.
        extra_target_keys / pad_spec: per-example side tensors for the loss, and pads for the seq-shaped ones.
        pin_runner_columns: pin the runner's keys (plus ``extra_signature_columns``) as HF's signature columns.
        predictions_fn / eval_labels_fn: the eval output reduced on the last stage before the broadcast, and
            the labels paired with it, a fixed number of entries per example.
        metrics_fn: ``() -> {name: 0-dim tensor}`` per step, named from config: only values cross the chain.
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


def interleave_halves(concatenated: torch.Tensor) -> torch.Tensor:
    """:func:`interleave_pairs` over a ``[chosen ⧺ rejected]`` concatenation."""
    return interleave_pairs(*concatenated.chunk(2))


def valid_pairs(labels: torch.Tensor) -> torch.Tensor:
    """Float ``[pairs]`` mask of an interleaved batch, ``0`` for the PP eval's filler pairs."""
    return rows_with_labels(labels)[0::2].float()


def pair_placeholder_labels(inputs: Mapping[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    """One zero label per interleaved pair, the eval target a per-pair prediction is paired with."""
    input_ids = inputs["input_ids"]
    return torch.zeros(input_ids.size(0) // 2, dtype=dtype, device=input_ids.device)


def last_non_pad_positions(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Rightmost position whose token != ``pad_token_id``, per row (an all-pad row yields 0).

    Transformers' exact sequence-classification pooling rule (``GenericForSequenceClassification``), left- and
    right-padding alike: PP must score the tokens the unsplit model scores.
    """
    non_pad_mask = (input_ids != pad_token_id).to(torch.int32)
    token_indices = torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.int32)
    return (token_indices * non_pad_mask).argmax(dim=-1)


def encode_pooling_plane(input_ids: torch.Tensor, pad_token_id: int, values: int | torch.Tensor) -> torch.Tensor:
    """``[B, S]`` labels plane: ``LABEL_IGNORE_INDEX`` except ``values`` at each row's pooled position.

    The plane is the only per-row channel the pipeline runner ships unconditionally, so it hands the last stage
    the position and the row's target with no side tensors, and makes the eval's inert row padding
    self-identifying — which ``input_ids`` are not, their filler carrying the tokenizer's pad id.
    """
    plane = torch.full_like(input_ids, LABEL_IGNORE_INDEX)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    plane[rows, last_non_pad_positions(input_ids, pad_token_id)] = values
    return plane


class PoolingPlane(NamedTuple):
    """A decoded labels plane: the marked rows, each row's pooled position and the value marked there."""

    valid: torch.Tensor
    positions: torch.Tensor
    values: torch.Tensor


def decode_pooling_plane(plane: torch.Tensor) -> PoolingPlane:
    """Inverse of :func:`encode_pooling_plane` on a batch or any microbatch of it; unmarked rows are the PP eval's
    row padding."""
    positions = (plane != LABEL_IGNORE_INDEX).int().argmax(dim=-1)
    rows = torch.arange(plane.size(0), device=plane.device)
    return PoolingPlane(rows_with_labels(plane), positions, plane[rows, positions])


def pooled_logits(logits: torch.Tensor, plane: PoolingPlane) -> torch.Tensor:
    """Each row's ``[B, S, ...]`` head output at its pooled position."""
    return logits[torch.arange(logits.size(0), device=logits.device), plane.positions]
