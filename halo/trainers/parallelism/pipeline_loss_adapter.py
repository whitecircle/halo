"""A trainer's pipeline-loss contract. The last stage's loss runs once per microbatch, so an objective must sum to the
full-batch loss under a whole-batch normalizer and survive the ``tensor_split(dim=0)`` split (pairs interleaved)."""

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
    """``token_loss_fn`` is ``(output, target) -> summed loss``, target being labels or ``{"labels", **extras}``. A
    ``None`` normalizer is the DP-global loss-token count / dp, a ``None`` eval one this rank's count over real rows.
    ``metrics_fn`` must name its metrics from config: only values cross the chain."""

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
        """Only the causal-LM CE with raw-logit predictions fuses into the last stage's forward: it decomposes exactly
        over token chunks, while every other objective reads the assembled output."""
        if fuse_head and self.token_loss_fn is causal_lm_token_loss and self.predictions_fn is None:
            return FusedHeadLoss()
        return TokenLoss(self.token_loss_fn)


def interleave_pairs(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    """``[c0, r0, c1, r1, ...]``, so an even-sized ``tensor_split`` microbatch never separates a pair."""
    if chosen.shape != rejected.shape:
        raise ValueError(f"chosen {tuple(chosen.shape)} and rejected {tuple(rejected.shape)} must match")
    stacked = torch.stack((chosen, rejected), dim=1)
    return stacked.reshape(2 * chosen.size(0), *chosen.shape[1:])


def split_pairs(interleaved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(chosen, rejected)`` by strided indexing, valid on the whole batch or on any microbatch of it."""
    if interleaved.size(0) % 2 != 0:
        raise ValueError(
            f"A paired batch must have an even leading dimension, got {interleaved.size(0)}. "
            f"Under a pipeline this means per_device_train_batch_size / pipeline_microbatches must "
            f"be even, so no pair is split across microbatches."
        )
    return interleaved[0::2], interleaved[1::2]


def rows_with_labels(labels: torch.Tensor) -> torch.Tensor:
    """``False`` for the PP eval's all-ignore filler rows. Pair and row losses are nonzero at zero log-probs
    (``-logsigmoid(0)``, ``σ(0)``), so they multiply their per-row terms by it before summing."""
    return (labels != LABEL_IGNORE_INDEX).any(dim=-1)


def row_count_normalizer(inputs: Mapping[str, torch.Tensor]) -> float:
    """Chain-local row count for unpaired per-row objectives (KTO), whose TRL losses are a
    per-replica mean over rows."""
    return float(inputs["input_ids"].size(0))


def pair_count_normalizer(inputs: Mapping[str, torch.Tensor]) -> float:
    return row_count_normalizer(inputs) / 2


def interleave_halves(concatenated: torch.Tensor) -> torch.Tensor:
    return interleave_pairs(*concatenated.chunk(2))


def valid_pairs(labels: torch.Tensor) -> torch.Tensor:
    """``0`` for the PP eval's filler pairs."""
    return rows_with_labels(labels)[0::2].float()


def pair_placeholder_labels(inputs: Mapping[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    """One zero label per interleaved pair, the eval target a per-pair prediction is paired with."""
    input_ids = inputs["input_ids"]
    return torch.zeros(input_ids.size(0) // 2, dtype=dtype, device=input_ids.device)


def last_non_pad_positions(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Rightmost non-pad position per row (an all-pad row yields 0). Transformers' ``GenericForSequenceClassification``
    pooling rule, since PP must score the tokens the unsplit model scores."""
    non_pad_mask = (input_ids != pad_token_id).to(torch.int32)
    token_indices = torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.int32)
    return (token_indices * non_pad_mask).argmax(dim=-1)


def encode_pooling_plane(input_ids: torch.Tensor, pad_token_id: int, values: int | torch.Tensor) -> torch.Tensor:
    """``LABEL_IGNORE_INDEX`` except ``values`` at each row's pooled position. The only per-row channel the runner
    ships unconditionally, and it makes the eval's inert row padding self-identifying."""
    plane = torch.full_like(input_ids, LABEL_IGNORE_INDEX)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    plane[rows, last_non_pad_positions(input_ids, pad_token_id)] = values
    return plane


class PoolingPlane(NamedTuple):
    valid: torch.Tensor
    positions: torch.Tensor
    values: torch.Tensor


def decode_pooling_plane(plane: torch.Tensor) -> PoolingPlane:
    """Inverse of :func:`encode_pooling_plane` on a batch or microbatch; unmarked rows are the PP eval's row
    padding."""
    positions = (plane != LABEL_IGNORE_INDEX).int().argmax(dim=-1)
    rows = torch.arange(plane.size(0), device=plane.device)
    return PoolingPlane(rows_with_labels(plane), positions, plane[rows, positions])


def pooled_logits(logits: torch.Tensor, plane: PoolingPlane) -> torch.Tensor:
    return logits[torch.arange(logits.size(0), device=logits.device), plane.positions]
