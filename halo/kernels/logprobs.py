"""Chunked log-probs and cross-entropy sums over a logits plane, in at least fp32, and their next-token label math.
Whole, the fp32 upcast and the ``log_softmax`` output saved for backward are two fp32 planes, ~26 GB at ``V=201088``,
``S=8192``. The hidden-state form is :mod:`halo.kernels.head_logprobs`."""

from collections.abc import Callable, Iterator

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# fp32 elements per chunk of a [tokens, V] plane, as a fixed row count would scale the held state with the vocabulary.
# 128M elements = 512 MB fp32, i.e. 4096 rows at V=32k.
LOGIT_CHUNK_ELEMENTS = 128 * 1024 * 1024

LABEL_IGNORE_INDEX = -100


def logit_chunk_rows(vocab_size: int) -> int:
    return max(1, LOGIT_CHUNK_ELEMENTS // max(vocab_size, 1))


def at_least_fp32(logits: torch.Tensor) -> torch.Tensor:
    return logits.to(torch.promote_types(logits.dtype, torch.float32))


def checkpointed_chunks(
    fn: Callable[..., torch.Tensor], *tensors: torch.Tensor, chunk_rows: int
) -> Iterator[torch.Tensor]:
    """Each chunk under a non-reentrant checkpoint, so backward holds no chunk's fp32 state. One ``split`` per input,
    as its backward concatenates gradients once; a slice per chunk zero-fills a whole-input gradient each time."""
    for chunk in zip(*(tensor.split(chunk_rows) for tensor in tensors), strict=True):
        yield checkpoint(fn, *chunk, use_reentrant=False)


def map_token_chunks(fn: Callable[..., torch.Tensor], *tensors: torch.Tensor) -> torch.Tensor:
    """``fn`` over aligned ``[T, ...]`` or ``[B, T, ...]`` tensors, the first a logits plane whose width sizes the
    chunks. Rows go one sequence at a time, so a non-contiguous ``logits[..., :-1, :]`` is never copied whole."""
    first = tensors[0]
    if first.dim() == 2:
        chunk_rows = logit_chunk_rows(first.size(-1))
        return torch.cat(list(checkpointed_chunks(fn, *tensors, chunk_rows=chunk_rows)))
    if first.size(0) == 0:
        # One empty chunk keeps the result's dtype.
        return map_token_chunks(fn, *(tensor.flatten(0, 1) for tensor in tensors)).view(first.shape[:2])
    return torch.stack([map_token_chunks(fn, *row) for row in zip(*tensors, strict=True)])


def chunked_token_sum(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rows: torch.Tensor,
    flat_labels: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    """Accumulated in at least fp32 (an fp64 chunk result widens the sum). A single chunk needs no checkpoint."""
    total = rows.new_zeros((), dtype=torch.float32)
    if flat_labels.numel() <= chunk_rows:
        return total + fn(rows, flat_labels)
    for value in checkpointed_chunks(fn, rows, flat_labels, chunk_rows=chunk_rows):
        total = total + value
    return total


def cross_entropy_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(at_least_fp32(logits), labels, ignore_index=LABEL_IGNORE_INDEX, reduction="sum")


def _logprob_chunk(chunk_logits: torch.Tensor, chunk_labels: torch.Tensor) -> torch.Tensor:
    logps = torch.log_softmax(at_least_fp32(chunk_logits), dim=-1)
    return logps.gather(-1, chunk_labels.unsqueeze(-1)).squeeze(-1)


def selective_logprobs(
    logits: torch.Tensor, index: torch.Tensor, row_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """``[B, T]`` log-probs of ``index`` under aligned ``[B, T, V]`` logits. TRL's ``selective_log_softmax`` contract
    minus its bf16 branch, which returns bf16 log-probs; a ``row_mask`` zero yields a zero log-prob."""
    if logits.dim() != 3 or index.shape != logits.shape[:-1]:
        raise ValueError(
            f"selective_logprobs takes [B, T, V] logits and one index per position ([B, T]), got "
            f"logits {tuple(logits.shape)} and index {tuple(index.shape)}."
        )
    if row_mask is not None and row_mask.shape != index.shape:
        raise ValueError(f"row_mask {tuple(row_mask.shape)} must match index {tuple(index.shape)}.")
    logps = map_token_chunks(_logprob_chunk, logits, index)
    return logps if row_mask is None else logps.masked_fill(row_mask == 0, 0.0)


def shift_labels_left(labels: torch.Tensor) -> torch.Tensor:
    """Slicing the logits instead would turn the caller's ``reshape`` view into a ``[B, S, V]`` copy."""
    shifted = torch.full_like(labels, LABEL_IGNORE_INDEX)
    shifted[:, :-1] = labels[:, 1:]
    return shifted


def next_token_targets(labels: torch.Tensor, start: int = 0, length: int | None = None) -> torch.Tensor:
    return shift_labels_left(labels)[:, start : None if length is None else start + length]


def next_token_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, start: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-prob each position of ``logits`` gives its next token in ``labels``, plus the non-ignored mask. ``logits``
    starts at ``start`` of the full labels, so a CP rank's last position scores against the next chunk's first token.
    Ignored positions carry the finite log-prob of label 0, so clamp or reweight before masking."""
    if logits.size(0) != labels.size(0) or start < 0 or start + logits.size(1) > labels.size(1):
        raise ValueError(
            f"logits {tuple(logits.shape[:2])} at start={start} do not lie inside labels "
            f"{tuple(labels.shape)}: the rows must match and positions [start, start + "
            f"{logits.size(1)}) must fall within the labels' {labels.size(1)}."
        )
    targets = next_token_targets(labels, start, logits.size(1))
    mask = targets != LABEL_IGNORE_INDEX
    return selective_logprobs(logits, targets.masked_fill(~mask, 0)), mask


def token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The final position is dropped here, as slicing it off the logits would rebuild a full plane in backward."""
    logps, mask = next_token_logprobs(logits, labels)
    return logps[:, :-1], mask[:, :-1]


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logps, mask = token_logprobs(logits, labels)
    return (logps * mask).sum(dim=-1)


def loss_token_counts_per_row(labels: torch.Tensor) -> torch.Tensor:
    """Raw labels lose position 0 to the next-token shift, so an unshifted count mis-scales a token-normalized loss."""
    return (labels[:, 1:] != LABEL_IGNORE_INDEX).sum(dim=-1)
