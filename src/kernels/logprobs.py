"""Chunked log-probs over a logits plane, in at least fp32, and the next-token label math they score.

Whole, the fp32 upcast of a logits plane and the ``log_softmax`` output saved for backward are two
fp32 planes, ~26 GB at ``V=201088``, ``S=8192``, so every caller chunks under the one budget below. The
hidden-state form, which never builds the plane, is :mod:`src.kernels.head_logprobs`.
"""

from collections.abc import Callable, Iterator

import torch
from torch.utils.checkpoint import checkpoint

# fp32 elements per chunk of a [tokens, V] plane; a non-reentrant checkpoint bounds the held fp32 state
# to one chunk. Budgeted in elements rather than token rows because the plane is tokens×V, so a fixed
# row count would scale the held state with the vocabulary. 128M elements = 512 MB fp32, i.e. 4096
# rows at V=32k.
LOGIT_CHUNK_ELEMENTS = 128 * 1024 * 1024

# The HF cross_entropy ignore_index: every label plane marks its unscored positions with it.
LABEL_IGNORE_INDEX = -100


def logit_chunk_rows(vocab_size: int) -> int:
    """Token rows whose fp32 [rows, V] plane fits the chunk budget; at least one row."""
    return max(1, LOGIT_CHUNK_ELEMENTS // max(vocab_size, 1))


def at_least_fp32(logits: torch.Tensor) -> torch.Tensor:
    """``logits`` in fp32, or in their own wider dtype (an fp64 oracle stays fp64)."""
    return logits.to(torch.promote_types(logits.dtype, torch.float32))


def checkpointed_chunks(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rows: torch.Tensor,
    labels: torch.Tensor,
    chunk_rows: int,
) -> Iterator[torch.Tensor]:
    """``fn(rows[chunk], labels[chunk])`` per chunk of ``chunk_rows`` along dim 0, each under a
    non-reentrant checkpoint, so backward holds no chunk's fp32 state.

    The chunks come from one ``split`` per input, whose backward concatenates their gradients once; a
    slice per chunk would instead zero-fill a whole-input gradient in each chunk's backward.
    """
    for row_chunk, label_chunk in zip(rows.split(chunk_rows), labels.split(chunk_rows), strict=True):
        yield checkpoint(fn, row_chunk, label_chunk, use_reentrant=False)


def _logprob_chunk(chunk_logits: torch.Tensor, chunk_labels: torch.Tensor) -> torch.Tensor:
    """Log-probs of ``chunk_labels`` under one ``[rows, V]`` chunk (:func:`at_least_fp32`), a ``[rows]`` vector."""
    logps = torch.log_softmax(at_least_fp32(chunk_logits), dim=-1)
    return logps.gather(-1, chunk_labels.unsqueeze(-1)).squeeze(-1)


def selective_logprobs(
    logits: torch.Tensor, index: torch.Tensor, row_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """``[B, T]`` log-probs of ``index`` under already-aligned ``[B, T, V]`` logits, in at least fp32.

    TRL's ``selective_log_softmax`` contract for one index per position, minus that function's bf16
    branch, which returns bf16 log-probs: a ``row_mask`` zero yields a zero log-prob there. Every
    chunk is checkpointed, short rows included, so backward holds no fp32 state; rows are taken one at
    a time, so a non-contiguous view such as ``logits[..., :-1, :]`` is never copied whole.
    """
    if logits.dim() != 3 or index.shape != logits.shape[:-1]:
        raise ValueError(
            f"selective_logprobs takes [B, T, V] logits and one index per position ([B, T]), got "
            f"logits {tuple(logits.shape)} and index {tuple(index.shape)}."
        )
    if row_mask is not None and row_mask.shape != index.shape:
        raise ValueError(f"row_mask {tuple(row_mask.shape)} must match index {tuple(index.shape)}.")
    chunk_rows = logit_chunk_rows(logits.size(-1))
    logps = torch.stack(
        [
            torch.cat(list(checkpointed_chunks(_logprob_chunk, row_logits, row_index, chunk_rows)))
            for row_logits, row_index in zip(logits, index, strict=True)
        ]
    )
    return logps if row_mask is None else logps.masked_fill(row_mask == 0, 0.0)


def shift_labels_left(labels: torch.Tensor) -> torch.Tensor:
    """Next-token targets: ``labels`` rolled one position left, the last position ignored.

    Shifting the labels rather than slicing the logits keeps the caller's ``reshape`` a view instead
    of a ``[B, S, V]`` copy.
    """
    shifted = torch.full_like(labels, LABEL_IGNORE_INDEX)
    shifted[:, :-1] = labels[:, 1:]
    return shifted


def next_token_targets(labels: torch.Tensor, start: int = 0, length: int | None = None) -> torch.Tensor:
    """The next-token targets of positions ``[start, start + length)`` of the full ``[B, S]`` labels."""
    return shift_labels_left(labels)[:, start : None if length is None else start + length]


def next_token_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, start: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-prob each position of ``logits`` gives its next token in ``labels``, plus the non-ignored mask.

    ``logits`` covers positions ``[start, start + logits.size(1))`` of the full ``[B, S]`` ``labels``,
    so a sequence chunk (a CP rank's) is scored against the labels it was cut from, its last
    position against the next chunk's first token. Returns two ``[B, logits.size(1)]`` tensors, the
    log-probs in fp32 (fp64 logits keep fp64); the sequence's final position has no next token and
    is masked. The log-probs are not pre-masked: ignored positions carry the (finite) log-prob of
    label 0, so callers that clamp or reweight per token do so before applying the mask.
    """
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
    """Shifted ``[B, S-1]`` per-token log-probs of ``labels`` under ``logits`` plus the non-ignored mask.

    The final, ignore-only position is dropped from the ``[B, S]`` result rather than sliced off the
    logits, whose backward would rebuild a full plane.
    """
    logps, mask = next_token_logprobs(logits, labels)
    return logps[:, :-1], mask[:, :-1]


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sequence summed log-probability of ``labels`` under ``logits`` (ignore_index aware)."""
    logps, mask = token_logprobs(logits, labels)
    return (logps * mask).sum(dim=-1)


def loss_token_counts_per_row(labels: torch.Tensor) -> torch.Tensor:
    """Per-row ``[rows]`` count of non-ignored next-token targets in ``labels``.

    Raw labels lose position 0 to the next-token shift, so counting them unshifted mis-scales a
    token-normalized loss by a token per row.
    """
    return (labels[:, 1:] != LABEL_IGNORE_INDEX).sum(dim=-1)
