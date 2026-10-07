"""Chunked log-probs over a logits plane, in at least fp32.

Whole, the fp32 upcast of a logits plane and the ``log_softmax`` output saved for backward are two
fp32 planes, ~26 GB at ``V=201088``, ``S=8192``. The pipeline-parallel last-stage losses, SMPO's
loss paths and the DPO/KTO loss paths (``src/trainers/preference/logprobs.py``) all chunk under the
one budget below.
"""

from collections.abc import Callable, Iterator

import torch
from torch.utils.checkpoint import checkpoint

# fp32 elements per chunk of a [tokens, V] plane; a non-reentrant checkpoint bounds the held fp32 state
# to one chunk. Budgeted in elements rather than token rows because the plane is tokens×V, so a fixed
# row count would scale the held state with the vocabulary. 128M elements = 512 MB fp32, i.e. 4096
# rows at V=32k.
LOGIT_CHUNK_ELEMENTS = 128 * 1024 * 1024


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


def selective_logprobs(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``[B, T]`` log-probs of ``index`` under already-aligned ``[B, T, V]`` logits, in at least fp32.

    TRL's ``selective_log_softmax`` contract for one index per position, minus that function's bf16
    branch, which returns bf16 log-probs. Every chunk is checkpointed, short rows included, so
    backward holds no fp32 state; rows are taken one at a time, so a non-contiguous view such as
    ``logits[..., :-1, :]`` is never copied whole.
    """
    if logits.dim() != 3 or index.shape != logits.shape[:-1]:
        raise ValueError(
            f"selective_logprobs takes [B, T, V] logits and one index per position ([B, T]), got "
            f"logits {tuple(logits.shape)} and index {tuple(index.shape)}."
        )
    chunk_rows = logit_chunk_rows(logits.size(-1))
    return torch.stack(
        [
            torch.cat(list(checkpointed_chunks(_logprob_chunk, row_logits, row_index, chunk_rows)))
            for row_logits, row_index in zip(logits, index, strict=True)
        ]
    )
