"""Chunked fp32 log-probs over a ``[B, T, V]`` logits plane.

Whole, the ``.float()`` upcast of a logits plane and the ``log_softmax`` output saved for backward are
two fp32 planes, ~26 GB at ``V=201088``, ``S=8192``. The pipeline-parallel last-stage losses and the
DPO/KTO loss paths (``src/trainers/preference/logprobs.py``) all chunk under the one budget below.
"""

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


def _logprob_chunk(chunk_logits: torch.Tensor, chunk_labels: torch.Tensor) -> torch.Tensor:
    """fp32 log-probs of ``chunk_labels`` under one ``[rows, V]`` chunk — a ``[rows]`` vector."""
    logps = torch.log_softmax(chunk_logits.float(), dim=-1)
    return logps.gather(-1, chunk_labels.unsqueeze(-1)).squeeze(-1)


def selective_logprobs(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """fp32 ``[B, T]`` log-probs of ``index`` under already-aligned ``[B, T, V]`` logits.

    TRL's ``selective_log_softmax`` contract for one index per position, minus that function's bf16
    branch, which returns bf16 log-probs. Every chunk runs under a non-reentrant checkpoint, short rows
    included, so backward holds no fp32 state; rows are taken one at a time, so a non-contiguous view
    such as ``logits[..., :-1, :]`` is never copied whole.
    """
    if logits.dim() != 3 or index.shape != logits.shape[:-1]:
        raise ValueError(
            f"selective_logprobs takes [B, T, V] logits and one index per position ([B, T]), got "
            f"logits {tuple(logits.shape)} and index {tuple(index.shape)}."
        )
    chunk_rows = logit_chunk_rows(logits.size(-1))
    rows = []
    for row_logits, row_index in zip(logits, index, strict=True):
        # One split per row: per-chunk slices would each zero-fill a whole-row gradient in backward.
        chunks = [
            checkpoint(_logprob_chunk, chunk_logits, chunk_index, use_reentrant=False)
            for chunk_logits, chunk_index in zip(
                row_logits.split(chunk_rows), row_index.split(chunk_rows), strict=True
            )
        ]
        rows.append(torch.cat(chunks))
    return torch.stack(rows)
