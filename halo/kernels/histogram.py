"""Sync-free integer histograms for MoE routing. ``torch.bincount`` sizes its output from the largest index, so CUDA
reads that value back to the host, a full device sync per MoE layer per forward."""

from __future__ import annotations

import torch


def sync_free_bincount(
    indices: torch.Tensor, size: int, *, dtype: torch.dtype = torch.float32, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """The caller filters out-of-range values, which ``scatter_add_`` would write out of bounds. ``dtype`` is fp32
    where counts feed metric math, int64 where they become offsets (integer adds are exactly associative)."""
    flat = indices.flatten().long()
    counts = torch.zeros(size, dtype=dtype, device=flat.device)
    occurrences = torch.ones_like(flat, dtype=dtype) if weights is None else weights.flatten().to(dtype)
    counts.scatter_add_(0, flat, occurrences)
    return counts


def accumulate_bincount(
    counter: torch.Tensor | None, indices: torch.Tensor, size: int, *, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """``counter`` is ``None`` until the first forward that routes and is added into in place afterwards."""
    counts = sync_free_bincount(indices, size, weights=weights)
    return counts if counter is None else counter.add_(counts)
