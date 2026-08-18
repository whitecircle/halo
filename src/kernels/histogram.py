"""Sync-free integer histograms for MoE routing.

``torch.bincount`` sizes its output from the largest index, so CUDA reads that value back to the host —
a full device sync per MoE layer per forward on the routing hot path. Counting into a pre-sized buffer
with ``scatter_add_`` needs no round-trip.
"""

from __future__ import annotations

import torch


def sync_free_bincount(indices: torch.Tensor, size: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Occurrences of each value in ``[0, size)`` across ``indices``, without a device sync.

    ``indices`` is flattened and cast to ``int64``; the caller must filter out-of-range values, which
    ``scatter_add_`` would write out of bounds. ``dtype`` is the COUNTER's: fp32 where the counts feed
    metric math, int64 where they become offsets (integer adds are exactly associative).
    """
    flat = indices.flatten().long()
    counts = torch.zeros(size, dtype=dtype, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=dtype))
    return counts


def accumulate_bincount(counter: torch.Tensor | None, indices: torch.Tensor, size: int) -> torch.Tensor:
    """Add ``indices``' counts into ``counter``, creating it on the first call and returning it.

    The create-or-add tail every per-step load counter shares: ``None`` until the first forward that
    routes, added into in place afterwards. Counts are fp32 — these feed metric math, never offsets.
    """
    counts = sync_free_bincount(indices, size)
    return counts if counter is None else counter.add_(counts)
