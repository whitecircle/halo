#!/usr/bin/env python
"""Pins for ``cp_boundary_shift`` — the shared CP causal-shift helper (SMPO loss + SFT CP metrics).

A non-final CP rank's last logit predicts the NEXT chunk's first token, so the boundary pair is
appended instead of dropped; the last rank (or a rank without a boundary label) uses the standard
shift. These constants freeze the behavior both call sites depend on.

    python tests/cpu/parallelism/test_cp_boundary_shift.py
"""

import pytest
import torch

from src.distributed.context_parallel.config import cp_boundary_shift


def test_last_rank_standard_causal_shift():
    logits = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    labels = torch.arange(2 * 4).reshape(2, 4)
    shift_logits, shift_labels = cp_boundary_shift(logits, labels, boundary_labels=None, is_last_rank=True)
    assert torch.equal(shift_logits, logits[:, :-1, :])
    assert torch.equal(shift_labels, labels[:, 1:])


def test_missing_boundary_falls_back_to_standard_shift():
    # boundary_labels is None on non-final ranks only when CP is off — same standard shift.
    logits = torch.randn(1, 3, 5)
    labels = torch.randint(0, 5, (1, 3))
    shift_logits, shift_labels = cp_boundary_shift(logits, labels, boundary_labels=None, is_last_rank=False)
    assert torch.equal(shift_logits, logits[:, :-1, :])
    assert torch.equal(shift_labels, labels[:, 1:])


def test_non_last_rank_appends_boundary_pair():
    logits = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    labels = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    boundary = torch.tensor([[14], [24]])  # next chunk's first token per batch row
    shift_logits, shift_labels = cp_boundary_shift(logits, labels, boundary, is_last_rank=False)
    # Every local logit keeps supervision: positions [0..n-2] pair with labels[1:], the final logit
    # pairs with the boundary label — so shift_logits is value-equal to the full local chunk.
    assert torch.equal(shift_logits, logits)
    assert torch.equal(shift_labels, torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]]))
    # Shifted length stays chunk-length aligned (the pre-extraction invariant of both call sites).
    assert shift_logits.shape[1] == logits.shape[1]
    assert shift_labels.shape == labels.shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
