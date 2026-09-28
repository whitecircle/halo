#!/usr/bin/env python
"""``cp_shift_against_full_labels``: the CP ranks' shifted chunks tile the unsplit causal shift.

The CP wrapper's loss, SMPO's CP forward and SFT's CP metrics all pair a rank's local logits with the
full pre-split labels through this one helper. Concatenated over the ranks, the ``(logit, label)``
pairs must be exactly the unsplit ``logits[:, :-1]`` / ``labels[:, 1:]`` pairs: every non-final rank
keeps its last logit, supervised by the next chunk's first label, and only the final rank drops one.

    python tests/cpu/parallelism/test_cp_full_label_shift.py
"""

import pytest
import torch

from src.distributed.context_parallel.config import cp_shift_against_full_labels

BATCH, SEQ, VOCAB = 3, 24, 11


@pytest.mark.parametrize("cp_size", [2, 3, 4])
def test_rank_shifts_tile_the_unsplit_shift(cp_size):
    generator = torch.Generator().manual_seed(cp_size)
    logits = torch.randn(BATCH, SEQ, VOCAB, generator=generator)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator)
    chunk = SEQ // cp_size

    shifted = [
        cp_shift_against_full_labels(logits[:, rank * chunk : (rank + 1) * chunk], labels, rank, cp_size)
        for rank in range(cp_size)
    ]
    for rank, (shift_logits, shift_labels) in enumerate(shifted):
        expected_width = chunk - 1 if rank == cp_size - 1 else chunk
        assert shift_logits.shape[1] == shift_labels.shape[1] == expected_width, f"rank {rank}"

    assert torch.equal(torch.cat([s_logits for s_logits, _ in shifted], dim=1), logits[:, :-1])
    assert torch.equal(torch.cat([s_labels for _, s_labels in shifted], dim=1), labels[:, 1:])


def test_a_sequence_the_ranks_cannot_split_evenly_is_refused():
    logits = torch.zeros(BATCH, SEQ // 2, VOCAB)
    labels = torch.zeros(BATCH, SEQ + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="divisible by context_parallel_size 2"):
        cp_shift_against_full_labels(logits, labels, 0, 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
