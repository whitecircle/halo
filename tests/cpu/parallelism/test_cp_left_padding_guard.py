"""CP must refuse a left-padded batch.

The Ulysses attention path never uses ``attention_mask``: both dense-attention call sites pass only
``causal=``, and the package has no varlen path. That is safe for TRAILING padding, which a causal
mask ignores and whose labels are ``-100``, and silently wrong for LEADING padding — every real token
attends the pads, and the identical batch computes a different loss without CP, where HF's FA2 unpads
via the mask.

This is reachable from a shipped trainer: ``DataCollatorForSMPO`` left-pads prompts unconditionally
and SMPO is one of only two CP-enabled trainers. The repo already rejects ``padding_free`` and
``packing`` under CP for the very same mechanism, so this closes the remaining case.

    python tests/cpu/parallelism/test_cp_left_padding_guard.py
"""

import sys

import pytest
import torch

from src.distributed.context_parallel.wrapper import _reject_left_padding


def test_left_padding_in_any_row_raises():
    """One offending row is enough — it is the row whose loss is wrong."""
    with pytest.raises(ValueError, match="LEFT-padded"):
        _reject_left_padding(torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]]))


def test_smpo_collator_shape_is_what_the_guard_catches():
    """The concrete reachable case: SMPO left-pads prompts, then concatenates right-padded
    completions, so a batch with unequal prompt lengths carries leading pads."""
    # prompt lengths 2 and 4, left-padded to 4; completions right-padded to 3.
    prompt_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    completion_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    with pytest.raises(ValueError, match="LEFT-padded"):
        _reject_left_padding(torch.cat([prompt_mask, completion_mask], dim=1))


def test_equal_length_prompts_produce_no_leading_pad():
    """Anti-vacuity: the guard must not fire on the batch=1 / equal-length case the error message
    recommends as the workaround, or that advice would be wrong."""
    prompt_mask = torch.tensor([[1, 1, 1, 1]])
    completion_mask = torch.tensor([[1, 1, 0]])
    _reject_left_padding(torch.cat([prompt_mask, completion_mask], dim=1))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
