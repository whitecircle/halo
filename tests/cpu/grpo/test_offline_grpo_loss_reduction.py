"""Offline GRPO's per-loss-type reduction (``loss_numerator`` / ``loss_normalizer``), the one dispatch
the non-PP loss and the pipeline's microbatch numerator + per-step normalizer share.

Hand-derived constants on a two-row batch: row 0 has three loss tokens and group size 2 (weight 0.5),
row 1 has two loss tokens and group size 4 (weight 0.25).

    python tests/cpu/grpo/test_offline_grpo_loss_reduction.py
"""

import pytest
import torch

from src.trainers.grpo.objective.offline import LOSS_TYPES, loss_normalizer, loss_numerator

PER_TOKEN_LOSS = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0]])
MASK = torch.tensor([[1, 1, 1], [1, 1, 0]])
GROUP_SIZES = torch.tensor([2, 4])
MAX_COMPLETION_LENGTH = 3

# (numerator, denominator) per loss type; the masked 7.0 never enters.
EXPECTED = {
    # per-row means of the weighted tokens: (0.5 + 1 + 1.5) / 3 + (1 + 1.25) / 2; summed weights 0.75
    "grpo": (2.125, 0.75),
    # weighted token sum 3 + 2.25; weighted token count 0.5 * 3 + 0.25 * 2
    "bnpo": (5.25, 2.0),
    # weighted token sum; summed weights x max_completion_length
    "dr_grpo": (5.25, 2.25),
}


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_reduction_matches_hand_derived_values(loss_type):
    numerator = loss_numerator(PER_TOKEN_LOSS, MASK, GROUP_SIZES, loss_type)
    denominator = loss_normalizer(GROUP_SIZES, MASK.sum(dim=1), loss_type, MAX_COMPLETION_LENGTH)
    assert (numerator.item(), denominator.item()) == pytest.approx(EXPECTED[loss_type])


def test_microbatch_numerators_sum_to_the_batch_numerator():
    """The pipeline divides once per step, so each loss type's numerator must be row-local."""
    for loss_type in LOSS_TYPES:
        whole = loss_numerator(PER_TOKEN_LOSS, MASK, GROUP_SIZES, loss_type)
        split = sum(
            loss_numerator(PER_TOKEN_LOSS[i : i + 1], MASK[i : i + 1], GROUP_SIZES[i : i + 1], loss_type)
            for i in range(2)
        )
        assert torch.allclose(whole, split), loss_type


def test_empty_rows_and_batches_divide_by_at_least_one():
    empty = torch.zeros_like(MASK)
    assert loss_numerator(PER_TOKEN_LOSS, empty, GROUP_SIZES, "grpo").item() == 0.0
    assert loss_normalizer(GROUP_SIZES, empty.sum(dim=1), "bnpo", MAX_COMPLETION_LENGTH).item() == 1.0


def test_unknown_loss_type_is_refused():
    with pytest.raises(ValueError, match="Unknown loss type"):
        loss_normalizer(GROUP_SIZES, MASK.sum(dim=1), "nonsense", MAX_COMPLETION_LENGTH)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
