"""Per-row importance-sampling ratio for env-GRPO (``compute_is_ratio``).

A rollout error drops a trajectory's vLLM logprobs. That row must fall back to ratio 1 *on its own*
without perturbing the other rows: an all-or-nothing gate costs the whole batch its trust region,
silently removing the correction on any step containing one bad episode.
"""

import sys

import pytest
import torch

from src.trainers.grpo.objective.logratio import compute_is_ratio

CLIP_MAX = 3.0


def _run(recompute, sampling, mask, has_row, clip_max=CLIP_MAX):
    return compute_is_ratio(
        torch.tensor(recompute, dtype=torch.float32),
        torch.tensor(sampling, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.long),
        torch.tensor(has_row, dtype=torch.bool),
        clip_max,
    )


def test_corrected_row_matches_exp_of_logratio():
    ratio, diff, corrected = _run([[-1.0, -2.0]], [[-1.5, -2.25]], [[1, 1]], [True])
    torch.testing.assert_close(diff, torch.tensor([[0.5, 0.25]]))
    torch.testing.assert_close(ratio, torch.exp(torch.tensor([[0.5, 0.25]])))
    assert corrected.all()


def test_row_without_sampling_logps_is_exactly_one():
    """The regression this guards: a logprob-less row must be ratio 1, not exp(garbage)."""
    # sampling_logps for the bad row are zeros (the placeholder the trainer fills in).
    ratio, diff, corrected = _run([[-4.0, -5.0]], [[0.0, 0.0]], [[1, 1]], [False])
    assert torch.equal(ratio, torch.ones(1, 2)), ratio
    assert torch.equal(diff, torch.zeros(1, 2))
    assert not corrected.any()


def test_bad_row_does_not_perturb_good_rows():
    """One errored episode must not disable (or alter) the correction for the rest of the batch."""
    recompute = [[-1.0, -2.0], [-4.0, -5.0]]
    sampling = [[-1.5, -2.25], [0.0, 0.0]]  # row 1 = placeholder zeros
    ratio_mixed, _, corrected = _run(recompute, sampling, [[1, 1], [1, 1]], [True, False])

    # The good row is bit-identical to computing it alone.
    ratio_alone, _, _ = _run([recompute[0]], [sampling[0]], [[1, 1]], [True])
    torch.testing.assert_close(ratio_mixed[0], ratio_alone[0])
    # The bad row is neutral.
    assert torch.equal(ratio_mixed[1], torch.ones(2))
    assert corrected[0].all() and not corrected[1].any()


def test_non_policy_tokens_are_neutral():
    """Masked (padding / dummy-row) positions carry ratio 1 regardless of the logps there."""
    ratio, diff, corrected = _run([[-1.0, -9.0]], [[-1.5, 0.0]], [[1, 0]], [True])
    assert ratio[0, 1].item() == pytest.approx(1.0)
    assert diff[0, 1].item() == 0.0
    assert corrected[0, 0] and not corrected[0, 1]


def test_ratio_is_truncated_at_clip_max():
    """A huge positive log-ratio is clamped so a negative-advantage term can't blow up."""
    ratio, _, _ = _run([[0.0]], [[-20.0]], [[1]], [True])
    assert ratio.item() == pytest.approx(CLIP_MAX)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
