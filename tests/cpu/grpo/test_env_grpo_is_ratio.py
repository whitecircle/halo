"""Per-row importance-sampling ratio for env-GRPO (``compute_is_ratio``).

A rollout error drops a trajectory's vLLM logprobs. That row must fall back to ratio 1 *on its own*
without perturbing the other rows: an all-or-nothing gate costs the whole batch its trust region,
silently removing the correction on any step containing one bad episode.
"""

import pytest
import torch

from src.trainers.grpo.objective.logratio import compute_is_ratio, zero_engine_forced_closes

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


def test_sampler_certain_token_is_uncorrected():
    """A token the engine emitted at probability 1 (sampling logprob exactly 0 — a budget-forced
    ``</think>``, a collapsed nucleus) was no sampling choice: ratio exactly 1, outside the corrected
    set, while its neighbours stay corrected."""
    ratio, diff, corrected = _run([[-1.0, -9.0, -2.0]], [[-1.5, 0.0, -2.25]], [[1, 1, 1]], [True])
    assert ratio[0, 1].item() == 1.0 and diff[0, 1].item() == 0.0
    assert not corrected[0, 1]
    assert corrected[0, 0] and corrected[0, 2]
    torch.testing.assert_close(ratio[0, [0, 2]], torch.exp(torch.tensor([0.5, 0.25])))


def test_ratio_is_truncated_at_clip_max():
    """A huge positive log-ratio is clamped so a negative-advantage term can't blow up."""
    ratio, _, _ = _run([[0.0]], [[-20.0]], [[1]], [True])
    assert ratio.item() == pytest.approx(CLIP_MAX)


_END = 7


def test_only_forced_reasoning_closes_lose_their_policy_gradient():
    """A reasoning close the engine forced at the thinking budget (the end token at probability 1) gets ratio 0:
    trained with the episode's advantage it moves the model's own close probability. A naturally certain token
    of any other id, an unforced close, and a row without sampling logprobs all keep their ratio."""
    sampling = torch.tensor([[-0.3, 0.0, 0.0, -0.4], [0.0, 0.0, 0.0, 0.0]])
    ids = torch.tensor([[5, _END, 9, _END], [_END, 5, 5, 5]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    ratio, forced = zero_engine_forced_closes(
        torch.ones(2, 4), sampling, mask, torch.tensor([True, False]), ids, (_END,)
    )
    assert forced.tolist() == [[False, True, False, False], [False, False, False, False]]
    assert ratio.tolist() == [[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]


_OPENER = (70, 71, 72, 73, 74)


def test_a_forced_multi_token_close_loses_its_policy_gradient_as_one_run():
    """gpt-oss's budget appends a five-token opener, every token at probability 1. The whole run gets ratio 0;
    the same ids with one token the model chose (a natural close), a run cut short, and a lone certain id of
    the run keep their ratio."""
    certain, chosen = 0.0, -0.2
    sampling = torch.tensor(
        [
            [-0.5, certain, certain, certain, certain, certain, -0.4, -0.3],
            [-0.5, certain, certain, chosen, certain, certain, -0.4, -0.3],
            [-0.5, -0.1, certain, certain, certain, -0.2, -0.3, -0.1],
            [-0.5, -0.1, -0.3, -0.2, certain, certain, certain, certain],
        ]
    )
    ids = torch.tensor(
        [[1, *_OPENER, 2, 3], [1, *_OPENER, 2, 3], [1, 9, 72, 73, 74, 2, 3, 4], [1, 9, 8, 7, *_OPENER[:4]]]
    )
    mask = torch.ones_like(ids)
    ratio, forced = zero_engine_forced_closes(
        torch.ones(4, 8), sampling, mask, torch.ones(4, dtype=torch.bool), ids, _OPENER
    )
    assert forced.int().tolist() == [[0, 1, 1, 1, 1, 1, 0, 0], [0] * 8, [0] * 8, [0] * 8]
    assert ratio[0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert ratio[1:].eq(1.0).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
