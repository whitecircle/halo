#!/usr/bin/env python
"""``tests.common.first_step`` grades a recorded first step the way the SMPO CP GPU rows rely on.

Those rows run only on the GPU tiers, so a check that stopped failing there would go unnoticed. These
pins drive :func:`first_step_checks` and :func:`first_step_gradient_checks` with each miscount they exist
to catch and assert that exactly its own verdict flips, and drive :func:`recording_autograd_fallbacks`
with a real backward through an in-place gloo all-reduce.

Run: python tests/cpu/conventions/test_first_step_checks.py
"""

import warnings

import pytest
import torch
import torch.distributed as dist

from tests.common.first_step import (
    AUTOGRAD_FALLBACK_WARNING,
    FirstStep,
    first_step_checks,
    first_step_gradient_checks,
    recording_autograd_fallbacks,
)
from tests.common.gloo import run_gloo_ranks

CP_SIZE = 2
LOSS_RTOL = 0.1
LOSSES = [1.7, 2.3]


def _grads() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {name: torch.randn(4, 3, generator=generator) for name in ("embed.weight", "q_proj.weight", "norm.weight")}


def _step(losses=LOSSES, grads=None, logged_loss=None, autograd_fallbacks=()) -> FirstStep:
    return FirstStep(
        losses=list(losses),
        grads=_grads() if grads is None else grads,
        logged_loss=sum(losses) / len(losses) if logged_loss is None else logged_loss,
        autograd_fallbacks=list(autograd_fallbacks),
    )


def _failed(record: FirstStep, reference: FirstStep | None = None) -> list[str]:
    reference = reference or _step()
    checks, _ = first_step_checks(record, reference, miscount_factor=CP_SIZE, loss_rtol=LOSS_RTOL)
    grad_checks, _ = first_step_gradient_checks(record, reference)
    return [name for name, ok in (checks | grad_checks).items() if not ok]


def test_a_matching_step_passes_every_check():
    assert _failed(_step()) == []


def test_losses_within_the_relative_bound_pass():
    assert _failed(_step(losses=[(1 + LOSS_RTOL / 2) * loss for loss in LOSSES])) == []


def test_a_loss_counted_once_per_cp_rank_fails_both_loss_checks():
    assert _failed(_step(losses=[CP_SIZE * loss for loss in LOSSES])) == [
        "step1_losses_match_reference",
        "logged_step1_loss_is_the_reference_loss",
    ]


def test_a_logged_loss_off_the_trained_one_fails_only_the_logged_check():
    true_loss = sum(LOSSES) / len(LOSSES)
    assert _failed(_step(logged_loss=CP_SIZE * true_loss)) == ["logged_step1_loss_is_the_reference_loss"]


@pytest.mark.parametrize("scale", [CP_SIZE, 1 / CP_SIZE])
def test_a_gradient_scaled_by_the_cp_size_fails_only_the_norm_check(scale):
    grads = {name: scale * grad for name, grad in _grads().items()}
    assert _failed(_step(grads=grads)) == ["step1_grad_norm_matches_reference"]


def test_a_reoriented_gradient_fails_only_the_direction_check():
    grads = _grads()
    grads["q_proj.weight"] = grads["q_proj.weight"].flip(0)
    assert _failed(_step(grads=grads)) == ["step1_grad_direction_matches_reference"]


def test_a_parameter_left_out_of_the_step_fails_every_gradient_check():
    grads = _grads()
    del grads["norm.weight"]
    assert _failed(_step(grads=grads)) == [
        "step1_grads_cover_the_reference_params",
        "step1_grad_direction_matches_reference",
        "step1_grad_norm_matches_reference",
    ]


def test_a_backward_through_the_c10d_fallback_fails_its_check():
    assert _failed(_step(autograd_fallbacks=[AUTOGRAD_FALLBACK_WARNING])) == [
        "backward_never_took_the_c10d_autograd_fallback"
    ]


def test_a_loss_too_small_to_show_a_miscount_fails_the_resolution_precondition():
    tiny = [1e-3, 2e-3]
    assert _failed(_step(losses=tiny), _step(losses=tiny)) == ["loss_bound_resolves_a_miscounted_loss"]


def _backward_through_an_in_place_all_reduce(rank: int) -> None:
    del rank
    fallbacks: list[str] = []
    # The recorder inside, so a warning it passes on reaches the outer log.
    with warnings.catch_warnings(record=True) as shown, recording_autograd_fallbacks(fallbacks):
        warnings.simplefilter("always")
        partial = torch.ones(3, requires_grad=True) * 2.0
        dist.all_reduce(partial)
        partial.sum().backward()
        warnings.warn("an unrelated warning", UserWarning, stacklevel=1)
    assert fallbacks, "the in-place all-reduce's backward raised no autograd-fallback warning"
    assert all(AUTOGRAD_FALLBACK_WARNING in message for message in fallbacks)
    assert [str(w.message) for w in shown] == ["an unrelated warning"], "other warnings must still show"


def test_the_recorder_catches_a_backward_through_an_in_place_collective():
    run_gloo_ranks(_backward_through_an_in_place_all_reduce, 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
