#!/usr/bin/env python
"""``tests.common.utils.training_run_checks`` grades a finished run the way its GPU callers rely on.

The SFT and LoRA mode suites share this helper and run only on the GPU tiers, so a check that stopped
failing there would go unnoticed. These pins drive it with a stand-in trainer state and assert each
verdict flips on the input that should flip it.

Run: python tests/cpu/conventions/test_training_run_checks.py
"""

import math
from types import SimpleNamespace

import pytest

from tests.common.utils import LM_TRAINING_LOSS_BAND, training_run_checks

MAX_STEPS = 3
STEP_LOSSES = [2.5, 2.1, 1.8]
EVAL_ENTRY = {"eval_loss": math.nan, "epoch": 1.0}


def _run(step_losses, *, training_loss=2.1, global_step=MAX_STEPS, extra_entries=(), grad_norms=None):
    history = [{"loss": loss, "step": step} for step, loss in enumerate(step_losses, start=1)]
    for entry, norm in zip(history, grad_norms or (), strict=False):
        entry["grad_norm"] = norm
    trainer = SimpleNamespace(state=SimpleNamespace(log_history=[*history, *extra_entries]))
    return SimpleNamespace(training_loss=training_loss, global_step=global_step), trainer


def test_a_healthy_run_passes_every_check():
    result, trainer = _run(STEP_LOSSES, extra_entries=[EVAL_ENTRY])
    checks = training_run_checks(result, trainer, MAX_STEPS, loss_band=LM_TRAINING_LOSS_BAND)
    assert checks == {"loss_finite": True, "all_steps_finite": True, "steps_completed": True, "loss_reasonable": True}


def test_no_band_means_no_band_check():
    result, trainer = _run(STEP_LOSSES)
    assert "loss_reasonable" not in training_run_checks(result, trainer, MAX_STEPS)


@pytest.mark.parametrize(
    ("run_kwargs", "failed"),
    [
        # A NaN is outside every band too.
        ({"training_loss": math.nan}, ["loss_finite", "loss_reasonable"]),
        ({"step_losses": [2.5, math.inf, 1.8]}, ["all_steps_finite"]),
        ({"global_step": MAX_STEPS - 1}, ["steps_completed"]),
        ({"training_loss": LM_TRAINING_LOSS_BAND[0]}, ["loss_reasonable"]),
        ({"training_loss": LM_TRAINING_LOSS_BAND[1]}, ["loss_reasonable"]),
    ],
)
def test_each_check_fails_on_its_own_breach(run_kwargs, failed):
    kwargs = dict(run_kwargs)
    result, trainer = _run(kwargs.pop("step_losses", STEP_LOSSES), **kwargs)
    checks = training_run_checks(result, trainer, MAX_STEPS, loss_band=LM_TRAINING_LOSS_BAND)
    assert [name for name, ok in checks.items() if not ok] == failed


@pytest.mark.parametrize("bad_norm", [math.inf, math.nan])
def test_the_grad_norm_check_fails_on_a_non_finite_norm(bad_norm):
    result, trainer = _run(STEP_LOSSES, grad_norms=[1.5, bad_norm, 0.9])
    assert training_run_checks(result, trainer, MAX_STEPS, grad_norms=True)["grad_norms_finite"] is False
    assert "grad_norms_finite" not in training_run_checks(result, trainer, MAX_STEPS), "the check is opt-in"


def test_the_grad_norm_check_fails_when_no_norm_was_logged():
    result, trainer = _run(STEP_LOSSES)
    assert training_run_checks(result, trainer, MAX_STEPS, grad_norms=True)["grad_norms_finite"] is False


def test_the_grad_norm_check_passes_on_finite_norms():
    result, trainer = _run(STEP_LOSSES, grad_norms=[1.5, 1.2, 0.9])
    assert training_run_checks(result, trainer, MAX_STEPS, grad_norms=True)["grad_norms_finite"] is True


def test_a_non_finite_eval_entry_is_not_a_step_loss():
    result, trainer = _run(STEP_LOSSES, extra_entries=[{"loss": math.nan, "eval_loss": math.nan}])
    assert training_run_checks(result, trainer, MAX_STEPS)["all_steps_finite"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
