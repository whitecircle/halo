#!/usr/bin/env python
"""``tests.common.sft_modes.sft_metric_checks`` flips each verdict on the log that should flip it.

The dense SFT mode suites grade their logged metrics through this helper and run only on the GPU
tiers. A stand-in trainer state drives it here, one breach per case.

Run: python tests/cpu/conventions/test_sft_metric_checks.py
"""

import math
from types import SimpleNamespace

import pytest

from tests.common.sft_modes import sft_metric_checks

FIRST_LOSS_BAND = (1.0, 8.0)
MAX_GRAD_NORM = 500.0
LOSSES = [4.5, 3.9, 3.1]
GRAD_NORMS = [40.0, 31.0, 25.0]
ACCURACIES = [0.41, 0.47, 0.55]
EVAL_LOSSES = [3.6]


def _trainer(losses=LOSSES, grad_norms=GRAD_NORMS, accuracies=ACCURACIES, eval_losses=EVAL_LOSSES):
    history = [
        {"loss": loss, "grad_norm": norm, "mean_token_accuracy": accuracy}
        for loss, norm, accuracy in zip(losses, grad_norms, accuracies, strict=True)
    ]
    history += [{"eval_loss": loss} for loss in eval_losses]
    return SimpleNamespace(state=SimpleNamespace(log_history=history))


def _checks(trainer, **kwargs):
    return sft_metric_checks(trainer, first_loss_band=FIRST_LOSS_BAND, max_grad_norm=MAX_GRAD_NORM, **kwargs)


def test_a_healthy_run_passes_every_check():
    checks = _checks(_trainer(), token_accuracy_rises=True, eval_loss_logged=True)
    assert checks == {
        "initial_loss_reasonable": True,
        "grad_norms_reasonable": True,
        "token_accuracy_valid": True,
        "token_accuracy_increased": True,
        "eval_loss_finite": True,
    }


@pytest.mark.parametrize(
    ("trainer_kwargs", "failed"),
    [
        ({"losses": [9.5, 3.9, 3.1]}, ["initial_loss_reasonable"]),
        ({"losses": [0.2, 0.1, 0.1]}, ["initial_loss_reasonable"]),
        ({"grad_norms": [40.0, 16000.0, 25.0]}, ["grad_norms_reasonable"]),
        # A NaN norm mid-run must not be dropped by the max.
        ({"grad_norms": [40.0, math.nan, 25.0]}, ["grad_norms_reasonable"]),
        ({"accuracies": [0.41, 1.2, 0.55]}, ["token_accuracy_valid"]),
        ({"accuracies": [0.55, 0.47, 0.41]}, ["token_accuracy_increased"]),
        ({"eval_losses": [math.inf]}, ["eval_loss_finite"]),
        # An eval leg that logged nothing proves nothing.
        ({"eval_losses": []}, ["eval_loss_finite"]),
    ],
)
def test_each_check_fails_on_its_own_breach(trainer_kwargs, failed):
    checks = _checks(_trainer(**trainer_kwargs), token_accuracy_rises=True, eval_loss_logged=True)
    assert [name for name, ok in checks.items() if not ok] == failed


def test_the_trend_and_eval_checks_are_opt_in():
    assert set(_checks(_trainer())) == {"initial_loss_reasonable", "grad_norms_reasonable", "token_accuracy_valid"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
