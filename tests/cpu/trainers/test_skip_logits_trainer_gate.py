#!/usr/bin/env python
"""The SFT eval fast-path must not inject ``skip_logits`` when the loss reads the logits.

``DistributedSFTTrainer.prediction_step`` adds ``skip_logits=True`` to route loss-only eval through
Liger's fused linear cross-entropy. Subclasses that build their own loss forward ``inputs`` onward
minus ``labels`` — the injected key rides along, and Liger's ``lce_forward`` raises
``ValueError("skip_logits is True, but labels and shift_labels are None")``, killing every eval
step. Reachable whenever fused-linear-CE is on (the per-model default for Zaya / DeepSeek-V4 /
GLM-4-MoE-Lite) and ``compute_metrics is None``, which is when HF sets ``prediction_loss_only``.

The gate is the class-declared ``_loss_reads_logits`` contract.
"""

import pytest

from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from src.trainers.sft import DistributedSFTTrainer


class _Args:
    use_liger_kernel = True


class _Gate:
    """Exercises the real predicate without constructing a Trainer."""

    def __init__(self, *, loss_reads_logits, is_cp_mode=False, liger=True, forward_has_skip_logits=True):
        self._loss_reads_logits = loss_reads_logits
        self.is_cp_mode = is_cp_mode
        self.args = _Args()
        self.args.use_liger_kernel = liger
        self._forward_accepts_skip_logits = forward_has_skip_logits

    _should_skip_eval_logits = DistributedSFTTrainer._should_skip_eval_logits


INPUTS = {"input_ids": [[1, 2, 3]], "labels": [[1, 2, 3]]}


def test_plain_sft_takes_the_fused_ce_fast_path():
    """The optimization must still apply where it is safe — otherwise the gate is a silent regression."""
    assert _Gate(loss_reads_logits=False)._should_skip_eval_logits(True, INPUTS) is True


def test_logit_reading_trainer_is_excluded():
    """A trainer whose compute_loss consumes outputs.logits must never receive skip_logits."""
    assert _Gate(loss_reads_logits=True)._should_skip_eval_logits(True, INPUTS) is False


@pytest.mark.parametrize(
    "kwargs,inputs,prediction_loss_only",
    [
        ({"is_cp_mode": True}, INPUTS, True),  # CP metrics read the logits
        ({"liger": False}, INPUTS, True),  # nothing to route through
        ({"forward_has_skip_logits": False}, INPUTS, True),  # forward has no such parameter
        ({}, {"input_ids": [[1]]}, True),  # no labels -> lce_forward would raise
        ({}, {**INPUTS, "skip_logits": False}, True),  # caller already decided
        ({}, INPUTS, False),  # the eval loop wants the logits back
    ],
)
def test_existing_exclusions_still_hold(kwargs, inputs, prediction_loss_only):
    """The pre-existing guards must survive the added one."""
    gate = _Gate(loss_reads_logits=False, **kwargs)
    assert gate._should_skip_eval_logits(prediction_loss_only, inputs) is False


def test_trainers_declare_their_logit_contract():
    """SFT reads no logits; the self-distillation subclass does and must say so."""
    assert DistributedSFTTrainer._loss_reads_logits is False
    assert DistributedSelfDistillationTrainer._loss_reads_logits is True


def test_self_distillation_actually_inherits_the_gated_prediction_step():
    """The declaration only helps if the subclass goes through SFT's prediction_step."""
    assert DistributedSelfDistillationTrainer.prediction_step is DistributedSFTTrainer.prediction_step


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
