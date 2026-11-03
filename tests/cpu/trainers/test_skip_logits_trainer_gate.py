#!/usr/bin/env python
"""The SFT eval fast-path must not inject ``skip_logits`` when the loss reads the logits.
In a subclass building its own loss without ``labels``, Liger's ``lce_forward`` then raises on every eval step."""

import types
from unittest import mock

import pytest

from halo.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.sft import DistributedSFTTrainer


class _Gate(DistributedSFTTrainer):
    """The real ``prediction_step`` without constructing a Trainer; the spine's step below it records inputs."""

    def __init__(self, *, loss_reads_logits, is_cp_mode=False, liger=True, forward_has_skip_logits=True):
        self.LOSS_READS_LOGITS = loss_reads_logits
        self.parallelism_config = types.SimpleNamespace(is_cp_mode=is_cp_mode)
        self.args = types.SimpleNamespace(use_liger_kernel=liger)
        self.__dict__["_forward_accepts_skip_logits"] = forward_has_skip_logits

    def _injects_skip_logits(self, prediction_loss_only, inputs):
        seen = {}
        with mock.patch.object(
            DistributedTrainerMixin,
            "prediction_step",
            lambda _self, _m, step_inputs, *a, **k: seen.update(step_inputs),
        ):
            self.prediction_step(None, inputs, prediction_loss_only)
        return seen.get("skip_logits") is True


INPUTS = {"input_ids": [[1, 2, 3]], "labels": [[1, 2, 3]]}


def test_plain_sft_takes_the_fused_ce_fast_path():
    assert _Gate(loss_reads_logits=False)._injects_skip_logits(True, INPUTS) is True


def test_logit_reading_trainer_is_excluded():
    assert _Gate(loss_reads_logits=True)._injects_skip_logits(True, INPUTS) is False


@pytest.mark.parametrize(
    "kwargs,inputs,prediction_loss_only",
    [
        ({"is_cp_mode": True}, INPUTS, True),  # CP metrics read the logits
        ({"liger": False}, INPUTS, True),  # nothing to route through
        ({"forward_has_skip_logits": False}, INPUTS, True),
        ({}, {"input_ids": [[1]]}, True),  # no labels -> lce_forward would raise
        ({}, {**INPUTS, "skip_logits": False}, True),  # caller already decided
        ({}, INPUTS, False),  # the eval loop wants the logits back
    ],
)
def test_existing_exclusions_still_hold(kwargs, inputs, prediction_loss_only):
    gate = _Gate(loss_reads_logits=False, **kwargs)
    assert gate._injects_skip_logits(prediction_loss_only, inputs) is False


def test_trainers_declare_their_logit_contract():
    """SFT reads no logits; the self-distillation subclass does and must say so."""
    assert DistributedSFTTrainer.LOSS_READS_LOGITS is False
    assert DistributedSelfDistillationTrainer.LOSS_READS_LOGITS is True


def test_self_distillation_actually_inherits_the_gated_prediction_step():
    """The declaration only helps if the subclass goes through SFT's prediction_step."""
    assert DistributedSelfDistillationTrainer.prediction_step is DistributedSFTTrainer.prediction_step


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
