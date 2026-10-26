#!/usr/bin/env python
"""SMPO's margin losses: one per configurable ``loss_type``, each its own formula.

python tests/cpu/trainers/test_smpo_objective.py
"""

import typing

import pytest
import torch
import torch.nn.functional as F

from src.configs.preference import SmoothMarginPOConfig
from src.trainers.preference.smpo_objective import MARGIN_LOSSES, SMPOObjective

BETA = 1.5
TARGET_MARGIN = 0.3
CHOSEN_LOGPS = torch.tensor([-1.0, -2.5, -0.2, -3.0])
REJECTED_LOGPS = torch.tensor([-2.0, -1.0, -0.4, -3.1])
LOGITS = CHOSEN_LOGPS - REJECTED_LOGPS - TARGET_MARGIN
DEFINITIONS = {
    "sigmoid": -F.logsigmoid(BETA * LOGITS),
    "hinge": torch.relu(-BETA * LOGITS),
    "ipo": (BETA * LOGITS).pow(2),
    "smooth_lower_bound": torch.relu(-BETA * LOGITS).pow(2),
}


def test_every_configurable_loss_type_has_exactly_one_margin_loss():
    assert set(MARGIN_LOSSES) == set(typing.get_args(typing.get_type_hints(SmoothMarginPOConfig)["loss_type"]))


@pytest.mark.parametrize("loss_type", sorted(DEFINITIONS))
def test_each_margin_loss_is_its_definition(loss_type):
    objective = SMPOObjective(
        beta=BETA,
        loss_type=loss_type,
        chosen_sft_ratio=0.5,
        lower_clip_percentile=None,
        upper_clip_percentile=None,
        min_log_prob=None,
    )
    losses, chosen_rewards, rejected_rewards = objective.margin_loss(CHOSEN_LOGPS, REJECTED_LOGPS, TARGET_MARGIN)

    torch.testing.assert_close(losses, DEFINITIONS[loss_type], rtol=0, atol=0)
    torch.testing.assert_close(chosen_rewards, BETA * CHOSEN_LOGPS)
    torch.testing.assert_close(rejected_rewards, BETA * REJECTED_LOGPS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
