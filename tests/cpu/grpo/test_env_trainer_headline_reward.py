#!/usr/bin/env python
"""The headline ``reward`` / ``reward_std`` are computed over the VALID episodes — the rows that train.

An infra-errored or env-invalid episode carries a forced failure reward that says nothing about the
policy and is already excluded from the group baseline; averaged into the headline it read a grader
outage as a policy collapse. A step with no valid episode logs neither value rather than a NaN or a
zero that would fold into the logging window.

    python tests/cpu/grpo/test_env_trainer_headline_reward.py
"""

from collections import defaultdict

import pytest
import torch

from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer


def _host():
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host._metrics = {"train": defaultdict(list)}
    return host


def test_invalid_episodes_are_left_out_of_the_headline():
    host = _host()
    rewards = torch.tensor([1.0, 0.0, -1.0, -1.0])
    host._log_headline_rewards(rewards, torch.tensor([True, True, False, False]), "train")
    assert host._metrics["train"]["reward"] == [pytest.approx(0.5)]
    assert host._metrics["train"]["reward_std"] == [pytest.approx(torch.tensor([1.0, 0.0]).std().item())]


def test_a_single_valid_reward_reports_zero_std_not_nan():
    host = _host()
    host._log_headline_rewards(torch.tensor([0.7, -1.0]), torch.tensor([True, False]), "train")
    assert host._metrics["train"]["reward"] == [pytest.approx(0.7)]
    assert host._metrics["train"]["reward_std"] == [0.0]


def test_a_step_with_no_valid_episode_logs_no_headline():
    host = _host()
    host._log_headline_rewards(torch.tensor([-1.0, -1.0]), torch.tensor([False, False]), "train")
    assert "reward" not in host._metrics["train"] and "reward_std" not in host._metrics["train"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
