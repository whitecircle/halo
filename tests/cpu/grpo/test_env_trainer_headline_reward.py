#!/usr/bin/env python
"""The headline ``reward`` / ``reward_std`` / ``reward/within_group_std`` are computed over the VALID
episodes — the rows that train.

An infra-errored or env-invalid episode carries a forced failure reward that says nothing about the
policy and is already excluded from the group baseline; averaged into the headline it read a grader
outage as a policy collapse, and its spread against valid siblings as group contrast the advantage
never sees. A step with no valid episode logs none of them rather than a NaN or a zero that would fold
into the logging window.

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
    host._log_headline_rewards(rewards, torch.tensor([True, True, False, False]), 1, "train")
    assert host._metrics["train"]["reward"] == [pytest.approx(0.5)]
    assert host._metrics["train"]["reward_std"] == [pytest.approx(torch.tensor([1.0, 0.0]).std().item())]


def test_a_single_valid_reward_reports_zero_std_not_nan():
    host = _host()
    host._log_headline_rewards(torch.tensor([0.7, -1.0]), torch.tensor([True, False]), 1, "train")
    assert host._metrics["train"]["reward"] == [pytest.approx(0.7)]
    assert host._metrics["train"]["reward_std"] == [0.0]


def test_a_step_with_no_valid_episode_logs_no_headline():
    host = _host()
    host._log_headline_rewards(torch.tensor([-1.0, -1.0]), torch.tensor([False, False]), 2, "train")
    assert not {"reward", "reward_std", "reward/within_group_std"} & set(host._metrics["train"])


def test_within_group_std_reads_the_valid_members_the_advantage_divides_by():
    """Group A ties on its three valid members, so its advantages are zero: the invalid member's forced
    0.0 must not read as contrast. Group B's spread is the sample std of its valid members only."""
    host = _host()
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.5, -1.0])
    valid = torch.tensor([True, True, True, False, True, True, True, False])
    host._log_headline_rewards(rewards, valid, 4, "train")
    group_b = torch.tensor([1.0, 0.0, 0.5]).std().item()
    assert host._metrics["train"]["reward/within_group_std"] == [pytest.approx(group_b / 2)]


def test_within_group_std_of_a_group_with_one_valid_member_is_zero():
    """One valid member carries no contrast, the group-scaled advantage's own convention."""
    host = _host()
    host._log_headline_rewards(
        torch.tensor([1.0, -1.0, 0.0, 1.0]), torch.tensor([True, False, True, True]), 2, "train"
    )
    assert host._metrics["train"]["reward/within_group_std"] == [
        pytest.approx(torch.tensor([0.0, 1.0]).std().item() / 2)
    ]


def test_within_group_std_with_every_episode_valid_is_the_plain_group_std():
    host = _host()
    rewards = torch.tensor([1.0, 0.0, 0.25, 0.25])
    host._log_headline_rewards(rewards, torch.ones(4, dtype=torch.bool), 2, "train")
    expected = rewards.view(-1, 2).std(dim=1).mean().item()
    assert host._metrics["train"]["reward/within_group_std"] == [pytest.approx(expected)]


@pytest.mark.parametrize(
    ("rewards", "num_generations"),
    [([1.0, 0.0, 0.5], 1), ([1.0, 0.0, 0.5], 2)],
    ids=["groups-of-one", "ragged-batch"],
)
def test_no_within_group_std_without_whole_groups_of_several(rewards, num_generations):
    """A group of one has no spread to log (it would read as a 0.0 that folds into the window), and a batch
    that is not whole groups (a ragged eval batch) has no groups to read; the headline still logs."""
    host = _host()
    host._log_headline_rewards(torch.tensor(rewards), torch.ones(3, dtype=torch.bool), num_generations, "train")
    assert "reward/within_group_std" not in host._metrics["train"]
    assert host._metrics["train"]["reward"] == [pytest.approx(0.5)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
