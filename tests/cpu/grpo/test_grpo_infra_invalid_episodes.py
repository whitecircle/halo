#!/usr/bin/env python
"""Grading-infra-outage episodes must not poison the GRPO group baseline.

A ``CodeContestsEnvironment`` episode whose submission grade lost every test to backend errors
completes WITHOUT a ``RolloutResult.error``, so nothing stops the trainer reading its forced failure
reward as a genuine wrong answer — biasing every sibling's advantage. The protocol-level contract:

- the environment marks the trajectory via ``EPISODE_INVALID_KEY`` (``Trajectory.episode_invalid``);
- ``rollout_valid_mask`` excludes it from the baseline exactly like ``RolloutResult.error``;
- ``degenerate_group_mask`` computes degeneracy over VALID members only, so an invalid member's
  differing placeholder reward cannot let an all-alike group escape degenerate detection
  (a group with < 2 valid members is degenerate by definition).

    python tests/cpu/grpo/test_grpo_infra_invalid_episodes.py
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState

from src.environments.base import EPISODE_INVALID_KEY, Trajectory
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.episode import RolloutResult
from src.trainers.grpo.environmental import (
    EMPTY_ROLLOUT_STEP_LIMIT,
    DistributedAsyncEnvironmentalGRPOTrainer,
    rollout_valid_mask,
)
from src.trainers.grpo.objective.advantages import degenerate_group_mask, group_relative_advantages
from src.trainers.grpo.objective.application import degenerate_drop_rows

PartialState()  # the trainer warns through accelerate's logger, which refuses to log without it

_DEVICE = torch.device("cpu")


# --- Environment side: code_contests marks the outage on the trajectory ---


def _graded_traj(env, **info):
    traj = env._reset_single("Print a+b.", {"answer": {"tests": [{"input": "1\n2\n", "output": "3"}]}})
    traj.info.update(info)
    return traj


def test_outage_marks_episode_invalid():
    env = CodeContestsEnvironment(language="python")
    traj = _graded_traj(
        env,
        submission_result="graded",
        tests_total=3,
        tests_passed=0,
        tests_ran_ok=0,
        tests_infra_errors=3,
    )
    reward = env._compute_reward(traj)
    assert traj.episode_invalid is True
    assert traj.info[EPISODE_INVALID_KEY] is True
    assert reward == pytest.approx(env.failure_reward + env._shaped_base_reward(traj))


def test_genuine_failure_stays_valid():
    env = CodeContestsEnvironment(language="python")
    # tests_ran_ok=3 with zero passes is real policy signal, not an outage.
    traj = _graded_traj(
        env,
        submission_result="graded",
        tests_total=3,
        tests_passed=0,
        tests_ran_ok=3,
        tests_infra_errors=0,
    )
    env._compute_reward(traj)
    assert traj.episode_invalid is False


def test_never_submitted_stays_valid():
    env = CodeContestsEnvironment(language="python")
    traj = _graded_traj(env)  # no submission_result at all
    env._compute_reward(traj)
    assert traj.episode_invalid is False


# --- Trainer side: rollout_valid_mask consumes the flag like RolloutResult.error ---


def _result(error=None, invalid=False):
    traj = Trajectory(done=True)
    if invalid:
        traj.info[EPISODE_INVALID_KEY] = True
    return RolloutResult(prompt="p", trajectory=traj, error=error)


def test_rollout_valid_mask_excludes_error_and_invalid():
    results = [_result(), _result(error="boom"), _result(invalid=True), RolloutResult(prompt="p")]
    mask = rollout_valid_mask(results, _DEVICE)
    assert mask.tolist() == [True, False, False, True]


def test_outage_member_excluded_from_group_baseline():
    rewards = torch.tensor([1.0, 0.5, 1.5, 0.0])
    results = [_result(), _result(), _result(), _result(invalid=True)]
    mask = rollout_valid_mask(results, _DEVICE)

    adv = group_relative_advantages(rewards, 4, "none", valid_mask=mask)
    outage_free = rewards[:3] - rewards[:3].mean()  # baseline over valid members only
    assert torch.allclose(adv[:3], outage_free)

    # Without the mask, the outage placeholder's 0.0 drags the baseline down for every sibling.
    poisoned = group_relative_advantages(rewards, 4, "none")
    assert not torch.allclose(adv[:3], poisoned[:3])


# --- Degenerate-group detection over valid members only (F3-D) ---


def test_degenerate_detection_ignores_invalid_member_spread():
    # The invalid member's 0.0 is the group's only spread, so an unmasked check misses the degeneracy.
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0])
    valid = torch.tensor([True, True, True, False])
    assert degenerate_group_mask(rewards, 4).tolist() == [False] * 4
    assert degenerate_group_mask(rewards, 4, valid_mask=valid).tolist() == [True] * 4


def test_group_with_fewer_than_two_valid_members_is_degenerate():
    rewards = torch.tensor([1.0, 0.0])
    valid = torch.tensor([True, False])
    assert degenerate_group_mask(rewards, 2, valid_mask=valid).tolist() == [True, True]
    all_invalid = torch.tensor([False, False])
    assert degenerate_group_mask(rewards, 2, valid_mask=all_invalid).tolist() == [True, True]


def test_valid_spread_keeps_group_alive():
    rewards = torch.tensor([1.0, 0.5, 1.0, 0.0])
    valid = torch.tensor([True, True, True, False])
    assert degenerate_group_mask(rewards, 4, valid_mask=valid).tolist() == [False] * 4


def test_no_mask_behavior_unchanged():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.5])
    assert degenerate_group_mask(rewards, 2).tolist() == [True, True, False, False]


def test_degenerate_drop_rows_threads_valid_mask():
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0])
    valid = torch.tensor([True, True, True, False])
    drop, frac = degenerate_drop_rows(rewards, 4, valid_mask=valid)
    assert drop.tolist() == [True] * 4
    assert frac == pytest.approx(1.0)


# --- A step in which NOTHING survived: zero gradient, so it must not pass silently ---


def _all_invalid_results(error: str | None):
    """Four rollouts that all carry ``error`` (or are all env-marked invalid when it is None)."""
    if error is not None:
        return [RolloutResult(prompt="p", error=error) for _ in range(4)]
    invalid = Trajectory(info={EPISODE_INVALID_KEY: True})
    return [RolloutResult(prompt="p", trajectory=invalid) for _ in range(4)]


def _guard_stub():
    """What the guard reads off the trainer: the streak counter ``__init__`` declares."""
    return SimpleNamespace(_empty_rollout_steps=0)


def _check(stub, results):
    DistributedAsyncEnvironmentalGRPOTrainer._check_step_has_valid_episodes(
        stub, results, rollout_valid_mask(results, _DEVICE)
    )


def test_consecutive_empty_rollout_steps_halt_the_run_and_name_the_rollout_error():
    """Every episode failing means an all-masked batch — a zero gradient logged as a plausible
    ``loss=0, reward=0``. One step is a blip; the limit-th consecutive one has to stop the run, and
    carry the engine's own message, which is where the remedy is."""
    stub = _guard_stub()
    results = _all_invalid_results("vLLM error (status 400): thinking_token_budget is not supported")
    for _ in range(EMPTY_ROLLOUT_STEP_LIMIT - 1):
        _check(stub, results)  # under the limit: warn, keep training
    with pytest.raises(RuntimeError) as excinfo:
        _check(stub, results)
    assert "thinking_token_budget" in str(excinfo.value)


def test_env_marked_invalid_episodes_halt_too_and_say_so():
    """The other way a step empties out carries no ``RolloutResult.error`` at all (a grading
    outage), so the message must not promise an engine error that does not exist."""
    stub = _guard_stub()
    results = _all_invalid_results(None)
    for _ in range(EMPTY_ROLLOUT_STEP_LIMIT - 1):
        _check(stub, results)
    with pytest.raises(RuntimeError, match="environment marked every episode invalid"):
        _check(stub, results)


def test_one_survivor_resets_the_streak():
    """The guard fires on CONSECUTIVE empty steps: a single valid episode means the step trained on
    something, so a later isolated empty step must not inherit the earlier count."""
    stub = _guard_stub()
    empty = _all_invalid_results("boom")
    for _ in range(EMPTY_ROLLOUT_STEP_LIMIT - 1):
        _check(stub, empty)
    _check(stub, [RolloutResult(prompt="p", total_reward=1.0), *empty])
    for _ in range(EMPTY_ROLLOUT_STEP_LIMIT - 1):
        _check(stub, empty)  # streak restarted, so this must not raise


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
