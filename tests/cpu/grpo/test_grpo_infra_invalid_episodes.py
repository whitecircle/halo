#!/usr/bin/env python
"""Grading-infra-outage episodes must not poison the GRPO group baseline. Such an episode completes without a
``RolloutResult.error``, so the trainer would read its forced 0 grade as a genuine wrong answer."""

from types import SimpleNamespace

import pytest
import torch

from halo.environments.episode import RolloutResult
from halo.environments.episode_keys import EPISODE_INVALID_KEY, EPISODE_INVALID_REASON_KEY, REWARD_COMPONENTS_KEY
from halo.environments.messages import Trajectory
from halo.environments.tasks.coding.code_contests import CodeContestsEnvironment
from halo.rewards.terms import OBJECTIVE_REWARD_KEY
from halo.trainers.grpo.environmental.step_batch import EMPTY_ROLLOUT_STEP_LIMIT, rollout_valid_mask
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.objective.advantages import degenerate_group_mask, group_relative_advantages

_DEVICE = torch.device("cpu")


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
    env._settle_grade(traj, None)
    assert traj.episode_invalid is True
    assert traj.info[EPISODE_INVALID_KEY] is True
    components = traj.info[REWARD_COMPONENTS_KEY]
    assert components[OBJECTIVE_REWARD_KEY] == 0.0  # the forced failure grade, no rung on top
    assert sum(components.values()) == pytest.approx(0.0)
    assert traj.total_reward == pytest.approx(0.0)


def test_genuine_failure_stays_valid():
    env = CodeContestsEnvironment(language="python")
    # Three judged failures with zero passes is real policy signal, not an outage.
    traj = _graded_traj(
        env,
        submission_result="graded",
        tests_total=3,
        tests_passed=0,
        tests_ran_ok=3,
        tests_infra_errors=0,
    )
    env._settle_grade(traj, None)
    assert traj.episode_invalid is False


def test_never_submitted_stays_valid():
    env = CodeContestsEnvironment(language="python")
    traj = _graded_traj(env)
    env._settle_grade(traj, None)
    assert traj.episode_invalid is False


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
    outage_free = rewards[:3] - rewards[:3].mean()
    assert torch.allclose(adv[:3], outage_free)

    # Without the mask, the outage placeholder's 0.0 drags the baseline down for every sibling.
    poisoned = group_relative_advantages(rewards, 4, "none")
    assert not torch.allclose(adv[:3], poisoned[:3])


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


def _all_invalid_results(error: str | None):
    if error is not None:
        return [RolloutResult(prompt="p", error=error) for _ in range(4)]
    invalid = Trajectory(info={EPISODE_INVALID_KEY: True})
    return [RolloutResult(prompt="p", trajectory=invalid) for _ in range(4)]


def _guard_stub():
    return SimpleNamespace(_empty_rollout_steps=0)


def _check(stub, results):
    DistributedAsyncEnvironmentalGRPOTrainer._track_empty_rollout_step(
        stub, results, rollout_valid_mask(results, _DEVICE)
    )


def test_consecutive_empty_rollout_steps_halt_the_run_and_name_the_rollout_error():
    """An all-masked batch is a zero gradient logged as a plausible ``loss=0, reward=0``. The limit-th consecutive step
    stops the run and carries the engine's own message, which is where the remedy is."""
    stub = _guard_stub()
    results = _all_invalid_results("vLLM error (status 400): thinking_token_budget is not supported")
    for _ in range(EMPTY_ROLLOUT_STEP_LIMIT - 1):
        _check(stub, results)
    with pytest.raises(RuntimeError) as excinfo:
        _check(stub, results)
    assert "thinking_token_budget" in str(excinfo.value)


def test_env_marked_invalid_episodes_halt_too_and_say_so():
    """A grading outage carries no ``RolloutResult.error``, so the message must not promise an engine error."""
    stub = _guard_stub()
    results = _all_invalid_results(None)
    for _ in range(EMPTY_ROLLOUT_STEP_LIMIT - 1):
        _check(stub, results)
    with pytest.raises(RuntimeError, match="environment marked every episode invalid"):
        _check(stub, results)


def test_untrainable_episodes_halt_names_the_render_failure():
    """An episode the template could not re-render carries no ``RolloutResult.error`` either."""
    stub = _guard_stub()
    reason = "cannot locate the trained turn spans of tok inside its own chat-template render"
    results = [
        RolloutResult(
            prompt="p", trajectory=Trajectory(info={EPISODE_INVALID_KEY: True, EPISODE_INVALID_REASON_KEY: reason})
        )
        for _ in range(4)
    ]
    for _ in range(EMPTY_ROLLOUT_STEP_LIMIT - 1):
        _check(stub, results)
    with pytest.raises(RuntimeError, match="dropped as untrainable: cannot locate the trained turn spans"):
        _check(stub, results)


def test_one_survivor_resets_the_streak():
    stub = _guard_stub()
    empty = _all_invalid_results("boom")
    for _ in range(EMPTY_ROLLOUT_STEP_LIMIT - 1):
        _check(stub, empty)
    _check(stub, [RolloutResult(prompt="p", total_reward=1.0), *empty])
    for _ in range(EMPTY_ROLLOUT_STEP_LIMIT - 1):
        _check(stub, empty)  # streak restarted, so this must not raise


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
