"""The env-GRPO e2e's own task and verdicts, which only a GPU row with a live server would otherwise exercise."""

import math
import pickle
import types

import pytest
from ray import cloudpickle

from halo.environments.registry import build_environment
from tests.common.env_grpo_e2e import (
    DRAW_PROMPT,
    DRAW_RANGE,
    DRAW_TOOL,
    MAX_TURNS,
    DrawNumberEnvironment,
    record_first_step_gradient,
    record_routing_replay,
    routing_flip_bar,
)

_BAR_CHECK = "routing_replay_flip_rate_within_the_measured_bar"
# Probe ends of the shape the CPU forwards of the real checkpoints measured (Qwen3-30B-A3B floor 0.084, gpt-oss-20b
# 0.020; a one-token shift 0.94-0.995).
_PROBE = {"floor": 0.04, "misplaced": 0.95, "stale": 0.6}


def _environment() -> DrawNumberEnvironment:
    """The environment as a Ray rollout actor builds it: from the pickled ``(class, kwargs)`` spec."""
    blob = cloudpickle.dumps((DrawNumberEnvironment, {"max_turns": MAX_TURNS}))
    assert b"tests.common.env_grpo_e2e" in blob, "the class must pickle by reference, as the actors import it"
    return build_environment(pickle.loads(blob), {})


def _episode(env: DrawNumberEnvironment, answer: str | None) -> tuple[float, str]:
    """One episode: the draw call, then ``answer`` with ``{drawn}`` standing for the draw (``None``: no answer
    turn). Returns the episode's grade and its draw."""
    ids, _ = env.reset([DRAW_PROMPT], [None])
    call = [{"id": "call_1", "function": {"name": DRAW_TOOL, "arguments": "{}"}}]
    env.step(ids, [""], [{"tool_calls": call}])
    trajectory = env.get_trajectories(ids)[0]
    drawn = next(m.content for m in trajectory.messages if m.role == "tool")
    if answer is not None:
        env.step(ids, [answer.format(drawn=drawn)], [{"finish_reason": "stop"}])
    trajectory = env.get_trajectories(ids)[0]
    return env._grade_episode(trajectory).objective, drawn


def test_two_episodes_draw_apart_and_each_is_graded_on_its_own_draw():
    env = _environment()
    assert env.registry.names() == [DRAW_TOOL]
    (first, first_draw), (second, second_draw) = _episode(env, "{drawn}"), _episode(env, "The number is {drawn}.")
    assert first_draw != second_draw
    assert first == int(first_draw) / DRAW_RANGE and second == int(second_draw) / DRAW_RANGE
    assert first != second


@pytest.mark.parametrize(
    "answer",
    ["The number is 7.", "{drawn}0", None],
    ids=["another-number", "the-draw-inside-a-longer-number", "no-answer"],
)
def test_an_episode_that_does_not_report_its_draw_scores_zero(answer):
    grade, _ = _episode(_environment(), answer)
    assert grade == 0.0


def _routing(flips: list[float], acquired: list[bool], probe: dict) -> dict[str, bool]:
    history = [
        {"step": step, "routing/replay_flip_rate": flip, "routing/rollout_unresolved_frac": 0.0}
        for step, flip in enumerate(flips, start=1)
    ]
    trainer = types.SimpleNamespace(state=types.SimpleNamespace(log_history=history))
    checks: dict[str, bool] = {}
    record_routing_replay(trainer, checks, {}, probe=probe, acquired=acquired)
    return checks


def test_the_bar_sits_halfway_between_its_ends_in_log_space():
    assert math.isclose(routing_flip_bar(0.04, 0.9), 0.06 * math.sqrt(10))


def test_steps_on_their_own_weights_rollouts_pass_and_popped_rounds_are_left_out():
    # vLLM's measured step-1 flips on identical weights, then SGLang's prefetched steps after an update.
    assert all(_routing([0.0755, 0.0748], [], _PROBE).values())
    assert all(_routing([0.026, 0.80, 0.39], [False, True, True], _PROBE).values())


@pytest.mark.parametrize(
    ("flips", "acquired", "probe", "failed"),
    [
        ([0.026, 0.80, 0.39], [], _PROBE, _BAR_CHECK),
        ([0.96, 0.95], [], _PROBE, _BAR_CHECK),
        ([], [], _PROBE, _BAR_CHECK),
        ([0.07], [], {**_PROBE, "floor": 0.3}, "routing_probe_separates_noise_from_a_misplaced_mask"),
        ([0.0], [], {**_PROBE, "floor": 0.0}, "routing_probe_floor_flips_a_selection"),
    ],
    ids=["a-stale-step-read-as-fresh", "a-shifted-mask", "no-step-1-flip", "ends-too-close", "a-zero-floor"],
)
def test_a_routing_mismatch_or_a_degenerate_probe_fails(flips, acquired, probe, failed):
    assert not _routing(flips, acquired, probe)[failed]


@pytest.mark.parametrize(
    ("entry", "passes"),
    [
        ({"grad_norm": 23.67, "reward/within_group_std": 0.354}, True),
        ({"grad_norm": 0.0, "reward/within_group_std": 0.0}, False),
        ({"grad_norm": 1.0, "reward/within_group_std": 0.0}, False),
        ({"grad_norm": 1.0}, False),
        ({"grad_norm": math.nan, "reward/within_group_std": 0.3}, False),
    ],
    ids=["a-spread-step", "tied-groups", "tied-groups-with-a-norm", "no-spread-logged", "a-non-finite-norm"],
)
def test_step_one_must_carry_a_gradient(entry, passes):
    trainer = types.SimpleNamespace(state=types.SimpleNamespace(log_history=[{"loss": 0.0}, entry]))
    checks: dict[str, bool] = {}
    record_first_step_gradient(trainer, checks, {})
    assert all(checks.values()) is passes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
