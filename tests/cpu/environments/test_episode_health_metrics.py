#!/usr/bin/env python
"""CPU tests for the two episode-health metrics a natural-termination rate hides.

A turn that stops inside its reasoning ends the episode as an empty final answer, and reasoning that
drifts into another script reads as a normal turn; both are only visible in transcripts unless the
environment counts them.

Run: python tests/cpu/environments/test_episode_health_metrics.py  (or pytest)
"""

import pytest

from src.environments.base import EMPTY_FINAL_ANSWER_KEY, Message, Trajectory
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter


def _make_env(**kwargs):
    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="echo",
            description="echo",
            parameters=[ToolParameter("text", "string", "text")],
            handler=lambda text: text,
        )
    )
    kwargs.setdefault("max_turns", 4)
    return NativeToolUseEnvironment(tool_registry=registry, **kwargs)


def _episode(text: str, ctx: dict) -> tuple[NativeToolUseEnvironment, Trajectory]:
    env = _make_env()
    ids, _ = env.reset(["task"], [{}])
    env.step(ids, [text], [ctx])
    return env, env.get_trajectories(ids)[0]


def test_a_reasoning_only_final_turn_counts_as_an_empty_answer():
    env, traj = _episode("", {"finish_reason": "stop", "reasoning": "a thought that never reached an answer"})
    assert traj.done and traj.info[EMPTY_FINAL_ANSWER_KEY] is True
    assert env.rollout_metrics(traj)["episode/empty_answer_rate"] == 1.0


def test_a_text_answer_is_not_an_empty_answer():
    env, traj = _episode("the answer is 4", {"finish_reason": "stop"})
    assert traj.done and traj.info[EMPTY_FINAL_ANSWER_KEY] is False
    assert env.rollout_metrics(traj)["episode/empty_answer_rate"] == 0.0


def test_a_cut_turn_is_not_an_empty_answer():
    # The recovery path never finalizes a text answer, so the flag stays unset and the rate 0.
    env, traj = _episode("", {"finish_reason": "length", "reasoning": "half a thought"})
    assert not traj.done and EMPTY_FINAL_ANSWER_KEY not in traj.info
    assert env.rollout_metrics(traj)["episode/empty_answer_rate"] == 0.0


def _metrics_for(messages: list[Message]) -> dict[str, float]:
    return _make_env().rollout_metrics(Trajectory(messages=messages))


@pytest.mark.parametrize("drift", ["the 最长 path is bounded", "但实际上 the prefix sums", "カタカナ", "한글"])
def test_reasoning_in_another_script_is_counted(drift):
    metrics = _metrics_for([Message.user("task"), Message.assistant("answer", thinking=drift)])
    assert metrics["episode/reasoning_cjk_rate"] == 1.0


def test_latin_reasoning_and_non_reasoning_channels_are_not_counted():
    latin = _metrics_for(
        [Message.user("task"), Message.assistant("answer", thinking="plain reasoning, with symbols: ≤ ∑ π")]
    )
    assert latin["episode/reasoning_cjk_rate"] == 0.0
    # The metric watches the reasoning channel: a task statement or a tool result in another script is not drift.
    elsewhere = _metrics_for(
        [
            Message.user("任务"),
            Message.assistant("答案", thinking="reasoning in one script"),
            Message.tool("输出", "c0", "echo"),
        ]
    )
    assert elsewhere["episode/reasoning_cjk_rate"] == 0.0
    assert _metrics_for([Message.user("task"), Message.assistant("answer")])["episode/reasoning_cjk_rate"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
