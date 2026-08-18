"""CPU test: tool observations are capped (``max_observation_chars``) at the source.

An unbounded tool output (e.g. a python_repl that prints megabytes) otherwise bloats the trajectory
and makes the per-turn chat-template re-render take minutes — stalling the whole training step. The cap
is applied where the observation is built, so the rollout and the trainer's recompute see identical text.

    python tests/cpu/environments/test_observation_truncation.py
"""

import sys

import pytest

from src.environments.base import Trajectory
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.tools.definitions import NativeTool, NativeToolCall


def _env(cap):
    return CodeContestsEnvironment(
        language="python", success_reward=1.0, sandbox_backend="local", max_observation_chars=cap
    )


def _traj():
    t = Trajectory()
    t.info["total_tool_calls"] = 0
    t.info["successful_tool_calls"] = 0
    return t


def test_truncate_helper_caps_and_marks():
    env = _env(1000)
    out = env._truncate_observation("x" * 500_000)
    assert len(out) <= 1000 + 40  # limit + short marker
    assert out.startswith("x" * 1000)
    assert "truncated" in out and "499000" in out  # reports dropped-char count


def test_truncate_helper_leaves_small_output_untouched():
    env = _env(1000)
    small = "ok: 3 tests passed"
    assert env._truncate_observation(small) == small


def test_execute_tool_calls_applies_cap():
    """The cap must be wired into execution, not just available as a helper."""
    env = _env(1000)
    huge = "y" * 500_000
    env.registry.register(NativeTool(name="big", description="returns a huge string", handler=lambda: huge))
    results, _ = env._execute_tool_calls([NativeToolCall(id="1", name="big", arguments={})], _traj())
    assert results[0].success
    assert len(results[0].content) <= 1000 + 40, f"observation not capped: {len(results[0].content)}"
    assert "truncated" in results[0].content


def test_execute_tool_calls_caps_huge_error():
    env = _env(1000)

    def _boom():
        raise RuntimeError("E" * 500_000)

    env.registry.register(NativeTool(name="boom", description="raises", handler=_boom))
    results, _ = env._execute_tool_calls([NativeToolCall(id="1", name="boom", arguments={})], _traj())
    assert not results[0].success
    assert len(results[0].content) <= 1000 + 40


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
