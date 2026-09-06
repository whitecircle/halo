"""An engine-aborted generation (vLLM finish reason ``abort``: a pause in abort mode, an engine restart)
must be recorded like a length cut — a truncated turn the trainer never trains as a natural stop — and
not as an ordinary completed turn. A fragment scored as a natural termination teaches the policy to end
its reasoning wherever the engine happened to stop it.

    python tests/cpu/environments/test_engine_abort_is_a_cut_turn.py
"""

from types import SimpleNamespace

import pytest

from src.environments.base import BaseEnvironment, Message, Trajectory
from src.inference.response import FINISH_REASON_ABORT, FINISH_REASON_LENGTH


def _record_turn(finish_reason: str | None) -> Message:
    env = SimpleNamespace(max_tool_calls_per_turn=None)
    trajectory = Trajectory()
    trajectory.add_message(Message.user("q"))
    BaseEnvironment._add_action_message(env, trajectory, "partial reasoning", {"finish_reason": finish_reason})
    return trajectory.messages[-1]


@pytest.mark.parametrize("finish_reason", [FINISH_REASON_LENGTH, FINISH_REASON_ABORT])
def test_engine_cut_turns_are_truncated(finish_reason):
    assert _record_turn(finish_reason).truncated is True


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", None])
def test_completed_turns_are_not_truncated(finish_reason):
    assert _record_turn(finish_reason).truncated is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
