"""An engine-aborted generation (vLLM finish reason ``abort``) is recorded like a length cut, which the trainer
never trains as a natural stop; a fragment scored as one teaches the policy to end its reasoning wherever the
engine stopped it."""

from types import SimpleNamespace

import pytest

from halo.environments.environment_base import EnvironmentBase
from halo.environments.messages import Message, Trajectory
from halo.inference.response import FINISH_REASON_ABORT, FINISH_REASON_LENGTH


def _record_turn(finish_reason: str | None) -> Message:
    env = SimpleNamespace(max_tool_calls_per_turn=None)
    trajectory = Trajectory()
    trajectory.add_message(Message.user("q"))
    EnvironmentBase._add_action_message(env, trajectory, "partial reasoning", {"finish_reason": finish_reason})
    return trajectory.messages[-1]


@pytest.mark.parametrize("finish_reason", [FINISH_REASON_LENGTH, FINISH_REASON_ABORT])
def test_engine_cut_turns_are_truncated(finish_reason):
    assert _record_turn(finish_reason).truncated is True


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", None])
def test_completed_turns_are_not_truncated(finish_reason):
    assert _record_turn(finish_reason).truncated is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
