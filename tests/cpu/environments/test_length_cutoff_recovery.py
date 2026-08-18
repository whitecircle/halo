#!/usr/bin/env python
"""CPU tests for length-cutoff turns: recoverable and priced, never a silent episode death.

When the engine cuts a turn off at its token cap (``finish_reason == "length"``) before the turn
produced anything, the text is a fragment. Finalizing it as a plain-text answer ends the episode and
books the failure as a NATURAL termination — invisible in every health metric. Instead the episode
nudges (in its own protocol's words) and retries within ``max_turns``, unpriced; the trainer skips
the fragment.

Run: python tests/cpu/environments/test_length_cutoff_recovery.py  (or pytest)
"""

import pytest

from src.environments.base import Message, Trajectory
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.envs.protocols.react import ReActEnvironment
from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter


def _echo_registry():
    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="echo",
            description="echo",
            parameters=[ToolParameter("text", "string", "text")],
            handler=lambda text: text,
        )
    )
    return registry


def _make_env(**kwargs):
    kwargs.setdefault("turn_overflow_penalty", 0.1)
    kwargs.setdefault("max_turns", 4)
    return NativeToolUseEnvironment(tool_registry=_echo_registry(), **kwargs)


def _make_react_env(**kwargs):
    kwargs.setdefault("max_turns", 4)
    return ReActEnvironment(tool_registry=_echo_registry(), **kwargs)


def _reset(env):
    ids, _ = env.reset(["task"], [{}])
    return ids[0]


def test_length_cutoff_keeps_the_episode_alive_and_is_unpriced():
    env = _make_env()
    eid = _reset(env)
    step = env.step([eid], ["a thought that ran out of room"], [{"finish_reason": "length"}])[0]

    assert step.done is False
    # Unpriced on purpose: an added penalty is only avoidable by reasoning short of the budget.
    assert step.reward == 0.0
    traj = env.get_trajectories([eid])[0]
    assert traj.info["length_cutoff_turns"] == 1
    assert traj.messages[-1].role == "user"
    assert traj.messages[-1].content == NativeToolUseEnvironment.LENGTH_CUTOFF_NUDGE
    assert env.rollout_metrics(traj)["episode/length_cutoff_turns"] == 1.0


def test_react_length_cutoff_keeps_the_episode_alive_and_is_unpriced():
    """The ReAct protocol recovers a cut-off turn too, instead of hinting at a format the model got
    right — an unrecoverable turn there is indistinguishable in the logs from a format failure."""
    env = _make_react_env()
    ids, _ = env.reset(["task"], [{}])
    eid = ids[0]

    step = env.step([eid], ["Thought: I should start by computing the ra"], [{"finish_reason": "length"}])[0]

    assert step.done is False
    # Unpriced in full: the thought bonus would pay for a turn the model never finished.
    assert step.reward == 0.0
    traj = env.get_trajectories([eid])[0]
    assert traj.info["length_cutoff_turns"] == 1
    assert traj.messages[-1].role == "user"
    assert traj.messages[-1].content == ReActEnvironment.LENGTH_CUTOFF_NUDGE
    assert env.rollout_metrics(traj)["episode/length_cutoff_turns"] == 1.0


@pytest.mark.parametrize("protocol", [NativeToolUseEnvironment, ReActEnvironment])
def test_the_nudge_never_asks_for_shorter_reasoning(protocol):
    # The nudge is trained on wherever recovery succeeds, so a terseness ask becomes a global lesson.
    lowered = protocol.LENGTH_CUTOFF_NUDGE.lower()
    assert "think brief" not in lowered
    assert "short" not in lowered and "concise" not in lowered and "briefly" not in lowered


def test_cut_off_turn_is_flagged_for_the_trainer():
    env = _make_env()
    eid = _reset(env)
    env.step([eid], ["fragment"], [{"finish_reason": "length"}])
    traj = env.get_trajectories([eid])[0]
    assert [m.truncated for m in traj.messages if m.role == "assistant"] == [True]


def test_a_completed_text_turn_still_ends_the_episode():
    env = _make_env()
    eid = _reset(env)
    step = env.step([eid], ["my final answer"], [{"finish_reason": "stop"}])[0]
    assert step.done is True
    traj = env.get_trajectories([eid])[0]
    assert traj.info["completed"] is True
    assert traj.info.get("length_cutoff_turns", 0) == 0
    assert [m.truncated for m in traj.messages if m.role == "assistant"] == [False]


def test_length_finish_with_a_tool_call_takes_the_normal_tool_path():
    # The call landed before the cap, so the turn produced real work: no penalty, no nudge.
    env = _make_env(tool_success_reward=0.02)
    eid = _reset(env)
    call = {"id": "c0", "function": {"name": "echo", "arguments": '{"text": "hi"}'}}
    step = env.step([eid], ["calling"], [{"finish_reason": "length", "tool_calls": [call]}])[0]
    assert step.done is False
    assert step.reward == pytest.approx(0.02)
    traj = env.get_trajectories([eid])[0]
    assert traj.info.get("length_cutoff_turns", 0) == 0
    assert traj.messages[-1].role == "tool"


def test_repeated_cutoffs_still_end_at_max_turns():
    env = _make_env(max_turns=2)
    eid = _reset(env)
    env.step([eid], ["fragment"], [{"finish_reason": "length"}])
    step = env.step([eid], ["fragment"], [{"finish_reason": "length"}])[0]
    assert step.done is True and step.truncated is True
    assert env.get_trajectories([eid])[0].info["length_cutoff_turns"] == 2


def test_unknown_tool_error_names_the_real_tools():
    # A drifted policy invents plausible names; the correction has to reach it in the observation.
    env = _make_env()
    eid = _reset(env)
    call = {"id": "c0", "function": {"name": "test_tool", "arguments": "{}"}}
    env.step([eid], ["calling"], [{"finish_reason": "stop", "tool_calls": [call]}])
    tool_msg = env.get_trajectories([eid])[0].messages[-1]
    assert "Unknown tool 'test_tool'" in tool_msg.content
    assert "Available tools: echo" in tool_msg.content


def test_turn_whose_every_call_is_unknown_is_flagged_and_not_trained():
    env = _make_env()
    eid = _reset(env)
    bogus = {"id": "c0", "function": {"name": "test_tool", "arguments": "{}"}}
    env.step([eid], ["calling"], [{"finish_reason": "stop", "tool_calls": [bogus]}])
    real = {"id": "c1", "function": {"name": "echo", "arguments": '{"text": "hi"}'}}
    env.step([eid], ["calling"], [{"finish_reason": "stop", "tool_calls": [real]}])
    flags = [m.calls_rejected for m in env.get_trajectories([eid])[0].messages if m.role == "assistant"]
    assert flags == [True, False]


def test_partially_valid_turn_is_not_flagged():
    # One real call did work, so the turn produced something: it must stay trainable.
    env = _make_env()
    eid = _reset(env)
    calls = [
        {"id": "c0", "function": {"name": "test_tool", "arguments": "{}"}},
        {"id": "c1", "function": {"name": "echo", "arguments": '{"text": "hi"}'}},
    ]
    env.step([eid], ["calling"], [{"finish_reason": "stop", "tool_calls": calls}])
    asst = [m for m in env.get_trajectories([eid])[0].messages if m.role == "assistant"][-1]
    assert asst.calls_rejected is False


def test_a_tools_own_not_found_message_is_not_a_model_rejection():
    """The rejection channel is structural, never the error text.

    A tool's ``error`` is free text its backend controls — the MCP client re-raises the SERVER's
    message verbatim — so a server answering "Tool not found: x" describes a REAL tool failing. Read
    as the env's own invented-tool rejection it would silently drop the turn from training and book a
    broken tool as a model mistake."""
    env = _make_env()

    def _server_error(**_kwargs):
        raise RuntimeError("Tool not found: mcp_read")

    env.registry.register(NativeTool(name="mcp_read", description="mcp", parameters=[], handler=_server_error))
    eid = _reset(env)
    call = {"id": "c0", "function": {"name": "mcp_read", "arguments": "{}"}}
    env.step([eid], ["calling"], [{"finish_reason": "stop", "tool_calls": [call]}])

    traj = env.get_trajectories([eid])[0]
    asst = [m for m in traj.messages if m.role == "assistant"][-1]
    assert asst.calls_rejected is False  # a registered tool ran and failed: the turn stays trainable
    assert traj.info["tool_results"][-1]["success"] is False
    assert traj.info["successful_tool_calls"] == 0


def test_trainer_skips_the_cut_off_turn_row():
    from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer

    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer._rollout_routing_replay = False
    trainer._batch_build_error = None
    trainer._warned_capture_missing = False
    trainer._context_limit = lambda: 100_000

    traj = Trajectory(
        messages=[
            Message.user("task"),
            Message.assistant("fragment", token_ids=[1, 2, 3], prompt_token_ids=[9], truncated=True),
            Message.assistant("bogus call", token_ids=[6], prompt_token_ids=[9], calls_rejected=True),
            Message.assistant("real turn", token_ids=[4, 5], prompt_token_ids=[9, 9]),
        ]
    )
    rows = trainer._tokenize_trajectory_turns(type("R", (), {"trajectory": traj})())
    assert len(rows) == 1  # both the fragment and the rejected-call turn are skipped
    assert rows[0].completion_ids.tolist() == [4, 5]


def test_all_turns_excluded_trains_a_zero_weight_row():
    """No surviving turn must not re-tokenize the whole trajectory at FULL weight.

    ``_tokenize_trajectory`` masks IN every assistant span, truncated and rejected ones included, so
    falling back to it here inverts the exclusion into full-weight training on exactly the runaway or
    invented call it exists to suppress. The row must survive (rank-uniform row counts) at zero loss
    weight, and this is not a batch error — the episode was simply unusable.
    """
    from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer

    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer._rollout_routing_replay = False
    trainer._batch_build_error = None
    trainer._warned_capture_missing = False
    trainer._context_limit = lambda: 100_000
    trainer.eos_token_id = 2
    trainer.pad_token_id = 0

    traj = Trajectory(
        messages=[
            Message.user("task"),
            Message.assistant("fragment", token_ids=[1, 2, 3], prompt_token_ids=[9], truncated=True),
            Message.assistant("bogus call", token_ids=[6], prompt_token_ids=[9], calls_rejected=True),
        ]
    )
    rows = trainer._tokenize_trajectory_turns(type("R", (), {"trajectory": traj})())

    assert len(rows) == 1
    assert rows[0].completion_mask.sum().item() == 0
    assert trainer._batch_build_error is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
