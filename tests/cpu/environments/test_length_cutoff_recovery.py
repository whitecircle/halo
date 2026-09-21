#!/usr/bin/env python
"""CPU tests for unproductive turns: recoverable and priced, never a silent episode death.

When the engine cuts a turn short before it produced anything — at its token cap
(``finish_reason == "length"``) or by aborting it (``"abort"``) — the text is a fragment. Finalizing
it as a plain-text answer ends the episode and books the failure as a NATURAL termination —
invisible in every health metric. Instead the episode nudges (in its own protocol's words) and
retries within ``max_turns``; the trainer skips the fragment. A turn the model ends with neither a
tool call nor visible content recovers the same way, under the same cap, and is skipped for the same
reason. A recovered turn is unpriced by default and pays ``length_cutoff_penalty`` where a protocol
configures it; the turn that exhausts the recovery cap pays the overflow price instead.

Run: python tests/cpu/environments/test_length_cutoff_recovery.py  (or pytest)
"""

import pytest

from src.environments.base import REWARD_COMPONENTS_KEY, Message, Trajectory
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.envs.protocols.react import ReActEnvironment
from src.environments.episode import TurnGeneration, step_context_from_generation
from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter
from src.inference.response import ENGINE_CUT_FINISH_REASONS

# Token ids of the synthetic prefix-monotone template the whole-trajectory tests render with.
_ROLE_TOKENS = {"system": 100, "user": 101, "assistant": 102, "tool": 103}
_END_TOKEN = 199


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


def _echo_call_step(env, eid):
    """One completed turn whose echo call executes: the native protocol reads calls off the context."""
    call = {"id": "c0", "function": {"name": "echo", "arguments": '{"text": "x"}'}}
    return env.step([eid], ["calling"], [{"finish_reason": "stop", "tool_calls": [call]}])[0]


def test_length_cutoff_keeps_the_episode_alive_and_is_unpriced():
    env = _make_env()
    eid = _reset(env)
    step = env.step([eid], ["a thought that ran out of room"], [{"finish_reason": "length"}])[0]

    assert step.done is False
    # Unpriced by default: the price is a knob, and off it is only avoidable by reasoning short of the budget.
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


@pytest.mark.parametrize("finish_reason", ENGINE_CUT_FINISH_REASONS)
def test_every_engine_cut_reason_takes_the_recovery_path(finish_reason):
    """An aborted turn is as unfinished as a length-capped one, and the recovery is what keeps it out
    of the grade: flagging the message alone still lets the fragment terminate the episode as the
    model's final answer."""
    env = _make_env()
    eid = _reset(env)
    step = env.step([eid], ["a fragment"], [{"finish_reason": finish_reason}])[0]

    assert step.done is False
    traj = env.get_trajectories([eid])[0]
    assert traj.info["completed"] is False
    assert traj.info["length_cutoff_turns"] == 1
    assert traj.messages[-1].content == NativeToolUseEnvironment.LENGTH_CUTOFF_NUDGE


_SALVAGED_CALL = [{"id": "c1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}]


def _generation(finish_reason: str) -> TurnGeneration:
    return TurnGeneration(
        text="Let me implement this", tool_calls=_SALVAGED_CALL, reasoning="", tokens=4300, finish_reason=finish_reason
    )


@pytest.mark.parametrize("finish_reason", ENGINE_CUT_FINISH_REASONS)
def test_a_cut_turn_executes_nothing_the_parser_salvaged(finish_reason):
    """A turn cut inside its tool call reaches the driver with the call's name and empty arguments.
    Executed, it books a malformed call the model never finished, and because the label says the
    turn completed, the fragment trains as a normal row and the model retries into the same cap."""
    ctx = step_context_from_generation({}, _generation(finish_reason))
    assert "tool_calls" not in ctx

    env = _make_env()
    eid = _reset(env)
    step = env.step([eid], ["Let me implement this"], [ctx])[0]

    assert step.done is False
    traj = env.get_trajectories([eid])[0]
    assert traj.info["length_cutoff_turns"] == 1
    assert traj.info.get("total_tool_calls", 0) == 0
    fragment, nudge = traj.messages[-2], traj.messages[-1]
    assert fragment.role == "assistant" and fragment.truncated is True and not fragment.tool_calls
    assert nudge.content == NativeToolUseEnvironment.LENGTH_CUTOFF_NUDGE


def test_a_completed_turn_keeps_its_tool_calls():
    ctx = step_context_from_generation({}, _generation("tool_calls"))
    assert ctx["tool_calls"] == _SALVAGED_CALL


@pytest.mark.parametrize("finish_reason", ENGINE_CUT_FINISH_REASONS)
def test_react_every_engine_cut_reason_takes_the_recovery_path(finish_reason):
    # Unrecovered, the fragment reads as a ReAct format failure and earns the format hint instead.
    env = _make_react_env()
    eid = _reset(env)
    step = env.step([eid], ["Thought: I should start by computing the ra"], [{"finish_reason": finish_reason}])[0]

    assert step.done is False
    traj = env.get_trajectories([eid])[0]
    assert traj.info["length_cutoff_turns"] == 1
    assert traj.messages[-1].content == ReActEnvironment.LENGTH_CUTOFF_NUDGE
    assert "no_action" not in step.info


@pytest.mark.parametrize(
    "text", ['Thought: ready\nAction: echo(text="hi")', "Thought: done\nFinal Answer: 42"], ids=["action", "answer"]
)
def test_react_cut_turn_executes_nothing_the_parser_salvaged(text):
    """The base flags every cut turn untrainable, so an Action executed or a Final Answer graded off
    a cut turn earns a reward on the one turn the trainer then excludes."""
    env = _make_react_env()
    eid = _reset(env)
    step = env.step([eid], [text], [{"finish_reason": "length"}])[0]
    assert step.done is False and step.reward == 0.0
    traj = env.get_trajectories([eid])[0]
    assert traj.info["total_tool_calls"] == 0 and traj.info["completed"] is False
    assert traj.info["length_cutoff_turns"] == 1
    assert traj.messages[-1].content == ReActEnvironment.LENGTH_CUTOFF_NUDGE
    assert [m.truncated for m in traj.messages if m.role == "assistant"] == [True]


def test_react_invented_tool_turn_is_flagged_untrainable():
    env = _make_react_env()
    eid = _reset(env)
    env.step([eid], ['Thought: t\nAction: bogus(text="x")'], [{"finish_reason": "stop"}])
    env.step([eid], ['Thought: t\nAction: echo(text="x")'], [{"finish_reason": "stop"}])
    assistant = [m for m in env.get_trajectories([eid])[0].messages if m.role == "assistant"]
    assert [m.calls_rejected for m in assistant] == [True, False]
    assert [m.untrainable for m in assistant] == [True, False]


@pytest.mark.parametrize(
    "nudge",
    [
        NativeToolUseEnvironment.LENGTH_CUTOFF_NUDGE,
        NativeToolUseEnvironment.EMPTY_TURN_NUDGE,
        ReActEnvironment.LENGTH_CUTOFF_NUDGE,
        ReActEnvironment.EMPTY_TURN_NUDGE,
    ],
    ids=["native-cut", "native-empty", "react-cut", "react-empty"],
)
def test_the_nudge_never_asks_for_shorter_reasoning(nudge):
    # The nudge is trained on wherever recovery succeeds, so a terseness ask becomes a global lesson.
    lowered = nudge.lower()
    assert "think brief" not in lowered
    assert "short" not in lowered and "concise" not in lowered and "briefly" not in lowered
    # It answers an engine abort as well, so it must not name a cause it cannot know.
    assert "length limit" not in lowered


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


@pytest.mark.parametrize("action", ["", "  \n"], ids=["empty", "whitespace"])
def test_an_empty_turn_is_recovered_like_a_cut_and_flagged_untrainable(action):
    """The model closed its reasoning and stopped with nothing visible and no call. Finalized, that is a
    natural termination graded on nothing; recovered, the stop is a turn the trainer skips — weighted,
    a recovering episode would reinforce it."""
    env = _make_env()
    eid = _reset(env)
    step = env.step([eid], [action], [{"finish_reason": "stop", "reasoning": "a thought that reached no plan"}])[0]

    assert step.done is False and step.reward == 0.0
    # Stamped as its own kind: an eval record must not book a stop the engine never cut as a cut.
    assert step.info["empty"] is True and "length_cutoff" not in step.info
    traj = env.get_trajectories([eid])[0]
    assert traj.info["completed"] is False
    assert traj.info["empty_turns"] == 1 and "length_cutoff_turns" not in traj.info
    turn, nudge = traj.messages[-2], traj.messages[-1]
    assert turn.role == "assistant" and turn.empty is True and turn.truncated is False and turn.untrainable
    assert nudge.role == "user" and nudge.content == NativeToolUseEnvironment.EMPTY_TURN_NUDGE


def test_react_empty_turn_is_recovered_and_is_not_a_format_failure():
    # Unrecovered, an empty turn earns the format hint: uncapped, unpriced and trained on.
    env = _make_react_env()
    eid = _reset(env)
    step = env.step([eid], [""], [{"finish_reason": "stop"}])[0]
    assert step.done is False and step.reward == 0.0 and "no_action" not in step.info
    traj = env.get_trajectories([eid])[0]
    assert traj.info["empty_turns"] == 1
    assert traj.messages[-1].content == ReActEnvironment.EMPTY_TURN_NUDGE
    assert [m.empty for m in traj.messages if m.role == "assistant"] == [True]


def test_a_recovered_empty_turn_pays_the_cut_price():
    env = _make_env(length_cutoff_penalty=0.05, max_turns=6)
    eid = _reset(env)
    assert not env.step([eid], [""], [{"finish_reason": "stop"}])[0].done
    _echo_call_step(env, eid)
    env.step([eid], ["done"], [{"finish_reason": "stop"}])
    shaping, traj = _settled_tool_shaping(env, eid)
    assert traj.done and not traj.truncated and traj.info["total_tool_calls"] == 1
    assert shaping == pytest.approx(-0.05), shaping
    assert traj.total_reward == pytest.approx(sum(traj.info[REWARD_COMPONENTS_KEY].values()))


@pytest.mark.parametrize(
    ("action", "ctx"),
    [("", {"finish_reason": "stop"}), ("a thought that ran out of room", {"finish_reason": "length"})],
    ids=["empty", "cut"],
)
def test_an_unproductive_last_turn_pays_the_overflow_price_and_not_the_cut_price(action, ctx):
    """No turn is left to retry into, so the episode ends truncated at once: no dangling nudge, and the
    turn is not booked as recovered — recovered and overflow would otherwise both be charged."""
    env = _make_env(length_cutoff_penalty=0.05, turn_overflow_penalty=0.1, max_turns=1)
    eid = _reset(env)
    step = env.step([eid], [action], [ctx])[0]
    assert step.done and step.truncated and step.info.get("unrecovered_turn") is True
    traj = env.get_trajectories([eid])[0]
    assert traj.messages[-1].role == "assistant" and traj.messages[-1].untrainable
    shaping, _ = _settled_tool_shaping(env, eid)
    assert shaping == pytest.approx(-0.1), shaping


def test_cuts_and_empty_turns_share_the_recovery_cap():
    """One cap bounds what an episode may burn on turns that produced nothing, whichever way; the
    turn past it ends the episode truncated and pays the overflow price, not the cut price."""
    env = _make_env(length_cutoff_penalty=0.05, turn_overflow_penalty=0.1, max_length_cutoff_recoveries=1)
    eid = _reset(env)
    assert not env.step([eid], ["a thought that ran out of room"], [{"finish_reason": "length"}])[0].done
    second = env.step([eid], [""], [{"finish_reason": "stop"}])[0]
    assert second.done and second.truncated and second.info["unrecovered_turn"]
    traj = env.get_trajectories([eid])[0]
    assert traj.info["length_cutoff_turns"] == 1 and traj.info["empty_turns"] == 1
    last = traj.messages[-1]
    assert last.role == "assistant", "no nudge for a turn that ends the episode"
    assert last.empty is True and last.untrainable, "the exhausting stop is skipped like a recovered one"
    shaping, _ = _settled_tool_shaping(env, eid)
    assert shaping == pytest.approx(-0.15), shaping


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
    trainer._rollout_template_kwargs = {}
    trainer._carry_reasoning = False
    trainer._max_train_row_tokens = None
    trainer._rows_over_cap = 0
    trainer._context_limit = lambda: 100_000

    traj = Trajectory(
        messages=[
            Message.user("task"),
            Message.assistant("fragment", token_ids=[1, 2, 3], prompt_token_ids=[9], truncated=True),
            Message.assistant("bogus call", token_ids=[6], prompt_token_ids=[9], calls_rejected=True),
            Message.assistant("", token_ids=[7, 8], prompt_token_ids=[9], empty=True),
            Message.assistant("real turn", token_ids=[4, 5], prompt_token_ids=[9, 9]),
        ]
    )
    rows = trainer._tokenize_trajectory_turns(type("R", (), {"trajectory": traj})())
    assert len(rows) == 1  # the fragment, the rejected-call turn and the empty turn are all skipped
    assert rows[0].completion_ids.tolist() == [4, 5]


def _flat_render(msgs, add_generation_prompt, _include_thinking):
    """A minimal prefix-monotone chat template: ``<role> <content chars> <end>`` per message.

    Stands in for a real tokenizer so the span locator anchors every boundary by construction and the
    assertions below are about the loss mask alone.
    """
    ids = []
    for m in msgs:
        ids.append(_ROLE_TOKENS[m.role])
        ids.extend(ord(c) for c in (m.content or ""))
        ids.append(_END_TOKEN)
    if add_generation_prompt:
        ids.append(_ROLE_TOKENS["assistant"])
    return ids


def _render_trainer():
    from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer

    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer._batch_build_error = None
    trainer._rollout_template_kwargs = {}
    trainer._carry_reasoning = False
    trainer._max_train_row_tokens = None
    trainer._rows_over_cap = 0
    trainer._context_limit = lambda: 100_000
    trainer._tokenizer = type("T", (), {"name_or_path": "fake"})()
    trainer.eos_token_id = 2
    trainer.pad_token_id = 0
    trainer._render_messages_to_ids = lambda msgs, agp, _kwargs, include_thinking=True: _flat_render(
        msgs, agp, include_thinking
    )
    return trainer


def _trained_tokens(trainer, traj):
    _prompt, completion_ids, mask = trainer._tokenize_trajectory(type("R", (), {"trajectory": traj})())
    return [tid for tid, keep in zip(completion_ids.tolist(), mask.tolist(), strict=True) if keep]


def _two_turn_trajectory(first_truncated: bool = False, first_empty: bool = False):
    return Trajectory(
        messages=[
            Message.user("task"),
            Message.assistant("aaaa", truncated=first_truncated, empty=first_empty),
            Message.user("obs"),
            Message.assistant("bb"),
        ]
    )


@pytest.mark.parametrize("first", [{"first_truncated": True}, {"first_empty": True}], ids=["cut", "empty"])
def test_whole_trajectory_render_does_not_weight_an_unproductive_turn(first):
    """The re-tokenized path must exclude the same turns the per-turn path does.

    It is the fallback whenever the engine returned no sampled ids, and it is the only path at
    ``train_on_sampled_tokens: false`` — weighting the fragment there reinforces exactly the runaway
    the exclusion exists to suppress, at whatever advantage the recovered episode earns.
    """
    trained = _trained_tokens(_render_trainer(), _two_turn_trajectory(**first))

    assert ord("a") not in trained  # the unproductive turn's own tokens carry no loss
    assert trained.count(ord("b")) == 2  # the turn that finished still trains


def test_whole_trajectory_row_over_the_train_row_cap_is_one_masked_row():
    trainer = _render_trainer()
    trainer._max_train_row_tokens = 5
    _prompt, completion_ids, mask = trainer._tokenize_trajectory(
        type("R", (), {"trajectory": _two_turn_trajectory()})()
    )
    assert mask.tolist() == [0] and completion_ids.numel() == 1
    assert trainer._rows_over_cap == 1


def test_whole_trajectory_render_weights_every_finished_turn():
    trained = _trained_tokens(_render_trainer(), _two_turn_trajectory())

    assert trained.count(ord("a")) == 4
    assert trained.count(ord("b")) == 2


def test_whole_trajectory_render_of_an_all_cut_episode_is_one_masked_row():
    """Nothing survives, so the row must not carry the trajectory's width into the padded batch."""
    traj = Trajectory(
        messages=[
            Message.user("task"),
            Message.assistant("aaaa", truncated=True),
            Message.user("obs"),
            Message.assistant("bb", calls_rejected=True),
        ]
    )
    trainer = _render_trainer()
    _prompt, completion_ids, mask = trainer._tokenize_trajectory(type("R", (), {"trajectory": traj})())

    assert mask.tolist() == [0]
    assert completion_ids.numel() == 1


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
    trainer._rollout_template_kwargs = {}
    trainer._carry_reasoning = False
    trainer._max_train_row_tokens = None
    trainer._rows_over_cap = 0
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


def test_recovery_cap_ends_the_episode_truncated_at_the_cut_past_it():
    env = _make_env(max_length_cutoff_recoveries=1)
    eid = _reset(env)
    first = env.step([eid], ["a thought that ran out of room"], [{"finish_reason": "length"}])[0]
    assert not first.done and first.info["length_cutoff"], "the first cut is recovered as before"
    traj = env.get_trajectories([eid])[0]
    assert traj.messages[-1].content == NativeToolUseEnvironment.LENGTH_CUTOFF_NUDGE
    second = env.step([eid], ["another thought that ran out of room"], [{"finish_reason": "length"}])[0]
    assert second.done and second.truncated, "the cut past the cap ends the episode like a max_turns overflow"
    assert second.info["unrecovered_turn"]
    traj = env.get_trajectories([eid])[0]
    assert traj.info["length_cutoff_turns"] == 2
    assert traj.messages[-1].content != NativeToolUseEnvironment.LENGTH_CUTOFF_NUDGE, (
        "no nudge for a turn that ends the episode"
    )


def _settled_tool_shaping(env, eid):
    traj = env.get_trajectories([eid])[0]
    env._settle_grade(traj, None)
    return traj.info[REWARD_COMPONENTS_KEY]["reward/tool_shaping"], traj


def test_a_recovered_cut_pays_the_length_cutoff_penalty_once_per_cut():
    """Two recovered cuts, a tool call, then the text answer that ends the episode: the price lands in
    ``reward/tool_shaping`` twice, beside no overflow, and the decomposition still sums."""
    env = _make_env(length_cutoff_penalty=0.05, max_turns=6)
    eid = _reset(env)
    for _ in range(2):
        assert not env.step([eid], ["a thought that ran out of room"], [{"finish_reason": "length"}])[0].done
    _echo_call_step(env, eid)
    env.step([eid], ["done"], [{"finish_reason": "stop"}])
    shaping, traj = _settled_tool_shaping(env, eid)
    assert traj.done and not traj.truncated and traj.info["total_tool_calls"] == 1
    assert shaping == pytest.approx(-0.1), shaping
    assert traj.total_reward == pytest.approx(sum(traj.info[REWARD_COMPONENTS_KEY].values()))


def test_the_cut_that_exhausts_the_cap_pays_the_overflow_price_not_the_cut_price():
    env = _make_env(length_cutoff_penalty=0.05, turn_overflow_penalty=0.1, max_length_cutoff_recoveries=1)
    eid = _reset(env)
    env.step([eid], ["a thought that ran out of room"], [{"finish_reason": "length"}])
    second = env.step([eid], ["another thought that ran out of room"], [{"finish_reason": "length"}])[0]
    assert second.truncated
    shaping, _ = _settled_tool_shaping(env, eid)
    # one recovered cut (0.05) + the overflow (0.1); the exhausting cut is not charged twice
    assert shaping == pytest.approx(-0.15), shaping


def test_without_the_knob_a_recovered_cut_still_costs_nothing():
    env = _make_env(max_turns=6)
    eid = _reset(env)
    env.step([eid], ["a thought that ran out of room"], [{"finish_reason": "length"}])
    _echo_call_step(env, eid)
    env.step([eid], ["done"], [{"finish_reason": "stop"}])
    shaping, _ = _settled_tool_shaping(env, eid)
    assert shaping == 0.0


@pytest.mark.parametrize("bad", [-0.05, float("nan"), float("inf")])
def test_the_cut_price_is_a_finite_magnitude(bad):
    with pytest.raises(ValueError, match="length_cutoff_penalty"):
        _make_env(length_cutoff_penalty=bad)


def test_carried_reasoning_puts_the_cut_thought_in_the_retry_observation():
    """With ``carry_reasoning`` the retry conditions on the thought the cap interrupted; without it the
    engine sees only the fragment's visible text and the retry restarts from nothing."""
    carrying = _make_env(carry_reasoning=True)
    eid = _reset(carrying)
    step = carrying.step([eid], ["visible fragment"], [{"finish_reason": "length", "reasoning": "half a thought"}])[0]
    fragment = step.observation[-2]
    assert fragment["role"] == "assistant" and fragment["reasoning_content"] == "half a thought"
    assert step.observation[-1] == {"role": "user", "content": NativeToolUseEnvironment.LENGTH_CUTOFF_NUDGE}

    plain = _make_env()
    assert plain.carry_reasoning is False
    eid = _reset(plain)
    step = plain.step([eid], ["visible fragment"], [{"finish_reason": "length", "reasoning": "half a thought"}])[0]
    assert step.observation[-2] == {"role": "assistant", "content": "visible fragment"}


def test_only_the_previous_turns_reasoning_is_carried():
    """A request grows by one reasoning budget at most: the turn before the next one carries its thought,
    the turns before that revert to their visible text."""
    env = _make_env(carry_reasoning=True)
    eid = _reset(env)
    call = {"id": "c0", "function": {"name": "echo", "arguments": '{"text": "hi"}'}}
    env.step([eid], ["first"], [{"finish_reason": "stop", "tool_calls": [call], "reasoning": "thought one"}])
    step = env.step([eid], ["second"], [{"finish_reason": "stop", "tool_calls": [call], "reasoning": "thought two"}])[
        0
    ]
    assistants = [m for m in step.observation if m["role"] == "assistant"]
    assert "reasoning_content" not in assistants[0] and assistants[0]["content"] == "first"
    assert assistants[1]["reasoning_content"] == "thought two"


def test_carried_reasoning_reaches_every_observation_of_a_tool_round():
    env = _make_env(carry_reasoning=True)
    eid = _reset(env)
    call = {"id": "c0", "function": {"name": "echo", "arguments": '{"text": "hi"}'}}
    step = env.step([eid], ["calling"], [{"finish_reason": "stop", "tool_calls": [call], "reasoning": "why echo"}])[0]
    assistant = next(m for m in step.observation if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "why echo" and assistant["tool_calls"]
    assert step.observation[-1]["role"] == "tool"


def test_recovery_cap_of_zero_ends_the_episode_at_the_first_cut_and_negative_is_refused():
    env = _make_env(max_length_cutoff_recoveries=0)
    eid = _reset(env)
    step = env.step([eid], ["a fragment"], [{"finish_reason": "length"}])[0]
    assert step.done and step.truncated
    with pytest.raises(ValueError, match="max_length_cutoff_recoveries must be >= 0"):
        _make_env(max_length_cutoff_recoveries=-1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
