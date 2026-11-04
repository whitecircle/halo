#!/usr/bin/env python
"""The shared eval runner: trajectory recording, the episode driver's failure handling and retries, and the scores.
A recording keeps the verdict but never the answer key; a generation the driver lost is a generation error, out
of every score, while one the episode's own request lost is graded on what it earned."""

import itertools
import json
import logging
import math
import random
import types
from typing import Any

import httpx
import pytest
from openai import (
    NOT_GIVEN,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
)

import halo.environments.eval_runner as eval_runner
from halo.environments.episode_keys import EPISODE_ERROR_KEY, REWARD_COMPONENTS_KEY, SOLVE_RATE_KEY
from halo.environments.eval_runner import (
    collect_results,
    report,
    run_episode,
    serialize_trajectory,
    summarize,
    trajectory_path,
    write_trajectories_jsonl,
)
from halo.environments.messages import Message, Trajectory
from halo.environments.protocols.native_tool_use import NativeToolUseEnvironment
from halo.environments.reasoning_effort import stable_reasoning_effort
from halo.environments.rollout_config import RANDOM_REASONING_EFFORT, RolloutConfig
from halo.environments.tools.definitions import NativeTool, NativeToolRegistry
from halo.environments.tools.toolsets import build_simulated_file_tools
from halo.rewards.terms import OBJECTIVE_REWARD_KEY
from tests.common.fake_chat_client import chat_reply, scripted_chat


def test_trajectory_path_slugifies_per_run():
    p = trajectory_path("/tmp/traj", "anthropic/claude-opus-4.8", "codeforces", "test", "python")
    assert p == "/tmp/traj/anthropic-claude-opus-4.8__codeforces__test__python.jsonl"


def _leaky_traj() -> Trajectory:
    """A finished trajectory whose info carries both the verdict and the answer key."""
    return Trajectory(
        messages=[
            Message.system("You are an expert competitive programmer."),
            Message.user("Solve: print 1"),
            Message.assistant("```python\nprint(1)\n```"),
        ],
        total_reward=1.0,
        done=True,
        truncated=False,
        info={
            "tests_passed": 2,
            "tests_total": 2,
            "submission_result": "Passed 2/2 test cases.",
            "_test_cases": [{"input": "1", "output": "EXPECTED_OUTPUT_42"}],  # answer key (underscored)
            "context": {"answer": '{"tests": [{"output": "EXPECTED_OUTPUT_42"}]}'},  # answer key (context)
            "_eval_stats": {"generations": 1, "completion_tokens": 50},
        },
    )


def test_serialize_trajectory_keeps_messages_and_verdict():
    s = serialize_trajectory(_leaky_traj())
    assert [m["role"] for m in s["messages"]] == ["system", "user", "assistant"]
    assert s["total_reward"] == 1.0 and s["done"] is True and s["truncated"] is False
    assert s["info"]["tests_passed"] == 2 and s["info"]["submission_result"] == "Passed 2/2 test cases."
    assert s["info"]["eval_stats"]["completion_tokens"] == 50  # telemetry renamed from _eval_stats


def test_serialize_trajectory_keeps_each_turns_cap_and_reasoning_count():
    """A record of an eval shows beside each turn's render the reasoning cap it ran under and the reasoning it
    sampled, as training counts it; a turn that recorded neither carries no key."""
    trajectory = _leaky_traj()
    trajectory.messages.append(Message.assistant("again", thinking_cap=18000, reasoning_tokens=17999))
    turns = [m for m in serialize_trajectory(trajectory)["messages"] if m["role"] == "assistant"]
    assert "thinking_cap" not in turns[0] and "reasoning_tokens" not in turns[0]
    assert turns[1]["thinking_cap"] == 18000 and turns[1]["reasoning_tokens"] == 17999


def test_serialize_trajectory_drops_answer_key_and_internals():
    s = serialize_trajectory(_leaky_traj())
    assert "_test_cases" not in s["info"]
    assert "context" not in s["info"]
    assert "EXPECTED_OUTPUT_42" not in json.dumps(s)


def test_write_trajectories_jsonl_meta_then_episodes(tmp_path):
    results = [
        {
            "group": 800,
            "id": "cf-1900-A",
            "samples": [
                {
                    "reward": 1.0,
                    "success": True,
                    "stats": {"generations": 1},
                    "trajectory": serialize_trajectory(_leaky_traj()),
                }
            ],
        }
    ]
    path = str(tmp_path / "traj.jsonl")
    meta = {"model": "m", "tools": [{"function": {"name": "submit_solution"}}], "system_prompt": "sys"}
    n = write_trajectories_jsonl(path, meta, results)

    with open(path) as f:
        lines = [json.loads(line) for line in f]
    assert n == 1 and len(lines) == 2
    assert lines[0]["type"] == "meta" and lines[0]["model"] == "m" and lines[0]["tools"]
    ep = lines[1]
    assert ep["type"] == "episode" and ep["group"] == 800 and ep["success"] is True
    assert ep["index"] == 0 and ep["id"] == "cf-1900-A"  # addressable per-episode
    assert [m["role"] for m in ep["messages"]] == ["system", "user", "assistant"]
    assert "EXPECTED_OUTPUT_42" not in json.dumps(lines)  # no leak through the writer either
    assert ep[eval_runner.GENERATION_ERROR_KEY] is None


def test_a_generation_error_is_recorded_with_its_episode(tmp_path):
    """The file is what the offline re-grader reads: without the marker it could not tell a sample the
    driver lost from a scored miss, and would count it in ``n``."""
    lost = {"reward": None, "success": None, "stats": {}, eval_runner.GENERATION_ERROR_KEY: "NotFoundError: gone"}
    path = str(tmp_path / "traj.jsonl")
    write_trajectories_jsonl(path, {"model": "m"}, [{"group": None, "id": "p", "samples": [lost]}])
    with open(path) as f:
        episode = [json.loads(line) for line in f][1]
    assert episode[eval_runner.GENERATION_ERROR_KEY] == "NotFoundError: gone"
    assert episode["reward"] is None and episode["success"] is None


async def test_an_episode_that_raises_records_its_exception_type(monkeypatch):
    """A bare ``TimeoutError`` stringifies to ``""``: the error must still name what was raised."""

    async def _raises(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(eval_runner, "run_episode", _raises)
    (row,) = await collect_results(
        _tooled_env(), [{"prompt": "q", "context": {}}], client=object(), rollout=_RETRYING, num_samples=1
    )
    assert row["samples"][0]["error"] == "TimeoutError"


def _tooled_env(**kw) -> NativeToolUseEnvironment:
    registry = NativeToolRegistry()
    registry.register(NativeTool(name="echo", description="echo", parameters=[], handler=lambda **a: "ok"))
    kw.setdefault("tool_success_reward", 0.05)
    return NativeToolUseEnvironment(tool_registry=registry, **kw)


def _tool_call_response():
    fn = types.SimpleNamespace(name="echo", arguments="{}")
    return chat_reply(
        answer="",
        finish_reason="tool_calls",
        completion_tokens=3,
        tool_calls=[types.SimpleNamespace(id="c1", function=fn)],
        reasoning=None,
    )


async def _run_scripted_episode(env, script, monkeypatch):
    monkeypatch.setattr(eval_runner, "chat_completion", scripted_chat(script))
    return await run_episode(
        env, "task", {}, client=object(), rollout=RolloutConfig(model_name="m", temperature=0.0, max_tokens=16)
    )


async def test_generation_failure_is_truncation_not_completion(monkeypatch):
    """The plain-text terminal path would mark it ``completed`` and pay completion-graded envs the full objective
    for an HTTP error."""
    env = _tooled_env()
    traj = await _run_scripted_episode(env, [RuntimeError("http boom")], monkeypatch)
    assert traj.done and traj.truncated
    assert traj.info["completed"] is False
    assert traj.info[REWARD_COMPONENTS_KEY][OBJECTIVE_REWARD_KEY] == 0.0
    assert traj.total_reward == pytest.approx(0.0)


async def test_generation_failure_keeps_earned_tool_reward(monkeypatch):
    """A mid-episode death keeps the per-call tool rewards and adds no completion payout."""
    env = _tooled_env()
    traj = await _run_scripted_episode(env, [_tool_call_response(), RuntimeError("http boom")], monkeypatch)
    assert traj.done and traj.truncated and traj.info["completed"] is False
    assert traj.info["successful_tool_calls"] == 1
    assert traj.total_reward == pytest.approx(0.05)


async def test_generation_failure_is_not_priced_as_a_turn_overflow(monkeypatch):
    """A lost generation is the driver's fault: stamped under ``EPISODE_ERROR_KEY``, with no
    ``turn_overflow_penalty``."""
    env = _tooled_env(turn_overflow_penalty=0.3)
    traj = await _run_scripted_episode(env, [_tool_call_response(), RuntimeError("http boom")], monkeypatch)
    assert traj.done and traj.truncated
    assert traj.info[EPISODE_ERROR_KEY] == "RuntimeError: http boom"
    assert traj.total_reward == pytest.approx(0.05)
    assert traj.info["_eval_stats"]["tool_calls"] == 1, "the driver's stamp lands after the env's payload drop"


async def test_a_real_turn_cap_overflow_still_pays(monkeypatch):
    env = _tooled_env(max_turns=1, turn_overflow_penalty=0.3)
    traj = await _run_scripted_episode(env, [_tool_call_response()], monkeypatch)
    assert traj.done and traj.truncated and EPISODE_ERROR_KEY not in traj.info
    assert traj.total_reward == pytest.approx(0.05 - 0.3)


async def test_completed_episode_still_pays_success(monkeypatch):
    env = _tooled_env()
    final = chat_reply(answer="done", finish_reason="stop", completion_tokens=2, tool_calls=None, reasoning=None)
    traj = await _run_scripted_episode(env, [final], monkeypatch)
    assert traj.done and not traj.truncated
    assert traj.info["completed"] is True
    assert traj.total_reward == pytest.approx(1.0)


async def test_length_cutoff_is_recovered_offline_exactly_as_online(monkeypatch):
    """Without the engine's ``finish_reason`` in the step context, a cut turn arrives as a text answer and the env
    grades the fragment, diverging from online."""
    env = _tooled_env()
    cut_off = chat_reply(
        answer="a thought that ran out of",
        finish_reason="length",
        completion_tokens=9,
        tool_calls=None,
        reasoning=None,
    )
    final = chat_reply(answer="done", finish_reason="stop", completion_tokens=2, tool_calls=None, reasoning=None)
    traj = await _run_scripted_episode(env, [cut_off, final], monkeypatch)

    assert traj.info["length_cutoff_turns"] == 1
    assert traj.info["final_response"] == "done"  # the fragment was never scored as the answer
    assert NativeToolUseEnvironment.LENGTH_CUTOFF_NUDGE in [m.content for m in traj.messages]
    # The same flag the trainer reads to skip a fragment's row.
    assert [m.truncated for m in traj.messages if m.role == "assistant"] == [True, False]
    assert traj.total_reward == pytest.approx(1.0)


async def test_the_eval_keeps_a_cut_call_on_its_turn_and_in_the_record_without_running_or_resending_it(
    monkeypatch,
):
    """The eval driver salvages a turn cut inside its call as the training actor does: the call lands on the
    turn and in the JSONL apart from the ``tool_calls`` the re-grader replays, never runs, and is absent from
    what the endpoint is sent next."""
    partial = types.SimpleNamespace(
        id="c1", function=types.SimpleNamespace(name="echo", arguments='{"x": "# still re')
    )
    cut = chat_reply(
        answer="Computing", finish_reason="length", completion_tokens=16, tool_calls=[partial], reasoning=None
    )
    done = chat_reply(answer="done", finish_reason="stop", completion_tokens=1, tool_calls=None, reasoning=None)
    script, sent, ran = [cut, done], [], []

    async def _generate(client, **request):
        sent.append(request["messages"])
        return script.pop(0)

    monkeypatch.setattr(eval_runner, "chat_completion", _generate)
    registry = NativeToolRegistry()
    registry.register(NativeTool(name="echo", description="echo", parameters=[], handler=lambda **a: ran.append(a)))
    traj = await run_episode(
        NativeToolUseEnvironment(tool_registry=registry),
        "task",
        {},
        client=object(),
        rollout=RolloutConfig(model_name="m", temperature=0.0, max_tokens=16),
    )

    salvaged = {"id": "c1", "type": "function", "function": {"name": "echo", "arguments": '{"x": "# still re'}}
    assert ran == [] and traj.info["total_tool_calls"] == 0
    assert [m.cut_tool_calls for m in traj.messages if m.role == "assistant"] == [[salvaged], None]
    assert next(m for m in sent[1] if m["role"] == "assistant") == {"role": "assistant", "content": "Computing"}
    assert "still re" not in json.dumps(sent[1])
    recorded = [m for m in serialize_trajectory(traj)["messages"] if m["role"] == "assistant"]
    assert recorded[0]["cut_tool_calls"] == [salvaged] and "tool_calls" not in recorded[0]
    assert "cut_tool_calls" not in recorded[1]


async def test_eval_records_the_reasoning_channel_without_writing_it_to_the_jsonl(monkeypatch):
    """``serialize_trajectory`` renders through the plain ``to_dict``; ``include_thinking`` would put every
    episode's CoT into the file that gets shared."""
    env = _tooled_env()
    final = chat_reply(
        answer="done",
        finish_reason="stop",
        completion_tokens=2,
        tool_calls=None,
        reasoning="secret CoT",
    )
    traj = await _run_scripted_episode(env, [final], monkeypatch)

    assert [m.thinking for m in traj.messages if m.role == "assistant"] == ["secret CoT"]
    assert "secret CoT" not in json.dumps(serialize_trajectory(traj))


async def test_run_episode_reads_the_tool_schema_after_reset(monkeypatch):
    """An MCP environment registers its tools at the first reset, so an earlier read sends ``tools=None`` and scores
    a toolless episode as the model's failure."""
    env = _tooled_env()
    seen: dict[str, Any] = {}
    real_reset, real_schema = env.reset, env.get_tools_schema

    def _reset(prompts, contexts):
        seen["reset_done"] = True
        return real_reset(prompts, contexts)

    def _schema():
        return real_schema() if seen.get("reset_done") else []

    async def _generate(client, **request):
        seen["tools"] = request.get("tools")
        return chat_reply(answer="done", finish_reason="stop", completion_tokens=1, tool_calls=None, reasoning=None)

    monkeypatch.setattr(env, "reset", _reset)
    monkeypatch.setattr(env, "get_tools_schema", _schema)
    monkeypatch.setattr(eval_runner, "chat_completion", _generate)
    await run_episode(
        env, "task", {}, client=object(), rollout=RolloutConfig(model_name="m", temperature=0.0, max_tokens=16)
    )
    assert seen["tools"], "the generation ran without the tools the environment registers on reset"


async def test_a_random_level_is_drawn_from_the_problem_so_reruns_agree(monkeypatch):
    """A rerun or the next checkpoint's eval scores each problem at the level it got before. ``random.choice`` walks
    the levels in turn here, so a per-episode draw cannot match by chance."""
    env = _tooled_env()
    env.reasoning_effort = RANDOM_REASONING_EFFORT
    turn = itertools.count()
    monkeypatch.setattr(random, "choice", lambda seq: seq[next(turn) % len(seq)])
    levels: list[str] = []
    real_reset = env.reset

    def _reset(prompts, contexts):
        levels.append(contexts[0]["reasoning_effort"])
        return real_reset(prompts, contexts)

    async def _generate(client, **request):
        return chat_reply(answer="done", finish_reason="stop", completion_tokens=1, tool_calls=None, reasoning=None)

    monkeypatch.setattr(env, "reset", _reset)
    monkeypatch.setattr(eval_runner, "chat_completion", _generate)
    rollout = RolloutConfig(model_name="m", temperature=0.0, max_tokens=16)
    problems = [f"problem {i}" for i in range(12)]
    for _ in range(2):
        for problem in problems:
            await run_episode(env, problem, {}, client=object(), rollout=rollout)
    stable = [stable_reasoning_effort(problem) for problem in problems]
    assert len(set(stable)) > 1, "every problem draws one level, so matching it proves nothing"
    assert levels == stable * 2


def test_a_conversation_with_no_user_turn_has_no_level_to_draw():
    """The draw is keyed by the task; drawn as an empty task, every such problem would share one level."""
    with pytest.raises(ValueError, match="no 'user' message"):
        stable_reasoning_effort([{"role": "system", "content": "frame"}])


async def test_a_captured_turn_records_the_reasoning_it_sampled(monkeypatch):
    """Under a resolved reasoning close the eval asks for the sampled ids and counts each turn's reasoning up to
    the close, as training does, so ``episode/thinking_cap_turns`` reads the same off an eval record."""
    end, sent = 7, []

    async def _chat(client, **request):
        sent.append(request)
        return chat_reply("done", completion_tokens=6, token_ids=[3, 3, 3, end, 9, 9])

    monkeypatch.setattr(eval_runner, "chat_completion", _chat)
    rollout = RolloutConfig(
        model_name="m", max_tokens=16, max_thinking_tokens=8, capture_token_ids=True, reasoning_end_token_id=end
    )
    traj = await run_episode(_tooled_env(), "task", {}, client=object(), rollout=rollout)
    turn = next(m for m in serialize_trajectory(traj)["messages"] if m["role"] == "assistant")
    assert (turn["reasoning_tokens"], turn["thinking_cap"]) == (4, 8)
    assert sent[0]["extra_body"]["logprobs"] is True and sent[0]["extra_body"]["return_token_ids"] is True


async def test_a_call_cut_at_the_cap_is_recorded_as_the_policy_wrote_it(monkeypatch):
    """As in training, the eval decodes the ids a cut turn wrote past its reasoning close (vLLM's ``/detokenize``,
    at the server root under the client's ``/v1`` base) and records them as the cut call's arguments."""
    end, ids, decoded = 99, [11, 99, 12, 99, 21, 22, 23], []
    partial = types.SimpleNamespace(id="c1", function=types.SimpleNamespace(name="echo", arguments='{"x": "1+'))
    cut = chat_reply("", finish_reason="tool_calls", completion_tokens=len(ids), tool_calls=[partial], token_ids=ids)

    class _Client:
        base_url = httpx.URL("http://x:8000/v1/")

        async def post(self, url, *, cast_to, body, options):
            decoded.append((url, body["tokens"]))
            return types.SimpleNamespace(json=lambda: {"prompt": "<call>echo 1+ # cut"})

    monkeypatch.setattr(eval_runner, "chat_completion", scripted_chat([cut, chat_reply("done")]))
    rollout = RolloutConfig(model_name="m", max_tokens=len(ids), capture_token_ids=True, reasoning_end_token_id=end)
    traj = await run_episode(_tooled_env(), "task", {}, client=_Client(), rollout=rollout)
    turn = next(m for m in traj.messages if m.role == "assistant")
    assert decoded == [("http://x:8000/detokenize", [21, 22, 23])]
    assert turn.cut_tool_calls[0]["function"]["arguments"] == "<call>echo 1+ # cut"


async def test_run_episode_cleans_up_on_mid_episode_exception(monkeypatch):
    """Otherwise every failed episode leaks its trajectory and sandbox session for the run's lifetime."""
    env = _tooled_env()

    async def _boom(client, **request):
        raise RuntimeError("boom")

    monkeypatch.setattr(eval_runner, "chat_completion", _boom)
    monkeypatch.setattr(env, "finalize_truncated", lambda eids: (_ for _ in ()).throw(RuntimeError("finalize boom")))
    with pytest.raises(RuntimeError, match="finalize boom"):
        await run_episode(
            env, "task", {}, client=object(), rollout=RolloutConfig(model_name="m", temperature=0.0, max_tokens=16)
        )
    assert env._trajectories == {}


async def test_collect_results_forwards_the_whole_rollout_contract(monkeypatch):
    """``top_p`` hardest: omitted, the eval samples at the server's default nucleus while training uses
    ``rollout_top_p``, measuring a policy the trainer never optimized."""
    seen: list[dict] = []

    async def _capture(client, **request):
        seen.append({"client": client, **request})
        return chat_reply(answer="done", finish_reason="stop", completion_tokens=1, tool_calls=None, reasoning=None)

    monkeypatch.setattr(eval_runner, "chat_completion", _capture)
    rollout = RolloutConfig(
        model_name="served-model", temperature=0.3, top_p=0.71, max_tokens=128, request_timeout=42.0
    )
    results = await collect_results(
        _tooled_env(), [{"prompt": "q", "context": {}}], client=object(), rollout=rollout, num_samples=1
    )

    assert len(seen) == 1, seen
    assert seen[0]["top_p"] == pytest.approx(0.71)
    assert seen[0]["temperature"] == pytest.approx(0.3)
    assert seen[0]["max_tokens"] == 128
    assert seen[0]["model"] == "served-model"
    assert seen[0]["timeout"] == pytest.approx(42.0)
    assert results[0]["samples"][0]["success"] is True


async def test_a_conversation_prompt_hands_the_environment_its_last_user_turn(monkeypatch):
    """The whole list would open the episode on the dataset's framing turns and grade a different task than
    training."""
    convo = [
        {"role": "system", "content": "dataset framing"},
        {"role": "user", "content": "an earlier turn"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "the task"},
    ]
    done = chat_reply(answer="done", finish_reason="stop", completion_tokens=1, tool_calls=None, reasoning=None)
    monkeypatch.setattr(eval_runner, "chat_completion", scripted_chat([done]))
    (row,) = await collect_results(
        _tooled_env(),
        [{"prompt": convo, "context": {}}],
        client=object(),
        rollout=_RETRYING,
        collect_trajectories=True,
    )
    messages = row["samples"][0]["trajectory"]["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [("user", "the task"), ("assistant", "done")]


async def test_a_conversation_without_a_user_turn_is_refused_before_any_episode(monkeypatch):
    async def _never(client, **request):
        raise AssertionError("no episode may run for a prompt with no task in it")

    monkeypatch.setattr(eval_runner, "chat_completion", _never)
    with pytest.raises(ValueError, match="no 'user' message"):
        await collect_results(
            _tooled_env(),
            [{"prompt": [{"role": "system", "content": "s"}], "context": {}}],
            object(),
            rollout=_RETRYING,
        )


async def test_a_content_part_user_turn_hands_the_environment_its_text(monkeypatch):
    """Handed the part list, the environment's reset reads each part as a message and raises ``KeyError: 'role'``."""
    convo = [
        {"role": "system", "content": "dataset framing"},
        {"role": "user", "content": [{"type": "text", "text": "the "}, {"type": "text", "text": "task"}]},
    ]
    done = chat_reply(answer="done", finish_reason="stop", completion_tokens=1, tool_calls=None, reasoning=None)
    monkeypatch.setattr(eval_runner, "chat_completion", scripted_chat([done]))
    (row,) = await collect_results(
        _tooled_env(),
        [{"prompt": convo, "context": {}}],
        client=object(),
        rollout=_RETRYING,
        collect_trajectories=True,
    )
    messages = row["samples"][0]["trajectory"]["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [("user", "the task"), ("assistant", "done")]


async def test_a_content_part_the_environment_cannot_carry_is_refused_before_any_episode(monkeypatch):
    """An environment's conversation is text, so an image part would be dropped and the task graded without it."""

    async def _never(client, **request):
        raise AssertionError("no episode may run for a task the environment cannot carry")

    monkeypatch.setattr(eval_runner, "chat_completion", _never)
    content = [{"type": "text", "text": "what is shown?"}, {"type": "image_url", "image_url": {"url": "x"}}]
    with pytest.raises(ValueError, match=r"other than text \(\['image_url'\]\)"):
        await collect_results(
            _tooled_env(),
            [{"prompt": [{"role": "user", "content": content}], "context": {}}],
            object(),
            rollout=_RETRYING,
        )


async def test_a_recorded_native_episode_carries_its_tool_calls_once(monkeypatch):
    """A second log in ``info`` would be pickled through Ray and the TP broadcast and written to the JSONL for
    nothing."""
    done = chat_reply(answer="done", finish_reason="stop", completion_tokens=1, tool_calls=None, reasoning=None)
    monkeypatch.setattr(eval_runner, "chat_completion", scripted_chat([_tool_call_response(), done]))
    (row,) = await collect_results(
        _tooled_env(), [{"prompt": "q", "context": {}}], client=object(), rollout=_RETRYING, collect_trajectories=True
    )
    trajectory = row["samples"][0]["trajectory"]
    assert [m["role"] for m in trajectory["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert trajectory["messages"][1]["tool_calls"][0]["id"] == "c1" and trajectory["messages"][2]["content"] == "ok"
    assert not {"tool_calls", "tool_results"} & trajectory["info"].keys()


async def test_collect_results_omits_the_model_for_a_single_model_endpoint(monkeypatch):
    """A single-model server 404s a null model, so unset sends NOT_GIVEN."""
    seen: list[dict] = []

    async def _capture(client, **request):
        seen.append({"client": client, **request})
        return chat_reply(answer="done", finish_reason="stop", completion_tokens=1, tool_calls=None, reasoning=None)

    monkeypatch.setattr(eval_runner, "chat_completion", _capture)
    await collect_results(
        _tooled_env(), [{"prompt": "q", "context": {}}], client=object(), rollout=RolloutConfig(), num_samples=1
    )
    assert seen[0]["model"] is NOT_GIVEN


async def test_eval_driver_keeps_contextvar_writes_across_turns(monkeypatch):
    """Per-call ``asyncio.to_thread`` copies a fresh context, so a file written on turn 1 would be gone on turn 2,
    which online (one context per episode) keeps."""
    registry = build_simulated_file_tools()
    env = NativeToolUseEnvironment(tool_registry=registry, max_turns=3)

    def _call(name, **arguments):
        fn = types.SimpleNamespace(name=name, arguments=json.dumps(arguments))
        return chat_reply(
            answer="",
            finish_reason="tool_calls",
            completion_tokens=1,
            tool_calls=[types.SimpleNamespace(id=f"c-{name}", function=fn)],
            reasoning=None,
        )

    monkeypatch.setattr(
        eval_runner,
        "chat_completion",
        scripted_chat(
            [
                _call("write_file", path="/home/user/x.txt", content="persisted"),
                _call("read_file", path="/home/user/x.txt"),
            ]
        ),
    )
    traj = await run_episode(
        env, "task", {}, client=object(), rollout=RolloutConfig(model_name="m", temperature=0.0, max_tokens=16)
    )
    observations = [m.content for m in traj.messages if m.role == "tool"]
    assert observations[-1] == "persisted", observations


def test_summarize_empty_results_fails_loud():
    with pytest.raises(ValueError, match="zero episodes"):
        summarize([], num_samples=1)


def test_summarize_counts_the_samples_that_carry_no_signal():
    """An invalid grade or a raised episode scores 0 and stays in the means; counted apart, a sandbox outage does
    not read as a weak model."""
    rows = [
        {"samples": [{"reward": 1.0, "success": True}, {"reward": 0.0, "success": False, "error": "outage"}]},
        {"samples": [{"reward": 0.0, "success": False}, {"reward": 0.0, "success": False, "error": "raised"}]},
    ]
    summary = summarize(rows, num_samples=2)
    assert summary["invalid"] == 2
    assert summary["mean_reward"] == pytest.approx(0.25)


def _status_error(error_cls: type[APIStatusError], status: int, message: str) -> APIStatusError:
    body = json.dumps({"object": "error", "message": message, "type": error_cls.__name__, "code": status})
    response = httpx.Response(status, text=body, request=httpx.Request("POST", "http://x/v1/chat/completions"))
    return error_cls(f"Error code: {status} - {body}", response=response, body=json.loads(body))


def _bad_request(message: str) -> BadRequestError:
    return _status_error(BadRequestError, 400, message)


def _final(answer="done"):
    return chat_reply(answer=answer, finish_reason="stop", completion_tokens=2, tool_calls=None, reasoning=None)


_RETRYING = RolloutConfig(model_name="m", temperature=0.0, max_tokens=16, max_retries=2, retry_base_wait=0.0)


async def _episode_under_retries(env, script, monkeypatch, rollout=_RETRYING):
    calls = []
    scripted = scripted_chat(script)

    async def _counting(client, **request):
        calls.append({"client": client, **request})
        return await scripted(client, **request)

    monkeypatch.setattr(eval_runner, "chat_completion", _counting)
    traj = await run_episode(env, "task", {}, client=object(), rollout=rollout)
    return traj, len(calls)


async def test_an_engine_serialization_fault_is_retried_like_a_training_rollout(monkeypatch):
    """A 400 for a NaN log-prob the engine could not serialise is its own fault: training retries it, so the eval
    does too."""
    fault = _bad_request("Out of range float values are not JSON compliant: nan")
    traj, calls = await _episode_under_retries(_tooled_env(), [fault, _final()], monkeypatch)

    assert calls == 2
    assert EPISODE_ERROR_KEY not in traj.info
    assert traj.info["completed"] is True and traj.total_reward == pytest.approx(1.0)


async def test_an_engine_abort_is_re_issued_not_stepped(monkeypatch):
    aborted = chat_reply(answer="partial", finish_reason="abort", completion_tokens=3, tool_calls=None, reasoning=None)
    traj, calls = await _episode_under_retries(_tooled_env(), [aborted, _final()], monkeypatch)

    assert calls == 2
    assert traj.info["completed"] is True
    assert traj.info.get("length_cutoff_turns", 0) == 0, "the abort fragment reached the environment"


async def test_a_genuine_client_error_is_terminal(monkeypatch):
    traj, calls = await _episode_under_retries(
        _tooled_env(), [_bad_request("max_tokens must be positive"), _final()], monkeypatch
    )

    assert calls == 1
    assert "BadRequestError" in traj.info[EPISODE_ERROR_KEY]


_TIMEOUT = APITimeoutError(request=httpx.Request("POST", "http://x/v1/chat/completions"))


@pytest.mark.parametrize(
    "failure",
    [_bad_request("This model's maximum context length is 4096 tokens"), _TIMEOUT],
    ids=["context-overflow", "timeout"],
)
async def test_a_failure_the_episode_itself_caused_is_graded_not_dropped(monkeypatch, failure):
    """Context overflow or timeouts follow the episode's own length; dropping them would score only episodes short
    enough to finish."""
    monkeypatch.setattr(eval_runner, "chat_completion", scripted_chat([_tool_call_response(), failure]))

    (row,) = await collect_results(
        _tooled_env(), [{"prompt": "q", "context": {}}], client=object(), rollout=_RETRYING, num_samples=1
    )

    (sample,) = row["samples"]
    assert eval_runner.GENERATION_ERROR_KEY not in sample
    assert sample["success"] is False and sample["reward"] == pytest.approx(0.05)
    assert summarize([row], num_samples=1)["generation_errors"] == 0


@pytest.mark.parametrize(
    "failure",
    [
        _status_error(NotFoundError, 404, "The model `m` does not exist."),
        _status_error(AuthenticationError, 401, "Invalid API key"),
        _status_error(PermissionDeniedError, 403, "Forbidden"),
        _bad_request("max_tokens must be positive"),
    ],
    ids=["unknown-model", "rejected-key", "forbidden", "malformed-request"],
)
async def test_a_client_error_the_episode_did_not_cause_is_a_generation_error(monkeypatch, failure):
    """An unknown model or route, a rejected key, a request built wrong says nothing about the episode; graded as a
    miss it scores misconfiguration as the policy's failure."""
    monkeypatch.setattr(eval_runner, "chat_completion", scripted_chat([_tool_call_response(), failure]))

    (row,) = await collect_results(
        _tooled_env(), [{"prompt": "q", "context": {}}], client=object(), rollout=_RETRYING, num_samples=1
    )

    (sample,) = row["samples"]
    assert type(failure).__name__ in sample[eval_runner.GENERATION_ERROR_KEY]
    assert sample["reward"] is None and sample["success"] is None
    assert summarize([row], num_samples=1)["generation_errors"] == 1


async def test_an_engine_fault_past_the_retries_is_a_generation_error_row(monkeypatch):
    fault = _bad_request("Out of range float values are not JSON compliant: nan")
    monkeypatch.setattr(eval_runner, "chat_completion", scripted_chat([fault] * 3))

    (row,) = await collect_results(
        _tooled_env(), [{"prompt": "q", "context": {}}], client=object(), rollout=_RETRYING, num_samples=1
    )

    (sample,) = row["samples"]
    assert "BadRequestError" in sample[eval_runner.GENERATION_ERROR_KEY]
    assert sample["reward"] is None and sample["success"] is None, "a lost generation is not a scored miss"
    assert "error" not in sample, "a generation error is not an invalid grade"


def test_generation_errors_leave_every_mean_and_are_counted():
    lost = {"reward": None, "success": None, eval_runner.GENERATION_ERROR_KEY: "BadRequestError: nan"}
    rows = [
        {"samples": [lost, {"reward": 1.0, "success": True}]},
        {"samples": [{"reward": 0.0, "success": False}, {"reward": 0.5, "success": False}]},
        {"samples": [lost, lost]},
    ]
    summary = summarize(rows, num_samples=2)

    assert summary["generation_errors"] == 3
    assert summary["invalid"] == 0
    assert summary["success@1"] == pytest.approx(0.5), "each row's first scored sample: solved, then missed"
    assert summary["success@2"] == pytest.approx(0.5), "the all-error row leaves the denominator"
    assert summary["mean_reward"] == pytest.approx((1.0 + 0.25) / 2)


def test_success_at_one_never_exceeds_success_at_k():
    lost = {"reward": None, "success": None, eval_runner.GENERATION_ERROR_KEY: "RuntimeError: gone"}
    rows = [
        {"samples": [{"reward": 1.0, "success": True}, lost]},
        {"samples": [lost, {"reward": 0.0, "success": False}]},
    ]
    summary = summarize(rows, num_samples=2)
    assert summary["success@1"] == pytest.approx(0.5) and summary["success@2"] == pytest.approx(0.5)


def test_a_bucket_of_only_generation_errors_reads_nan_not_zero():
    lost = {"reward": None, "success": None, eval_runner.GENERATION_ERROR_KEY: "RuntimeError: gone"}
    summary = summarize([{"samples": [lost]}], num_samples=1)
    assert math.isnan(summary["success@1"]) and math.isnan(summary["mean_reward"])
    assert summary["generation_errors"] == 1


class _VerdictEnv(NativeToolUseEnvironment):
    """A native env that reports the solve verdict ``solved`` whatever its shaped reward."""

    solved = 1.0

    def rollout_metrics(self, trajectory):
        return {**super().rollout_metrics(trajectory), SOLVE_RATE_KEY: self.solved}


def _cut_then_final():
    cut = chat_reply(
        answer="a thought that ran",
        finish_reason="length",
        completion_tokens=9,
        tool_calls=None,
        reasoning=None,
    )
    return [cut, _final()]


async def _one_sample(env, monkeypatch, success_threshold=1.0):
    monkeypatch.setattr(eval_runner, "chat_completion", scripted_chat(_cut_then_final()))
    (row,) = await collect_results(
        env,
        [{"prompt": "q", "context": {}}],
        client=object(),
        rollout=RolloutConfig(model_name="m", temperature=0.0, max_tokens=16),
        success_threshold=success_threshold,
    )
    return row["samples"][0]


def _verdict_env(solved: float) -> _VerdictEnv:
    registry = NativeToolRegistry()
    registry.register(NativeTool(name="echo", description="echo", parameters=[], handler=lambda **a: "ok"))
    env = _VerdictEnv(tool_registry=registry, tool_success_reward=0.05, length_cutoff_penalty=0.2)
    env.solved = solved
    return env


async def test_a_solve_that_paid_a_shaping_penalty_counts_on_the_envs_verdict(monkeypatch):
    sample = await _one_sample(_verdict_env(solved=1.0), monkeypatch)
    assert sample["reward"] == pytest.approx(0.8), "the recovered cut must have paid its penalty"
    assert sample["success"] is True, "a solved episode read as unsolved through its shaped total"


async def test_the_verdict_also_refuses_a_total_that_clears_the_threshold(monkeypatch):
    sample = await _one_sample(_verdict_env(solved=0.0), monkeypatch, success_threshold=0.5)
    assert sample["reward"] >= 0.5
    assert sample["success"] is False


async def test_without_a_verdict_the_threshold_decides(monkeypatch):
    sample = await _one_sample(_tooled_env(length_cutoff_penalty=0.2), monkeypatch)
    assert sample["reward"] == pytest.approx(0.8)
    assert sample["success"] is False


def test_report_states_the_generation_error_count(caplog):
    lost = {"reward": None, "success": None, eval_runner.GENERATION_ERROR_KEY: "RuntimeError: gone"}
    with caplog.at_level(logging.INFO, logger="halo.environments.eval_runner"):
        report([{"samples": [lost, {"reward": 1.0, "success": True}]}], num_samples=2, title="lost")
    assert "generation_errors=1" in caplog.records[-1].getMessage()


def test_report_states_the_invalid_count_even_when_none_is(caplog):
    clean = [{"samples": [{"reward": 1.0, "success": True}, {"reward": 0.0, "success": False}]}]
    lost = [{"samples": [{"reward": 0.0, "success": False, "error": "outage"}]}]
    with caplog.at_level(logging.INFO, logger="halo.environments.eval_runner"):
        report(clean, num_samples=2, title="clean")
        report(lost, num_samples=1, title="lost")
    clean_log, lost_log = (r.getMessage() for r in caplog.records)
    assert "invalid=0" in clean_log and "invalid=1" in lost_log


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
