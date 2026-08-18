#!/usr/bin/env python
"""CPU tests for the shared eval runner's dataset loading (load_hf_split) and trajectory recording
(serialize_trajectory + write_trajectories_jsonl).

The critical invariant: a recorded trajectory must keep the conversation and the grading verdict but
must NOT leak the answer key (the hidden test cases live in ``info["_test_cases"]`` / the graded
``info["context"]["answer"]`` payload). A drop-list that stops covering one of those keys writes it
into the recorded file.

Run:
    python tests/cpu/environments/test_eval_runner.py
"""

import json
import sys
import types
from typing import Any

import pytest
from datasets import Dataset, DatasetDict
from openai import NOT_GIVEN

import src.environments.eval_runner as eval_runner
from src.configs.rollout_config import RolloutConfig
from src.environments.base import Message, Trajectory
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.eval_runner import (
    collect_results,
    load_hf_split,
    run_episode,
    serialize_trajectory,
    summarize,
    trajectory_path,
    write_trajectories_jsonl,
)
from src.environments.tools.definitions import NativeTool, NativeToolRegistry
from src.environments.tools.factories import create_native_file_tools


def test_trajectory_path_slugifies_per_run():
    p = trajectory_path("/tmp/traj", "anthropic/claude-opus-4.8", "codeforces", "test", "python")
    assert p == "/tmp/traj/anthropic-claude-opus-4.8__codeforces__test__python.jsonl"


def _leaky_traj() -> Trajectory:
    """A finished trajectory whose info carries both the verdict AND the answer key, to prove the
    serializer keeps the former and drops the latter."""
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
            "tool_calls": [{"name": "submit_solution"}],  # raw per-turn log
            "_eval_stats": {"generations": 1, "completion_tokens": 50},
        },
    )


def test_serialize_trajectory_keeps_messages_and_verdict():
    s = serialize_trajectory(_leaky_traj())
    assert [m["role"] for m in s["messages"]] == ["system", "user", "assistant"]
    assert s["total_reward"] == 1.0 and s["done"] is True and s["truncated"] is False
    assert s["info"]["tests_passed"] == 2 and s["info"]["submission_result"] == "Passed 2/2 test cases."
    assert s["info"]["eval_stats"]["completion_tokens"] == 50  # telemetry renamed from _eval_stats


def test_serialize_trajectory_drops_answer_key_and_internals():
    s = serialize_trajectory(_leaky_traj())
    assert "_test_cases" not in s["info"]
    assert "context" not in s["info"]
    assert "tool_calls" not in s["info"]
    assert "EXPECTED_OUTPUT_42" not in json.dumps(s)
    assert serialize_trajectory(None) is None


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


# run_episode failure handling


def _tooled_env(**kw) -> NativeToolUseEnvironment:
    registry = NativeToolRegistry()
    registry.register(NativeTool(name="echo", description="echo", parameters=[], handler=lambda **a: "ok"))
    kw.setdefault("success_reward", 1.0)
    kw.setdefault("failure_reward", 0.0)
    kw.setdefault("tool_success_reward", 0.05)
    return NativeToolUseEnvironment(tool_registry=registry, **kw)


def _tool_call_response():
    fn = types.SimpleNamespace(name="echo", arguments="{}")
    return types.SimpleNamespace(
        answer="",
        finish_reason="tool_calls",
        completion_tokens=3,
        tool_calls=[types.SimpleNamespace(id="c1", function=fn)],
        reasoning=None,
    )


def _scripted_generate(script):
    """Return an async stand-in for generate_openai_response driven by a list of responses/exceptions."""
    queue = list(script)

    async def _generate(**kwargs):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return _generate


async def _run_scripted_episode(env, script, monkeypatch):
    monkeypatch.setattr(eval_runner, "generate_openai_response", _scripted_generate(script))
    return await run_episode(
        env, "task", {}, client=object(), rollout=RolloutConfig(model_name="m", temperature=0.0, max_tokens=16)
    )


async def test_generation_failure_is_truncation_not_completion(monkeypatch):
    """An episode that dies on a generation error must NOT take the plain-text terminal path: that
    marks it ``completed`` and pays completion-rewarded envs full ``success_reward`` for an HTTP error."""
    env = _tooled_env()
    traj = await _run_scripted_episode(env, [RuntimeError("http boom")], monkeypatch)
    assert traj.done and traj.truncated
    assert traj.info["completed"] is False
    assert traj.total_reward == pytest.approx(env.failure_reward)


async def test_generation_failure_keeps_earned_tool_reward(monkeypatch):
    """A mid-episode death preserves what the episode already earned (per-call tool rewards), it just
    doesn't add the completion payout."""
    env = _tooled_env()
    traj = await _run_scripted_episode(env, [_tool_call_response(), RuntimeError("http boom")], monkeypatch)
    assert traj.done and traj.truncated and traj.info["completed"] is False
    assert traj.info["successful_tool_calls"] == 1
    assert traj.total_reward == pytest.approx(0.05)


async def test_completed_episode_still_pays_success(monkeypatch):
    """The happy path is untouched: a real final text answer completes and earns success_reward."""
    env = _tooled_env()
    final = types.SimpleNamespace(
        answer="done", finish_reason="stop", completion_tokens=2, tool_calls=None, reasoning=None
    )
    traj = await _run_scripted_episode(env, [final], monkeypatch)
    assert traj.done and not traj.truncated
    assert traj.info["completed"] is True
    assert traj.total_reward == pytest.approx(1.0)


async def test_length_cutoff_is_recovered_offline_exactly_as_online(monkeypatch):
    """The eval must stamp the engine's ``finish_reason`` into the step context the way the training
    rollout does (``EnvironmentActor.run_episode``).

    Without it a turn the engine cut off at its token cap arrives as an ordinary text answer: the env
    finalizes the mid-sentence fragment and grades it as the deliberate one, so an offline eval scores
    what online would have nudged and retried — diverging the identical online/offline verdict the
    grading module is built to promise."""
    env = _tooled_env()
    cut_off = types.SimpleNamespace(
        answer="a thought that ran out of",
        finish_reason="length",
        completion_tokens=9,
        tool_calls=None,
        reasoning=None,
    )
    final = types.SimpleNamespace(
        answer="done", finish_reason="stop", completion_tokens=2, tool_calls=None, reasoning=None
    )
    traj = await _run_scripted_episode(env, [cut_off, final], monkeypatch)

    assert traj.info["length_cutoff_turns"] == 1
    assert traj.info["final_response"] == "done"  # the fragment was never scored as the answer
    assert NativeToolUseEnvironment.LENGTH_CUTOFF_NUDGE in [m.content for m in traj.messages]
    # The same flag the trainer reads to skip a fragment's row.
    assert [m.truncated for m in traj.messages if m.role == "assistant"] == [True, False]
    assert traj.total_reward == pytest.approx(1.0)


async def test_eval_records_the_reasoning_channel_without_writing_it_to_the_jsonl(monkeypatch):
    """The eval stamps reasoning onto the assistant turn exactly as the training rollout does — and
    the persisted trajectory still ships no chain-of-thought.

    ``serialize_trajectory`` renders messages through the plain ``to_dict``, the same reasoning-free
    form the generation request uses; flipping it to ``include_thinking`` would put every episode's
    CoT into the ``--save_trajectories`` file, and the file is what gets shared.
    """
    env = _tooled_env()
    final = types.SimpleNamespace(
        answer="done", finish_reason="stop", completion_tokens=2, tool_calls=None, reasoning="secret CoT"
    )
    traj = await _run_scripted_episode(env, [final], monkeypatch)

    assert [m.thinking for m in traj.messages if m.role == "assistant"] == ["secret CoT"]
    assert "secret CoT" not in json.dumps(serialize_trajectory(traj))


async def test_run_episode_reads_the_tool_schema_after_reset(monkeypatch):
    """The schema is read after ``reset``: an MCP environment only registers its tools when the first
    reset connects to the server, so a read before it would send ``tools=None`` and score a toolless
    episode as the model's failure."""
    env = _tooled_env()
    seen: dict[str, Any] = {}
    real_reset, real_schema = env.reset, env.get_tools_schema

    def _reset(prompts, contexts):
        seen["reset_done"] = True
        return real_reset(prompts, contexts)

    def _schema():
        return real_schema() if seen.get("reset_done") else []

    async def _generate(**kwargs):
        seen["tools"] = kwargs["tools"]
        return types.SimpleNamespace(
            answer="done", finish_reason="stop", completion_tokens=1, tool_calls=None, reasoning=None
        )

    monkeypatch.setattr(env, "reset", _reset)
    monkeypatch.setattr(env, "get_tools_schema", _schema)
    monkeypatch.setattr(eval_runner, "generate_openai_response", _generate)
    await run_episode(
        env, "task", {}, client=object(), rollout=RolloutConfig(model_name="m", temperature=0.0, max_tokens=16)
    )
    assert seen["tools"], "the generation ran without the tools the environment registers on reset"


async def test_run_episode_cleans_up_on_mid_episode_exception(monkeypatch):
    """An exception escaping the episode loop must still release the trajectory (try/finally), else
    every failed episode leaks its trajectory + sandbox session for the run's lifetime."""
    env = _tooled_env()

    async def _boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(eval_runner, "generate_openai_response", _boom)
    monkeypatch.setattr(env, "finalize_truncated", lambda eids: (_ for _ in ()).throw(RuntimeError("finalize boom")))
    with pytest.raises(RuntimeError, match="finalize boom"):
        await run_episode(
            env, "task", {}, client=object(), rollout=RolloutConfig(model_name="m", temperature=0.0, max_tokens=16)
        )
    assert env._trajectories == {}


def test_load_hf_split_reads_a_bare_saved_dataset(tmp_path):
    # save_to_disk on a bare Dataset writes one unnamed split; DatasetDict handling dies on .keys().
    path = str(tmp_path / "bare")
    Dataset.from_list([{"prompt": "p", "answer": "a"}]).save_to_disk(path)
    split = load_hf_split(path, None, "test")
    assert isinstance(split, Dataset)
    assert split[0] == {"prompt": "p", "answer": "a"}


def test_load_hf_split_reads_a_hub_id_from_the_hub_not_off_disk(tmp_path, monkeypatch):
    """``org/name`` is a Hub id even where a directory of that name sits in the working directory.

    Classifying by an existence probe reads whatever local copy happens to shadow the id the caller
    named — and on a non-shared filesystem it answers differently per rank.
    """
    shadow = tmp_path / "org" / "name"
    shadow.mkdir(parents=True)
    Dataset.from_list([{"prompt": "local-row"}]).save_to_disk(str(shadow))
    monkeypatch.chdir(tmp_path)

    seen: dict = {}

    def fake_load_dataset(path, config=None, *, split=None, **_kwargs):
        seen.update(path=path, config=config, split=split)
        return Dataset.from_list([{"prompt": "hub-row"}])

    monkeypatch.setattr(eval_runner, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(eval_runner, "load_from_disk", lambda *a, **k: pytest.fail("a Hub id was read off disk"))

    assert load_hf_split("org/name", None, "test")[0]["prompt"] == "hub-row"
    assert seen == {"path": "org/name", "config": None, "split": "test"}


def test_load_hf_split_refuses_an_s3_uri(monkeypatch):
    """An S3 URI names neither of the two sources this loader reads, and must say so — handed to the
    Hub loader instead, it fails on the Hub's name for a path the caller spelled correctly."""
    monkeypatch.setattr(eval_runner, "load_dataset", lambda *a, **k: pytest.fail("an S3 URI reached the Hub loader"))
    monkeypatch.setattr(eval_runner, "load_from_disk", lambda *a, **k: pytest.fail("an S3 URI was read off disk"))

    with pytest.raises(ValueError, match="S3 URI"):
        load_hf_split("s3://bucket/key", None, "train")


# collect_results — the generation contract every sample is drawn under


async def test_collect_results_forwards_the_whole_rollout_contract(monkeypatch):
    """Every sampling knob on the RolloutConfig must reach the request.

    ``top_p`` is the one this pins hardest: it was omitted at this call for the whole life of the
    runner, so every eval sampled at the SERVER's default nucleus while training sampled at
    ``rollout_top_p`` — the eval measured a policy the trainer never optimized, and nothing in the
    output said so. The rest ride along because the contract is passed as one object.
    """
    seen: list[dict] = []

    async def _capture(**kwargs):
        seen.append(kwargs)
        return types.SimpleNamespace(
            answer="done", finish_reason="stop", completion_tokens=1, tool_calls=None, reasoning=None
        )

    monkeypatch.setattr(eval_runner, "generate_openai_response", _capture)
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
    assert seen[0]["request_timeout"] == pytest.approx(42.0)
    assert results[0]["samples"][0]["success"] is True


async def test_collect_results_omits_the_model_for_a_single_model_endpoint(monkeypatch):
    """``model_name`` unset must send NOT_GIVEN, not ``None``: a single-model server 404s a null
    model instead of answering with what it serves."""
    seen: list[dict] = []

    async def _capture(**kwargs):
        seen.append(kwargs)
        return types.SimpleNamespace(
            answer="done", finish_reason="stop", completion_tokens=1, tool_calls=None, reasoning=None
        )

    monkeypatch.setattr(eval_runner, "generate_openai_response", _capture)
    await collect_results(
        _tooled_env(), [{"prompt": "q", "context": {}}], client=object(), rollout=RolloutConfig(), num_samples=1
    )
    assert seen[0]["model"] is NOT_GIVEN


async def test_eval_driver_keeps_contextvar_writes_across_turns(monkeypatch):
    """The eval and training drivers share one dispatcher, so a sync env's ContextVar store must
    survive from turn to turn here too.

    Per-call ``asyncio.to_thread`` copies a FRESH context each time, which re-seeded the simulated
    file tools every turn: a file written on turn 1 was gone on turn 2, so an offline eval scored a
    capability the online rollout (one context per episode) had.
    """
    registry = create_native_file_tools()
    env = NativeToolUseEnvironment(tool_registry=registry, max_turns=3)

    def _call(name, **arguments):
        fn = types.SimpleNamespace(name=name, arguments=json.dumps(arguments))
        return types.SimpleNamespace(
            answer="",
            finish_reason="tool_calls",
            completion_tokens=1,
            tool_calls=[types.SimpleNamespace(id=f"c-{name}", function=fn)],
            reasoning=None,
        )

    monkeypatch.setattr(
        eval_runner,
        "generate_openai_response",
        _scripted_generate(
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


def test_load_hf_split_picks_the_named_split_of_a_dataset_dict(tmp_path):
    path = str(tmp_path / "dict")
    DatasetDict(
        {
            "train": Dataset.from_list([{"prompt": "train-row"}]),
            "test": Dataset.from_list([{"prompt": "test-row"}]),
        }
    ).save_to_disk(path)
    assert load_hf_split(path, None, "test")[0]["prompt"] == "test-row"
    # An absent split falls back to the first one rather than raising.
    assert load_hf_split(path, None, "validation")[0]["prompt"] == "train-row"


def test_summarize_empty_results_fails_loud():
    with pytest.raises(ValueError, match="zero episodes"):
        summarize([], num_samples=1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
