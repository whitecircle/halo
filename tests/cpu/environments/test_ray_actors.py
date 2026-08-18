#!/usr/bin/env python
"""
Tests for Ray actors with local Ray (no GPU/vLLM required).

Run with:
    python tests/cpu/environments/test_ray_actors.py

These tests use Ray in local mode and mock the vLLM HTTP calls.
"""

import asyncio
import json
import sys
from types import SimpleNamespace

# Test: Environment Actor (Direct - Without Ray)


def test_env_reset_yields_one_episode_per_prompt():
    """The actor drives ``env.reset`` per episode, so it must return one id and one step per prompt.

    A reset that collapses or reorders prompts silently misattributes every reward the manager maps
    back positionally.
    """
    from src.environments.registry import create_environment

    env = create_environment("react_math", {"max_turns": 5})
    episode_ids, steps = env.reset(["2+2?", "3+3?"], [{"answer": "4"}, {"answer": "6"}])

    assert len(episode_ids) == len(steps) == 2
    assert len(set(episode_ids)) == 2, "episode ids must be distinct, else trajectories overwrite"
    assert [t.info["episode_id"] for t in env.get_trajectories(episode_ids)] == episode_ids


# Test: Rollout Manager (Mocked Ray)


def test_rollout_manager_round_robins_across_servers():
    """Multi-server rollout spreads load by cycling ``_next_url``. A selector that stopped advancing
    would pin every episode on one engine — same total throughput on paper, one server saturated."""
    from src.environments.ray_actors import RolloutConfig, RolloutManager

    urls = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
    manager = RolloutManager(
        num_workers=1,
        env_type="react_math",
        env_config={},
        server_urls=urls,
        rollout_config=RolloutConfig(),
    )

    picked = [manager._next_url() for _ in range(len(urls) * 2)]
    assert picked == urls * 2, f"expected round-robin over {urls}, got {picked}"


async def test_rollout_manager_start_shutdown():
    """Test RolloutManager start and shutdown with Ray.

    Note: Async Ray actors don't work in local_mode, so we skip the actual
    actor creation test when local_mode would be used.
    """
    import ray

    from src.environments.ray_actors import RolloutConfig, RolloutManager, ray_init_kwargs

    # Skip test if we can only use local mode (async actors not supported)
    # This test requires a real Ray cluster or non-local mode
    try:
        if ray.is_initialized():
            ray.shutdown()
        # ray_init_kwargs keeps the dashboard off (jobs-API CVE surface) and the tempdir short;
        # num_cpus caps Ray's core-count worker prestart on many-core hosts.
        ray.init(**ray_init_kwargs(num_cpus=4))
    except Exception as e:  # ray[default] is a hard dependency — a failed init is a real failure
        raise AssertionError(f"ray.init() failed, so the actor-lifecycle checks below never ran: {e}") from e

    try:
        manager = RolloutManager(
            num_workers=2,
            env_type="react_math",
            env_config={"max_turns": 3},
            server_urls=["http://localhost:8000"],
            rollout_config=RolloutConfig(),
        )

        # Start should create actors
        await manager.start()
        assert manager._started
        assert len(manager._actors) == 2

        # Second start should be idempotent
        await manager.start()
        assert len(manager._actors) == 2

        # Shutdown should clean up
        await manager.shutdown()
        assert not manager._started
        assert len(manager._actors) == 0

    except ray.exceptions.RaySystemError as e:
        if "Async actor" in str(e):
            print("  Skipping: Async actors not supported in local mode")
            return
        raise
    finally:
        ray.shutdown()


# Test: RolloutConfig


def test_rollout_config_defaults():
    """Fields RolloutConfig must NOT carry, and picklability (the actors receive it pickled).

    The default VALUES are pinned against their ``AsyncTrainingConfig`` counterparts in
    ``tests/cpu/config/test_rollout_config_mirror.py``. Echoing them here as well was the second
    source of truth that let the two sides drift 32x on ``max_tokens``.
    """
    from src.environments.ray_actors import RolloutConfig

    config = RolloutConfig()

    # Must NOT have removed fields
    assert not hasattr(config, "top_k")
    assert not hasattr(config, "min_p")
    assert not hasattr(config, "repetition_penalty")
    assert not hasattr(config, "max_concurrent_per_actor")

    # RolloutConfig should be fully picklable (no unpicklable callables)
    import pickle

    pickled = pickle.dumps(config)
    restored = pickle.loads(pickled)
    assert restored.temperature == config.temperature
    assert restored.max_retries == config.max_retries
    assert restored.model_name == config.model_name


# Test: Error Handling


# Test: Bounded Concurrency and Positional Mapping (fake actor pool, no Ray)


def _manager_over_fake_actors(actor, *, num_workers=2, max_concurrent_rollouts=None):
    """A started RolloutManager whose pool is ``actor``, so ``collect_rollouts`` runs without Ray."""
    from src.environments.ray_actors import RolloutConfig, RolloutManager

    manager = RolloutManager(
        num_workers=num_workers,
        env_type="react_math",
        env_config={},
        server_urls=["http://localhost:8000"],
        rollout_config=RolloutConfig(episode_timeout=30.0),
        max_concurrent_rollouts=max_concurrent_rollouts,
    )
    manager._actors = [actor]
    manager._started = True
    return manager


class _RecordingActor:
    """Ray-handle stand-in: ``run_episode.remote(...)`` returns an awaitable episode.

    Records peak in-flight episodes so the manager's own semaphore — not a locally constructed
    one — is what the bound is asserted against.
    """

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.inflight = 0
        self.peak = 0
        self.run_episode = SimpleNamespace(remote=self._remote)

    def _remote(self, *, prompt, context, server_url, config):
        from src.environments.base import Trajectory
        from src.environments.ray_actors import RolloutResult

        async def _episode():
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            try:
                await asyncio.sleep(0.02)
                if self.fail:
                    raise RuntimeError("no rollout server")
                return RolloutResult(prompt=prompt, trajectory=Trajectory(done=True), success=True)
            finally:
                self.inflight -= 1

        return _episode()


async def test_collect_rollouts_bounds_in_flight_episodes_by_max_concurrent():
    """The manager's semaphore must cap in-flight episodes — an unbounded fan-out stampedes the server."""
    actor = _RecordingActor()
    manager = _manager_over_fake_actors(actor, num_workers=3, max_concurrent_rollouts=3)

    results = await manager.collect_rollouts([f"p{i}" for i in range(12)])

    assert len(results) == 12
    assert actor.peak <= manager.max_concurrent, f"peak {actor.peak} exceeded the cap {manager.max_concurrent}"
    # Also assert the cap is REACHED: a semaphore stuck at 1 would satisfy the bound above alone.
    assert actor.peak == manager.max_concurrent


async def test_collect_rollouts_yields_one_result_per_prompt_when_every_episode_fails():
    """The trainer maps rollouts back to prompts positionally, so a failing server must still yield
    one RolloutResult per prompt, in order — dropping the failures shifts every reward by one."""
    from src.environments.ray_actors import RolloutResult

    prompts = ["a", "b", "c"]
    manager = _manager_over_fake_actors(_RecordingActor(fail=True), num_workers=2)

    results = await manager.collect_rollouts(prompts)

    assert [r.prompt for r in results] == prompts
    assert all(isinstance(r, RolloutResult) and not r.success for r in results)
    assert all("no rollout server" in (r.error or "") for r in results)


# Test: Tool Schema Detection


def test_tools_schema_detection():
    """Native tool-use environments must emit OpenAI-format tool schemas — the server rejects any
    other shape, so the key VALUES matter, not just their presence."""
    from src.environments.registry import create_environment

    native_env = create_environment("native_math", {"max_turns": 3})
    schema = native_env.get_tools_schema()

    assert isinstance(schema, list) and schema
    for tool in schema:
        assert tool["type"] == "function"
        assert set(tool["function"]) >= {"name", "description", "parameters"}
        assert tool["function"]["parameters"]["type"] == "object"


def test_tool_calls_in_step_context():
    """Test that tool_calls in context are attached to assistant messages."""
    from src.environments.registry import create_environment

    # Create a native tool use environment
    env = create_environment("native_math", {"max_turns": 5})

    # Reset with a prompt
    episode_ids, steps = env.reset(["What is 2+2?"], [{"answer": "4"}])
    episode_id = episode_ids[0]

    # Simulate a step with tool_calls in context (as vLLM would return)
    mock_tool_calls = [
        {"id": "call_123", "type": "function", "function": {"name": "calculate", "arguments": '{"expression": "2+2"}'}}
    ]

    step_context = {"answer": "4", "tool_calls": mock_tool_calls}
    env.step([episode_id], [""], [step_context])

    # The trajectory should have the assistant message with tool_calls
    traj = env._trajectories[episode_id]
    assistant_msgs = [m for m in traj.messages if m.role == "assistant"]
    assert len(assistant_msgs) >= 1
    assert assistant_msgs[-1].tool_calls == mock_tool_calls


# Test: _build_payload (what actually gets sent to vLLM)


def _make_actor(env_type="native_math", env_config=None):
    """Instantiate the REAL EnvironmentActor off-cluster.

    ``@ray.remote`` wraps the class; the original is kept at
    ``__ray_metadata__.modified_class``. ``__init__`` only sets attributes (env + tools
    schema are lazy), so this is CPU-safe and lets us exercise the genuine ``_build_payload``
    rather than a hand-rebuilt copy of its logic.
    """
    from src.environments.ray_actors import EnvironmentActor

    cls = EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type=env_type, env_config=env_config or {"max_turns": 3})
    return actor


def test_build_payload_sends_only_supported_params_and_env_tools():
    """The real _build_payload forwards the supported sampling params + OpenAI tools, nothing else."""
    from src.environments.ray_actors import RolloutConfig

    actor = _make_actor("native_math")  # native_math exposes OpenAI-format tools
    payload = actor._build_payload(
        [{"role": "user", "content": "2+2?"}],
        RolloutConfig(temperature=0.9, top_p=0.8, max_tokens=512),
    )

    assert payload["messages"][0]["content"] == "2+2?"
    assert payload["temperature"] == 0.9 and payload["top_p"] == 0.8 and payload["max_tokens"] == 512
    # Env tools are attached and in OpenAI function format.
    assert payload["tools"] and payload["tools"][0]["type"] == "function" and "function" in payload["tools"][0]
    # model is omitted when model_name is unset, and no unsupported sampling knob leaks through.
    assert "model" not in payload
    for leaked in ("top_k", "min_p", "repetition_penalty", "frequency_penalty"):
        assert leaked not in payload


def test_build_payload_gates_model_and_omits_empty_tools():
    """model is added iff model_name is set; tools key is dropped when the env exposes none."""
    from src.environments.ray_actors import RolloutConfig

    actor = _make_actor("native_math")
    actor._tools_schema = None  # prime the cache to the no-tools branch (skips env construction)
    payload = actor._build_payload([{"role": "user", "content": "hi"}], RolloutConfig(model_name="Qwen/Qwen3-4B"))

    assert payload["model"] == "Qwen/Qwen3-4B"
    assert "tools" not in payload


def test_build_payload_sends_no_tools_for_a_react_env():
    """ReAct parses its action out of the assistant TEXT, so the rollout must advertise no tools.

    Advertising them makes a server with a tool-call parser strip the call out of ``content``: ReAct
    then sees no ``Action:``, appends its format hint and burns the turn unpriced and uncounted.
    """
    from src.environments.ray_actors import RolloutConfig

    react_payload = _make_actor("react_math")._build_payload([{"role": "user", "content": "2+2?"}], RolloutConfig())
    native_payload = _make_actor("native_math")._build_payload([{"role": "user", "content": "2+2?"}], RolloutConfig())

    assert "tools" not in react_payload
    # Positive control: the native protocol still advertises its schema through the same code path.
    assert native_payload["tools"]


# Test: RolloutResult wire contract

# The fields a rollout is allowed to carry home. Every one is pickled through the Ray object store and
# again through the TP broadcast, once per episode, so an unread field ships the whole per-episode
# payload (code_contests: the hidden-test set) for nothing.
CONSUMED_ROLLOUT_RESULT_FIELDS = {
    "prompt",
    "trajectory",
    "episode_length",
    "total_reward",
    "success",
    "latency",
    "error",
    "generation_tokens",
    "metrics",
}

# Where a rollout result is read: the actor/manager that builds it and the trainer that consumes it.
ROLLOUT_RESULT_CONSUMERS = ("src/environments", "src/trainers/grpo")


def _bare_local_attribute_reads(root):
    """Attribute names read off a bare local (``r.error``) anywhere under ``root``.

    Bare locals only, ``self`` excluded: ``self.async_config.rollout_server_url`` reads a *config*
    field of the same name, and counting it would let a dead result field look consumed.
    """
    import ast

    names = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                base = node.value
                if isinstance(base, ast.Name) and base.id != "self":
                    names.add(node.attr)
    return names


def test_rollout_result_carries_only_consumed_fields():
    """RolloutResult must declare exactly the consumed set — and each field must have a real reader.

    Two halves: the first fails when a field is added or removed without the contract moving with it,
    the second fails when a field stops being read anywhere in the actor/trainer path. Neither passes
    on a field that only ever gets written.
    """
    import dataclasses

    from src.environments.ray_actors import RolloutResult
    from tests.common.utils import REPO_ROOT

    declared = {f.name for f in dataclasses.fields(RolloutResult)}
    assert declared == CONSUMED_ROLLOUT_RESULT_FIELDS, (
        f"RolloutResult fields drifted from the consumed set: only in the class {declared - CONSUMED_ROLLOUT_RESULT_FIELDS}, "
        f"only in the contract {CONSUMED_ROLLOUT_RESULT_FIELDS - declared}"
    )

    read = set()
    for relative in ROLLOUT_RESULT_CONSUMERS:
        read |= _bare_local_attribute_reads(REPO_ROOT / relative)
    unread = declared - read
    assert not unread, (
        f"RolloutResult fields written but never read in {list(ROLLOUT_RESULT_CONSUMERS)}: {sorted(unread)} — "
        f"each one pickles through the Ray object store and the TP broadcast every episode for nothing"
    )


class _FakeChatCompletionsSession:
    """An ``aiohttp.ClientSession`` stand-in serving one canned /v1/chat/completions body."""

    def __init__(self, body: dict):
        self._body = body

    def post(self, url, json):  # noqa: A002 — aiohttp's own keyword
        return self

    async def __aenter__(self):
        return SimpleNamespace(status=200, json=self._json, text=self._text)

    async def __aexit__(self, *exc_info):
        return False

    async def _json(self):
        return self._body

    async def _text(self):
        return ""


async def test_sglang_length_cutoff_survives_the_rollout_transport():
    """SGLang spells the token-cap cut-off ``stop_reason``, and this transport reads the response body
    as a plain dict — so a reader that knows only ``finish_reason`` clears the truncation and the env
    grades an engine-cut fragment as a deliberate final answer."""
    from src.environments.ray_actors import RolloutConfig
    from src.inference.response import FINISH_REASON_LENGTH

    actor = _make_actor("native_math")
    session = _FakeChatCompletionsSession(
        {
            "choices": [{"message": {"content": "partial answ"}, "stop_reason": FINISH_REASON_LENGTH}],
            "usage": {"completion_tokens": 3},
        }
    )

    generation = await actor._generate(
        session, "server:8000", [{"role": "user", "content": "2+2?"}], RolloutConfig(backend="sglang", max_retries=0)
    )

    assert generation.text == "partial answ"
    assert generation.finish_reason == FINISH_REASON_LENGTH, (
        "SGLang's stop_reason spelling was dropped: the turn reaches the env as an untruncated answer"
    )


# Test: stateful-env session cleanup across rollouts (leak fix)


async def test_actor_releases_session_when_episode_errors():
    """A stateful SweEnvironment episode that errors mid-way must still release its sandbox
    session + temp dir — run_episode cleans up in a finally, so a long-lived actor doesn't leak a
    working directory per failed rollout."""
    import os

    from src.environments.ray_actors import RolloutConfig, TurnGeneration

    actor = _make_actor("swe", {"max_turns": 5})
    env = actor._get_env()  # force construction so we can inspect its sessions

    created = []
    orig_open = env.sandbox.open_session
    env.sandbox.open_session = lambda: created.append(orig_open()) or created[-1]

    async def _fake_client(timeout):  # unused (we mock _generate), just must not hit the network
        return None

    actor._get_http_client = _fake_client

    calls = {"n": 0}

    async def _fake_generate(client, url, messages, config, reasoning_effort=None):
        calls["n"] += 1
        if calls["n"] == 1:  # turn 1: write a file -> creates the episode's session
            tc = [{"id": "c1", "function": {"name": "write_file", "arguments": '{"path": "f.txt", "content": "x"}'}}]
            return TurnGeneration("", tc, "", 1)
        raise RuntimeError("boom-mid-episode")  # turn 2: fail, leaving a live session

    actor._generate = _fake_generate

    result = await actor.run_episode("task", None, "http://x", RolloutConfig(max_retries=1))

    assert result.error and "boom" in result.error, "episode should surface the mid-episode error"
    assert created, "write_file should have opened a session (otherwise the test proves nothing)"
    assert env._sessions == {}, "session must be released on the error path — no leak"
    for session in created:
        assert not os.path.exists(session.workdir), "session temp dir must be deleted by cleanup"


async def test_actor_drives_async_env_via_step_async():
    """An async environment (async-only tool handlers, like MCP) must be driven through
    reset_async/step_async by the rollout — a sync step would hit NativeTool.execute's 'no sync
    handler' and the tool would never run. Verifies the actor detects AsyncBaseEnvironment and the
    async tool actually executes."""
    from src.environments.envs.protocols.native import AsyncNativeToolUseEnvironment
    from src.environments.ray_actors import RolloutConfig, TurnGeneration
    from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

    ran = {"called": False}

    async def _async_handler(x):
        ran["called"] = True
        return f"async-result-{x}"

    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="aecho",
            description="async echo",
            parameters=[ToolParameter("x", "string", "value")],
            async_handler=_async_handler,  # NOTE: no sync handler — only runs via the async path
        )
    )

    actor = _make_actor(
        env_type=(AsyncNativeToolUseEnvironment, {"tool_registry": registry}),
        env_config={"max_turns": 3},
    )

    async def _fake_client(timeout):
        return None

    actor._get_http_client = _fake_client

    calls = {"n": 0}

    async def _fake_generate(client, url, messages, config, reasoning_effort=None):
        calls["n"] += 1
        if calls["n"] == 1:  # turn 1: call the async tool
            return TurnGeneration("", [{"id": "c1", "function": {"name": "aecho", "arguments": '{"x": "hi"}'}}], "", 1)
        return TurnGeneration("final answer", [], "", 1)  # turn 2: no tool call -> finalize

    actor._generate = _fake_generate

    result = await actor.run_episode("task", None, "http://x", RolloutConfig(max_retries=1))

    assert ran["called"], "async tool must execute (sync stepping would NotImplementedError it)"
    assert result.error is None, f"async episode should not error: {result.error}"
    tool_results = [m.content for m in result.trajectory.messages if m.role == "tool"]
    assert any("async-result-hi" in r for r in tool_results), f"async tool output missing: {tool_results}"


# Test: per-EPISODE generation-token accounting (summed across turns)


async def test_run_episode_generation_tokens_sum_across_turns():
    """RolloutResult.generation_tokens must be the SUM of every turn's generated tokens for THIS
    episode — not the last turn, the first turn, or a per-turn mean. This is the source the trainer
    aggregates into episode/generation_tokens, so a regression to per-turn accounting would silently mis-report
    per-episode generation length. Drives the real multi-turn loop in run_episode (native_math), mocking
    only the vLLM call so each turn reports a distinct token count."""
    from src.environments.ray_actors import RolloutConfig, TurnGeneration

    actor = _make_actor("native_math", {"max_turns": 5})

    async def _fake_client(timeout):
        return None

    actor._get_http_client = _fake_client

    per_turn: list[int] = []
    calls = {"n": 0}

    async def _fake_generate(client, url, messages, config, reasoning_effort=None):
        i = calls["n"]
        calls["n"] += 1
        tokens = (i + 1) * 10  # turn 1 -> 10, turn 2 -> 20, turn 3 -> 30
        per_turn.append(tokens)
        if i < 2:  # first two turns call a tool -> loop continues
            tc = [{"id": f"c{i}", "function": {"name": "calculate", "arguments": '{"expression": "1+1"}'}}]
            return TurnGeneration("", tc, "", tokens)
        return TurnGeneration("final answer: 4", [], "", tokens)  # third turn: no tool call -> finalize

    actor._generate = _fake_generate

    result = await actor.run_episode("2+2?", {"answer": "4"}, "http://x", RolloutConfig(max_retries=1))

    assert calls["n"] == 3, f"expected a 3-turn episode (setup guard), got {calls['n']} turns"
    assert result.episode_length == 3
    # Per-EPISODE total = 10 + 20 + 30 = 60. A per-turn/last-turn/mean bug would give 30 or 20, not 60.
    assert result.generation_tokens == 60
    assert result.generation_tokens == sum(per_turn)


# Test: concurrent CodeContests episodes grade against their OWN tests


async def test_actor_grades_concurrent_codecontests_episodes_in_isolation():
    """Two concurrent CodeContests episodes sharing ONE actor (hence one env instance and one
    sandbox) must each be graded against their OWN hidden tests.

    The submit_solution handler reads the episode it is grading from the ``_ACTIVE_TRAJECTORY``
    ContextVar (set per step), not a shared instance attribute. If that isolation broke, one
    in-flight episode's submission would be graded against the other's tests — silently corrupting
    the RL reward. Episode A's code only solves A's problem and B's only solves B's, so a leak flips
    at least one reward from full-pass to fail. This is the reward-critical Ray-actor property, so
    it runs the genuine sync env + local subprocess grader, no mocks below the HTTP boundary."""
    from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
    from src.environments.ray_actors import RolloutConfig, TurnGeneration

    # A: double the input; B: increment it. Each problem's single hidden test is satisfied ONLY by
    # its own program — B's incrementing code scores 0 on A's doubling test and vice versa.
    problems = {
        "DOUBLE": {"code": "print(int(input()) * 2)", "answer": {"tests": [{"input": "21", "output": "42"}]}},
        "INCREMENT": {"code": "print(int(input()) + 1)", "answer": {"tests": [{"input": "21", "output": "22"}]}},
    }

    # max_turns=1: exactly one graded submission per episode, so a single mis-bound grade is fatal
    # (no later turn to silently re-grade against the correct tests and mask the leak).
    actor = _make_actor(
        env_type=(CodeContestsEnvironment, {"language": "python", "max_submissions": 1, "max_test_calls": 0}),
        env_config={"max_turns": 1},
    )
    assert actor._get_env() is actor._get_env(), "both episodes must share one cached env instance"

    async def _fake_client(timeout):
        return None

    actor._get_http_client = _fake_client

    async def _fake_generate(client, url, messages, config, reasoning_effort=None):
        # Yield so both episodes are in-flight before either submits (real concurrent interleaving),
        # then submit the program for whichever problem this episode's prompt names.
        await asyncio.sleep(0)
        task_text = " ".join(m.get("content") or "" for m in messages)
        kind = "DOUBLE" if "DOUBLE" in task_text else "INCREMENT"
        args = json.dumps({"code": problems[kind]["code"]})
        return TurnGeneration("", [{"id": "c1", "function": {"name": "submit_solution", "arguments": args}}], "", 1)

    actor._generate = _fake_generate

    async def _run(kind: str):
        return await actor.run_episode(
            f"Problem {kind}: transform the input.",
            {"answer": problems[kind]["answer"]},
            "http://x",
            RolloutConfig(max_retries=1),
        )

    result_a, result_b = await asyncio.gather(_run("DOUBLE"), _run("INCREMENT"))

    # Each solved its own problem → full reward. A leak would grade one against the other's test → 0.
    assert result_a.total_reward == 1.0, f"DOUBLE episode mis-graded (leak?): reward={result_a.total_reward}"
    assert result_b.total_reward == 1.0, f"INCREMENT episode mis-graded (leak?): reward={result_b.total_reward}"
    assert result_a.trajectory.info["tests_passed"] == result_a.trajectory.info["tests_total"] == 1
    assert result_b.trajectory.info["tests_passed"] == result_b.trajectory.info["tests_total"] == 1
    # The two episodes really were distinct (different stored tests), proving the test is not vacuous.
    assert result_a.trajectory.info["_test_cases"] != result_b.trajectory.info["_test_cases"]


# Test: sync env.step runs OFF the actor's event loop (no episode serialization)


async def test_slow_sync_step_does_not_block_concurrent_episodes():
    """A sync env's blocking step (sandboxed grading runs seconds-to-minutes) must run OFF the
    actor's event loop. Two concurrent episodes rendezvous inside step on a 2-party barrier:
    offloaded to worker threads both are inside step simultaneously and pass; stepped inline on the
    loop, the first step blocks the loop, the second episode's step never starts, and the barrier
    times out — surfacing as broken-barrier episode errors."""
    import threading

    from src.environments.envs.protocols.native import NativeToolUseEnvironment
    from src.environments.ray_actors import RolloutConfig, TurnGeneration
    from src.environments.tools.definitions import NativeToolRegistry

    barrier = threading.Barrier(2, timeout=10.0)

    class _BarrierStepEnv(NativeToolUseEnvironment):
        def step(self, episode_ids, actions, contexts=None):
            barrier.wait()  # BrokenBarrierError (-> episode error) unless both steps overlap
            return super().step(episode_ids, actions, contexts)

    actor = _make_actor(
        env_type=(_BarrierStepEnv, {"tool_registry": NativeToolRegistry()}),
        env_config={"max_turns": 2},
    )

    async def _fake_client(timeout):
        return None

    actor._get_http_client = _fake_client

    async def _fake_generate(client, url, messages, config, reasoning_effort=None):
        return TurnGeneration("final answer", [], "", 1)  # no tool call -> done after one step

    actor._generate = _fake_generate

    cfg = RolloutConfig(max_retries=1)
    result_a, result_b = await asyncio.gather(
        actor.run_episode("task A", None, "http://x", cfg),
        actor.run_episode("task B", None, "http://x", cfg),
    )
    assert result_a.error is None and result_b.error is None, (
        f"sync steps serialized on the event loop: {result_a.error} / {result_b.error}"
    )


async def test_sync_step_offload_preserves_cross_turn_contextvars():
    """Offloading must keep the per-episode ContextVar semantics steps had when they ran inline on
    the episode's own asyncio task: a value set during turn 1 (e.g. the simulated file tools'
    per-episode store) stays visible on turn 2 of the SAME episode, and never leaks into a
    concurrent episode. A naive per-call ``asyncio.to_thread`` would drop turn-1 writes (fresh
    context copy each call); a shared context would leak markers across episodes."""
    from contextvars import ContextVar

    from src.environments.envs.protocols.native import NativeToolUseEnvironment
    from src.environments.ray_actors import RolloutConfig, TurnGeneration
    from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

    turn_var: ContextVar = ContextVar("test_episode_marker", default=None)
    second_turn_reads: dict[str, str | None] = {}

    class _CtxProbeEnv(NativeToolUseEnvironment):
        def step(self, episode_ids, actions, contexts=None):
            marker = contexts[0]["marker"]
            if turn_var.get() is None:
                turn_var.set(marker)  # turn 1: write into the episode's context
            else:
                second_turn_reads[marker] = turn_var.get()  # turn 2: record what survived
            return super().step(episode_ids, actions, contexts)

    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="echo",
            description="echo",
            parameters=[ToolParameter("x", "string", "value")],
            handler=lambda x: x,
        )
    )
    actor = _make_actor(env_type=(_CtxProbeEnv, {"tool_registry": registry}), env_config={"max_turns": 3})

    async def _fake_client(timeout):
        return None

    actor._get_http_client = _fake_client

    calls = {"A": 0, "B": 0}

    async def _fake_generate(client, url, messages, config, reasoning_effort=None):
        await asyncio.sleep(0)  # interleave the two episodes
        text = " ".join(m.get("content") or "" for m in messages)
        key = "A" if "task A" in text else "B"
        calls[key] += 1
        if calls[key] == 1:  # turn 1: call the tool so the episode continues
            return TurnGeneration("", [{"id": "c1", "function": {"name": "echo", "arguments": '{"x": "hi"}'}}], "", 1)
        return TurnGeneration("done", [], "", 1)

    actor._generate = _fake_generate

    cfg = RolloutConfig(max_retries=1)
    result_a, result_b = await asyncio.gather(
        actor.run_episode("task A", {"marker": "A"}, "http://x", cfg),
        actor.run_episode("task B", {"marker": "B"}, "http://x", cfg),
    )
    assert result_a.error is None and result_b.error is None, f"{result_a.error} / {result_b.error}"
    assert second_turn_reads == {"A": "A", "B": "B"}, (
        f"cross-turn ContextVar semantics broken: {second_turn_reads} — an empty/partial dict means "
        "per-call context copies dropped turn-1 writes; swapped values mean a cross-episode leak"
    )


# Test: _is_client_error


def test_is_client_error():
    """Only a genuine, non-transient 4xx from the rollout server abandons the batch."""
    from src.environments.ray_actors import RolloutHTTPError, _is_client_error

    def _http(status, body="boom"):
        return RolloutHTTPError(status, "vllm", body)

    # 4xx errors should NOT be retried
    assert _is_client_error(_http(400, "Bad request"))
    assert _is_client_error(_http(422, "Unprocessable"))
    assert _is_client_error(_http(404, "Not found"))

    # 5xx errors SHOULD be retried (giveup returns False)
    assert not _is_client_error(_http(500, "Internal error"))
    assert not _is_client_error(_http(503, "Unavailable"))

    # Timeouts SHOULD be retried
    assert not _is_client_error(TimeoutError("Connection timed out"))
    assert not _is_client_error(ConnectionError("Connection refused"))

    # 429 (rate limit) and 408 (request timeout) are 4xx but transient — they MUST be
    # retried, not given up on. A `"status 4" in str(exc)` substring test gave up
    # immediately the instant the engine was briefly overloaded.
    assert not _is_client_error(_http(429, "Too Many Requests"))
    assert not _is_client_error(_http(408, "Request Timeout"))


def test_a_server_error_quoting_a_4xx_in_its_body_is_still_retried():
    """The status is read as data, never re-parsed out of the message.

    The response body is server-controlled text: a 503 from a proxy that quotes the upstream
    "status 400" it saw must not abandon the whole batch on the strength of that quote.
    """
    from src.environments.ray_actors import RolloutHTTPError, _is_client_error

    exc = RolloutHTTPError(503, "vllm", "upstream returned status 400: bad request")

    assert "status 400" in str(exc)
    assert not _is_client_error(exc)
    # And a plain RuntimeError carrying the same text is a transport fault, not a client error.
    assert not _is_client_error(RuntimeError("vllm error (status 400): Bad request"))


# Test: RolloutManager max_concurrent clamping


def test_rollout_manager_max_concurrent_clamping():
    """Test that max_concurrent is clamped to at least num_workers."""
    from src.environments.ray_actors import RolloutConfig, RolloutManager

    # max_concurrent_rollouts < num_workers should be clamped
    manager = RolloutManager(
        num_workers=8,
        env_type="react_math",
        env_config={},
        server_urls=["http://localhost:8000"],
        rollout_config=RolloutConfig(),
        max_concurrent_rollouts=2,  # Less than num_workers=8
    )
    assert manager.max_concurrent == 8  # Clamped to num_workers

    # Default: num_workers * 4
    manager2 = RolloutManager(
        num_workers=4,
        env_type="react_math",
        env_config={},
        server_urls=["http://localhost:8000"],
        rollout_config=RolloutConfig(),
    )
    assert manager2.max_concurrent == 16  # 4 * 4


# Test: AsyncTrainingConfig → RolloutConfig mapping


def test_async_training_config_to_rollout_config():
    """Test that AsyncTrainingConfig.get_rollout_config() maps all fields."""
    from src.configs.async_training_config import AsyncTrainingConfig

    config = AsyncTrainingConfig(
        rollout_temperature=0.9,
        rollout_top_p=0.8,
        rollout_max_tokens=2048,
        model_name="test-model",
        request_timeout=60.0,
        max_retries=5,
        retry_base_wait=2.0,
    )

    rc = config.get_rollout_config()
    assert rc.temperature == 0.9
    assert rc.top_p == 0.8
    assert rc.max_tokens == 2048
    assert rc.model_name == "test-model"
    assert rc.request_timeout == 60.0
    assert rc.max_retries == 5
    assert rc.retry_base_wait == 2.0

    # Verify removed fields don't exist on AsyncTrainingConfig
    assert not hasattr(config, "rollout_top_k")
    assert not hasattr(config, "rollout_min_p")
    assert not hasattr(config, "rollout_repetition_penalty")


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
