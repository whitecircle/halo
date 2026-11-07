#!/usr/bin/env python
import ast
import asyncio
import dataclasses
import json
import os
import pickle
import tempfile
import threading
from contextvars import ContextVar
from types import SimpleNamespace

import pytest
import ray

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.environments.episodes.messages import Trajectory
from halo.environments.protocols.native_tool_use import AsyncNativeToolUseEnvironment, NativeToolUseEnvironment
from halo.environments.registry import build_environment
from halo.environments.rollout_config import RolloutConfig
from halo.environments.rollouts import ray_actors
from halo.environments.rollouts.episode_loop import RolloutResult, TurnGeneration, step_context_from_generation
from halo.environments.rollouts.ray_actors import (
    EnvironmentActor,
    RolloutHTTPError,
    RolloutManager,
    _should_giveup,
    ray_init_kwargs,
)
from halo.environments.tasks.coding.code_contests import CodeContestsEnvironment
from halo.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter
from halo.environments.tools.toolsets import build_math_tools
from halo.inference.response import FINISH_REASON_LENGTH
from halo.rewards.samples import CUT_TOOL_CALLS_KEY
from tests.common.repo import REPO_ROOT


def test_env_reset_yields_one_episode_per_prompt():
    """A reset that collapses or reorders prompts misattributes every reward the manager maps back positionally."""
    env = build_environment("react_math", {"max_turns": 5})
    episode_ids, steps = env.reset(["2+2?", "3+3?"], [{"answer": "4"}, {"answer": "6"}])

    assert len(episode_ids) == len(steps) == 2
    assert len(set(episode_ids)) == 2, "episode ids must be distinct, else trajectories overwrite"
    assert [t.info["episode_id"] for t in env.get_trajectories(episode_ids)] == episode_ids


async def test_env_batch_calls_refuse_contexts_that_do_not_pair():
    """Truncating to the shorter list drops the unpaired episodes."""
    env = build_environment("react_math", {"max_turns": 5})
    with pytest.raises(ValueError, match="shorter"):
        env.reset(["2+2?", "3+3?"], [{"answer": "4"}])
    episode_ids, _ = env.reset(["2+2?"], [{"answer": "4"}])
    with pytest.raises(ValueError, match="shorter"):
        env.step(episode_ids, ["Final Answer: 4"], [])

    async_env = AsyncNativeToolUseEnvironment(tool_registry=build_math_tools())
    with pytest.raises(ValueError, match="shorter"):
        await async_env.reset_async(["2+2?", "3+3?"], [{}])
    episode_ids, _ = await async_env.reset_async(["2+2?"], [{}])
    with pytest.raises(ValueError, match="shorter"):
        await async_env.step_async(episode_ids, ["4"], [])


@pytest.fixture
def deep_temp_root(tmp_path, monkeypatch):
    """A temp root too deep for Ray's socket paths, with no RAY_TMPDIR set; removes the link it gets."""
    root = tmp_path / ("deep" * 15)
    root.mkdir()
    monkeypatch.delenv("RAY_TMPDIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    yield root
    temp_dir = ray_init_kwargs().get("_temp_dir")
    if temp_dir is not None and os.path.islink(temp_dir):
        os.unlink(temp_dir)


def test_a_deep_temp_root_gets_a_short_link_onto_its_own_volume(deep_temp_root):
    """Ray keeps its session logs and spilled objects under ``_temp_dir``: a short path on the root
    filesystem in its place would move them off the large volume TMPDIR names."""
    temp_dir = ray_init_kwargs()["_temp_dir"]

    assert len(temp_dir) <= ray_actors._MAX_RAY_TEMP_DIR_LEN
    assert os.path.realpath(temp_dir) == os.path.realpath(deep_temp_root / "ray")
    assert ray_init_kwargs()["_temp_dir"] == temp_dir, "a second launch over the root must reuse the link"


def test_a_short_ray_tmpdir_is_used_as_is(deep_temp_root, monkeypatch):
    monkeypatch.setenv("RAY_TMPDIR", "/tmp/rt")
    assert "_temp_dir" not in ray_init_kwargs(), "Ray's own short RAY_TMPDIR must not be overridden"


def test_rollout_manager_round_robins_across_servers():
    """A selector that stopped advancing would pin every episode on one engine."""
    urls = ["http://server1:8000", "http://server2:8000", "http://server3:8000"]
    manager = RolloutManager(
        num_workers=1,
        env_type="react_math",
        env_config={},
        server_urls=urls,
        rollout_config=RolloutConfig(),
        max_concurrent_rollouts=1,
    )

    picked = [manager._next_url() for _ in range(len(urls) * 2)]
    assert picked == urls * 2, f"expected round-robin over {urls}, got {picked}"


def _round_load(*, ranks: list[int], servers: int, rows: int) -> list[list[int]]:
    """Episodes per server in each of two rounds, one real selector per training rank at ``first_server`` = its DP rank
    (partners under TP or ETP share theirs)."""
    urls = [f"http://server{i}:8000" for i in range(servers)]
    managers = [
        RolloutManager(
            num_workers=1,
            env_type="react_math",
            env_config={},
            server_urls=urls,
            rollout_config=RolloutConfig(),
            first_server=rank,
            max_concurrent_rollouts=1,
        )
        for rank in ranks
    ]
    load = []
    for _ in range(2):
        picked = [manager._next_url() for manager in managers for _ in range(rows)]
        load.append([picked.count(url) for url in urls])
    return load


@pytest.mark.parametrize(
    ("ranks", "servers", "rows"),
    [
        (list(range(8)), 4, 2),
        (list(range(4)), 2, 3),
        ([0, 0, 1, 1, 2, 2, 3, 3], 2, 3),
        ([0, 0, 1, 1], 4, 1),
    ],
    ids=["8-ranks-4-servers", "odd-rows", "tp2-partners-2-servers", "tp2-partners-4-servers"],
)
def test_ranks_spread_a_narrow_round_over_the_servers(ranks, servers, rows):
    """Every rank starting at server 0 put all 16 episodes of 8 ranks × 2 rows on two of four servers each round."""
    for round_load in _round_load(ranks=ranks, servers=servers, rows=rows):
        assert max(round_load) - min(round_load) <= 2 * max(1, len(ranks) // len(set(ranks))) - 1, round_load
        assert sum(round_load) == len(ranks) * rows


async def test_rollout_manager_start_shutdown():
    try:
        if ray.is_initialized():
            ray.shutdown()
        # ray_init_kwargs keeps the dashboard off (jobs-API CVE surface) and the tempdir short;
        # num_cpus caps Ray's core-count worker prestart on many-core hosts.
        ray.init(**ray_init_kwargs(num_cpus=4))
    except Exception as e:  # ray[default] is a hard dependency, so a failed init is a real failure
        raise AssertionError(f"ray.init() failed, so the actor-lifecycle checks below never ran: {e}") from e

    try:
        manager = RolloutManager(
            num_workers=2,
            env_type="react_math",
            env_config={"max_turns": 3},
            server_urls=["http://localhost:8000"],
            rollout_config=RolloutConfig(),
            max_concurrent_rollouts=2,
        )

        await manager.start()
        assert manager._started
        assert len(manager._actors) == 2

        await manager.start()
        assert len(manager._actors) == 2

        # The actors' copy of the pause clock is a live actor the manager's windows reach, in order.
        manager.begin_engine_pause()
        manager.end_engine_pause(2.5)
        assert await manager._actor_pause_clock.paused_seconds.remote() == 2.5

        await manager.shutdown()
        assert not manager._started
        assert len(manager._actors) == 0
        assert manager._actor_pause_clock is None
    finally:
        ray.shutdown()


class _Spawner:
    """Stands in for a ``@ray.remote`` class: records each spawn with the scheduling strategy it was placed under."""

    def __init__(self, name: str, spawned: list[tuple[str, object]], strategy: object = None):
        self.name, self.spawned, self.strategy = name, spawned, strategy

    def options(self, *, scheduling_strategy):
        return _Spawner(self.name, self.spawned, scheduling_strategy)

    def remote(self, *args):
        self.spawned.append((self.name, self.strategy))
        return SimpleNamespace()


async def test_the_pool_and_its_clock_prefer_this_node_and_spill_off_a_busy_one(monkeypatch):
    """The actors and their pause clock are placed under a soft affinity to this node that also spills when
    the node is merely saturated: ``soft`` alone pins every actor to a live, busy training node, which stalls."""
    spawned: list[tuple[str, object]] = []
    runtime = SimpleNamespace(get_node_id=lambda: "node")
    monkeypatch.setattr(
        ray_actors, "ray", SimpleNamespace(is_initialized=lambda: True, get_runtime_context=lambda: runtime)
    )
    monkeypatch.setattr(ray_actors, "NodeAffinitySchedulingStrategy", lambda **kwargs: kwargs)
    monkeypatch.setattr(ray_actors, "EnvironmentActor", _Spawner("actor", spawned))
    monkeypatch.setattr(ray_actors, "_RemoteEnginePauseClock", _Spawner("clock", spawned))
    manager = ray_actors.RolloutManager(
        num_workers=2,
        env_type="react_math",
        env_config={},
        server_urls=["http://localhost:8000"],
        rollout_config=ray_actors.RolloutConfig(),
        max_concurrent_rollouts=2,
    )

    await manager.start()

    affinity = {"node_id": "node", "soft": True, "_spill_on_unavailable": True}
    assert spawned == [("clock", affinity), ("actor", affinity), ("actor", affinity)]


def test_rollout_config_defaults():
    """The default values are pinned against ``AsyncRolloutConfig`` in
    ``tests/cpu/config/test_rollout_config_mirror.py``."""
    config = RolloutConfig()

    assert not hasattr(config, "max_concurrent_per_actor")

    # The actors receive it pickled.
    pickled = pickle.dumps(config)
    restored = pickle.loads(pickled)
    assert restored.temperature == config.temperature
    assert restored.max_retries == config.max_retries
    assert restored.model_name == config.model_name


def _manager_over_fake_actors(actor, *, num_workers=2, max_concurrent_rollouts=8):
    """A started RolloutManager whose pool is ``actor``, so ``collect_rollouts`` runs without Ray."""
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
    """Ray-handle stand-in recording peak in-flight episodes, so the bound is asserted on the manager's own
    semaphore."""

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.inflight = 0
        self.peak = 0
        self.run_episode = SimpleNamespace(remote=self._remote)

    def _remote(self, *, prompt, context, server_url, config):
        async def _episode():
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            try:
                await asyncio.sleep(0.02)
                if self.fail:
                    raise RuntimeError("no rollout server")
                return RolloutResult(prompt=prompt, trajectory=Trajectory(done=True), terminated_naturally=True)
            finally:
                self.inflight -= 1

        return _episode()


async def test_collect_rollouts_bounds_in_flight_episodes_by_max_concurrent():
    actor = _RecordingActor()
    manager = _manager_over_fake_actors(actor, num_workers=3, max_concurrent_rollouts=3)

    results = await manager.collect_rollouts([f"p{i}" for i in range(12)])

    assert len(results) == 12
    assert actor.peak <= manager.max_concurrent, f"peak {actor.peak} exceeded the cap {manager.max_concurrent}"
    # Also assert the cap is REACHED: a semaphore stuck at 1 would satisfy the bound above alone.
    assert actor.peak == manager.max_concurrent


async def test_collect_rollouts_yields_one_result_per_prompt_when_every_episode_fails():
    """The trainer maps rollouts back positionally, so a failing server still yields one result per prompt, in
    order."""
    prompts = ["a", "b", "c"]
    manager = _manager_over_fake_actors(_RecordingActor(fail=True), num_workers=2)

    results = await manager.collect_rollouts(prompts)

    assert [r.prompt for r in results] == prompts
    assert all(isinstance(r, RolloutResult) and not r.terminated_naturally for r in results)
    assert all("no rollout server" in (r.error or "") for r in results)


async def test_collect_rollouts_refuses_contexts_that_do_not_pair_with_prompts():
    """A short list raises before any episode is submitted; a raise after the first tasks started would leave them
    running on the actors."""
    actor = _RecordingActor()
    manager = _manager_over_fake_actors(actor)

    with pytest.raises(ValueError, match="shorter"):
        await manager.collect_rollouts(["a", "b", "c"], [{"answer": "1"}])
    await asyncio.sleep(0.05)  # an orphaned task would run its episode here
    assert actor.peak == 0


@pytest.mark.parametrize("url", ["http://localhost:8000", "127.0.1.1:8000", "http://[::1]:8000"])
def test_a_loopback_server_url_is_refused_on_several_nodes(url):
    """Every loopback spelling the weight-sync client treats as loopback, ``127.x`` included, reaches another engine or
    none from an actor on another node, and that node's engine takes no push."""

    def manager(*urls):
        return ray_actors.RolloutManager(
            num_workers=1,
            env_type="react_math",
            env_config={},
            server_urls=list(urls),
            rollout_config=ray_actors.RolloutConfig(),
            max_concurrent_rollouts=1,
        )

    with pytest.raises(ValueError, match="loopback"):
        manager("http://10.0.0.2:8000", url).reject_loopback_servers(num_nodes=2)
    # One node reaches its own engine, which the forwarding rank's client pushes to.
    manager("http://10.0.0.2:8000", url).reject_loopback_servers(num_nodes=1)
    manager("http://10.0.0.2:8000").reject_loopback_servers(num_nodes=2)


def test_tools_schema_detection():
    """The server rejects any other shape, so the key values matter."""
    native_env = build_environment("native_math", {"max_turns": 3})
    schema = native_env.get_tools_schema()

    assert isinstance(schema, list) and schema
    for tool in schema:
        assert tool["type"] == "function"
        assert set(tool["function"]) >= {"name", "description", "parameters"}
        assert tool["function"]["parameters"]["type"] == "object"


def test_tool_calls_in_step_context():
    env = build_environment("native_math", {"max_turns": 5})

    episode_ids, steps = env.reset(["What is 2+2?"], [{"answer": "4"}])
    episode_id = episode_ids[0]

    # Simulate a step with tool_calls in context (as vLLM would return)
    mock_tool_calls = [
        {"id": "call_123", "type": "function", "function": {"name": "calculate", "arguments": '{"expression": "2+2"}'}}
    ]

    step_context = {"answer": "4", "tool_calls": mock_tool_calls}
    env.step([episode_id], [""], [step_context])

    traj = env._trajectories[episode_id]
    assistant_msgs = [m for m in traj.messages if m.role == "assistant"]
    assert len(assistant_msgs) >= 1
    assert assistant_msgs[-1].tool_calls == mock_tool_calls


def _make_actor(env_type="native_math", env_config=None):
    """The real EnvironmentActor off-cluster: ``@ray.remote`` keeps the class at
    ``__ray_metadata__.modified_class``, and ``__init__`` only sets attributes."""
    cls = EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type=env_type, env_config=env_config or {"max_turns": 3})
    return actor


def test_build_payload_sends_only_supported_params_and_env_tools():
    actor = _make_actor("native_math")  # native_math exposes OpenAI-format tools
    payload = actor._build_payload(
        [{"role": "user", "content": "2+2?"}],
        RolloutConfig(temperature=0.9, top_p=0.8, max_tokens=512),
    )

    assert payload["messages"][0]["content"] == "2+2?"
    assert payload["temperature"] == 0.9 and payload["top_p"] == 0.8 and payload["max_tokens"] == 512
    assert payload["tools"] and payload["tools"][0]["type"] == "function" and "function" in payload["tools"][0]
    # model is omitted when model_name is unset, and no unsupported sampling knob leaks through (the
    # sampler filters each request carries are pinned in tests/cpu/weight_sync/test_rollout_backend_selection.py).
    assert "model" not in payload
    for leaked in ("frequency_penalty", "presence_penalty"):
        assert leaked not in payload


def test_build_payload_gates_model_and_omits_empty_tools():
    """model is added iff model_name is set; tools key is dropped when the env exposes none."""
    actor = _make_actor("native_math")
    actor._tools_schema = None  # prime the cache to the no-tools branch (skips env construction)
    payload = actor._build_payload([{"role": "user", "content": "hi"}], RolloutConfig(model_name="Qwen/Qwen3-4B"))

    assert payload["model"] == "Qwen/Qwen3-4B"
    assert "tools" not in payload


def test_build_payload_carries_the_chat_template_kwargs_only_when_set():
    """The run's template variables ride nested; the effort level never does (it travels top-level)."""
    actor = _make_actor("native_math")
    actor._tools_schema = None
    messages = [{"role": "user", "content": "hi"}]
    carried = actor._build_payload(messages, RolloutConfig(chat_template_kwargs={"preserve_thinking": True}), "low")
    assert carried["chat_template_kwargs"] == {"preserve_thinking": True}
    assert carried["reasoning_effort"] == "low"
    assert "chat_template_kwargs" not in actor._build_payload(messages, RolloutConfig(), "low")


def test_build_payload_sends_no_tools_for_a_react_env():
    """ReAct parses its action out of the text; a server's tool-call parser would strip it out of ``content`` and
    burn the turn."""
    react_payload = _make_actor("react_math")._build_payload([{"role": "user", "content": "2+2?"}], RolloutConfig())
    native_payload = _make_actor("native_math")._build_payload([{"role": "user", "content": "2+2?"}], RolloutConfig())

    assert "tools" not in react_payload
    # Positive control: the native protocol still advertises its schema through the same code path.
    assert native_payload["tools"]


# The fields a rollout is allowed to carry home. Every one is pickled through the Ray object store and
# again through the TP broadcast, once per episode, so an unread field ships the whole per-episode
# payload (code_contests: the hidden-test set) for nothing.
CONSUMED_ROLLOUT_RESULT_FIELDS = {
    "prompt",
    "trajectory",
    "episode_length",
    "total_reward",
    "terminated_naturally",
    "latency",
    "error",
    "generation_tokens",
    "requests_expired_in_sync",
    "metrics",
}

# Where a rollout result is read: the actor/manager that builds it and the trainer that consumes it.
ROLLOUT_RESULT_CONSUMERS = ("halo/environments", "halo/trainers/grpo")


def _bare_local_attribute_reads(root):
    """Attribute names read off a bare local anywhere under ``root``; ``self`` is excluded, since it reads config
    fields of the same names."""
    names = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                base = node.value
                if isinstance(base, ast.Name) and base.id != "self":
                    names.add(node.attr)
    return names


def test_rollout_result_carries_only_consumed_fields():
    """Fails when a field changes without the contract, and when a field stops being read in the actor/trainer path."""
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
    """SGLang spells the cut ``stop_reason`` and this transport reads the body as a plain dict; knowing only
    ``finish_reason`` clears the truncation and grades a fragment."""
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


async def test_a_turn_that_used_its_whole_cap_is_a_cut_even_when_vllm_says_tool_calls():
    """vLLM labels the finish ``tool_calls`` whenever its parser extracted a call, including one salvaged from a cut
    turn, so the transport reads the cap off ``usage.completion_tokens``."""
    actor = _make_actor("native_math")
    salvaged = {"id": "c1", "type": "function", "function": {"name": "python_repl", "arguments": "{}"}}
    session = _FakeChatCompletionsSession(
        {
            "choices": [
                {
                    "message": {"content": "Let me implement this", "tool_calls": [salvaged]},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"completion_tokens": 4300},
        }
    )

    generation = await actor._generate(
        session, "server:8000", [{"role": "user", "content": "2+2?"}], RolloutConfig(max_retries=0, max_tokens=4300)
    )

    assert generation.finish_reason == FINISH_REASON_LENGTH, (
        "a completion that consumed max_tokens was taken at vLLM's word: the salvaged call executes as a malformed one"
    )


_PARTIAL_CALL = {
    "id": "c1",
    "type": "function",
    "function": {"name": "calculate", "arguments": '{"expression": "1+ # still reasoning'},
}


@pytest.mark.parametrize(
    ("backend", "choice"),
    [
        ("vllm", {"finish_reason": "tool_calls"}),
        ("sglang", {"stop_reason": FINISH_REASON_LENGTH}),
    ],
)
async def test_a_call_cut_at_the_cap_reaches_the_env_as_a_cut_call_on_either_engine(backend, choice):
    """vLLM labels a turn its cap cut inside a call ``tool_calls``, SGLang spells the cut ``stop_reason``;
    either way the transport reads a length cut, and the salvaged call travels as a cut call, never as one
    to run."""
    actor = _make_actor("native_math")
    session = _FakeChatCompletionsSession(
        {
            "choices": [{"message": {"content": "Computing", "tool_calls": [_PARTIAL_CALL]}, **choice}],
            "usage": {"completion_tokens": 4300},
        }
    )
    config = RolloutConfig(backend=backend, max_retries=0, max_tokens=4300)
    generation = await actor._generate(session, "server:8000", [{"role": "user", "content": "2+2?"}], config)

    ctx = step_context_from_generation(None, generation)
    assert ctx[CUT_TOOL_CALLS_KEY] == [_PARTIAL_CALL] and "tool_calls" not in ctx


def test_a_row_column_named_like_a_turn_stamp_never_reads_as_the_turns_output():
    """A stamp is written only when the turn produced it, so a dataset column of that name would stand in for it: the
    row's calls would run on a plain text turn and its rationale would be priced as the turn's reasoning."""
    columns = {
        "tool_calls": [{"id": "c", "function": {"name": "echo", "arguments": '{"text": "from the row"}'}}],
        "reasoning": "the dataset's rationale",
        "token_ids": [1],
        "token_logprobs": [-0.5],
        "prompt_token_ids": [2],
        "routing_mask": "mask",
        "routing_prompt_tokens": 1,
        "thinking_cap": 8,
        "reasoning_tokens": 3,
        CUT_TOOL_CALLS_KEY: [{"id": "c"}],
        "last_turn": True,
        "finish_reason": "length",
    }
    row = {"answer": "4", "reasoning_effort": "high", **columns}
    turn = TurnGeneration(text="4", tool_calls=[], reasoning="", tokens=1, finish_reason="stop")
    assert step_context_from_generation(row, turn) == {
        "answer": "4",
        "reasoning_effort": "high",
        "finish_reason": "stop",
    }

    ran = []
    registry = NativeToolRegistry().register(
        NativeTool(name="echo", description="d", parameters=[ToolParameter("text", "string", "t")], handler=ran.append)
    )
    env = NativeToolUseEnvironment(tool_registry=registry, max_turns=3)
    ids, _ = env.reset(["q"], [row])
    step = env.step(ids, [turn.text], [step_context_from_generation(row, turn)])[0]
    assert step.done and not ran, "a text turn ends the episode, whatever the row's tool_calls column holds"
    assert step.trajectory.last_assistant_message().thinking is None
    assert step.trajectory.total_reward == 1.0, "the row's answer still grades the turn"


def test_a_turns_own_stamp_wins_over_a_row_column_of_the_same_name():
    row = {"reasoning": "the dataset's rationale", "tool_calls": [{"id": "row"}]}
    call = {"id": "turn", "function": {"name": "echo", "arguments": "{}"}}
    turn = TurnGeneration(text="", tool_calls=[call], reasoning="the turn's own", tokens=1, finish_reason="tool_calls")
    stamped = step_context_from_generation(row, turn)
    assert stamped["reasoning"] == "the turn's own" and stamped["tool_calls"] == [call]


class _FakeEngineSession(_FakeChatCompletionsSession):
    """Serves the canned completion and vLLM's ``/detokenize``, recording the ids each decode asked for."""

    def __init__(self, body: dict, *, decoded: str = "", decode_status: int = 200):
        super().__init__(body)
        self._decoded, self._decode_status, self._url = decoded, decode_status, ""
        self.decoded: list[list[int]] = []

    def post(self, url, json):  # noqa: A002 — aiohttp's own keyword
        self._url = url
        if url.endswith("/detokenize"):
            self.decoded.append(json["tokens"])
        return self

    async def __aenter__(self):
        if self._url.endswith("/detokenize"):
            return SimpleNamespace(status=self._decode_status, json=self._decode, text=self._text)
        return await super().__aenter__()

    async def _decode(self):
        return {"prompt": self._decoded}


_END = 99
_CUT_IDS = [11, _END, 12, _END, 21, 22, 23]
# The same ids where SGLang reports them, ``[logprob, id, text]`` per sampled token.
_SGLANG_CUT_META = {"output_token_logprobs": [[-0.1, i, f"t{i}"] for i in _CUT_IDS]}
_WRITTEN = "<tool_call>\n<function=calculate>\n<parameter=expression>\n1+ # still reasoning, so let me"


def _cut_turn_body(backend_choice: dict) -> dict:
    logprobs = {"content": [{"token": f"token_id:{i}", "logprob": -0.1} for i in _CUT_IDS]}
    message = {"content": "", "tool_calls": [_PARTIAL_CALL]}
    return {
        "choices": [{"message": message, "logprobs": logprobs, **backend_choice}],
        "usage": {"completion_tokens": len(_CUT_IDS)},
    }


async def test_a_call_cut_at_the_cap_reaches_the_judge_as_the_policy_wrote_it():
    """vLLM's parser keeps only the name and the closed arguments of a call cut at the cap; the turn's ids past its
    reasoning close are decoded and travel as that cut call's arguments, so the judge reads the whole call."""
    actor = _make_actor("native_math")
    session = _FakeEngineSession(_cut_turn_body({"finish_reason": "tool_calls"}), decoded=_WRITTEN)
    config = RolloutConfig(
        max_retries=0, max_tokens=len(_CUT_IDS), capture_token_ids=True, reasoning_end_token_id=_END
    )
    generation = await actor._generate(session, "server:8000", [{"role": "user", "content": "2+2?"}], config)

    assert session.decoded == [[21, 22, 23]], "only the ids past the last reasoning close are decoded"
    written = {**_PARTIAL_CALL, "function": {"name": "calculate", "arguments": _WRITTEN}}
    assert step_context_from_generation(None, generation)[CUT_TOOL_CALLS_KEY] == [written]


@pytest.mark.parametrize(
    ("backend", "choice", "decode_status", "end"),
    [
        ("vllm", {"finish_reason": "tool_calls"}, 500, _END),
        ("vllm", {"finish_reason": "tool_calls"}, 200, 7),
        ("sglang", {"stop_reason": FINISH_REASON_LENGTH, "meta_info": _SGLANG_CUT_META}, 200, _END),
    ],
    ids=["decode-fails", "no-reasoning-close-in-the-ids", "sglang"],
)
async def test_a_cut_call_keeps_the_parsers_salvage_where_it_cannot_be_decoded(backend, choice, decode_status, end):
    """A failed decode, ids holding no reasoning close, or SGLang (its ids captured, but the decode reads vLLM's
    ``/detokenize`` answer) leave the salvaged call as it came, still booked as a cut call."""
    actor = _make_actor("native_math")
    session = _FakeEngineSession(_cut_turn_body(choice), decoded=_WRITTEN, decode_status=decode_status)
    config = RolloutConfig(
        backend=backend, max_retries=0, max_tokens=len(_CUT_IDS), capture_token_ids=True, reasoning_end_token_id=end
    )
    generation = await actor._generate(session, "server:8000", [{"role": "user", "content": "2+2?"}], config)

    assert step_context_from_generation(None, generation)[CUT_TOOL_CALLS_KEY] == [_PARTIAL_CALL]
    assert session.decoded == ([[21, 22, 23]] if decode_status != 200 else [])


async def test_the_actor_keeps_a_cut_call_on_its_turn_without_running_or_resending_it():
    """Through the real episode loop: the call a turn was writing when its cap cut it lands on that turn's
    message, survives the pickle the Ray object store makes of the result, never runs, and is absent from
    what the engine is sent next."""
    actor = _make_actor("native_math", {"max_turns": 5})

    actor._http_client = None
    sent: list[list[dict]] = []

    async def _fake_generate(client, url, messages, config, reasoning_effort=None):
        sent.append(messages)
        if len(sent) == 1:
            return TurnGeneration("Computing", [_PARTIAL_CALL], "", 4300, finish_reason=FINISH_REASON_LENGTH)
        return TurnGeneration("final answer: 4", [], "", 5)

    actor._generate = _fake_generate
    result = await actor.run_episode("2+2?", {"answer": "4"}, "http://x", RolloutConfig(max_retries=1))

    assert len(sent) == 2 and result.error is None
    cut_turn = next(m for m in sent[1] if m["role"] == "assistant")
    assert cut_turn == {"role": "assistant", "content": "Computing"}
    assert "still reasoning" not in json.dumps(sent[1])
    assert result.trajectory.info["total_tool_calls"] == 0
    shipped = pickle.loads(pickle.dumps(result))
    turns = [m for m in shipped.trajectory.messages if m.role == "assistant"]
    assert [m.cut_tool_calls for m in turns] == [[_PARTIAL_CALL], None]
    assert [m.tool_calls for m in turns] == [None, None]


async def test_actor_releases_session_when_episode_errors():
    """run_episode cleans up in a finally, so a long-lived actor does not leak a working directory per failed
    rollout."""
    actor = _make_actor("swe", {"max_turns": 5})
    env = actor._env  # force construction so we can inspect its sessions

    created = []
    orig_open = env.sandbox.open_session
    env.sandbox.open_session = lambda: created.append(orig_open()) or created[-1]

    async def _fake_client():  # unused (we mock _generate), just must not hit the network
        return None

    actor._http_client = None

    calls = {"n": 0}

    async def _fake_generate(client, url, messages, config, reasoning_effort=None):
        calls["n"] += 1
        if calls["n"] == 1:  # turn 1: write a file -> creates the episode's session
            tc = [{"id": "c1", "function": {"name": "write_file", "arguments": '{"path": "f.txt", "content": "x"}'}}]
            return TurnGeneration("", tc, "", 1)
        raise RuntimeError("boom-mid-episode")  # turn 2: fail, leaving a live session

    actor._generate = _fake_generate

    result = await actor.run_episode("task", None, "http://x", RolloutConfig(max_retries=1, retry_base_wait=0.0))

    assert result.error and "boom" in result.error, "episode should surface the mid-episode error"
    assert created, "write_file should have opened a session (otherwise the test proves nothing)"
    assert env._sessions == {}, "session must be released on the error path — no leak"
    for session in created:
        assert not os.path.exists(session.workdir), "session temp dir must be deleted by cleanup"


async def test_actor_drives_async_env_via_step_async():
    """Async-only tool handlers must run through reset_async/step_async; a sync step would hit NativeTool.execute's
    no-sync-handler error."""
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
            async_handler=_async_handler,  # no sync handler: runs only via the async path
        )
    )

    actor = _make_actor(
        env_type=(AsyncNativeToolUseEnvironment, {"tool_registry": registry}),
        env_config={"max_turns": 3},
    )

    async def _fake_client():
        return None

    actor._http_client = None

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


async def test_run_episode_generation_tokens_sum_across_turns():
    """The source of episode/generation_tokens, so per-turn accounting would silently mis-report. Drives the real
    multi-turn loop, mocking only the vLLM call."""
    actor = _make_actor("native_math", {"max_turns": 5})

    async def _fake_client():
        return None

    actor._http_client = None

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


async def test_each_assistant_turn_records_the_runs_cap_and_the_reasoning_it_sampled():
    """400-token turns with a 100-token cap under a 1050-token episode budget: the third request narrows to 350 and
    50, and the turn records the run's cap."""
    end = 151668
    actor = _make_actor("native_math", {"max_turns": 5})

    async def _fake_client():
        return None

    actor._http_client = None
    caps_sent: list[tuple[int, int | None]] = []
    reasoning = [[1] * 30 + [end, 2], [1] * 60 + [end], [1] * 5 + [end]]
    sampled = [350, 350, 6]

    async def _fake_generate(client, url, messages, config, reasoning_effort=None):
        i = len(caps_sent)
        caps_sent.append((config.max_tokens, config.max_thinking_tokens))
        if i < 2:
            tc = [{"id": f"c{i}", "function": {"name": "calculate", "arguments": '{"expression": "1+1"}'}}]
            return TurnGeneration("", tc, "r", sampled[i], token_ids=reasoning[i])
        return TurnGeneration("final answer: 4", [], "r", sampled[i], token_ids=reasoning[i])

    actor._generate = _fake_generate
    config = RolloutConfig(
        max_retries=1,
        max_tokens=400,
        max_thinking_tokens=100,
        max_episode_tokens=1050,
        capture_token_ids=True,
        reasoning_end_token_id=end,
    )
    result = await actor.run_episode("2+2?", {"answer": "4"}, "http://x", config)

    assert caps_sent == [(400, 100), (400, 100), (350, 50)], "the setup no longer narrows the third request"
    turns = [m for m in result.trajectory.messages if m.role == "assistant"]
    assert [(m.thinking_cap, m.reasoning_tokens) for m in turns] == [(100, 31), (100, 61), (100, 6)]
    assert result.generation_tokens == sum(sampled)


async def test_actor_grades_concurrent_codecontests_episodes_in_isolation():
    """The submit handler reads its episode from the ``_ACTIVE_TRAJECTORY`` ContextVar set per step; if that broke,
    one episode's submission would grade against the other's tests. Runs the genuine sync env and local
    subprocess grader."""
    # A: double the input; B: increment it. Each problem's hidden test is satisfied only by its own program.
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
    assert actor._env is actor._env, "both episodes must share one cached env instance"

    async def _fake_client():
        return None

    actor._http_client = None

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
    # The two episodes really were distinct (each graded its own submission), proving the test is not
    # vacuous; the hidden tests themselves are dropped from a finished trajectory.
    calls_a, calls_b = ([m.tool_calls for m in r.trajectory.messages if m.tool_calls] for r in (result_a, result_b))
    assert calls_a and calls_a != calls_b
    assert "_test_cases" not in result_a.trajectory.info


async def test_slow_sync_step_does_not_block_concurrent_episodes():
    """Sandboxed grading blocks for seconds to minutes. Two episodes rendezvous inside step on a 2-party barrier:
    offloaded both pass; stepped inline, the barrier times out."""
    barrier = threading.Barrier(2, timeout=10.0)

    class _BarrierStepEnv(NativeToolUseEnvironment):
        def step(self, episode_ids, actions, contexts=None):
            barrier.wait()  # BrokenBarrierError (-> episode error) unless both steps overlap
            return super().step(episode_ids, actions, contexts)

    actor = _make_actor(
        env_type=(_BarrierStepEnv, {"tool_registry": NativeToolRegistry()}),
        env_config={"max_turns": 2},
    )

    async def _fake_client():
        return None

    actor._http_client = None

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
    """A value set in turn 1 stays visible on turn 2 of the same episode and never leaks to a concurrent one.
    Per-call ``asyncio.to_thread`` would drop writes, and a shared context would leak."""
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

    async def _fake_client():
        return None

    actor._http_client = None

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


def test_only_a_genuine_client_error_gives_up():
    def _http(status, body="boom"):
        return RolloutHTTPError(status, "vllm", body)

    assert _should_giveup(_http(400, "Bad request"))
    assert _should_giveup(_http(422, "Unprocessable"))
    assert _should_giveup(_http(404, "Not found"))

    assert not _should_giveup(_http(500, "Internal error"))
    assert not _should_giveup(_http(503, "Unavailable"))

    assert not _should_giveup(TimeoutError("Connection timed out"))
    assert not _should_giveup(ConnectionError("Connection refused"))

    # 429 (rate limit) and 408 (request timeout) are 4xx but transient, so they are retried.
    assert not _should_giveup(_http(429, "Too Many Requests"))
    assert not _should_giveup(_http(408, "Request Timeout"))


def test_an_engine_serialization_fault_under_a_400_is_retried():
    """vLLM reports a NaN log-prob it cannot serialise as a 400 though a fresh request succeeds; giving up ends
    every in-flight episode on that engine."""
    body = (
        '{"error":{"message":"Out of range float values are not JSON compliant: nan",'
        '"type":"BadRequestError","param":null,"code":400}}'
    )
    fault = RolloutHTTPError(400, "vllm", body)
    assert not _should_giveup(fault) and not _should_giveup(fault)
    # A genuine bad request under the same status is still terminal.
    assert _should_giveup(RolloutHTTPError(400, "vllm", '{"error":{"message":"max_tokens must be positive"}}'))


def test_a_server_error_quoting_a_4xx_in_its_body_is_still_retried():
    """The status is read as data: a 503 from a proxy quoting the upstream status 400 must not abandon the batch."""
    exc = RolloutHTTPError(503, "vllm", "upstream returned status 400: bad request")

    assert "status 400" in str(exc)
    assert not _should_giveup(exc)
    # And a plain RuntimeError carrying the same text is a transport fault, not a client error.
    assert not _should_giveup(RuntimeError("vllm error (status 400): Bad request"))


def test_async_rollout_config_builds_the_rollout_config():
    config = AsyncRolloutConfig(
        rollout_temperature=0.9,
        rollout_top_p=0.8,
        rollout_top_k=40,
        rollout_min_p=0.02,
        rollout_repetition_penalty=1.05,
        rollout_max_tokens=2048,
        model_name="test-model",
        request_timeout=60.0,
        max_retries=5,
        retry_base_wait=2.0,
    )

    rc = config.get_rollout_config()
    assert rc.temperature == 0.9
    assert rc.top_p == 0.8
    assert (rc.top_k, rc.min_p, rc.repetition_penalty) == (40, 0.02, 1.05)
    assert rc.max_tokens == 2048
    assert rc.model_name == "test-model"
    assert rc.request_timeout == 60.0
    assert rc.max_retries == 5
    assert rc.retry_base_wait == 2.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
