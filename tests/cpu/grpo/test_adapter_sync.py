#!/usr/bin/env python
"""Adapter-only sync as the trainer drives it, against fake SGLang servers: the base goes out once with the adapters
left out, each sync loads the adapter under a new name that new episodes select, a version leaves only once no episode
holds it, and a restarted server gets the base and the adapter again. And every model or server the engine cannot serve
as an adapter is refused with its reason."""

import asyncio
import time
from types import SimpleNamespace

import pytest
import requests
import torch
from huggingface_hub import constants
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.environments.episodes.messages import Trajectory
from halo.environments.rollout_config import RolloutConfig
from halo.environments.rollouts.engine_wire import build_payload
from halo.environments.rollouts.episode_loop import RolloutResult
from halo.environments.rollouts.ray_actors import EnvironmentActor, RolloutManager
from halo.models.introspection.unwrapping import PEFT_BASE_MODEL_PREFIX
from halo.trainers.grpo.rollout import adapter_sync
from halo.trainers.grpo.rollout.adapter_sync import (
    ADAPTER_NAME_PREFIX,
    AdapterSync,
    build_adapter_requirements,
    reject_unadaptable_config,
    reject_unadaptable_model,
    resolve_adapter_serving,
    resolve_live_versions,
)
from halo.trainers.grpo.rollout.engine_preflight import AdapterRequirements, verify_adapter_servers
from halo.weight_sync.adapter_routes import AdapterRoutes
from halo.weight_sync.adapter_servers import AdapterServers
from halo.weight_sync.adapter_versions import AdapterVersions
from halo.weight_sync.clients.base import ServedBase
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.adapter_sync import build_adapter_sync
from tests.common.fake_engines import FakeSGLangServer
from tests.common.tiny_models import TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, tiny_family_model

_LORA = {
    "max_lora_rank": 16,
    "lora_target_modules": ["q_proj", "v_proj"],
    "max_loaded_loras": 8,
    "max_loras_per_batch": 3,
}
_NO_PARALLELISM = SimpleNamespace(is_tp_mode=False, is_pp_mode=False, is_cp_mode=False)
_BASE_MODEL = "org/policy"
# What the fake servers launch with: the trainer's base when a sync is given it.
_LAUNCH_BASE = ServedBase(model_id=_BASE_MODEL, path="/models/policy", revision=None)


def _peft_model(*, targets=("q_proj", "v_proj"), dtype=torch.bfloat16, **lora) -> torch.nn.Module:
    torch.manual_seed(0)
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
    )
    model = get_peft_model(
        Qwen3ForCausalLM(config), LoraConfig(r=8, lora_alpha=24, target_modules=list(targets), **lora)
    ).to(dtype)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    return model


@pytest.fixture
def servers():
    fakes = [FakeSGLangServer(), FakeSGLangServer()]
    for fake in fakes:
        fake.lora = dict(_LORA)
    pool = AdapterServers([fake.url for fake in fakes], client_cls=SGLangWeightSyncClient, connection_timeout=5.0)
    yield pool, fakes
    pool.close()
    for fake in fakes:
        fake.close()


class _RecordingPool:
    def __init__(self, events: list):
        self.events = events
        self.marked: set[str] = set()

    def mark_servers_restarted(self, urls) -> None:
        self.marked.update(urls)

    def reconnect_servers(self, urls=()) -> None:
        urls = self.marked | set(urls)
        self.marked.clear()
        if urls:
            self.events.append(("reconnect", sorted(urls)))


@pytest.fixture
def events():
    """Reconnects and base pushes in the order a sync made them."""
    return []


@pytest.fixture
def client_pool(events):
    return _RecordingPool(events)


@pytest.fixture
def base_pushes(monkeypatch, events):
    """The full pushes the sync asked for, as the ``fold_adapters`` each carried."""
    pushes = []

    def push(trainer, pool, *, fold_adapters=True):
        pushes.append(fold_adapters)
        events.append(("push", fold_adapters))

    monkeypatch.setattr(adapter_sync, "sync_trainer_weights", push)
    return pushes


def _adapter_sync(model, pool, launch_base=None, base_refusal=None, adapter_dir=None) -> AdapterSync:
    routes = AdapterRoutes(SGLangWeightSyncClient.is_adapter_missing)
    serving = resolve_adapter_serving(model)
    sync = AdapterSync(serving, AdapterVersions(), routes, launch_base, base_refusal, adapter_dir)
    sync.servers = pool
    return sync


@pytest.fixture
def sync_of(servers):
    pool, _ = servers
    model = _peft_model()
    return _adapter_sync(model, pool), SimpleNamespace(model=model)


@pytest.fixture
def same_base_sync_of(servers):
    """A sync whose trainer base is the one the fake servers launch with."""
    pool, _ = servers
    model = _peft_model()
    return _adapter_sync(model, pool, _LAUNCH_BASE), SimpleNamespace(model=model)


def _held_everywhere(fakes) -> list[list[str]]:
    return [sorted(fake.adapters) for fake in fakes]


def test_the_first_sync_pushes_the_base_unfolded_then_loads_the_adapter(sync_of, servers, base_pushes, client_pool):
    (sync, trainer), (_, fakes) = sync_of, servers

    sync.sync(trainer, client_pool)

    assert base_pushes == [False], "the base alone: folding the adapter in would count its delta twice"
    [name] = sync.versions.held() | {sync.versions.current}
    assert name.startswith(f"{ADAPTER_NAME_PREFIX}-")
    assert _held_everywhere(fakes) == [[name], [name]]


def test_the_wire_adapter_is_the_trainers_with_the_scaling_folded_into_b(sync_of, servers, base_pushes, client_pool):
    (sync, trainer), (_, [fake, _]) = sync_of, servers

    sync.sync(trainer, client_pool)

    config, tensors = fake.adapters[sync.versions.current]
    assert config["lora_alpha"] == config["r"] == 8 and config["target_modules"] == ["q_proj", "v_proj"]
    live = dict(trainer.model.named_parameters())
    layer0_q = "model.layers.0.self_attn.q_proj"
    assert set(tensors) == {
        f"base_model.model.model.layers.{layer}.self_attn.{proj}.lora_{ab}.weight"
        for layer in (0, 1)
        for proj in ("q_proj", "v_proj")
        for ab in "AB"
    }
    a = live[f"base_model.model.{layer0_q}.lora_A.default.weight"].detach()
    b = live[f"base_model.model.{layer0_q}.lora_B.default.weight"].detach()
    assert torch.equal(tensors[f"base_model.model.{layer0_q}.lora_A.weight"], a)
    assert torch.equal(tensors[f"base_model.model.{layer0_q}.lora_B.weight"], (b.float() * (24 / 8)).bfloat16())


def test_a_later_sync_loads_a_new_name_and_retires_the_old_one(sync_of, servers, base_pushes, client_pool):
    (sync, trainer), (_, fakes) = sync_of, servers
    sync.sync(trainer, client_pool)
    first = sync.versions.current

    sync.sync(trainer, client_pool)

    assert base_pushes == [False], "the base is pushed once"
    assert sync.versions.current != first
    assert _held_everywhere(fakes) == [[sync.versions.current]] * 2


def test_a_version_an_episode_holds_stays_loaded_until_it_ends(sync_of, servers, base_pushes, client_pool):
    (sync, trainer), (_, fakes) = sync_of, servers
    sync.sync(trainer, client_pool)

    with sync.versions.lease() as running:
        sync.sync(trainer, client_pool)
        assert _held_everywhere(fakes) == [sorted([running, sync.versions.current])] * 2

    sync.sync(trainer, client_pool)
    assert _held_everywhere(fakes) == [[sync.versions.current]] * 2


def test_a_restarted_server_gets_the_base_and_the_adapter_again(sync_of, servers, base_pushes, client_pool):
    (sync, trainer), (_, [fake, restarted]) = sync_of, servers
    sync.sync(trainer, client_pool)
    restarted.restart()

    sync.sync(trainer, client_pool)

    assert base_pushes == [False, False], "the restart's base push is the unfolded one too"
    assert _held_everywhere([fake, restarted]) == [[sync.versions.current]] * 2


def test_a_restarted_servers_client_reconnects_before_the_base_is_sent(
    sync_of, servers, base_pushes, events, client_pool
):
    """A push into the restarted server's old communicator waits out the broadcast deadline for a peer that is gone."""
    (sync, trainer), (_, [_, restarted]) = sync_of, servers
    sync.sync(trainer, client_pool)
    events.clear()
    restarted.restart()

    sync.sync(trainer, client_pool)

    assert events == [("reconnect", [restarted.url]), ("push", False)]


def test_servers_that_kept_their_adapter_are_not_reconnected(sync_of, base_pushes, events, client_pool):
    sync, trainer = sync_of
    sync.sync(trainer, client_pool)
    sync.sync(trainer, client_pool)

    assert events == [("push", False)], "only the first sync pushes the base, and nothing reconnects"


def test_a_predecessors_versions_are_retired_with_the_first_push(sync_of, servers, base_pushes, client_pool):
    (sync, trainer), (pool, [fake, _]) = sync_of, servers
    left = _adapter_sync(trainer.model, pool)
    left.sync(trainer, client_pool)
    stale = left.versions.current

    sync.sync(trainer, client_pool)

    assert stale not in fake.adapters and sync.versions.current in fake.adapters


def test_requests_name_the_newest_version_for_a_whole_episode(sync_of, servers, base_pushes, client_pool):
    (sync, trainer), (_, [fake, _]) = sync_of, servers
    manager = RolloutManager(
        num_workers=1,
        env_type="unused",
        env_config={},
        server_urls=[fake.url],
        rollout_config=RolloutConfig(backend="sglang", model_name=_BASE_MODEL),
        max_concurrent_rollouts=1,
        adapter_versions=sync.versions,
        adapter_model_name=lambda adapter: SGLangWeightSyncClient.adapter_request_model(_BASE_MODEL, adapter),
    )
    sync.sync(trainer, client_pool)
    first = sync.versions.current

    with manager._episode_config() as (old_episode, old_version):
        sync.sync(trainer, client_pool)
        with manager._episode_config() as (new_episode, new_version):
            pass

    assert (old_version, new_version) == (first, sync.versions.current)
    assert build_payload([], old_episode)["model"] == f"{_BASE_MODEL}:{first}"
    assert build_payload([], new_episode)["model"] == f"{_BASE_MODEL}:{sync.versions.current}"
    assert first != sync.versions.current


class _FakeActor:
    """Plays an episode as one chat completion on the server it is given, as an actor's first request would."""

    def __init__(self, failure: str | None = None):
        self.run_episode = SimpleNamespace(remote=self._run)
        self.failure = failure

    def _run(self, *, prompt, context, server_url, config):
        return asyncio.to_thread(self._post, prompt, server_url, config.model_name)

    def _post(self, prompt, url, model):
        if self.failure is not None:
            return RolloutResult.errored(prompt, self.failure)
        reply = requests.post(f"{url}/v1/chat/completions", json={"model": model, "messages": []}, timeout=10)
        if reply.status_code != 200:
            return RolloutResult.errored(
                prompt, f"RolloutHTTPError: sglang error (status {reply.status_code}): {reply.text}"
            )
        return RolloutResult(prompt=prompt, trajectory=Trajectory(done=True))


def _manager_over(sync, fakes, actor=None) -> RolloutManager:
    manager = RolloutManager(
        num_workers=1,
        env_type="unused",
        env_config={},
        server_urls=[fake.url for fake in fakes],
        rollout_config=RolloutConfig(backend="sglang", model_name=_BASE_MODEL),
        max_concurrent_rollouts=1,
        adapter_versions=sync.versions,
        adapter_model_name=lambda adapter: SGLangWeightSyncClient.adapter_request_model(_BASE_MODEL, adapter),
        adapter_routes=sync.routes,
    )
    manager._actors = [actor or _FakeActor()]
    manager._started = True
    return manager


def _play(manager, episodes: int) -> list[RolloutResult]:
    return asyncio.run(manager.collect_rollouts([f"p{i}" for i in range(episodes)]))


def test_a_restarted_server_is_left_out_of_routing_once_a_probe_finds_it(sync_of, servers, base_pushes, client_pool):
    (sync, trainer), (_, [kept, restarted]) = sync_of, servers
    sync.sync(trainer, client_pool)
    restarted.restart()

    sync.recover(client_pool)
    results = _play(_manager_over(sync, [kept, restarted]), 4)

    assert all(result.error is None for result in results)
    assert restarted.completion_models == [], "no episode was sent to the server without the leased version"
    assert len(kept.completion_models) == 4


def test_a_server_that_answers_not_loaded_is_excluded_and_the_episode_replayed(
    sync_of, servers, base_pushes, client_pool
):
    """No probe saw the restart, so the first episode routed there learns it from the engine's own answer."""
    (sync, trainer), (_, [kept, restarted]) = sync_of, servers
    sync.sync(trainer, client_pool)
    restarted.restart()

    results = _play(_manager_over(sync, [kept, restarted]), 4)

    assert all(result.error is None for result in results), "the refused episode ran again on a server that holds it"
    assert len(restarted.completion_models) == 1, "one refusal, then the server is skipped"
    assert len(kept.completion_models) == 4


class _EnvironmentActorPlaying:
    """The real actor's episode loop and chat request, off-cluster, standing in for the Ray actor a manager calls."""

    def __init__(self):
        cls = EnvironmentActor.__ray_metadata__.modified_class
        self.actor = cls.__new__(cls)
        self.actor.__init__(actor_id=0, env_type="native_math", env_config={"max_turns": 1})
        self.run_episode = SimpleNamespace(remote=self.actor.run_episode)


def test_the_engines_unloaded_adapter_answer_reaches_the_routing_through_the_actors_request(
    sync_of, servers, base_pushes, client_pool
):
    """The fake's 400 body is SGLang's own; it goes out through the actor's chat request and error row, not a
    hand-built message, and the routing excludes the server from it."""
    (sync, trainer), (_, [kept, restarted]) = sync_of, servers
    sync.sync(trainer, client_pool)
    restarted.restart()
    manager = _manager_over(sync, [restarted, kept], _EnvironmentActorPlaying())
    manager.rollout_config = RolloutConfig(backend="sglang", model_name=_BASE_MODEL, max_retries=0)

    results = _play(manager, 1)

    assert restarted.completion_models == [f"{_BASE_MODEL}:{sync.versions.current}"], "the request named the adapter"
    assert sync.routes.eligible([kept.url, restarted.url], sync.versions.current) == [kept.url]
    assert len(kept.completion_models) == 1, "and the episode ran again on the server that holds it"
    assert len(results) == 1


def test_an_error_that_is_not_a_missing_adapter_is_not_routed_around(sync_of, servers, base_pushes, client_pool):
    (sync, trainer), (_, fakes) = sync_of, servers
    sync.sync(trainer, client_pool)
    manager = _manager_over(sync, fakes, _FakeActor(failure="RolloutHTTPError: sglang error (status 500): boom"))

    results = _play(manager, 2)

    assert [result.error for result in results] == ["RolloutHTTPError: sglang error (status 500): boom"] * 2
    assert sync.routes.eligible([fake.url for fake in fakes], sync.versions.current) == [fake.url for fake in fakes]


def test_every_server_lacking_the_version_errors_the_episode(sync_of, servers, base_pushes, client_pool):
    (sync, trainer), (_, fakes) = sync_of, servers
    sync.sync(trainer, client_pool)
    for fake in fakes:
        fake.restart()

    [result] = _play(_manager_over(sync, fakes), 1)

    assert result.error is not None and "never been loaded" in result.error


def test_a_restarted_server_that_serves_the_trainers_base_gets_the_adapter_at_once(
    same_base_sync_of, servers, base_pushes, events, client_pool
):
    (sync, trainer), (_, [kept, restarted]) = same_base_sync_of, servers
    sync.sync(trainer, client_pool)
    version = sync.versions.current
    events.clear()
    restarted.restart()

    sync.recover(client_pool)

    assert version in restarted.adapters and events == [], "loaded again, with no base push and no reconnect"
    assert restarted.adapters[version][1].keys() == kept.adapters[version][1].keys()
    results = _play(_manager_over(sync, [kept, restarted]), 4)
    assert all(result.error is None for result in results)
    assert len(restarted.completion_models) == 2, "routable again for the version it holds"

    sync.sync(trainer, client_pool)

    assert events == [], "the next sync pushes no base either, as the server's base never changed"
    assert _held_everywhere([kept, restarted]) == [[sync.versions.current]] * 2


@pytest.mark.parametrize(
    "restart_as",
    [
        {"model_path": "/models/other"},
        {"served_model_name": "org/other"},
        {"revision": "abc123"},
    ],
)
def test_a_restarted_server_that_serves_another_base_gets_the_full_push(
    restart_as, same_base_sync_of, servers, base_pushes, events, client_pool
):
    (sync, trainer), (_, [kept, restarted]) = same_base_sync_of, servers
    sync.sync(trainer, client_pool)
    version = sync.versions.current
    events.clear()
    restarted.restart()
    for field, value in restart_as.items():
        setattr(restarted, field, value)

    sync.recover(client_pool)

    assert version not in restarted.adapters and events == [], "no adapter on a base that is not the trainer's"
    [result] = _play(_manager_over(sync, [restarted, kept]), 1)
    assert result.error is None and restarted.completion_models == [], "and nothing is routed to it"

    sync.sync(trainer, client_pool)

    assert events == [("reconnect", [restarted.url]), ("push", False)]
    assert _held_everywhere([kept, restarted]) == [[sync.versions.current]] * 2


_BASE_REFUSAL = "no full push into this family is validated"


@pytest.fixture
def unpushable_sync_of(servers):
    """A sync whose model's base no engine takes, served from the launch base the fake servers hold."""
    pool, _ = servers
    model = _peft_model()
    return _adapter_sync(model, pool, _LAUNCH_BASE, _BASE_REFUSAL), SimpleNamespace(model=model)


def test_an_unpushable_base_is_never_pushed_and_each_sync_loads_the_adapter(
    unpushable_sync_of, servers, base_pushes, events
):
    """No group either: the trainer forms none for a base no push carries, so the sync runs without a pool."""
    (sync, trainer), (_, fakes) = unpushable_sync_of, servers
    sync.sync(trainer, None)
    first = sync.versions.current
    sync.sync(trainer, None)

    assert base_pushes == [] and events == [], "no push and no reconnect: the servers launched on the base"
    assert first != sync.versions.current
    assert _held_everywhere(fakes) == [[sync.versions.current]] * 2


@pytest.mark.parametrize("pushable", [True, False], ids=["pushed-base", "unpushable-base"])
def test_no_collective_runs_inside_a_rank0_probe(pushable, request, monkeypatch, client_pool, base_pushes):
    """A probe runs on rank 0 alone: a collective inside it pairs with the other ranks' outer one and desyncs them."""
    depth = []
    verdict = adapter_sync.rank0_verdict

    def tracked(probe, what, *args, **kwargs):
        assert not depth, f"{what!r} is a collective inside a rank-0 probe"
        depth.append(what)
        try:
            return verdict(probe, what, *args, **kwargs)
        finally:
            depth.pop()

    monkeypatch.setattr(adapter_sync, "rank0_verdict", tracked)
    sync, trainer = request.getfixturevalue("sync_of" if pushable else "unpushable_sync_of")
    pool = client_pool if pushable else None
    sync.sync(trainer, pool)
    sync.sync(trainer, pool)
    sync.recover(pool)


@pytest.mark.parametrize("between_syncs", [False, True], ids=["at-the-sync", "between-syncs"])
def test_an_unpushable_base_gets_its_adapter_back_on_a_restart_onto_the_launch_base(
    between_syncs, unpushable_sync_of, servers, base_pushes, events
):
    (sync, trainer), (_, [kept, restarted]) = unpushable_sync_of, servers
    sync.sync(trainer, None)
    restarted.restart()

    if between_syncs:
        sync.recover(None)
    else:
        sync.sync(trainer, None)

    assert base_pushes == []
    assert _held_everywhere([kept, restarted]) == [[sync.versions.current]] * 2


@pytest.mark.parametrize("launched_as", [{"model_path": "/models/other"}, {"served_model_name": "org/other"}])
def test_an_unpushable_base_refuses_a_server_on_another_base(launched_as, unpushable_sync_of, servers, base_pushes):
    (sync, trainer), (_, [_, foreign]) = unpushable_sync_of, servers
    for field, value in launched_as.items():
        setattr(foreign, field, value)

    with pytest.raises(Exception, match=_BASE_REFUSAL) as raised:
        sync.sync(trainer, None)

    assert foreign.url in str(raised.value) and base_pushes == []


def _rebound(sync: AdapterSync, fakes) -> AdapterServers:
    """The servers a second ``train()`` binds after the first one's ``reset()`` closed its own."""
    sync.servers = AdapterServers(
        [fake.url for fake in fakes], client_cls=SGLangWeightSyncClient, connection_timeout=5.0
    )
    return sync.servers


def test_a_second_train_over_an_unpushable_base_retires_the_version_the_first_left(
    unpushable_sync_of, servers, base_pushes
):
    """No base push marks the second run's first sync, and its ``reset()`` forgot the versions it had loaded."""
    (sync, trainer), (_, fakes) = unpushable_sync_of, servers
    sync.sync(trainer, None)
    sync.reset()
    rebound = _rebound(sync, fakes)
    try:
        sync.sync(trainer, None)
        sync.sync(trainer, None)
    finally:
        rebound.close()

    assert base_pushes == []
    assert _held_everywhere(fakes) == [[sync.versions.current]] * 2


def _published(directory) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


@pytest.mark.parametrize("pushable", [True, False], ids=["pushed-base", "unpushable-base"])
def test_a_version_directory_leaves_with_its_version(pushable, servers, base_pushes, client_pool, tmp_path):
    """At a later sync, at a second ``train()`` and for a restarted trainer, whose predecessor's versions it finds on
    the servers."""
    pool, fakes = servers
    model = _peft_model()
    trainer = SimpleNamespace(model=model)
    launch = (_LAUNCH_BASE, None if pushable else _BASE_REFUSAL)
    client_pool = client_pool if pushable else None
    sync = _adapter_sync(model, pool, *launch, adapter_dir=str(tmp_path))

    sync.sync(trainer, client_pool)
    first = sync.versions.current
    assert _published(tmp_path) == [first]
    assert fakes[0].adapters[first][0]["r"] == 8, "the servers loaded the directory by path"
    sync.sync(trainer, client_pool)
    assert _published(tmp_path) == [sync.versions.current] and first != sync.versions.current

    sync.reset()
    rebound = _rebound(sync, fakes)
    try:
        sync.sync(trainer, client_pool)
        assert _published(tmp_path) == [sync.versions.current]

        successor = _adapter_sync(model, rebound, *launch, adapter_dir=str(tmp_path))
        successor.sync(trainer, client_pool)
        assert _published(tmp_path) == [successor.versions.current]
        assert _held_everywhere(fakes) == [[successor.versions.current]] * 2
    finally:
        rebound.close()


def test_a_load_a_server_refuses_leaves_no_directory(servers, base_pushes, client_pool, tmp_path):
    pool, [_, refusing] = servers
    refusing.lora["max_lora_rank"] = 4
    model = _peft_model()
    sync = _adapter_sync(model, pool, adapter_dir=str(tmp_path))

    with pytest.raises(Exception, match="incompatible with the current LoRA memory pool"):
        sync.sync(SimpleNamespace(model=model), client_pool)

    assert _published(tmp_path) == []
    assert sync.versions.current is None


def test_a_trainer_base_no_engine_launched_with_is_never_reloaded_at_once(sync_of, servers, base_pushes, client_pool):
    """A from-scratch or vocabulary-grown model has no launch base (``None``), whatever the server reports."""
    (sync, trainer), (_, [_, restarted]) = sync_of, servers
    sync.sync(trainer, client_pool)
    restarted.restart()

    sync.recover(client_pool)

    assert restarted.adapters == {}


def test_the_launch_base_is_the_weights_the_trainer_read(monkeypatch, tmp_path):
    model = _peft_model()
    model.config._commit_hash = "abc123"
    monkeypatch.setattr(adapter_sync, "weights_read_from", lambda _: f"{tmp_path}/")
    config = SimpleNamespace(model_name=_BASE_MODEL)

    assert adapter_sync.resolve_launch_base(model, config) == ServedBase(_BASE_MODEL, str(tmp_path), "abc123")

    monkeypatch.setattr(adapter_sync, "weights_read_from", lambda _: None)
    assert adapter_sync.resolve_launch_base(model, config) is None
    monkeypatch.setattr(adapter_sync, "weights_read_from", lambda _: str(tmp_path))
    monkeypatch.setattr(adapter_sync, "vocabulary_grown", lambda _: True)
    assert adapter_sync.resolve_launch_base(model, config) is None


_COMMIT = "c0ffee" * 6 + "abcd"
_HUB_ID = "org/policy"


def _hub_cache(root, commit=_COMMIT):
    snapshot = root / "models--org--policy" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (root / "models--org--policy" / "refs").mkdir()
    (root / "models--org--policy" / "refs" / "main").write_text(commit)
    return snapshot


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    return _hub_cache(tmp_path)


def test_a_hub_id_resolves_to_its_cached_snapshot_and_commit(hub_cache, monkeypatch):
    monkeypatch.setattr(adapter_sync, "weights_read_from", lambda _: _HUB_ID)

    launch = adapter_sync.resolve_launch_base(_peft_model(), SimpleNamespace(model_name=_HUB_ID))

    assert launch == ServedBase(_HUB_ID, str(hub_cache), _COMMIT)


def test_a_hub_id_the_cache_lacks_has_no_launch_base(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(adapter_sync, "weights_read_from", lambda _: _HUB_ID)

    assert adapter_sync.resolve_launch_base(_peft_model(), SimpleNamespace(model_name=_HUB_ID)) is None


def _sglang_reports(path, revision=None) -> ServedBase:
    return ServedBase(_HUB_ID, path, revision)


def test_a_server_that_loaded_the_snapshot_serves_the_hub_id_the_trainer_launched_with(hub_cache, tmp_path):
    """SGLang reports the snapshot path and no revision for a base it loaded by hub id."""
    launch = ServedBase(_HUB_ID, str(hub_cache), _COMMIT)
    elsewhere = f"/mnt/other/hf/hub/models--org--policy/snapshots/{_COMMIT}"

    assert adapter_sync.serves_launch_base(_sglang_reports(str(hub_cache)), launch)
    assert adapter_sync.serves_launch_base(_sglang_reports(elsewhere), launch), "another mount of the same cache"
    assert not adapter_sync.serves_launch_base(_sglang_reports(elsewhere.replace(_COMMIT, "f" * 40)), launch)
    assert not adapter_sync.serves_launch_base(_sglang_reports(elsewhere.replace("--policy", "--other")), launch)
    assert not adapter_sync.serves_launch_base(_sglang_reports(elsewhere, revision="f" * 40), launch)
    assert not adapter_sync.serves_launch_base(_sglang_reports(None), launch)
    assert not adapter_sync.serves_launch_base(_sglang_reports("/models/policy"), launch), "a path naming no commit"


def test_two_spellings_of_one_local_directory_are_one_base(tmp_path):
    real = tmp_path / "policy"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    launch = ServedBase(_HUB_ID, str(real), None)

    assert adapter_sync.serves_launch_base(_sglang_reports(f"{link}"), launch)
    assert not adapter_sync.serves_launch_base(_sglang_reports(str(tmp_path / "other")), launch)


def test_a_restarted_server_that_loaded_the_launch_snapshot_gets_the_adapter_at_once(
    hub_cache, servers, base_pushes, events, client_pool, monkeypatch
):
    """The trainer launched by hub id and the server reports the snapshot path: the pair the GPU run saw differ."""
    pool, [kept, restarted] = servers
    model = _peft_model()
    monkeypatch.setattr(adapter_sync, "weights_read_from", lambda _: _HUB_ID)
    launch = adapter_sync.resolve_launch_base(model, SimpleNamespace(model_name=_BASE_MODEL))
    sync = _adapter_sync(model, pool, launch_base=launch)
    for fake in (kept, restarted):
        fake.model_path, fake.revision = str(hub_cache), None
    sync.sync(SimpleNamespace(model=model), client_pool)
    events.clear()
    restarted.restart()

    sync.recover(client_pool)

    assert sync.versions.current in restarted.adapters and events == []


def test_a_down_server_is_reported_within_seconds(sync_of, servers, base_pushes, client_pool, caplog):
    (sync, trainer), (pool, [_, down]) = sync_of, servers
    sync.sync(trainer, client_pool)
    down.close()

    started = time.monotonic()
    with caplog.at_level("WARNING"):
        holdings = pool.probe()
        pool.probe()
    elapsed = time.monotonic() - started

    assert holdings[down.url] is None and holdings[servers[1][0].url] is not None
    assert elapsed < 10, "a refused connection is not retried for a minute"
    assert [r.getMessage() for r in caplog.records].count(
        f"Rollout server {down.url} did not answer the adapter probe (down or restarting)."
    ) == 1, "named once per outage"
    with pytest.raises(RuntimeError, match="did not answer the adapter probe"):
        sync.sync(trainer, client_pool)


def test_recovery_skips_a_server_that_is_down_and_keeps_routing_to_the_rest(
    same_base_sync_of, servers, base_pushes, client_pool
):
    (sync, trainer), (_, [kept, down]) = same_base_sync_of, servers
    sync.sync(trainer, client_pool)
    down.close()

    sync.recover(client_pool)
    results = _play(_manager_over(sync, [kept, down]), 3)

    assert all(result.error is None for result in results) and len(kept.completion_models) == 3


def test_recovery_survives_a_server_answering_5xx_while_it_warms_up(
    same_base_sync_of, servers, base_pushes, client_pool
):
    (sync, trainer), (_, [kept, warming]) = same_base_sync_of, servers
    sync.sync(trainer, client_pool)
    warming.refuse.update({"/v1/models", "/server_info"})

    sync.recover(client_pool)
    results = _play(_manager_over(sync, [kept, warming]), 3)

    assert all(result.error is None for result in results) and len(kept.completion_models) == 3


def test_the_routes_know_only_what_a_probe_or_a_refusal_told_them():
    routes = AdapterRoutes(lambda error, adapter: False)
    urls = ["a", "b"]

    routes.learn({"a": ["v1"], "b": ["v2"]}, ["v1", "v2"])
    assert routes.eligible(urls, "v1") == ["a"] and routes.eligible(urls, "v2") == ["b"]
    assert routes.eligible(urls, "v3") == urls, "a version no probe has seen is routable everywhere"

    routes.learn({"b": None}, ["v1", "v2"])
    assert routes.eligible(urls, "v2") == [], "a server that did not answer lacks every tracked version"

    routes.exclude("a", "v3")
    assert routes.eligible(urls, "v3") == ["b"]
    routes.forget_except(["v1"])
    assert routes.eligible(urls, "v3") == urls


def test_a_manager_without_adapters_keeps_its_config(servers):
    config = RolloutConfig(backend="sglang", model_name=_BASE_MODEL)
    manager = RolloutManager(
        num_workers=1,
        env_type="unused",
        env_config={},
        server_urls=["http://x"],
        rollout_config=config,
        max_concurrent_rollouts=1,
    )

    with manager._episode_config() as (episode, version):
        assert episode is config and version is None


def _refusal_cases():
    def model(**kwargs):
        return lambda monkeypatch: _peft_model(**kwargs)

    def patched(patch):
        def build(monkeypatch):
            patch(monkeypatch)
            return _peft_model()

        return build

    def grown(monkeypatch):
        monkeypatch.setattr(adapter_sync, "vocabulary_grown", lambda model: True)
        return _peft_model()

    def trainable_norm(monkeypatch):
        peft_model = _peft_model()
        peft_model.base_model.model.model.norm.weight.requires_grad_(True)
        return peft_model

    return {
        "dora": (model(use_dora=True), "DoRA"),
        "modules_to_save": (model(modules_to_save=["lm_head"]), "modules_to_save"),
        "embedding adapter": (model(targets=("q_proj", "embed_tokens")), "not one SGLang 0.5.22 serves"),
        "lm_head adapter": (model(targets=("q_proj", "lm_head")), "not one SGLang"),
        "fp32 adapters": (model(dtype=torch.float32), "adapters in"),
        "rank_pattern": (model(rank_pattern={"q_proj": 4}), "rank_pattern"),
        "alpha_pattern": (model(alpha_pattern={"q_proj": 4}), "alpha_pattern"),
        "grown vocabulary": (grown, "grown vocabulary"),
        "other trainable parameters": (trainable_norm, "trainable parameters other than the adapters"),
        "qlora": (patched(lambda m: m.setattr(adapter_sync, "has_packed_quantized_params", lambda _: True)), "QLoRA"),
        "expert lora": (
            patched(lambda m: m.setattr(adapter_sync, "model_has_expert_lora", lambda _: True)),
            "native expert LoRA",
        ),
    }


@pytest.mark.parametrize("case", list(_refusal_cases()))
def test_a_model_the_engine_cannot_serve_as_an_adapter_is_refused_with_its_reason(case, monkeypatch):
    build, reason = _refusal_cases()[case]

    with pytest.raises(ValueError, match=f"rollout_adapter_sync cannot serve.*{reason}"):
        reject_unadaptable_model(build(monkeypatch), _NO_PARALLELISM, SGLangWeightSyncClient)


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("is_tp_mode", "tensor parallelism"),
        ("is_pp_mode", "pipeline parallelism"),
        ("is_cp_mode", "context parallelism"),
    ],
)
def test_an_unvalidated_parallelism_mode_is_refused(mode, reason):
    parallelism = SimpleNamespace(**{**vars(_NO_PARALLELISM), mode: True})

    with pytest.raises(ValueError, match=reason):
        reject_unadaptable_model(_peft_model(), parallelism, SGLangWeightSyncClient)


def test_a_model_with_no_adapter_is_refused():
    bare = Qwen3ForCausalLM(Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1, vocab_size=64))

    with pytest.raises(ValueError, match="no PEFT adapter"):
        reject_unadaptable_model(bare, _NO_PARALLELISM, SGLangWeightSyncClient)


def test_a_plain_attention_lora_is_served():
    model = _peft_model(targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))

    reject_unadaptable_model(model, _NO_PARALLELISM, SGLangWeightSyncClient)

    serving = resolve_adapter_serving(model)
    assert (serving.rank, serving.scaling) == (8, 3.0)
    assert serving.modules == ("down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj")


def test_rslora_scales_by_the_square_root_of_the_rank():
    assert resolve_adapter_serving(_peft_model(use_rslora=True)).scaling == pytest.approx(24 / 8**0.5)


def _config(**overrides):
    fields = {
        "rollout_backend": "sglang",
        "routing_replay": "none",
        "model_name": _BASE_MODEL,
        "rollout_adapter_dir": None,
    }
    return SimpleNamespace(**{**fields, **overrides})


_SHARED_DIR = {"rollout_adapter_dir": "/shared/adapters"}
_INKLING2_CONFIG = SimpleNamespace(model_type="inkling2")
_DENSE_CONFIG = SimpleNamespace(model_type="qwen3")


@pytest.mark.parametrize(
    ("overrides", "model_config", "reason"),
    [
        ({"rollout_backend": "vllm", **_SHARED_DIR}, None, "ran over 5% slower than the merged weights"),
        ({"rollout_backend": "vllm", **_SHARED_DIR}, _DENSE_CONFIG, "ran over 5% slower than the merged weights"),
        ({"rollout_backend": "vllm"}, _INKLING2_CONFIG, "loads an adapter only from a directory"),
        ({"routing_replay": "rollout"}, None, "R3"),
        ({"model_name": None}, None, "must be set and hold no colon"),
        ({"model_name": "org:policy"}, None, "must be set and hold no colon"),
    ],
)
def test_a_config_the_engine_cannot_serve_is_refused(overrides, model_config, reason):
    with pytest.raises(ValueError, match=f"rollout_adapter_sync cannot serve this run.*{reason}"):
        reject_unadaptable_config(_config(**overrides), model_config)


@pytest.mark.parametrize(
    ("overrides", "model_config"),
    [
        ({}, None),
        (_SHARED_DIR, _DENSE_CONFIG),
        ({"rollout_backend": "vllm", **_SHARED_DIR}, _INKLING2_CONFIG),
    ],
    ids=["sglang-tensors", "sglang-directory", "vllm-inkling2"],
)
def test_a_servable_config_passes(overrides, model_config):
    reject_unadaptable_config(_config(**overrides), model_config)


@pytest.mark.parametrize(("prefetch_depth", "active", "live"), [(3, True, 5), (1, True, 3), (3, False, 2)])
def test_the_live_versions_cover_the_rounds_in_flight_and_the_retiring_one(prefetch_depth, active, live):
    async_config = SimpleNamespace(prefetch_depth=prefetch_depth, prefetch_active=lambda: active)
    assert resolve_live_versions(async_config) == live
    serving = resolve_adapter_serving(_peft_model())
    assert build_adapter_requirements(serving, async_config).live_versions == live


_NEEDS = AdapterRequirements(rank=8, modules=("q_proj", "v_proj"), live_versions=3)


def _verify(fake):
    verify_adapter_servers([fake.url], _NEEDS, backend="sglang", connection_timeout=5.0)


@pytest.mark.parametrize(
    ("launch", "reason"),
    [
        (None, "without --enable-lora"),
        ({"max_lora_rank": 4}, r"--max-lora-rank is 4, below the run's lora_r 8"),
        ({"lora_target_modules": ["q_proj"]}, r"omit \['v_proj'\]"),
        ({"max_loaded_loras": 2}, r"--max-loaded-loras is 2, below the 3 versions"),
        ({"max_loras_per_batch": 1}, "--max-loras-per-batch is 1"),
        ({"speculative_algorithm": "EAGLE"}, "speculative decoding"),
        ({"dp_size": 2}, "data-parallel replicas"),
    ],
)
def test_a_server_launched_without_what_the_adapters_need_is_refused(launch, reason):
    fake = FakeSGLangServer()
    try:
        fake.lora = None if launch is None else {**_LORA, **launch}
        with pytest.raises(ValueError, match=reason):
            _verify(fake)
    finally:
        fake.close()


def test_a_server_launched_for_adapters_is_accepted():
    fake = FakeSGLangServer()
    try:
        fake.lora = dict(_LORA)
        _verify(fake)
    finally:
        fake.close()


def test_the_all_modules_launch_covers_every_adapted_projection():
    fake = FakeSGLangServer()
    try:
        fake.lora = {**_LORA, "lora_target_modules": ["all"]}
        _verify(fake)
    finally:
        fake.close()


# SGLang 0.5.22 tokenizer_manager: the 400 body's message for an adapter it never loaded.
_SGLANG_UNLOADED_ADAPTER_BODY = (
    '{"object":"error","message":"Got LoRA adapter that has never been loaded: halo-adapter-bogus\\n'
    "All loaded adapters: dict_keys(['halo-adapter-4b3e0f84-0', 'halo-adapter-4b3e0f84-1']).\","
    '"type":"BadRequestError","param":null,"code":400}'
)


def test_sglang_matches_the_engines_unloaded_adapter_error_on_the_adapter_name():
    is_missing = SGLangWeightSyncClient.is_adapter_missing

    assert is_missing(_SGLANG_UNLOADED_ADAPTER_BODY, "halo-adapter-bogus")
    assert not is_missing(_SGLANG_UNLOADED_ADAPTER_BODY, "halo-adapter-4b3e0f84-0")
    assert not is_missing(_SGLANG_UNLOADED_ADAPTER_BODY, "halo-adapter-bog")


def test_the_gpu_e2e_builds_its_sync_with_the_class_signature():
    sync = build_adapter_sync(resolve_adapter_serving(_peft_model()), AdapterVersions())

    assert isinstance(sync, AdapterSync)


_INKLING2_ATTENTION = ("q_proj", "k_proj", "v_proj", "r_proj", "o_proj")


def _inkling2_peft(family: str, targets: tuple[str, ...]) -> torch.nn.Module:
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES.get(family) or TINY_MOE_VLM_FAMILIES[family])
    return get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=list(targets))).to(torch.bfloat16)


@pytest.mark.parametrize("family", ["inkling2_text", "inkling2"])
def test_inkling2_attention_adapters_reach_the_release_names_vllm_packs(family):
    """vLLM loads Inkling 2's adapters under the release's names and packs wq_du/wk_dv/wv_dv/wr_du into qkvr."""
    model = _inkling2_peft(family, _INKLING2_ATTENTION)
    parallelism = SimpleNamespace(is_tp_mode=False, is_pp_mode=False, is_cp_mode=False)
    assert adapter_sync._unadaptable_reasons(model, parallelism, VLLMWeightSyncClient) == []
    state = {name: param for name, param in model.named_parameters() if ".lora_" in name}
    hub = adapter_sync._hub_adapter_state(model, state)
    hub = {key: value for key, value in hub.items() if ".visual." not in key}
    layer0 = {key.split(".attn.")[1].split(".")[0] for key in hub if ".layers.0.attn." in key}
    assert layer0 == {"wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud"}
    assert all(".model.llm.layers." in key for key in hub), "the release's layer path"


class _ExpertLoraLayer:
    """An EP layer with native expert LoRA at rank 8 and the given scaling."""

    def __init__(self, scaling: float):
        self.ep_config = SimpleNamespace(expert_lora=SimpleNamespace(r=8))
        self.expert_lora_scaling = scaling


@pytest.mark.parametrize(
    ("family", "client_cls", "scaling", "reason"),
    [
        ("inkling2_text", VLLMWeightSyncClient, 2.0, None),
        ("inkling2_text", VLLMWeightSyncClient, 1.0, "another rank or scaling"),
        ("inkling2_text", SGLangWeightSyncClient, 2.0, "serves no routed-expert adapter"),
        ("qwen3_moe", VLLMWeightSyncClient, 2.0, "serves no routed-expert adapter"),
    ],
)
def test_native_expert_lora_is_served_only_where_the_engine_takes_it(family, client_cls, scaling, reason, monkeypatch):
    """vLLM's Inkling 2 experts take an adapter (with Halo's source patch); one config carries every module's rank."""
    model = _inkling2_peft(family, ("q_proj", "o_proj"))
    monkeypatch.setattr(adapter_sync, "expert_lora_layers", lambda _: [("mlp", _ExpertLoraLayer(scaling))])
    reasons = adapter_sync._unadaptable_reasons(model, _NO_PARALLELISM, client_cls)
    if reason is None:
        assert reasons == []
    else:
        assert any(reason in each for each in reasons), reasons


def test_expert_adapters_reach_vllm_per_expert_with_gate_and_up_interleaved():
    """Halo's grouped ``[E, in, r]`` / ``[E, r, out]`` adapters become vLLM's ``[E, r, in]`` / ``[E, out, r]`` on
    ``w13_weight`` / ``w2_weight`` under the release's layer path, the w13 B rows alternating gate and up."""
    model = tiny_family_model(TINY_MOE_VLM_FAMILIES["inkling2"])
    layer = next(name for name, _ in model.named_modules() if name.endswith("layers.0.mlp"))
    experts, hidden, width, rank = 4, 8, 6, 2
    gate_up_a, gate_up_b = torch.randn(experts, hidden, rank), torch.randn(experts, rank, 2 * width)
    down_a, down_b = torch.randn(experts, width, rank), torch.randn(experts, rank, hidden)
    gathered = {
        f"{layer}.experts.gate_up_proj.lora_A": gate_up_a,
        f"{layer}.experts.gate_up_proj.lora_B": gate_up_b,
        f"{layer}.experts.down_proj.lora_A": down_a,
        f"{layer}.experts.down_proj.lora_B": down_b,
    }

    engine = adapter_sync._engine_expert_adapters(model, gathered)

    prefix = f"{PEFT_BASE_MODEL_PREFIX}model.llm.layers.0.mlp.experts"
    assert engine.keys() == {
        f"{prefix}.{weight}.lora_{side}.weight" for weight in ("w13_weight", "w2_weight") for side in "AB"
    }
    assert torch.equal(engine[f"{prefix}.w13_weight.lora_A.weight"], gate_up_a.transpose(1, 2))
    assert torch.equal(engine[f"{prefix}.w2_weight.lora_A.weight"], down_a.transpose(1, 2))
    assert torch.equal(engine[f"{prefix}.w2_weight.lora_B.weight"], down_b.transpose(1, 2))
    # The engine's w13 delta, rows (g0, u0, g1, u1, ...), against the layer's [gate | up] one.
    delta = engine[f"{prefix}.w13_weight.lora_B.weight"][1] @ engine[f"{prefix}.w13_weight.lora_A.weight"][1]
    halo_delta = (gate_up_a[1] @ gate_up_b[1]).T
    assert torch.allclose(delta[0::2], halo_delta[:width]) and torch.allclose(delta[1::2], halo_delta[width:])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
