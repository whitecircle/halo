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
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.environments.engine_wire import build_payload
from halo.environments.episode import RolloutResult
from halo.environments.messages import Trajectory
from halo.environments.ray_actors import RolloutManager
from halo.environments.rollout_config import RolloutConfig
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
from tests.common.fake_engines import FakeSGLangServer
from tests.gpu.trainers.grpo.test_adapter_sync_e2e import build_adapter_sync

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

    def reconnect_servers(self, urls) -> None:
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


def _adapter_sync(model, pool, launch_base=None) -> AdapterSync:
    routes = AdapterRoutes(SGLangWeightSyncClient.is_adapter_missing)
    sync = AdapterSync(resolve_adapter_serving(model), AdapterVersions(), routes, launch_base)
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

    sync.recover()
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

    sync.recover()

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

    sync.recover()

    assert version not in restarted.adapters and events == [], "no adapter on a base that is not the trainer's"
    [result] = _play(_manager_over(sync, [restarted, kept]), 1)
    assert result.error is None and restarted.completion_models == [], "and nothing is routed to it"

    sync.sync(trainer, client_pool)

    assert events == [("reconnect", [restarted.url]), ("push", False)]
    assert _held_everywhere([kept, restarted]) == [[sync.versions.current]] * 2


def test_a_trainer_base_no_engine_launched_with_is_never_reloaded_at_once(sync_of, servers, base_pushes, client_pool):
    """A from-scratch or vocabulary-grown model has no launch base (``None``), whatever the server reports."""
    (sync, trainer), (_, [_, restarted]) = sync_of, servers
    sync.sync(trainer, client_pool)
    restarted.restart()

    sync.recover()

    assert restarted.adapters == {}


def test_the_launch_base_is_the_weights_the_trainer_read(monkeypatch):
    model = _peft_model()
    model.config._commit_hash = "abc123"
    monkeypatch.setattr(adapter_sync, "weights_read_from", lambda _: "/models/policy/")
    config = SimpleNamespace(model_name=_BASE_MODEL)

    assert adapter_sync.resolve_launch_base(model, config) == ServedBase(_BASE_MODEL, "/models/policy", "abc123")

    monkeypatch.setattr(adapter_sync, "weights_read_from", lambda _: None)
    assert adapter_sync.resolve_launch_base(model, config) is None
    monkeypatch.setattr(adapter_sync, "weights_read_from", lambda _: "/models/policy")
    monkeypatch.setattr(adapter_sync, "vocabulary_grown", lambda _: True)
    assert adapter_sync.resolve_launch_base(model, config) is None


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

    sync.recover()
    results = _play(_manager_over(sync, [kept, down]), 3)

    assert all(result.error is None for result in results) and len(kept.completion_models) == 3


def test_recovery_survives_a_server_answering_5xx_while_it_warms_up(
    same_base_sync_of, servers, base_pushes, client_pool
):
    (sync, trainer), (_, [kept, warming]) = same_base_sync_of, servers
    sync.sync(trainer, client_pool)
    warming.refuse.update({"/v1/models", "/server_info"})

    sync.recover()
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
    fields = {"rollout_backend": "sglang", "routing_replay": "none", "model_name": _BASE_MODEL}
    return SimpleNamespace(**{**fields, **overrides})


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"rollout_backend": "vllm"}, VLLMWeightSyncClient.ADAPTER_SYNC_REFUSAL[:30]),
        ({"routing_replay": "rollout"}, "R3"),
        ({"model_name": None}, "must be set and hold no colon"),
        ({"model_name": "org:policy"}, "must be set and hold no colon"),
    ],
)
def test_a_config_the_engine_cannot_serve_is_refused(overrides, reason):
    with pytest.raises(ValueError, match=f"rollout_adapter_sync cannot serve this run.*{reason}"):
        reject_unadaptable_config(_config(**overrides))


def test_a_servable_config_passes():
    reject_unadaptable_config(_config())


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


def test_a_quantized_server_is_refused():
    fake = FakeSGLangServer()
    try:
        fake.lora = dict(_LORA)
        fake.quantization = "fp8"
        with pytest.raises(ValueError, match="base is quantized"):
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


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
