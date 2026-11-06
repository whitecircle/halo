#!/usr/bin/env python
"""Adapter-only sync as the trainer drives it, against fake SGLang servers: the base goes out once with the adapters
left out, each sync loads the adapter under a new name that new episodes select, a version leaves only once no episode
holds it, and a restarted server gets the base and the adapter again. And every model or server the engine cannot serve
as an adapter is refused with its reason."""

from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.environments.engine_wire import build_payload
from halo.environments.ray_actors import RolloutManager
from halo.environments.rollout_config import RolloutConfig
from halo.trainers.grpo.rollout import adapter_sync
from halo.trainers.grpo.rollout.adapter_sync import (
    ADAPTER_NAME_PREFIX,
    AdapterSync,
    reject_unadaptable_config,
    reject_unadaptable_model,
    resolve_adapter_serving,
)
from halo.trainers.grpo.rollout.engine_preflight import AdapterRequirements, verify_adapter_servers
from halo.weight_sync.adapter_servers import AdapterServers
from halo.weight_sync.adapter_versions import AdapterVersions
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.fake_engines import FakeSGLangServer

_LORA = {
    "max_lora_rank": 16,
    "lora_target_modules": ["q_proj", "v_proj"],
    "max_loaded_loras": 8,
    "max_loras_per_batch": 3,
}
_NO_PARALLELISM = SimpleNamespace(is_tp_mode=False, is_pp_mode=False, is_cp_mode=False)
_BASE_MODEL = "org/policy"


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


@pytest.fixture
def base_pushes(monkeypatch):
    """The full pushes the sync asked for, as the ``fold_adapters`` each carried."""
    pushes = []
    monkeypatch.setattr(
        adapter_sync, "sync_trainer_weights", lambda trainer, pool, *, fold_adapters=True: pushes.append(fold_adapters)
    )
    return pushes


@pytest.fixture
def sync_of(servers):
    pool, _ = servers
    model = _peft_model()
    sync = AdapterSync(resolve_adapter_serving(model), AdapterVersions())
    sync.servers = pool
    return sync, SimpleNamespace(model=model)


def _held_everywhere(fakes) -> list[list[str]]:
    return [sorted(fake.adapters) for fake in fakes]


def test_the_first_sync_pushes_the_base_unfolded_then_loads_the_adapter(sync_of, servers, base_pushes):
    (sync, trainer), (_, fakes) = sync_of, servers

    sync.sync(trainer, None)

    assert base_pushes == [False], "the base alone: folding the adapter in would count its delta twice"
    [name] = sync.versions.held() | {sync.versions.current}
    assert name.startswith(f"{ADAPTER_NAME_PREFIX}-")
    assert _held_everywhere(fakes) == [[name], [name]]


def test_the_wire_adapter_is_the_trainers_with_the_scaling_folded_into_b(sync_of, servers, base_pushes):
    (sync, trainer), (_, [fake, _]) = sync_of, servers

    sync.sync(trainer, None)

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


def test_a_later_sync_loads_a_new_name_and_retires_the_old_one(sync_of, servers, base_pushes):
    (sync, trainer), (_, fakes) = sync_of, servers
    sync.sync(trainer, None)
    first = sync.versions.current

    sync.sync(trainer, None)

    assert base_pushes == [False], "the base is pushed once"
    assert sync.versions.current != first
    assert _held_everywhere(fakes) == [[sync.versions.current]] * 2


def test_a_version_an_episode_holds_stays_loaded_until_it_ends(sync_of, servers, base_pushes):
    (sync, trainer), (_, fakes) = sync_of, servers
    sync.sync(trainer, None)

    with sync.versions.lease() as running:
        sync.sync(trainer, None)
        assert _held_everywhere(fakes) == [sorted([running, sync.versions.current])] * 2

    sync.sync(trainer, None)
    assert _held_everywhere(fakes) == [[sync.versions.current]] * 2


def test_a_restarted_server_gets_the_base_and_the_adapter_again(sync_of, servers, base_pushes):
    (sync, trainer), (_, [fake, restarted]) = sync_of, servers
    sync.sync(trainer, None)
    restarted.restart()

    sync.sync(trainer, None)

    assert base_pushes == [False, False], "the restart's base push is the unfolded one too"
    assert _held_everywhere([fake, restarted]) == [[sync.versions.current]] * 2


def test_a_predecessors_versions_are_retired_with_the_first_push(sync_of, servers, base_pushes):
    (sync, trainer), (pool, [fake, _]) = sync_of, servers
    left = AdapterSync(resolve_adapter_serving(trainer.model), AdapterVersions())
    left.servers = pool
    left.sync(trainer, None)
    stale = left.versions.current

    sync.sync(trainer, None)

    assert stale not in fake.adapters and sync.versions.current in fake.adapters


def test_requests_name_the_newest_version_for_a_whole_episode(sync_of, servers, base_pushes):
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
    sync.sync(trainer, None)
    first = sync.versions.current

    with manager._episode_config() as old_episode:
        sync.sync(trainer, None)
        with manager._episode_config() as new_episode:
            pass

    assert build_payload([], old_episode)["model"] == f"{_BASE_MODEL}:{first}"
    assert build_payload([], new_episode)["model"] == f"{_BASE_MODEL}:{sync.versions.current}"
    assert first != sync.versions.current


def test_a_manager_without_adapters_keeps_its_config(servers):
    config = RolloutConfig(backend="sglang", model_name=_BASE_MODEL)
    manager = RolloutManager(
        num_workers=1, env_type="unused", env_config={}, server_urls=["http://x"], rollout_config=config
    )

    with manager._episode_config() as episode:
        assert episode is config


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
