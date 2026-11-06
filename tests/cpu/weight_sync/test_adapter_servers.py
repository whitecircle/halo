#!/usr/bin/env python
"""What adapter sync sends an SGLang server: the wire tensors carry the scaling exactly once, a load lands on every
server or none, and a version is dropped only where it is held. Pinned against a fake server that unpickles the payload
as the engine does."""

import pytest
import torch

from halo.weight_sync.adapter_payload import build_adapter_config, serialize_adapter_tensors, wire_adapter_tensors
from halo.weight_sync.adapter_servers import AdapterServers
from halo.weight_sync.adapter_versions import AdapterVersions
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from tests.common.fake_engines import FakeSGLangServer

_LORA = {
    "max_lora_rank": 16,
    "lora_target_modules": ["q_proj", "v_proj"],
    "max_loaded_loras": 8,
    "max_loras_per_batch": 3,
}
_CONFIG = build_adapter_config(rank=8, target_modules=["q_proj", "v_proj"])
_KEY_A = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
_KEY_B = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"


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


def _payload(scaling: float = 1.0) -> tuple[dict[str, torch.Tensor], str]:
    generator = torch.Generator().manual_seed(0)
    state = {_KEY_A: torch.randn(8, 32, generator=generator), _KEY_B: torch.randn(32, 8, generator=generator)}
    return state, serialize_adapter_tensors(wire_adapter_tensors(state, scaling))


def test_b_is_scaled_in_fp32_and_rounded_once():
    state, _ = _payload()
    scaling = 64 / 48
    wire = wire_adapter_tensors(state, scaling)

    assert torch.equal(wire[_KEY_B], (state[_KEY_B].float() * scaling).to(torch.bfloat16))
    # An engine that rounds B and then scales it rounds twice: a different tensor, which is why the trainer scales.
    assert not torch.equal(
        wire[_KEY_B], state[_KEY_B].to(torch.bfloat16) * torch.tensor(scaling, dtype=torch.bfloat16)
    )
    assert torch.equal(wire[_KEY_A], state[_KEY_A].to(torch.bfloat16))
    assert _CONFIG["lora_alpha"] == _CONFIG["r"], "the engine's own scaling must be exactly 1"


def test_the_engine_reads_the_tensors_back_from_the_payload(servers):
    pool, [fake, _] = servers
    state, payload = _payload(scaling=2.0)

    pool.load("v0", _CONFIG, payload)

    config, tensors = fake.adapters["v0"]
    assert config == _CONFIG
    assert {key: tensor.dtype for key, tensor in tensors.items()} == dict.fromkeys(state, torch.bfloat16)
    assert torch.equal(tensors[_KEY_B], (state[_KEY_B] * 2.0).to(torch.bfloat16))


def test_one_serialized_copy_is_sent_per_tensor_parallel_rank(servers):
    pool, [fake, other] = servers
    fake.tp_size = 4
    _, payload = _payload()

    pool.load("v0", _CONFIG, payload)

    [(_, body)] = [(path, body) for path, body in fake.posted if path == "/load_lora_adapter_from_tensors"]
    assert body["serialized_named_tensors"] == [payload] * 4
    assert body["lora_name"] == "v0"
    assert "v0" in other.adapters


def test_a_server_that_refuses_the_load_leaves_none_holding_it(servers):
    pool, [fake, refusing] = servers
    refusing.lora["max_lora_rank"] = 4
    _, payload = _payload()

    with pytest.raises(RuntimeError, match=r"failed on 1/2 rollout server.*incompatible with the current LoRA memory"):
        pool.load("v0", _CONFIG, payload)

    assert fake.adapters == {} and refusing.adapters == {}, "one policy on one server and none on the other"


def test_a_server_launched_without_lora_refuses_by_name(servers):
    pool, [fake, _] = servers
    fake.lora = None
    _, payload = _payload()

    with pytest.raises(RuntimeError, match="Please set `--enable-lora`"):
        pool.load("v0", _CONFIG, payload)


def test_unload_drops_only_what_a_server_holds(servers):
    pool, [fake, other] = servers
    _, payload = _payload()
    pool.load("v0", _CONFIG, payload)
    pool.load("v1", _CONFIG, payload)
    del other.adapters["v0"]

    pool.unload(["v0"])

    assert list(fake.adapters) == ["v1"] and list(other.adapters) == ["v1"]
    assert fake.requests.count("POST /unload_lora_adapter") == 1, "a server without the version is not asked"


def test_a_refused_unload_is_loud(servers):
    pool, [fake, _] = servers
    _, payload = _payload()
    pool.load("v0", _CONFIG, payload)
    fake.status["/unload_lora_adapter"] = 500

    with pytest.raises(RuntimeError, match="Retiring adapters"):
        pool.unload(["v0"])


def test_a_restarted_server_is_the_one_lacking_the_adapter(servers):
    pool, [fake, other] = servers
    _, payload = _payload()
    pool.load("v0", _CONFIG, payload)
    assert pool.lacking("v0") == []

    other.restart()

    assert pool.lacking("v0") == [other.url]
    assert list(fake.adapters) == ["v0"]


def test_a_score_request_names_the_adapter_it_is_served_through():
    fake = FakeSGLangServer()
    try:
        client = SGLangWeightSyncClient(base_url=fake.url, connection_timeout=5.0)
        client.score_completion_logprobs([1, 2, 3], [4, 5])
        client.adapter_source = lambda: "v7"
        client.score_completion_logprobs([1, 2, 3], [4, 5])
        client.adapter_source = lambda: None
        client.score_completion_logprobs([1, 2, 3], [4, 5])
        assert fake.generate_lora_paths == [None, "v7", None]
    finally:
        fake.close()


def test_a_request_before_any_version_is_published_is_refused():
    versions = AdapterVersions()

    with pytest.raises(RuntimeError, match="would serve the frozen base"):
        with versions.lease():
            pass


def test_a_lease_keeps_its_version_across_a_publish():
    versions = AdapterVersions()
    versions.publish("v0")

    with versions.lease() as old:
        versions.publish("v1")
        with versions.lease() as new:
            assert (old, new) == ("v0", "v1")
            assert versions.held() == {"v0", "v1"}
        assert versions.held() == {"v0"}
    assert versions.held() == frozenset()
    assert versions.current == "v1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
