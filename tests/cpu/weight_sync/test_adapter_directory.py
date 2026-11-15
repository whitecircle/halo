"""Publishing an adapter version as a directory the engines load from disk."""

import json

import pytest
import torch
from safetensors.torch import load_file

from halo.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.weight_sync.adapter_payload import publish_adapter_directory, remove_adapter_directory
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.fake_engines import FakeVLLMServer


def test_a_published_version_is_a_whole_peft_directory_and_leaves_no_stage(tmp_path):
    tensors = {"base_model.model.x.lora_A.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}
    published = publish_adapter_directory(tmp_path, "v1", tensors, {"r": 2})

    assert published == tmp_path / "v1"
    assert json.loads((published / ADAPTER_CONFIG_FILE).read_text()) == {"r": 2}
    assert torch.equal(
        load_file(published / ADAPTER_SAFETENSORS_FILE)["base_model.model.x.lora_A.weight"],
        tensors["base_model.model.x.lora_A.weight"],
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["v1"], "the stage was renamed, not left beside it"
    remove_adapter_directory(tmp_path, "v1")
    assert not published.exists()


@pytest.mark.parametrize(
    ("kwargs", "accepted"),
    [
        ({"rollout_adapter_sync": True, "rollout_adapter_dir": "/shared/adapters"}, True),
        ({"rollout_adapter_sync": True, "rollout_adapter_dir": "adapters"}, False),
        ({"rollout_adapter_sync": False, "rollout_adapter_dir": "/shared/adapters"}, False),
    ],
)
def test_the_adapter_directory_is_absolute_and_serves_adapter_sync(kwargs, accepted):
    if accepted:
        AsyncRolloutConfig(**kwargs)
        return
    with pytest.raises(ValueError, match="rollout_adapter_dir"):
        AsyncRolloutConfig(**kwargs)


@pytest.fixture
def vllm_server():
    server = FakeVLLMServer()
    server.lora = {"max_lora_rank": 16, "max_loras": 2, "max_cpu_loras": 4}
    yield server
    server.close()


def test_vllm_loads_a_published_directory_lists_it_and_unloads_it(vllm_server, tmp_path):
    client = VLLMWeightSyncClient(base_url=vllm_server.url)
    tensors = {"base_model.model.x.lora_A.weight": torch.zeros(2, 3, dtype=torch.bfloat16)}
    published = publish_adapter_directory(tmp_path, "halo-adapter-1", tensors, {"r": 2})

    client.load_adapter("halo-adapter-1", {"r": 2}, published)

    assert vllm_server.adapters == {"halo-adapter-1": str(published)}
    assert client.loaded_adapters() == ["halo-adapter-1"]
    assert client.adapter_request_model("fake-model", "halo-adapter-1") == "halo-adapter-1"
    client.unload_adapter("halo-adapter-1")
    assert client.loaded_adapters() == []


def test_vllm_refuses_an_adapter_sent_as_tensors_and_names_a_failed_load(vllm_server, tmp_path):
    client = VLLMWeightSyncClient(base_url=vllm_server.url)

    with pytest.raises(ValueError, match="rollout_adapter_dir"):
        client.load_adapter("halo-adapter-1", {"r": 2}, "c2VyaWFsaXplZA==")
    with pytest.raises(RuntimeError, match="could not load adapter 'halo-adapter-1': HTTP 400"):
        client.load_adapter("halo-adapter-1", {"r": 2}, tmp_path / "never-published")


def test_vllm_reports_its_lora_launch(vllm_server):
    flags = VLLMWeightSyncClient(base_url=vllm_server.url).read_server_flags()

    assert flags.extras["enable_lora"] is True
    assert (flags.extras["max_lora_rank"], flags.extras["max_loras_per_batch"], flags.extras["max_loaded_loras"]) == (
        16,
        2,
        4,
    )


def test_vllm_matches_its_unknown_model_answer_on_the_adapter_name():
    # vLLM's OpenAI server: the 404 for a request naming a model (here an adapter) it does not serve.
    body = '{"error":{"message":"The model `halo-adapter-bogus` does not exist.","type":"NotFoundError","code":404}}'

    assert VLLMWeightSyncClient.is_adapter_missing(body, "halo-adapter-bogus")
    assert not VLLMWeightSyncClient.is_adapter_missing(body, "halo-adapter-bog")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
