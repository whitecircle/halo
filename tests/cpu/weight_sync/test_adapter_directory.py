"""An adapter version as a directory the engines load by path: published whole, loaded on either engine, and named with
its mount when a server cannot read it."""

import json
import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from halo.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.weight_sync import adapter_directory
from halo.weight_sync.adapter_directory import publish_adapter_directory, remove_adapter_directories
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.fake_engines import FakeSGLangServer, FakeVLLMServer


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
    remove_adapter_directories(tmp_path, ["v1", "never-published"])
    assert not published.exists()


def test_a_version_that_cannot_be_removed_is_logged_and_the_rest_still_go(tmp_path, monkeypatch, caplog):
    tensors = {"base_model.model.x.lora_A.weight": torch.zeros(1, 1, dtype=torch.bfloat16)}
    for name in ("v1", "v2"):
        publish_adapter_directory(tmp_path, name, tensors, {"r": 1})
    remove_tree = adapter_directory.shutil.rmtree

    def denied_on_v1(path, *args, **kwargs):
        if Path(path).name == "v1":
            raise PermissionError(13, "Permission denied", str(path))
        remove_tree(path, *args, **kwargs)

    monkeypatch.setattr(adapter_directory.shutil, "rmtree", denied_on_v1)
    with caplog.at_level(logging.WARNING, logger=adapter_directory.__name__):
        remove_adapter_directories(tmp_path, ["v1", "never-published", "v2"])

    assert (tmp_path / "v1").exists() and not (tmp_path / "v2").exists()
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert str(tmp_path / "v1") in caplog.text and "Permission denied" in caplog.text


def test_a_write_that_raises_leaves_no_staging_directory(tmp_path, monkeypatch):
    def full_disk(tensors, path):
        raise OSError("No space left on device")

    monkeypatch.setattr(adapter_directory, "save_file", full_disk)

    with pytest.raises(OSError, match="No space left"):
        publish_adapter_directory(tmp_path, "v1", {"x.lora_A.weight": torch.zeros(2, 3)}, {"r": 2})

    assert list(tmp_path.iterdir()) == []


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


@pytest.fixture
def sglang_server():
    server = FakeSGLangServer()
    server.lora = {"max_lora_rank": 16, "lora_target_modules": ["q_proj"], "max_loaded_loras": 4}
    yield server
    server.close()


def test_sglang_loads_a_published_directory_by_path_lists_it_and_unloads_it(sglang_server, tmp_path):
    client = SGLangWeightSyncClient(base_url=sglang_server.url)
    key = "base_model.model.x.lora_A.weight"
    tensors = {key: torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}
    published = publish_adapter_directory(tmp_path, "halo-adapter-1", tensors, {"r": 2})

    client.load_adapter("halo-adapter-1", {"r": 2}, published)

    [(route, body)] = [(path, body) for path, body in sglang_server.posted if "load_lora_adapter" in path]
    assert (route, body) == ("/load_lora_adapter", {"lora_name": "halo-adapter-1", "lora_path": str(published)})
    config, loaded = sglang_server.adapters["halo-adapter-1"]
    assert config == {"r": 2} and torch.equal(loaded[key], tensors[key])
    assert client.loaded_adapters() == ["halo-adapter-1"]
    client.unload_adapter("halo-adapter-1")
    assert client.loaded_adapters() == []


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_a_directory_a_server_cannot_read_names_the_setting_and_the_mount(engine, request, tmp_path):
    server = request.getfixturevalue(f"{engine}_server")
    client_cls = {"vllm": VLLMWeightSyncClient, "sglang": SGLangWeightSyncClient}[engine]
    unmounted = tmp_path / "halo-adapter-1"

    with pytest.raises(RuntimeError) as raised:
        client_cls(base_url=server.url).load_adapter("halo-adapter-1", {"r": 2}, unmounted)

    message = str(raised.value)
    assert "HTTP 400" in message, "the engine's own answer stays in the error"
    assert f"reads the adapter from {unmounted}" in message
    assert (
        f"mount rollout_adapter_dir ({tmp_path}) at the same path in the trainer and in every rollout server"
        in message
    )


def test_an_unload_the_engine_refuses_names_no_mount(sglang_server):
    with pytest.raises(RuntimeError, match="could not unload adapter 'gone'") as raised:
        SGLangWeightSyncClient(base_url=sglang_server.url).unload_adapter("gone")

    assert "rollout_adapter_dir" not in str(raised.value)


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
