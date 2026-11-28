#!/usr/bin/env python
"""The in-memory merge folds every tensor of the adapter file under its saved name, where PEFT's own load respells the
key through the base's checkpoint renames and drops the tensor with a warning, also where accelerate offloads it, and
refuses a tensor the wrap builds nothing for, which would leave the merged model without it, and a base on disk."""

import json
import os
import warnings

import accelerate.utils.modeling
import pytest
import torch
import transformers.integrations.accelerate
from accelerate import cpu_offload
from accelerate.utils import align_module_device
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file

from halo.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE
from halo.checkpoint.tools.adapter_merge import apply_adapter, merge_peft_adapter
from halo.model_loading.model_preparation import auto_load_model
from tests.common.adapters import randomize_adapters
from tests.common.tiny_models import (
    TINY_DENSE_FAMILY,
    TINY_MOE_FAMILIES,
    build_tiny_family_checkpoint,
    word_level_tokenizer,
)


def _save_adapter(base_dir: str, adapter_dir: str, config: LoraConfig) -> PeftModel:
    """A trained-looking adapter on the checkpoint at ``base_dir``, recorded against it; returns the live model."""
    torch.manual_seed(1)
    peft_model = get_peft_model(auto_load_model(base_dir, dtype=torch.float32), config)
    randomize_adapters(peft_model, std=0.1)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "modules_to_save" in name:
                param.normal_(std=0.1)
    peft_model.save_pretrained(adapter_dir)
    path = os.path.join(adapter_dir, ADAPTER_CONFIG_FILE)
    with open(path) as fh:
        recorded = json.load(fh)
    recorded["base_model_name_or_path"] = base_dir
    with open(path, "w") as fh:
        json.dump(recorded, fh)
    word_level_tokenizer().save_pretrained(adapter_dir)
    return peft_model


def _weights(directory: str) -> dict[str, torch.Tensor]:
    return {
        key: tensor
        for name in sorted(os.listdir(directory))
        if name.endswith(".safetensors")
        for key, tensor in load_file(os.path.join(directory, name)).items()
    }


def _left_by_peft(base_dir: str, adapter_dir: str) -> list[str]:
    """The saved adapter keys whose tensor stock PEFT's load leaves at another value."""
    saved = load_file(os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = PeftModel.from_pretrained(auto_load_model(base_dir, dtype=torch.float32), adapter_dir)
    state = get_peft_model_state_dict(loaded)
    return sorted(key for key, tensor in saved.items() if key not in state or not torch.equal(state[key], tensor))


@pytest.mark.parametrize(
    ("family", "targets", "renamed"),
    [
        ("laguna", ["gate_proj", "q_proj"], ".mlp.shared_experts.gate_proj."),
        ("inkling2_text", ["weight_proj", "q_proj"], ".weight_proj."),
    ],
)
def test_the_merge_folds_the_tensors_peft_renames_away(tmp_path, family, targets, renamed):
    base_dir, adapter_dir, merged_dir = (str(tmp_path / name) for name in ("base", "adapter", "merged"))
    build_tiny_family_checkpoint(TINY_MOE_FAMILIES[family], base_dir, word_level_tokenizer(), seed=0)
    trained = _save_adapter(base_dir, adapter_dir, LoraConfig(r=4, lora_alpha=8, target_modules=targets))
    left = _left_by_peft(base_dir, adapter_dir)
    assert left and all(renamed in key for key in left), (
        f"premise: stock PEFT drops the {renamed} tensors, left {left}"
    )
    expected = trained.merge_and_unload().state_dict()
    base = auto_load_model(base_dir, dtype=torch.float32).state_dict()

    merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32, verbose=False)

    merged = auto_load_model(merged_dir, dtype=torch.float32).state_dict()
    for key in sorted({key.split(".lora_")[0].removeprefix("base_model.model.") + ".weight" for key in left}):
        assert not torch.equal(merged[key], base[key]), f"{key} merged without its adapter delta"
    assert merged.keys() == expected.keys()
    wrong = [key for key in sorted(expected) if not torch.allclose(merged[key], expected[key])]
    assert not wrong, f"merged weights differ from PEFT's merge of the trained model: {wrong[:5]}"


def _report_host_memory(monkeypatch, cpu_bytes: int) -> None:
    """The probe transformers' and accelerate's ``auto`` device maps read reports ``cpu_bytes`` of host memory."""
    probe = accelerate.utils.modeling.get_max_memory

    def small_host(max_memory=None):
        return {**probe(max_memory), "cpu": cpu_bytes}

    monkeypatch.setattr(accelerate.utils.modeling, "get_max_memory", small_host)
    monkeypatch.setattr(transformers.integrations.accelerate, "accelerate_max_memory", small_host)


def test_a_base_the_device_map_places_on_disk_is_refused_before_pefts_load(tmp_path, monkeypatch):
    """``--device_map auto`` on a host too small for the base places its last layers on ``disk``, and PEFT's load over
    them writes a copy of each shard holding them at its path taken relative to the working directory: from / that is
    the base's own shard."""
    base_dir, adapter_dir = str(tmp_path / "base"), str(tmp_path / "adapter")
    build_tiny_family_checkpoint(TINY_DENSE_FAMILY, base_dir, word_level_tokenizer(), seed=0)
    _save_adapter(base_dir, adapter_dir, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    base = auto_load_model(base_dir, dtype=torch.float32)
    _report_host_memory(monkeypatch, sum(param.numel() * 4 for param in base.parameters()) // 2)
    placed = auto_load_model(base_dir, dtype=torch.float32, device_map="auto").hf_device_map
    assert "disk" in placed.values(), f"premise: the auto map offloads part of the base, got {placed}"
    shards = _weights(base_dir)
    # Where PEFT's copy would land without the refusal.
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match=r"placed \d+ module\(s\) of the base on disk(?s:.*)--streaming"):
        merge_peft_adapter(
            adapter_dir, str(tmp_path / "merged"), dtype=torch.float32, device_map="auto", verbose=False
        )

    kept = _weights(base_dir)
    assert kept.keys() == shards.keys() and all(torch.equal(kept[key], shards[key]) for key in shards)


def test_a_tensor_peft_dropped_on_an_offloaded_module_is_written_where_the_merge_onloads_it(tmp_path, monkeypatch):
    """A base placed on a GPU and on ``cpu`` leaves PEFT's load with ``meta`` adapter placeholders whose values the
    offload hooks keep; ``cpu_offload`` builds the same hooks over the whole wrap. A tensor PEFT's renames dropped is
    written to the hook, which PEFT's merge onloads from."""
    base_dir, adapter_dir = str(tmp_path / "base"), str(tmp_path / "adapter")
    build_tiny_family_checkpoint(TINY_MOE_FAMILIES["laguna"], base_dir, word_level_tokenizer(), seed=0)
    _save_adapter(base_dir, adapter_dir, LoraConfig(r=4, lora_alpha=8, target_modules=["gate_proj", "q_proj"]))
    left = _left_by_peft(base_dir, adapter_dir)
    assert left, "premise: stock PEFT drops the shared_experts tensors"
    load = PeftModel.from_pretrained

    def load_offloaded(model, model_id, **kwargs):
        return cpu_offload(load(model, model_id, **kwargs), execution_device=torch.device("cpu"))

    monkeypatch.setattr(PeftModel, "from_pretrained", load_offloaded)
    peft_model = apply_adapter(
        adapter_dir,
        auto_load_model(base_dir, dtype=torch.float32),
        adapter_processor=None,
        base_processor=None,
        log=lambda message: None,
    )

    saved = load_file(os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE))
    for key in left:
        module = peft_model.get_submodule(key.removesuffix(".weight") + ".default")
        assert module.weight.is_meta, f"premise: {key} is offloaded"
        with align_module_device(module):
            assert torch.equal(module.weight, saved[key]), f"{key}: the merge would onload PEFT's value"


def test_a_saved_tensor_the_wrap_builds_nothing_for_is_refused(tmp_path):
    """A ``modules_to_save: [gate]`` adapter an earlier PEFT saved holds ``shared_expert_gate`` too, which this PEFT
    matches only by whole name: the wrap builds no copy for it, and its trained weight would not reach the merge."""
    base_dir, adapter_dir = str(tmp_path / "base"), str(tmp_path / "adapter")
    build_tiny_family_checkpoint(TINY_MOE_FAMILIES["qwen3_next"], base_dir, word_level_tokenizer(), seed=0)
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], modules_to_save=["gate", "shared_expert_gate"])
    _save_adapter(base_dir, adapter_dir, config)
    path = os.path.join(adapter_dir, ADAPTER_CONFIG_FILE)
    with open(path) as fh:
        recorded = json.load(fh)
    recorded["modules_to_save"] = ["gate"]
    with open(path, "w") as fh:
        json.dump(recorded, fh)
    saved = load_file(os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE))
    gate_key = "base_model.model.model.layers.0.mlp.shared_expert_gate.weight"
    assert gate_key in saved, "premise: the file holds the trained shared_expert_gate"

    with pytest.raises(ValueError, match="builds no adapter tensor for.*shared_expert_gate"):
        merge_peft_adapter(adapter_dir, str(tmp_path / "merged"), dtype=torch.float32, verbose=False)

    merge_peft_adapter(adapter_dir, str(tmp_path / "streamed"), streaming=True, verbose=False)
    streamed = _weights(str(tmp_path / "streamed"))
    stored = gate_key.removeprefix("base_model.model.")
    assert torch.equal(streamed[stored], saved[gate_key].to(streamed[stored].dtype)), "the named remedy drops it too"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
