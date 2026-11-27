#!/usr/bin/env python
"""Inkling 2 adapters saved on Halo 1.3.0rc2, which shipped transformers 5.23: their keys name the attention linears
5.24 moved inside projection modules. Each tool that reads a saved adapter refuses one by name, and the engine export
refuses any name the base's checkpoint does not hold."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE
from halo.checkpoint.parallel.adapters import restore_adapters, save_peft_adapters
from halo.checkpoint.tools.adapter_merge import load_base_for_adapter
from halo.checkpoint.tools.adapter_stream_merge import merge_adapter_streaming
from halo.distributed.expert_parallel.moe_layer.expert_lora import EXPERT_LORA_RANK_MULTIPLE
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.entrypoints.engine_adapter_export import export_engine_adapter
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.models.moe.family_specs import INKLING2_PRE_524_ATTENTION_NAMES, inkling2_pre_524_adapter_reason
from halo.trainers.grpo.rollout.adapter_sync import hub_spellings_absent_from_base
from tests.common.checkpoint_io import save_context
from tests.common.parallelism import make_parallelism_config, single_process_ep_config
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, tiny_family_model

_RANK = EXPERT_LORA_RANK_MULTIPLE
_SPEC = ExpertLoraSpec(r=_RANK, alpha=2 * _RANK)
_INKLING2 = TINY_MOE_VLM_FAMILIES["inkling2"]
# The shipped recipes' PEFT targets: the text tower's attention, and o_proj beside the experts.
_ATTENTION = r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|relative_position_bias|o_proj)\.linear"
_RECIPES = {"attention": (None, _ATTENTION), "expert": (_SPEC, ["o_proj.linear"])}
_REFUSED = r"saved on Halo 1\.3\.0rc2, which shipped transformers 5\.23"
_LAYER = "model.language_model.layers.0.self_attn"


def _trained(base_dir: Path, recipe: str):
    """A PeftModel over the tiny model saved at ``base_dir``, under ``recipe``'s targets, its adapters off zero."""
    experts, targets = _RECIPES[recipe]
    torch.manual_seed(0)
    model = tiny_family_model(_INKLING2).to(torch.bfloat16)
    model.save_pretrained(base_dir)
    model.name_or_path = str(base_dir)
    if experts is not None:
        model = apply_ep_layers(model, single_process_ep_config(expert_lora=experts))
    peft_model = get_peft_model(model, LoraConfig(r=_RANK, lora_alpha=2 * _RANK, target_modules=targets))
    peft_model = peft_model.to(torch.bfloat16)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.05)
    return peft_model


def _as_saved_on_523(key: str) -> str:
    for old, path in INKLING2_PRE_524_ATTENTION_NAMES.items():
        key = key.replace(f".{path}.", f".{old}.")
    return key


def _saved_on_523(tmp_path: Path, recipe: str) -> Path:
    """``recipe``'s adapter saved on this release, its keys respelled as the 1.3.0rc2 layout had them."""
    experts, _ = _RECIPES[recipe]
    adapter = tmp_path / "rc2"
    run = make_parallelism_config(world_size=1, gpus_per_node=1, expert_lora=experts)
    mixed = experts is not None
    peft_model = _trained(tmp_path / "base", recipe)
    save_peft_adapters(save_context(peft_model, run, has_ep_layers=mixed, has_expert_lora=mixed), str(adapter))
    path = adapter / ADAPTER_SAFETENSORS_FILE
    saved = load_file(str(path))
    respelled = {_as_saved_on_523(key): tensor for key, tensor in saved.items()}
    assert respelled.keys() != saved.keys(), "premise: the recipe adapts a projection 5.24 wrapped"
    save_file(respelled, str(path), metadata={"format": "pt"})
    return adapter


def test_the_reason_names_an_inkling2_adapter_of_the_523_layout_alone():
    assert inkling2_pre_524_adapter_reason("inkling2_text", [f"{_LAYER}.r_proj"])
    assert inkling2_pre_524_adapter_reason("inkling2", [f"{_LAYER}.o_proj"])
    assert inkling2_pre_524_adapter_reason("inkling2", [f"{_LAYER}.o_proj.linear"]) is None
    assert inkling2_pre_524_adapter_reason("inkling2", ["model.language_model.layers.0.mlp.gate_proj"]) is None
    assert inkling2_pre_524_adapter_reason("qwen3_moe", [f"{_LAYER}.o_proj"]) is None


@pytest.mark.parametrize("recipe", list(_RECIPES))
def test_the_engine_export_refuses_it_by_name_before_any_write(tmp_path, recipe):
    """The expert recipe's ``o_proj`` still names a module on 5.24, the wrapper, so without the refusal it exported
    under a name the release does not hold."""
    adapter = _saved_on_523(tmp_path, recipe)
    with pytest.raises(ValueError, match=_REFUSED):
        export_engine_adapter(str(adapter), str(tmp_path / "exported"), rollout_backend="vllm")
    assert not (tmp_path / "exported").exists()


@pytest.mark.parametrize("recipe", list(_RECIPES))
def test_the_merge_and_bf16_check_loader_refuses_it_by_name(tmp_path, recipe):
    adapter = _saved_on_523(tmp_path, recipe)

    def load(base_model_path, *, excuse_task_head, text_only=False):
        return tiny_family_model(_INKLING2)

    with pytest.raises(ValueError, match=_REFUSED):
        load_base_for_adapter(str(adapter), str(tmp_path / "base"), load, excuse_task_head=False, log=print)


def test_the_streaming_merge_refuses_it_by_name_before_any_write(tmp_path):
    adapter = _saved_on_523(tmp_path, "attention")
    with pytest.raises(ValueError, match=_REFUSED):
        merge_adapter_streaming(str(adapter), str(tmp_path / "merged"), verbose=False)
    assert not (tmp_path / "merged").exists()


@pytest.mark.parametrize("recipe", list(_RECIPES))
def test_a_resume_refuses_it_by_name(tmp_path, recipe):
    """The restore's own mismatch would ask for the targets it was saved with, which 5.24 no longer builds."""
    adapter = _saved_on_523(tmp_path, recipe)
    with pytest.raises(RuntimeError, match=_REFUSED):
        restore_adapters(str(adapter), _trained(tmp_path / "fresh", recipe), is_cp_mode=False)


@pytest.mark.parametrize("live", [False, True], ids=["base", "peft"])
def test_only_a_name_the_base_checkpoint_holds_has_a_hub_spelling(tmp_path, live):
    model = _trained(tmp_path / "base", "attention") if live else tiny_family_model(_INKLING2)
    linear, wrapper = (f"base_model.model.{_LAYER}.o_proj.linear", f"base_model.model.{_LAYER}.o_proj")
    assert hub_spellings_absent_from_base(model, [linear]) == {}
    assert list(hub_spellings_absent_from_base(model, [linear, wrapper])) == [wrapper]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
