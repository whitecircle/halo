#!/usr/bin/env python
"""A CP-trained LoRA adapter merges into a plain base model, on disk, through the real script. PEFT drops unresolved
adapter keys silently, so weights must equal ``base + (B @ A) * alpha/r`` on targets and the base elsewhere."""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

from halo.checkpoint.format import ADAPTER_CONFIG_FILE, ROUTER_BALANCING_BIASES_FILE, TRAINING_PROVENANCE_FILE
from halo.checkpoint.parallel.adapters import _cp_normalized_config, save_peft_adapters
from halo.checkpoint.sidecars import PROVENANCE_GPT_OSS_SINKS, apply_training_sidecars
from halo.checkpoint.tools.adapter_merge import merge_peft_adapter
from halo.configs.script_arguments import ModelArguments
from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.model_loading.parallel.peft_setup import build_peft_config
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from tests.common.checkpoint_io import save_context
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import word_level_tokenizer

_TINY_QWEN3 = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
}
_LORA_R = 4
_LORA_ALPHA = 8
_TARGETS = ["q_proj", "v_proj"]
_UNTARGETED = "k_proj"


def _build_base(base_dir: Path) -> None:
    torch.manual_seed(0)
    Qwen3ForCausalLM(Qwen3Config(**_TINY_QWEN3)).save_pretrained(base_dir)
    # The merge refuses a checkpoint without a tokenizer.
    word_level_tokenizer(("hello", "world")).save_pretrained(base_dir)


def _cp_context(model):
    return save_context(model, make_parallelism_config(world_size=2, gpus_per_node=2, cp_size=2))


def _train_cp_adapter(base_dir: Path, adapter_dir: Path) -> dict[str, torch.Tensor]:
    """``cp_size=1`` still patches every attention layer, so saved keys carry the real CP artifacts without a
    process group."""
    inner = Qwen3ForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    inner.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
    wrapper = UlyssesModel(inner, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    peft_model = get_peft_model(wrapper, LoraConfig(r=_LORA_R, lora_alpha=_LORA_ALPHA, target_modules=_TARGETS))

    # lora_B initializes to zero, so an untrained adapter merges to a no-op and every check would pass on a merge that
    # resolved nothing.
    torch.manual_seed(11)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if ".lora_B." in name:
                param.add_(torch.randn_like(param))

    # The adapter stores LoRA at bf16, so the delta is built from rounded factors; fp32 originals would fail by ~1e-3.
    deltas = {}
    scaling = _LORA_ALPHA / _LORA_R
    for layer_index, layer in enumerate(peft_model.base_model.model.model.model.layers):
        for target in _TARGETS:
            module = getattr(layer.self_attn.original_attention, target)
            lora_a = module.lora_A["default"].weight.detach().to(torch.bfloat16).float()
            lora_b = module.lora_B["default"].weight.detach().to(torch.bfloat16).float()
            deltas[f"model.layers.{layer_index}.self_attn.{target}.weight"] = (lora_b @ lora_a) * scaling

    save_peft_adapters(_cp_context(peft_model), str(adapter_dir))
    return deltas


def _write_sidecars(adapter_dir: Path) -> None:
    """Neither file is in the base."""
    (adapter_dir / TRAINING_PROVENANCE_FILE).write_text(json.dumps({PROVENANCE_GPT_OSS_SINKS: SinksPolicy.LIVE}))
    torch.save(
        {"model.layers.0.mlp": torch.arange(4, dtype=torch.float32)}, adapter_dir / ROUTER_BALANCING_BIASES_FILE
    )


def test_cp_trained_adapter_merges_into_a_plain_model():
    PartialState()
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter, out = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "out"
        _build_base(base)
        deltas = _train_cp_adapter(base, adapter)

        merge_peft_adapter(adapter_dir=str(adapter), output_dir=str(out), dtype=torch.float32, verbose=False)

        merged = Qwen3ForCausalLM.from_pretrained(out, dtype=torch.float32).state_dict()
        original = Qwen3ForCausalLM.from_pretrained(base, dtype=torch.float32).state_dict()

    assert deltas, "premise: no targeted projections were found"
    assert all(delta.abs().max() > 1e-3 for delta in deltas.values()), "premise: the adapter is a no-op"

    for key, delta in deltas.items():
        assert torch.allclose(merged[key], original[key] + delta, atol=1e-5), (
            f"{key} did not receive its CP-trained LoRA delta — PEFT resolved no key for it and "
            f"merged silently, so the 'merged' model still serves base weights"
        )
    untargeted = [key for key in original if key.endswith(f"{_UNTARGETED}.weight")]
    assert untargeted, "premise: no untargeted control projections"
    assert all(torch.equal(merged[key], original[key]) for key in untargeted), (
        "an untargeted projection changed — the merge is not measuring what it claims"
    )


def test_cp_saved_exclusions_are_spelled_for_a_plain_model():
    """The saved config must spell it as the plain reload does, or PEFT warns and injects into the module the path
    excluded."""
    PartialState()
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter = Path(tmp) / "base", Path(tmp) / "adapter"
        _build_base(base)
        inner = Qwen3ForCausalLM.from_pretrained(base, dtype=torch.float32)
        inner.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
        # A container PEFT cannot adapt, spelled like a target.
        attention = inner.model.layers[0].self_attn
        attention.q_proj = Gemma4ClippableLinear(
            SimpleNamespace(use_clipped_linears=False), attention.q_proj.in_features, attention.q_proj.out_features
        )
        wrapper = UlyssesModel(inner, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
        model_config = ModelArguments(
            model_name_or_path=str(base),
            use_peft=True,
            lora_target_modules=_TARGETS,
            lora_r=_LORA_R,
            lora_alpha=_LORA_ALPHA,
        )
        peft_config = build_peft_config(wrapper, model_config)
        wrapped_spelling = ["model.model.layers.0.self_attn.original_attention.q_proj"]
        assert peft_config.exclude_modules == wrapped_spelling
        peft_model = get_peft_model(wrapper, peft_config)

        save_peft_adapters(_cp_context(peft_model), str(adapter))

        saved = json.loads((adapter / ADAPTER_CONFIG_FILE).read_text())
        assert saved["exclude_modules"] == ["model.layers.0.self_attn.q_proj"]
        assert peft_model.peft_config["default"].exclude_modules == wrapped_spelling, "the live config was respelled"

        plain = Qwen3ForCausalLM.from_pretrained(base, dtype=torch.float32)
        with warnings.catch_warnings():
            # PEFT only warns when an exclusion matches nothing; here that warning is the defect.
            warnings.simplefilter("error", UserWarning)
            reloaded = PeftModel.from_pretrained(plain, str(adapter))
    layers = reloaded.base_model.model.model.layers
    assert not hasattr(layers[0].self_attn.q_proj, "lora_A"), "the excluded module was adapted on the plain reload"
    assert hasattr(layers[1].self_attn.q_proj, "lora_A") and hasattr(layers[0].self_attn.v_proj, "lora_A")


def test_cp_saved_all_linear_targets_are_spelled_for_a_plain_model():
    """``all-linear`` rewrites ``target_modules`` in place to the wrapped tree's paths, which a plain reload cannot
    find."""
    wrapped = {
        "model.model.layers.0.self_attn.original_attention.q_proj",
        "model.model.layers.1.self_attn.original_attention.v_proj",
    }
    all_linear = LoraConfig(r=_LORA_R, lora_alpha=_LORA_ALPHA, target_modules=set(wrapped))

    saved = _cp_normalized_config(all_linear)

    assert saved.target_modules == ["model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.v_proj"]
    assert all_linear.target_modules == wrapped, "the live config was respelled"
    by_name = LoraConfig(r=_LORA_R, lora_alpha=_LORA_ALPHA, target_modules=list(_TARGETS))
    assert _cp_normalized_config(by_name) is by_name


def test_saved_cp_adapter_records_its_base_model():
    """``get_peft_model`` reads it from the wrapper's empty ``__dict__``, so without the backfill the merge has no
    base to load."""
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter = Path(tmp) / "base", Path(tmp) / "adapter"
        _build_base(base)
        _train_cp_adapter(base, adapter)
        with open(adapter / ADAPTER_CONFIG_FILE) as fh:
            recorded = json.load(fh)["base_model_name_or_path"]
    assert recorded == str(base), f"adapter points at {recorded!r}"


def test_merge_carries_the_training_sidecars_and_warns_about_an_unservable_bias():
    """``save_full_checkpoint`` copies aux files from the base, so sidecars come only through the merge's carry; an
    unservable bias must be named."""
    PartialState()
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter, out = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "out"
        _build_base(base)
        _train_cp_adapter(base, adapter)
        _write_sidecars(adapter)

        merged_model = Qwen3ForCausalLM.from_pretrained(base, dtype=torch.float32)
        actions = apply_training_sidecars(merged_model, str(adapter))
        assert any("TRANSIENT" in action for action in actions), (
            f"a dense model has no balancing slot — the merge must warn, got {actions}"
        )

        merge_peft_adapter(adapter_dir=str(adapter), output_dir=str(out), dtype=torch.float32, verbose=False)

        for name in (TRAINING_PROVENANCE_FILE, ROUTER_BALANCING_BIASES_FILE):
            assert (out / name).is_file(), f"{name} did not travel with the merged model"
        assert json.loads((out / TRAINING_PROVENANCE_FILE).read_text())[PROVENANCE_GPT_OSS_SINKS] == SinksPolicy.LIVE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
