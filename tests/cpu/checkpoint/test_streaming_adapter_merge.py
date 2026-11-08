#!/usr/bin/env python
"""The streaming adapter merge reproduces the in-memory merge tensor for tensor: PEFT's own fold for linear and DoRA
adapters (dense and composite bases), and the gather's merge-on-save fold for native expert LoRA on every expert
layout. A refusal leaves no output."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3Config, Qwen3ForCausalLM

from halo.checkpoint.adapters import write_mixed_adapter_config
from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE, MERGED_ADAPTER_CONFIG_DIR, TRAINING_PROVENANCE_FILE
from halo.checkpoint.parallel.adapters import save_expert_lora_adapters
from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.sidecars import PROVENANCE_GPT_OSS_SINKS
from halo.checkpoint.tools.adapter_merge import merge_peft_adapter
from halo.checkpoint.tools.adapter_stream_merge import _FUSED_GATE_UP_KINDS, _required_attrs
from halo.distributed.expert_parallel.moe_layer.expert_lora import EXPERT_LORA_RANK_MULTIPLE
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.moe.expert_layout import ExpertLayoutKind
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from tests.common.checkpoint_io import safetensors_state_dict, save_context
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN35_CONFIG
from tests.common.parallelism import make_parallelism_config, single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model, word_level_tokenizer

# One family per expert layout: interleaved fused with biases, per-expert separate, fused, per-expert renamed.
_EXPERT_FAMILIES = ("gpt_oss", "qwen3_moe", "qwen3_5_moe_text", "laguna", "cohere2_moe")
_PROJECTIONS = {"all": None, "down only": frozenset({"down"}), "gate only": frozenset({"gate"})}


def _assert_same_checkpoint(streamed: Path, reference: Path, *, changed: set[str] | None = None) -> None:
    got, want = safetensors_state_dict(streamed), safetensors_state_dict(reference)
    assert set(got) == set(want), f"key sets differ: {sorted(set(got) ^ set(want))[:6]}"
    for key, tensor in want.items():
        assert got[key].dtype == tensor.dtype, f"{key}: {got[key].dtype} vs {tensor.dtype}"
        assert torch.equal(got[key], tensor), f"{key} differs from the in-memory merge"
    if changed is not None:
        base = safetensors_state_dict(streamed.parent / "base")
        moved = {key for key in got if not torch.equal(got[key], base[key])}
        assert changed <= moved, f"premise: the adapter moves {sorted(changed - moved)[:4]}"


def _randomize_lora_b(model, std: float = 0.3) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name or "lora_magnitude" in name:
                param.normal_(std=std) if param.dim() > 1 else param.add_(torch.randn_like(param) * std)


def _dense_base(base_dir: Path, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    # Biased projections, so a LoRA bias has a base bias to fold into.
    config = Qwen3Config(**{**TINY_QWEN3_CONFIG, "num_hidden_layers": 3, "vocab_size": 96, "attention_bias": True})
    Qwen3ForCausalLM(config).to(dtype).save_pretrained(base_dir)


def _save_peft_adapter(model_cls, base_dir: Path, adapter_dir: Path, dtype: torch.dtype, **lora) -> set[str]:
    """A randomized adapter over ``base_dir``; returns the weight keys it targets."""
    model = model_cls.from_pretrained(base_dir, dtype=dtype)
    config = LoraConfig(r=4, lora_alpha=8, task_type="CAUSAL_LM", **{"target_modules": ["q_proj", "v_proj"], **lora})
    peft_model = get_peft_model(model, config)
    _randomize_lora_b(peft_model)
    peft_model.save_pretrained(adapter_dir)
    word_level_tokenizer(("hello", "world")).save_pretrained(adapter_dir)
    return {
        name.removeprefix("base_model.model.").replace(".base_layer", "")
        for name, _ in peft_model.named_parameters()
        if "base_layer.weight" in name
    }


def _merge_both(adapter_dir: Path, tmp_path: Path, dtype: torch.dtype) -> tuple[Path, Path]:
    streamed, reference = tmp_path / "streamed", tmp_path / "reference"
    merge_peft_adapter(str(adapter_dir), str(streamed), dtype=dtype, streaming=True, verbose=False)
    merge_peft_adapter(str(adapter_dir), str(reference), dtype=dtype, verbose=False)
    return streamed, reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "lora",
    [
        {},
        {"use_dora": True},
        {"use_rslora": True, "alpha_pattern": {"v_proj": 32}, "rank_pattern": {"q_proj": 2}},
        {"lora_bias": True},
    ],
    ids=["plain", "dora", "rslora-patterns", "lora-bias"],
)
def test_a_dense_linear_merge_equals_the_in_memory_merge(tmp_path, dtype, lora):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, dtype)
    targets = _save_peft_adapter(Qwen3ForCausalLM, base, adapter, dtype, **lora)
    streamed, reference = _merge_both(adapter, tmp_path, dtype)

    _assert_same_checkpoint(streamed, reference, changed={f"{key}" for key in targets})
    assert (streamed / MERGED_ADAPTER_CONFIG_DIR / "adapter_config.json").is_file()


def test_a_merge_leaves_the_untargeted_tensors_as_the_base_stored_them(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.bfloat16)
    targets = _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.bfloat16)
    streamed = tmp_path / "streamed"
    merge_peft_adapter(str(adapter), str(streamed), streaming=True, verbose=False)

    original, merged = safetensors_state_dict(base), safetensors_state_dict(streamed)
    assert targets and not any(torch.equal(merged[key], original[key]) for key in targets)
    assert all(torch.equal(merged[key], original[key]) for key in original if key not in targets)


def test_a_composite_base_merge_equals_the_in_memory_merge(tmp_path):
    torch.manual_seed(0)
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2}
    config = Qwen3_5Config(
        text_config={**TINY_QWEN35_CONFIG, "tie_word_embeddings": False},
        vision_config={**vision, "out_hidden_size": TINY_QWEN35_CONFIG["hidden_size"]},
    )
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    Qwen3_5ForConditionalGeneration(config).save_pretrained(base)
    word_level_tokenizer(("hello", "world")).save_pretrained(base)
    targets = _save_peft_adapter(
        Qwen3_5ForConditionalGeneration, base, adapter, torch.float32, target_modules=["q_proj", "o_proj"]
    )
    assert any(".language_model." in key for key in targets), "premise: the adapter addresses the wrapper's tree"
    streamed, reference = _merge_both(adapter, tmp_path, torch.float32)

    _assert_same_checkpoint(streamed, reference, changed=targets)


def _expert_run(tmp_path: Path, family: str, projections) -> tuple[Path, Path, Path]:
    """``(base, adapter, merged-on-save reference)`` of a tiny EP family with native expert LoRA."""
    torch.manual_seed(0)
    rank = EXPERT_LORA_RANK_MULTIPLE
    spec = ExpertLoraSpec(r=rank, alpha=2 * rank, **({} if projections is None else {"projections": projections}))
    model = tiny_family_model(TINY_MOE_FAMILIES[family]).to(torch.bfloat16)
    model = apply_ep_layers(model, single_process_ep_config(expert_lora=spec))
    _randomize_lora_b(model)
    plain = make_parallelism_config(world_size=1, gpus_per_node=1, expert_lora=spec)
    merging = make_parallelism_config(world_size=1, gpus_per_node=1, expert_lora=spec, merge_expert_lora_on_save=True)
    base, adapter, reference = (tmp_path / name for name in ("base", "adapter", "reference"))
    save_ep_gathered_checkpoint(save_context(model, plain, has_ep_layers=True), str(base))
    model.config._name_or_path = str(base)
    save_expert_lora_adapters(save_context(model, plain, has_ep_layers=True, has_expert_lora=True), str(adapter))
    save_ep_gathered_checkpoint(save_context(model, merging, has_ep_layers=True, has_expert_lora=True), str(reference))
    return base, adapter, reference


@pytest.mark.parametrize("projections", sorted(_PROJECTIONS))
@pytest.mark.parametrize("family", _EXPERT_FAMILIES)
def test_a_native_expert_lora_merge_equals_the_merge_on_save(tmp_path, family, projections):
    base, adapter, reference = _expert_run(tmp_path, family, _PROJECTIONS[projections])
    streamed = tmp_path / "streamed"
    # Automatic: the in-memory merge refuses this adapter.
    merge_peft_adapter(str(adapter), str(streamed), verbose=False)

    moved = {
        key
        for key, tensor in safetensors_state_dict(reference).items()
        if not torch.equal(tensor, safetensors_state_dict(base)[key])
    }
    assert moved and all(".experts." in key for key in moved), f"premise: only expert tensors move, got {moved}"
    _assert_same_checkpoint(streamed, reference, changed=moved)


def test_a_mixed_attention_and_expert_adapter_folds_both_halves(tmp_path):
    base, expert_adapter, reference = _expert_run(tmp_path, "qwen3_moe", None)
    peft_dir = tmp_path / "attention"
    from_base = tiny_family_model(TINY_MOE_FAMILIES["qwen3_moe"]).to(torch.bfloat16)
    stored = {
        key: tensor for key, tensor in load_file(str(base / "model.safetensors")).items() if ".experts." not in key
    }
    assert not from_base.load_state_dict(stored, strict=False).unexpected_keys
    peft_model = get_peft_model(
        from_base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    )
    _randomize_lora_b(peft_model)
    peft_model.save_pretrained(peft_dir)
    attention = load_file(str(peft_dir / ADAPTER_SAFETENSORS_FILE))
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    save_file(
        {**attention, **load_file(str(expert_adapter / ADAPTER_SAFETENSORS_FILE))},
        str(mixed / ADAPTER_SAFETENSORS_FILE),
    )
    spec = ExpertLoraSpec(r=EXPERT_LORA_RANK_MULTIPLE, alpha=2 * EXPERT_LORA_RANK_MULTIPLE)
    peft_model.peft_config["default"].base_model_name_or_path = str(base)
    write_mixed_adapter_config(peft_model.peft_config["default"], spec, str(mixed))

    streamed = tmp_path / "streamed"
    merge_peft_adapter(str(mixed), str(streamed), verbose=False)

    merged_attention = peft_model.merge_and_unload().state_dict()
    got, expert_reference = safetensors_state_dict(streamed), safetensors_state_dict(reference)
    for key in (k for k in got if k.endswith(("q_proj.weight", "v_proj.weight"))):
        assert torch.equal(got[key], merged_attention[key]), f"{key} lost its attention delta"
    for key in (k for k in got if ".experts." in k):
        assert torch.equal(got[key], expert_reference[key]), f"{key} lost its expert delta"


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("rename_module", "does not store"),
        ("embedding_lora", "does not fold"),
        ("grown_vocabulary", "resized"),
    ],
)
def test_a_merge_it_cannot_make_is_refused_before_anything_is_written(tmp_path, tamper, match):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.float32)
    _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.float32)
    tensors = load_file(str(adapter / ADAPTER_SAFETENSORS_FILE))
    if tamper == "rename_module":
        tensors = {key.replace("layers.0.", "layers.9."): tensor for key, tensor in tensors.items()}
    elif tamper == "embedding_lora":
        tensors["base_model.model.model.embed_tokens.lora_embedding_A"] = torch.zeros(4, 96)
    else:
        tensors["base_model.model.lm_head.weight"] = torch.zeros(100, TINY_QWEN3_CONFIG["hidden_size"])
    save_file(tensors, str(adapter / ADAPTER_SAFETENSORS_FILE))
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=match):
        merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)
    assert not out.exists(), "a refused merge left an output directory behind"


def test_an_adapter_directory_without_a_weight_file_is_refused_by_name(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.float32)
    _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.float32)
    (adapter / ADAPTER_SAFETENSORS_FILE).unlink()
    out = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match=ADAPTER_SAFETENSORS_FILE):
        merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)
    assert not out.exists()


def _adapter_recording_sinks(tmp_path: Path, policy: SinksPolicy) -> tuple[Path, Path]:
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.float32)
    _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.float32)
    (adapter / TRAINING_PROVENANCE_FILE).write_text(json.dumps({PROVENANCE_GPT_OSS_SINKS: str(policy)}))
    return adapter, tmp_path / "out"


def test_neutralized_sinks_in_the_provenance_refuse_the_merge_before_any_output(tmp_path):
    adapter, out = _adapter_recording_sinks(tmp_path, SinksPolicy.NEUTRALIZED)

    with pytest.raises(ValueError, match="neutralized GptOss attention sinks"):
        merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)
    assert not out.exists(), "a refused merge left an output directory behind"


def test_live_sinks_in_the_provenance_do_not_refuse_the_merge(tmp_path):
    adapter, out = _adapter_recording_sinks(tmp_path, SinksPolicy.LIVE)

    merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)
    assert out.is_dir()


@pytest.mark.parametrize("option", [{"task": "classification"}, {"device_map": "cpu"}, {"fp32_norms": True}])
def test_options_only_a_loaded_model_honours_are_refused(tmp_path, option):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.float32)
    _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.float32)

    with pytest.raises(ValueError, match="streaming merge loads no model"):
        merge_peft_adapter(str(adapter), str(tmp_path / "out"), streaming=True, verbose=False, **option)
    assert not (tmp_path / "out").exists()


def test_the_merged_checkpoint_carries_the_adapter_config_and_the_base_config(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.float32)
    _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.float32)
    out = tmp_path / "out"
    merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)

    assert json.loads((out / "config.json").read_text())["model_type"] == "qwen3"
    assert (out / MERGED_ADAPTER_CONFIG_DIR / "adapter_config.json").is_file()
    assert not (out / "adapter_config.json").exists(), "an adapter config in the root reads the merge as an adapter"
    assert os.path.isfile(out / "tokenizer.json"), "the adapter's tokenizer did not ship"


def test_every_expert_layout_kind_has_a_streaming_attribute_set():
    """A new ``ExpertLayoutKind`` must be classified in the streaming merge, not fall into the fused-gate-up branch."""
    separate = {ExpertLayoutKind.SEPARATE_GLU, ExpertLayoutKind.SEPARATE_MLP}
    assert separate | _FUSED_GATE_UP_KINDS == set(ExpertLayoutKind)
    for kind in ExpertLayoutKind:
        assert _required_attrs(kind, {"down_proj"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
