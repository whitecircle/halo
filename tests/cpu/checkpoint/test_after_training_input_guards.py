#!/usr/bin/env python
"""Checkpoint conversion tools refuse inputs they cannot handle; each failure is silent, exit 0 with the experts
gone. Missing keys only warn (``test_missing_keys_only_warn``)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from halo.checkpoint.format import SAFETENSORS_INDEX_FILE
from halo.checkpoint.reading import (
    checkpoint_shard_files,
    detect_model_type,
    is_ep_sharded_checkpoint,
    reject_partial_checkpoint,
)
from halo.checkpoint.tools import deepseek_v4_dequant, vocab_patch
from halo.checkpoint.tools.adapter_merge import merge_peft_adapter
from halo.checkpoint.tools.bf16_conversion import convert_to_bf16
from halo.checkpoint.tools.ep_shard_merge import merge_ep_shards
from halo.checkpoint.tools.expert_unfuse import unfuse_checkpoint
from halo.checkpoint.tools.lowp_export import quantize_checkpoint
from halo.checkpoint.tools.model_merge import merge_models
from halo.checkpoint.tools.sink_reset import reset_sinks
from halo.checkpoint.writing import StreamingShardWriter
from halo.models.moe.family_specs import per_expert_hub_model_types
from scripts.before_training import convert_deepseek_v4_bf16, convert_glm5_bf16, convert_mistral4_bf16, patch_vocab
from tests.common.repo import REPO_ROOT


def _write_ep_sharded_checkpoint(path, *, model_type="gpt_oss"):
    os.makedirs(path, exist_ok=True)
    shards = {}
    for rank in range(2):
        name = f"model-{rank:05d}-of-00002.safetensors"
        tensors = {f"model.layers.0.mlp.experts.gate_up_proj.shard_{rank}": torch.ones(2, 4, 8) * (rank + 1)}
        if rank == 0:
            tensors["model.embed_tokens.weight"] = torch.zeros(16, 8)
        save_file(tensors, os.path.join(path, name), metadata={"format": "pt"})
        shards.update({key: name for key in tensors})
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 1, "ep_size": 2, "format": "ep_sharded"}, "weight_map": shards}, f)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model_type": model_type, "hidden_size": 8, "intermediate_size": 4}, f)
    return path


def _write_gathered_checkpoint(path, tensors, *, model_type="qwen3_moe", config_extra=None):
    """``config_extra`` adds config fields a specific tool reads."""
    os.makedirs(path, exist_ok=True)
    save_file(tensors, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model_type": model_type, "hidden_size": 8, "intermediate_size": 4, **(config_extra or {})}, f)
    return path


_FUSED_MOE_CONFIG = {"hidden_size": 8, "moe_intermediate_size": 3}


def _fused_moe_tensors(prefix="model.layers.0.mlp", num_experts=2):
    return {
        f"{prefix}.experts.gate_up_proj": torch.randn(num_experts, 6, 8),
        f"{prefix}.experts.down_proj": torch.randn(num_experts, 8, 3),
    }


def _write_pp_per_node_checkpoint(path, *, model_type="qwen3_moe", complete=False):
    """One node's directory after a PP save on a non-shared filesystem: the world-wide index beside this node's
    parts only."""
    os.makedirs(path, exist_ok=True)
    parts = {
        "model-pp00000-of-00002-00001.safetensors": {"model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 6, 8)},
        "model-pp00001-of-00002-00001.safetensors": {"model.layers.1.mlp.experts.gate_up_proj": torch.randn(2, 6, 8)},
    }
    weight_map = {}
    for name, tensors in parts.items():
        if complete or name.startswith("model-pp00000"):
            save_file(tensors, os.path.join(path, name), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(tensors, name))
    with open(os.path.join(path, SAFETENSORS_INDEX_FILE), "w") as f:
        json.dump({"metadata": {"total_size": 1}, "weight_map": weight_map}, f)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({"model_type": model_type, **_FUSED_MOE_CONFIG}, f)
    return path


def test_fixture_is_recognized_as_per_rank_sharded(tmp_path):
    ep = _write_ep_sharded_checkpoint(tmp_path / "ep")
    assert is_ep_sharded_checkpoint(str(ep))
    assert not is_ep_sharded_checkpoint(str(_write_gathered_checkpoint(tmp_path / "ok", {"a": torch.zeros(2)})))


def test_reject_partial_checkpoint_raises_and_names_the_merge_scripts(tmp_path):
    ep = _write_ep_sharded_checkpoint(tmp_path / "ep")
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        reject_partial_checkpoint(str(ep))


def test_an_index_naming_absent_shards_is_refused_and_names_them(tmp_path):
    """A ``from_pretrained``-based tool would random-initialize the other stage's layers; a streaming one would die
    inside ``safe_open``."""
    node = _write_pp_per_node_checkpoint(tmp_path / "node0")
    with pytest.raises(ValueError, match="does not hold") as excinfo:
        reject_partial_checkpoint(str(node))
    assert "model-pp00001-of-00002-00001.safetensors" in str(excinfo.value)
    assert "merge_pp_stages.py" in str(excinfo.value)
    with pytest.raises(ValueError, match="does not hold"):
        checkpoint_shard_files(str(node))

    gathered = _write_pp_per_node_checkpoint(tmp_path / "gathered", complete=True)
    reject_partial_checkpoint(str(gathered))
    assert len(checkpoint_shard_files(str(gathered))) == 2


def _unfuse(src, out):
    unfuse_checkpoint(src, out)


def _quantize(src, out):
    quantize_checkpoint(src, out, "mxfp8")


def _convert(src, out):
    convert_to_bf16(src, out, "causal_lm")


@pytest.mark.parametrize("tool", [_unfuse, _quantize, _convert], ids=["unfuse", "quantize", "convert_to_bf16"])
def test_tools_refuse_a_per_node_pp_directory_before_writing(tmp_path, tool):
    """The gate runs ahead of ``os.makedirs``, so the refusal leaves no half-made output."""
    node = _write_pp_per_node_checkpoint(tmp_path / "node0")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="does not hold"):
        tool(str(node), str(out))
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_reset_sinks_refuses_a_per_rank_sharded_checkpoint(tmp_path):
    """Without the guard an ``--in_place`` run re-initializes the experts and writes the result over the source
    shards."""
    ep = _write_ep_sharded_checkpoint(tmp_path / "ep")
    before = sorted(os.listdir(ep))
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        reset_sinks(str(ep))
    assert sorted(os.listdir(ep)) == before, "the refused input directory must be left untouched"


def test_convert_to_bf16_refuses_a_per_rank_sharded_checkpoint(tmp_path):
    """--verify cannot catch this one: it counts dtypes, not whether the weights are real."""
    ep = _write_ep_sharded_checkpoint(tmp_path / "ep")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        convert_to_bf16(str(ep), str(out), "causal_lm")
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_unfuse_moe_experts_emits_the_familys_own_projection_names(tmp_path):
    """Emitting GLM-4's spelling exits 0 over keys nothing reads, and the whole expert bank re-initializes on load."""
    tensors = _fused_moe_tensors()
    src = _write_gathered_checkpoint(tmp_path / "lfm2", tensors, model_type="lfm2_moe", config_extra=_FUSED_MOE_CONFIG)
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))

    written = load_file(os.path.join(out, "model.safetensors"))
    gate_up = tensors["model.layers.0.mlp.experts.gate_up_proj"]
    for i in range(gate_up.shape[0]):
        for name, expected in (("w1", gate_up[i, :3]), ("w3", gate_up[i, 3:])):
            key = f"model.layers.0.mlp.experts.{i}.{name}.weight"
            assert key in written, f"missing {key}; got {sorted(written)[:4]}"
            assert torch.equal(written[key], expected)
    assert not [k for k in written if "gate_proj" in k or "up_proj" in k], "emitted GLM-4's spelling"


def test_unfuse_moe_experts_emits_glm4_spelling_for_glm4(tmp_path):
    src = _write_gathered_checkpoint(
        tmp_path / "glm4", _fused_moe_tensors(), model_type="glm4_moe_lite", config_extra=_FUSED_MOE_CONFIG
    )
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))

    written = load_file(os.path.join(out, "model.safetensors"))
    for i in range(2):
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert f"model.layers.0.mlp.experts.{i}.{name}.weight" in written


def test_unfuse_moe_experts_converts_a_fused_qwen3_moe_checkpoint(tmp_path):
    """Reachable with ``use_grouped_gemm: false`` and no EP, where the gathered save skips ``save_pretrained``'s
    revert; this is what the tool repairs."""
    src = _write_gathered_checkpoint(
        tmp_path / "qwen3", _fused_moe_tensors(), model_type="qwen3_moe", config_extra=_FUSED_MOE_CONFIG
    )
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))

    written = load_file(os.path.join(out, "model.safetensors"))
    for i in range(2):
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert f"model.layers.0.mlp.experts.{i}.{name}.weight" in written


@pytest.mark.parametrize("model_type", ["mistral4", "zaya", "step3p7"])
def test_unfuse_moe_experts_refuses_a_fused_only_family(tmp_path, model_type):
    """These families serve from fused tensors with these same names and shapes, so a per-expert rewrite produces
    keys their loaders never read."""
    src = _write_gathered_checkpoint(
        tmp_path / model_type, _fused_moe_tensors(), model_type=model_type, config_extra=_FUSED_MOE_CONFIG
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="does not store one tensor per expert"):
        unfuse_checkpoint(str(src), str(out))
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_unfuse_moe_experts_converts_a_qwen3_5_checkpoint(tmp_path):
    """A gathered save that bypassed the revert is what this script exists to repair."""
    src = _write_gathered_checkpoint(
        tmp_path / "qwen35", _fused_moe_tensors(), model_type="qwen3_5_moe", config_extra=_FUSED_MOE_CONFIG
    )
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))

    keys = set(load_file(os.path.join(out, "model.safetensors")))
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in keys
    assert "model.layers.0.mlp.experts.0.up_proj.weight" in keys
    assert "model.layers.0.mlp.experts.0.down_proj.weight" in keys


def test_unfuse_moe_experts_converts_a_glm5_next_composite_checkpoint(tmp_path):
    """The split must land under the family's ``gate/up/down_proj`` names at the wrapper prefix."""
    prefix = "model.language_model.layers.1.mlp"
    src = _write_gathered_checkpoint(
        tmp_path / "glm5",
        _fused_moe_tensors(prefix=prefix),
        model_type="glm5_next",
        config_extra={"text_config": {"model_type": "glm5_next_text", **_FUSED_MOE_CONFIG}},
    )
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))

    keys = set(load_file(os.path.join(out, "model.safetensors")))
    assert keys == {
        f"{prefix}.experts.{i}.{name}.weight" for i in range(2) for name in ("gate_proj", "up_proj", "down_proj")
    }


def test_unfuse_moe_experts_writes_deepseek_v4_under_its_own_w_names(tmp_path):
    """Emitting GLM-4's spelling would re-initialize the whole expert bank."""
    src = _write_gathered_checkpoint(
        tmp_path / "dsv4", _fused_moe_tensors(), model_type="deepseek_v4", config_extra=_FUSED_MOE_CONFIG
    )
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))

    keys = set(load_file(os.path.join(out, "model.safetensors")))
    assert "model.layers.0.mlp.experts.0.w1.weight" in keys
    assert "model.layers.0.mlp.experts.0.w3.weight" in keys
    assert "model.layers.0.mlp.experts.0.w2.weight" in keys
    assert not any(key.endswith(".gate_proj.weight") for key in keys)


def test_unfuse_moe_experts_refuses_an_unregistered_model_type(tmp_path):
    src = _write_gathered_checkpoint(
        tmp_path / "unknown", _fused_moe_tensors(), model_type="some_future_moe", config_extra=_FUSED_MOE_CONFIG
    )
    with pytest.raises(ValueError, match="no MoE family spec claims"):
        unfuse_checkpoint(str(src), str(tmp_path / "out"))


def test_unfuse_moe_experts_names_the_missing_config(tmp_path):
    """Reporting model_type '' as an unclaimed family sends the user looking for a roster entry instead of the
    missing file."""
    src = tmp_path / "no_config"
    os.makedirs(src)
    save_file(_fused_moe_tensors(), os.path.join(src, "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="no config.json"):
        unfuse_checkpoint(str(src), str(tmp_path / "out"))


def test_unfuse_moe_experts_diagnoses_a_per_rank_sharded_input_as_one(tmp_path):
    """Resolving the family first would answer "no per-expert hub layout" for a per-rank input."""
    ep = _write_ep_sharded_checkpoint(tmp_path / "ep", model_type="qwen3_5_moe")
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        unfuse_checkpoint(str(ep), str(tmp_path / "out"))


def test_unfuse_moe_experts_copies_through_an_already_per_expert_checkpoint(tmp_path):
    """A checkpoint with nothing fused needs no layout, so the family gate must not refuse a no-op."""
    tensors = {"model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(3, 8)}
    src = _write_gathered_checkpoint(
        tmp_path / "already", tensors, model_type="mistral4", config_extra=_FUSED_MOE_CONFIG
    )
    out = tmp_path / "out"
    unfuse_checkpoint(str(src), str(out))
    assert set(load_file(os.path.join(out, "model.safetensors"))) == set(tensors)


def test_unfuse_moe_experts_convertible_set_is_the_declared_per_expert_families():
    """Membership is checked against transformers' own converter registry, which decides the names
    ``from_pretrained`` reads."""
    convertible = set(per_expert_hub_model_types())
    assert {
        "glm4_moe_lite",
        "laguna",
        "lfm2_moe",
        "qwen3_moe",
        "bailing_moe",
        "bailing_moe_linear",
        "qwen3_5_moe",
        "deepseek_v4",
        "cohere2_moe",
        "cohere2_vision",
    } <= convertible
    assert not convertible & {"zaya", "gpt_oss", "gemma4", "mistral4"}


def test_convert_mistral4_bf16_refuses_a_per_rank_sharded_checkpoint(tmp_path, monkeypatch):
    """Without the refusal each partial expert slice is streamed through and re-indexed as the whole tensor."""
    ep = _write_ep_sharded_checkpoint(tmp_path / "ep", model_type="mistral4")
    before = sorted(os.listdir(ep))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["convert_mistral4_bf16.py", "--model_id", str(ep), "--output_dir", str(out)])
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        convert_mistral4_bf16.main()
    assert sorted(os.listdir(ep)) == before, "the refused input directory must be left untouched"
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_convert_glm5_bf16_refuses_a_per_rank_sharded_checkpoint(tmp_path, monkeypatch):
    """Without the refusal each partial expert slice is block-dequantized and re-indexed as the whole tensor."""
    ep = _write_ep_sharded_checkpoint(tmp_path / "ep", model_type="glm5_next")
    before = sorted(os.listdir(ep))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["convert_glm5_bf16.py", "--model_id", str(ep), "--output_dir", str(out)])
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        convert_glm5_bf16.main()
    assert sorted(os.listdir(ep)) == before, "the refused input directory must be left untouched"
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_convert_deepseek_v4_bf16_refuses_a_per_rank_sharded_checkpoint(tmp_path, monkeypatch):
    """``from_pretrained`` cannot tell a per-rank save from a whole one, so the converter would write ~580 GB with
    no experts in it."""
    ep = _write_ep_sharded_checkpoint(tmp_path / "ep", model_type="deepseek_v4")
    before = sorted(os.listdir(ep))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["convert_deepseek_v4_bf16.py", "--model_id", str(ep), "--output_dir", str(out)])
    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        convert_deepseek_v4_bf16.main()
    assert sorted(os.listdir(ep)) == before, "the refused input directory must be left untouched"
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_convert_deepseek_v4_bf16_refuses_an_in_place_conversion(tmp_path, monkeypatch):
    """``save_pretrained`` clears the ``model*.safetensors`` it does not overwrite, destroying the FP8 source the
    run still reads."""
    src = _write_gathered_checkpoint(tmp_path / "src", _fused_moe_tensors(), model_type="deepseek_v4")
    before = sorted(os.listdir(src))
    monkeypatch.setattr(sys, "argv", ["convert_deepseek_v4_bf16.py", "--model_id", str(src), "--output_dir", str(src)])
    with pytest.raises(ValueError, match="input and output directory are the same"):
        convert_deepseek_v4_bf16.main()
    assert sorted(os.listdir(src)) == before, "the refused input directory must be left untouched"


def test_convert_to_bf16_refuses_an_in_place_conversion(tmp_path, monkeypatch):
    """``save_pretrained`` clears the ``model*.safetensors`` it does not overwrite, destroying the source it is
    still reading."""
    src = _write_gathered_checkpoint(tmp_path / "src", _fused_moe_tensors(), model_type="qwen3_moe")
    before = sorted(os.listdir(src))
    with pytest.raises(ValueError, match="input and output directory are the same"):
        convert_to_bf16(str(src), str(src), task="causal_lm")
    assert sorted(os.listdir(src)) == before, "the refused input directory must be left untouched"


def test_merge_ep_shards_refuses_a_sibling_adapter(tmp_path):
    """The aux copy carries ``adapter_config.json`` but no weights, and the merge cannot tell whether the shards
    already hold the delta."""
    src = _write_ep_sharded_checkpoint(tmp_path / "ep")
    adapter = {"base_model.model.layers.0.q_proj.lora_A.weight": torch.zeros(2, 8)}
    save_file(adapter, os.path.join(src, "adapter_model.safetensors"))
    with open(os.path.join(src, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA", "r": 8}, f)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="adapter_model.safetensors"):
        merge_ep_shards(str(src), str(out), verbose=False)
    assert not out.exists(), "a refused merge must not create its output directory"


def test_merge_ep_shards_refuses_a_lone_adapter_config(tmp_path):
    """The copy across would make ``from_pretrained`` read the merged base as an adapter."""
    src = _write_ep_sharded_checkpoint(tmp_path / "ep")
    with open(os.path.join(src, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA", "r": 8}, f)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="adapter_config.json"):
        merge_ep_shards(str(src), str(out), verbose=False)
    assert not out.exists(), "a refused merge must not create its output directory"


def test_detect_model_type_reads_config_and_tolerates_absence(tmp_path):
    src = _write_gathered_checkpoint(tmp_path / "m", {"a": torch.zeros(2)}, model_type="glm4_moe_lite")
    assert detect_model_type(str(src)) == "glm4_moe_lite"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert detect_model_type(str(empty)) == ""


def test_index_less_shards_are_found_under_the_writers_own_pattern(tmp_path):
    """A reader that re-types the pattern reports "no weights found" for a real checkpoint once the spellings drift."""
    state = {f"model.layers.{i}.weight": torch.zeros(512) for i in range(3)}
    writer = StreamingShardWriter(str(tmp_path), "1KB", enabled=True)
    for key, tensor in state.items():
        writer.add(key, tensor)
    writer.close_as_hf_checkpoint()
    os.remove(tmp_path / SAFETENSORS_INDEX_FILE)

    found = checkpoint_shard_files(str(tmp_path))
    assert len(found) == 3, f"expected every shard, got {[os.path.basename(p) for p in found]}"
    recovered = {key for shard in found for key in load_file(shard)}
    assert recovered == set(state), "the glob found shards that do not add up to the checkpoint"


def test_merge_peft_adapters_refuses_in_place_output(tmp_path):
    """``--output_dir <adapter_dir>`` would have save_pretrained delete the adapter it just read."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(ValueError, match="same path"):
        merge_peft_adapter(str(adapter), str(adapter))


def _merge(*, models, output_dir):
    """``allow_missing_tokenizer``: these fixtures are bare weight directories; that contract is in
    test_after_training_tool_guards."""
    merge_models(models, output_dir, "linear", dtype="bfloat16", verbose=False, allow_missing_tokenizer=True)


def test_merge_models_refuses_asymmetric_key_sets(tmp_path):
    """A tensor only model B has is never visited by the reference-keyed loop, so it would be dropped with no
    warning."""
    shared = {"model.embed_tokens.weight": torch.zeros(4, 8)}
    a = _write_gathered_checkpoint(tmp_path / "a", dict(shared))
    b = _write_gathered_checkpoint(tmp_path / "b", {**shared, "lm_head.weight": torch.ones(4, 8)})
    with pytest.raises(ValueError, match=r"does not carry the merge's reference key set.* carries 1 more \['lm_head"):
        _merge(models=[str(a), str(b)], output_dir=str(tmp_path / "out"))


def test_merge_models_accepts_symmetric_key_sets(tmp_path):
    a = _write_gathered_checkpoint(tmp_path / "a", {"w": torch.zeros(4, 8)})
    b = _write_gathered_checkpoint(tmp_path / "b", {"w": torch.ones(4, 8) * 2})
    out = tmp_path / "out"
    _merge(models=[str(a), str(b)], output_dir=str(out))
    merged = load_file(os.path.join(out, "model.safetensors"))["w"]
    assert torch.allclose(merged.float(), torch.ones(4, 8), atol=1e-2), merged.float()[0, :2]


def _write_truncated_checkpoint(path):
    config = AutoConfig.for_model(
        "qwen3",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(config)
    os.makedirs(path, exist_ok=True)
    full_state = model.state_dict()
    state = {k: v for k, v in full_state.items() if "layers.1." not in k}
    save_file(state, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    config.save_pretrained(str(path))
    return str(path), sorted(set(full_state) - set(state))


def test_missing_keys_only_warn(tmp_path):
    """The premise behind every gate above: transformers does not raise on missing checkpoint keys."""
    src, dropped = _write_truncated_checkpoint(tmp_path / "trunc")
    assert dropped, "fixture dropped no layer-1 tensors"

    reloaded = AutoModelForCausalLM.from_pretrained(src)
    live = reloaded.state_dict()
    assert all(k in live for k in dropped)
    assert not any(live[k].is_meta for k in dropped), "missing keys were left on meta, not re-initialized"


def test_patch_vocab_refuses_a_truncated_checkpoint(tmp_path, monkeypatch):
    """A truncated source would otherwise be re-saved as a complete-looking checkpoint whose absent tensors are
    random."""
    src, dropped = _write_truncated_checkpoint(tmp_path / "trunc")
    assert dropped
    out = tmp_path / "out"
    monkeypatch.setattr(vocab_patch, "load_processing_class", lambda *a, **k: object())
    monkeypatch.setattr(sys, "argv", ["patch_vocab.py", "--model_id", src, "--output_dir", str(out)])
    with pytest.raises(RuntimeError, match="randomly initialized"):
        patch_vocab.main()
    assert not out.exists(), "a refused patch must not create its output directory"


def test_patch_vocab_refuses_a_source_with_no_tokenizer_file(tmp_path, monkeypatch):
    """transformers builds the family's tokenizer class with no vocabulary, so every pattern would read as already one
    token and the run would exit 0 having added nothing."""
    src, out = tmp_path / "src", tmp_path / "out"
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    ).save_pretrained(src)
    monkeypatch.setattr(
        sys,
        "argv",
        ["patch_vocab.py", "--model_id", str(src), "--output_dir", str(out), "--patterns", '["<tool_call>"]'],
    )
    with pytest.raises(ValueError, match="ships no tokenizer"):
        patch_vocab.main()
    assert not out.exists()


def test_a_pattern_the_tokenizer_encodes_to_nothing_is_refused():
    """Read as already one token, the pattern would be skipped and the run exit 0 having added nothing."""

    class _EncodesNothing:
        def encode(self, text, add_special_tokens=False):
            return []

    with pytest.raises(ValueError, match="encodes '<tool_call>' to nothing"):
        vocab_patch.filter_patterns_needing_multiple_tokens(_EncodesNothing(), ["<tool_call>"])


def test_patch_vocab_refuses_reset_sinks_on_a_family_that_has_none(tmp_path, monkeypatch):
    """Unrefused, the flag's whole effect drops silently and the artifact reads downstream as deliberately
    sink-free."""
    src, out = tmp_path / "src", tmp_path / "out"
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    ).save_pretrained(str(src))
    monkeypatch.setattr(vocab_patch, "load_processing_class", lambda *a, **k: object())
    monkeypatch.setattr(
        sys, "argv", ["patch_vocab.py", "--model_id", str(src), "--output_dir", str(out), "--reset_sinks"]
    )

    with pytest.raises(ValueError, match="carries no attention sinks"):
        patch_vocab.main()
    assert not out.exists(), "a refused patch must not create its output directory"


@pytest.mark.parametrize(
    ("tool", "model_type"),
    [(convert_mistral4_bf16, "mistral4"), (convert_glm5_bf16, "glm5_next"), (convert_deepseek_v4_bf16, "deepseek_v4")],
    ids=["mistral4", "glm5", "deepseek_v4"],
)
def test_a_converter_refuses_a_source_that_stores_no_fp8(tmp_path, monkeypatch, tool, model_type):
    """Each would otherwise write a plain copy of a checkpoint that needs no conversion and report success."""
    src = _write_gathered_checkpoint(tmp_path / "src", _fused_moe_tensors(), model_type=model_type)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [f"{tool.__name__}.py", "--model_id", str(src), "--output_dir", str(out)])
    with pytest.raises(ValueError, match="the checkpoint is not the fp8"):
        tool.main()
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_convert_deepseek_v4_bf16_loads_through_the_coverage_gate(tmp_path, monkeypatch):
    """A raw ``from_pretrained`` only warns, re-saving random experts as ~580 GB; the real FP8 load needs the hub
    checkpoint, so this pins the wiring."""
    tensors = {
        **_fused_moe_tensors(),
        "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4, dtype=torch.float8_e4m3fn),
    }
    src = _write_gathered_checkpoint(tmp_path / "src", tensors, model_type="deepseek_v4")
    out = tmp_path / "out"

    class _GateReached(Exception):
        pass

    def _gate(model_cls, model_id, **kwargs):
        assert model_id == str(src)
        raise _GateReached

    monkeypatch.setattr(deepseek_v4_dequant, "from_pretrained_verified", _gate)
    monkeypatch.setattr(sys, "argv", ["convert_deepseek_v4_bf16.py", "--model_id", str(src), "--output_dir", str(out)])
    with pytest.raises(_GateReached):
        convert_deepseek_v4_bf16.main()
    assert not out.exists(), "nothing may be written before the gated load completes"


@pytest.mark.parametrize(
    ("module", "progress_logger"),
    [
        ("scripts.before_training.convert_mistral4_bf16", "halo.checkpoint.tools.fp8_dequant"),
        ("scripts.before_training.convert_deepseek_v4_bf16", "halo.checkpoint.tools.deepseek_v4_dequant"),
        ("scripts.before_training.convert_glm5_bf16", "halo.checkpoint.tools.fp8_dequant"),
    ],
)
def test_converter_progress_lines_survive_the_src_root_handler(module, progress_logger):
    """Importing ``halo`` installs a root handler at WARNING and ``basicConfig`` is then a no-op, leaving converters
    mute through a multi-hour run. Checked in a subprocess because the root logger is process-wide."""
    probe = "\n".join(
        [
            "import contextlib, importlib, logging, sys",
            "import halo",
            f"tool = importlib.import_module({module!r})",
            "sys.argv = ['tool', '--help']",
            "with contextlib.suppress(SystemExit):",
            "    tool.main()",
            f"print('INFO_ENABLED', logging.getLogger({progress_logger!r}).isEnabledFor(logging.INFO))",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO_ROOT, check=True
    )
    assert "INFO_ENABLED True" in completed.stdout, completed.stdout + completed.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
