#!/usr/bin/env python
"""Config and weight-file integrity of standalone checkpoint-writing scripts. ``to_dict`` stamps ``model_type`` from
the class, empty on Bailing/Ling; ``save_pretrained`` deletes only shards its numbering regex matches."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.checkpoint.tools import adapter_merge as mpa
from halo.checkpoint.tools import bf16_conversion as cb
from halo.checkpoint.tools import model_merge as mm
from halo.checkpoint.tools import offline_export
from tests.common.checkpoint_io import weight_files
from tests.common.repo import load_script_module
from tests.common.tiny_models import word_level_tokenizer

# The scripts run the genuine load → transform → save path.
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

# The configuration module the auto_map names, plus a modeling stub.
_REMOTE_CONFIG_MODULE = '''"""Stand-in vendor config module: the class declares no model_type (Bailing/Ling shape)."""

from transformers import PretrainedConfig


class TinyRemoteConfig(PretrainedConfig):
    model_type = ""

    def __init__(self, hidden_size=8, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)
'''
_REMOTE_MODELING_MODULE = '"""Stand-in vendor modeling module; only its presence beside the auto_map matters."""\n'
_REMOTE_CONFIG_JSON = {
    "model_type": "tiny_remote",
    "architectures": ["TinyRemoteForCausalLM"],
    "auto_map": {
        "AutoConfig": "configuration_tiny_remote.TinyRemoteConfig",
        "AutoModelForCausalLM": "modeling_tiny_remote.TinyRemoteForCausalLM",
    },
    "hidden_size": 8,
    "torch_dtype": "float32",
}


pv = load_script_module("scripts/before_training/patch_vocab.py")


def _build_tiny_checkpoint(out_dir: Path, seed: int = 0) -> None:
    torch.manual_seed(seed)
    Qwen3ForCausalLM(Qwen3Config(**_TINY_QWEN3)).save_pretrained(out_dir)
    word_level_tokenizer(("hello", "world")).save_pretrained(out_dir)


def _build_lora_adapter(base_dir: Path, adapter_dir: Path) -> None:
    base = Qwen3ForCausalLM.from_pretrained(base_dir)
    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        base_model_name_or_path=str(base_dir),
    )
    get_peft_model(base, lora).save_pretrained(adapter_dir)


def _plant_stale_weights(out_dir: Path) -> None:
    """``from_pretrained`` prefers a single ``model.safetensors`` over the index, so a sharded save leaving it
    serves the old weights."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.safetensors").write_bytes(b"stale-single-file")
    (out_dir / "model-00001-of-00002.safetensors").write_bytes(b"stale-shard")
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"gone.weight": "model-00001-of-00002.safetensors"}})
    )


def _weight_files(out_dir: Path) -> list[str]:
    return sorted(weight_files(str(out_dir), include_index=True))


def _saved_model_type(out_dir: Path) -> str:
    with open(out_dir / "config.json") as f:
        return json.load(f).get("model_type", "")


class _VendorConfigWithoutClassModelType:
    """The instance keeps its ``model_type`` while ``to_dict``, reading the class, writes an empty one and the diff
    drops the key."""

    def __enter__(self):
        self._patch = patch.object(Qwen3Config, "model_type", "")
        self._patch.start()
        assert Qwen3Config(**_TINY_QWEN3).to_dict()["model_type"] == "", "class attribute did not take"
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


def _assert_saved_clean(out_dir: Path, expected_weights: list[str]) -> None:
    assert _saved_model_type(out_dir) == "qwen3", (
        f"{out_dir}/config.json lost model_type — the diff-based write dropped the key the live "
        f"config carries, and every model-type-keyed reader downstream sees no family"
    )
    assert _weight_files(out_dir) == expected_weights, (
        f"{out_dir} carries {_weight_files(out_dir)}, expected exactly {expected_weights}: a previous "
        f"save's weight file survived and shadows the one just written"
    )


def test_convert_to_bf16_keeps_model_type_and_clears_stale_weights():
    PartialState()
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        _build_tiny_checkpoint(src)
        _plant_stale_weights(out)

        with _VendorConfigWithoutClassModelType():
            cb.convert_to_bf16(str(src), str(out), "causal_lm")

        _assert_saved_clean(out, ["model.safetensors"])
        assert Qwen3ForCausalLM.from_pretrained(out).config.hidden_size == _TINY_QWEN3["hidden_size"]


def test_merge_peft_adapters_keeps_model_type_and_clears_stale_weights():
    PartialState()
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter, out = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "out"
        _build_tiny_checkpoint(base)
        _build_lora_adapter(base, adapter)
        _plant_stale_weights(out)

        with _VendorConfigWithoutClassModelType():
            # Below the model size to force the sharded write, where a leftover single file wins the lookup.
            mpa.merge_peft_adapter(
                adapter_dir=str(adapter),
                output_dir=str(out),
                dtype=torch.float32,
                max_shard_size="20KB",
                verbose=False,
            )

        written = _weight_files(out)
        assert "model.safetensors.index.json" in written, "expected a sharded save at max_shard_size=20KB"
        assert not (out / "model.safetensors").exists()
        assert _saved_model_type(out) == "qwen3"
        reloaded = Qwen3ForCausalLM.from_pretrained(out)
        assert all(not torch.all(p == 0) for p in reloaded.parameters() if p.numel() > 1), (
            "a reloaded tensor is all-zero — the planted stub shadowed the fresh sharded save"
        )


def test_convert_to_bf16_carries_source_aux_files_without_shadowing_the_fresh_config():
    """``dtype`` is the discriminator: a copy that ran last would claim float32 over bf16 weights. Skipping it loses
    the chat template, ``auto_map`` modules and router biases."""
    PartialState()
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        _build_tiny_checkpoint(src)
        (src / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }}{% endfor %}")
        biases = {"model.layers.0.mlp.gate.expert_bias": torch.arange(4, dtype=torch.float32)}
        torch.save(biases, src / "router_balancing_biases.pt")
        (src / "configuration_tiny_remote.py").write_text(_REMOTE_CONFIG_MODULE)
        with open(src / "config.json") as f:
            source_config = json.load(f)
        source_dtype = source_config.get("dtype", source_config.get("torch_dtype"))
        assert source_dtype == "float32", f"premise: the source predates the conversion, got {source_dtype}"

        cb.convert_to_bf16(str(src), str(out), "causal_lm")

        assert (out / "chat_template.jinja").read_text() == (src / "chat_template.jinja").read_text()
        assert (out / "configuration_tiny_remote.py").read_text() == _REMOTE_CONFIG_MODULE
        bias_key = "model.layers.0.mlp.gate.expert_bias"
        restored = torch.load(out / "router_balancing_biases.pt", weights_only=True)
        assert torch.equal(restored[bias_key], biases[bias_key]), (
            "the router biases a resume restores were not carried"
        )

        with open(out / "config.json") as f:
            written = json.load(f)
        assert written.get("dtype", written.get("torch_dtype")) == "bfloat16", (
            "config.json is the source's copy, not the one the save wrote: the aux copy ran after the save"
        )
        assert _saved_model_type(out) == "qwen3"
        # Otherwise the source's weights would shadow the converted ones.
        assert _weight_files(out) == ["model.safetensors"]
        assert Qwen3ForCausalLM.from_pretrained(out).dtype == torch.bfloat16


def test_convert_to_bf16_unmerged_peft_save_never_sweeps_the_output_dir():
    """--peft writes adapter files only, so the sweep would delete full-model weights the save does not replace."""
    PartialState()
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter, out = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "out"
        _build_tiny_checkpoint(base)
        _build_lora_adapter(base, adapter)
        _plant_stale_weights(out)
        planted = set(_weight_files(out))

        cb.convert_to_bf16(str(adapter), str(out), "causal_lm", is_peft=True)

        assert (out / "adapter_model.safetensors").exists(), "adapter save must still land"
        assert planted <= set(_weight_files(out)), "the unmerged-PEFT branch swept full-model weights it did not write"


def test_patch_vocab_keeps_model_type_and_clears_stale_weights():
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        _build_tiny_checkpoint(src)
        _plant_stale_weights(out)

        argv = ["patch_vocab.py", "--model_id", str(src), "--output_dir", str(out)]
        with _VendorConfigWithoutClassModelType(), patch.object(sys, "argv", argv):
            pv.main()

        _assert_saved_clean(out, ["model.safetensors"])


def test_merge_models_keeps_model_type_and_clears_stale_weights():
    with tempfile.TemporaryDirectory() as tmp:
        a, b, out = Path(tmp) / "a", Path(tmp) / "b", Path(tmp) / "out"
        _build_tiny_checkpoint(a, seed=0)
        _build_tiny_checkpoint(b, seed=1)
        _plant_stale_weights(out)

        with _VendorConfigWithoutClassModelType():
            mm.merge_models(
                model_specs=[str(a), str(b)],
                output_dir=str(out),
                method="linear",
                dtype="bfloat16",
                tokenizer_source=str(a),
                verbose=False,
            )

        _assert_saved_clean(out, ["model.safetensors"])


def _build_fake_remote_code_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "config.json").write_text(json.dumps(_REMOTE_CONFIG_JSON, indent=2))
    (repo_dir / "configuration_tiny_remote.py").write_text(_REMOTE_CONFIG_MODULE)
    (repo_dir / "modeling_tiny_remote.py").write_text(_REMOTE_MODELING_MODULE)


def test_merge_models_remote_code_source_ships_the_modules_its_auto_map_names():
    """Writing ``auto_map`` without the modules makes every consumer raise."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, out = Path(tmp) / "remote-code", Path(tmp) / "out"
        _build_fake_remote_code_repo(repo)
        out.mkdir()

        # This fake repo has no tokenizer, a separate contract from the auto_map one under test.
        mm._copy_aux_files(
            str(repo), str(out), "bfloat16", verbose=False, allow_missing_tokenizer=True, trust_remote_code=True
        )

        with open(out / "config.json") as f:
            written = json.load(f)
        for reference in written["auto_map"].values():
            module = reference.split(".")[0] + ".py"
            assert (out / module).is_file(), f"auto_map names {reference} but {module} was not shipped"
        assert written["model_type"] == "tiny_remote", "the vendor config's model_type was dropped on re-save"
        # ``torch_dtype`` is the pre-5 spelling.
        assert written.get("dtype", written.get("torch_dtype")) == "bfloat16", (
            "the merged dtype was not stamped into the copied config"
        )


def test_convert_to_bf16_honours_max_shard_size():
    """transformers' own default is 50GB in 5.20; every single-file reader would hold those files whole."""
    PartialState()
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        _build_tiny_checkpoint(src)

        cb.convert_to_bf16(str(src), str(out), "causal_lm", max_shard_size="20KB")

        written = _weight_files(out)
        assert "model.safetensors.index.json" in written, (
            f"expected a sharded save at max_shard_size=20KB, got {written}"
        )
        assert "model.safetensors" not in written
        assert Qwen3ForCausalLM.from_pretrained(out).config.hidden_size == _TINY_QWEN3["hidden_size"]


def test_patch_vocab_honours_max_shard_size():
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        _build_tiny_checkpoint(src)

        argv = ["patch_vocab.py", "--model_id", str(src), "--output_dir", str(out), "--max_shard_size", "20KB"]
        with patch.object(sys, "argv", argv):
            pv.main()

        written = _weight_files(out)
        assert "model.safetensors.index.json" in written, (
            f"expected a sharded save at max_shard_size=20KB, got {written}"
        )
        assert Qwen3ForCausalLM.from_pretrained(out).config.hidden_size == _TINY_QWEN3["hidden_size"]


def test_patch_vocab_preflights_the_full_model_load(monkeypatch, capsys):
    """Silence means the tool stopped calling the helper."""
    monkeypatch.setattr(offline_export, "available_host_ram_bytes", lambda: 1)
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        _build_tiny_checkpoint(src)

        argv = ["patch_vocab.py", "--model_id", str(src), "--output_dir", str(out)]
        with patch.object(sys, "argv", argv):
            pv.main()

        # The CLI's own logging setup writes to stderr.
        printed = capsys.readouterr().err
        assert "WARNING patch_vocab: estimated peak host RAM" in printed, printed
        assert _weight_files(out) == ["model.safetensors"], "the warn-only preflight aborted the patch"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
