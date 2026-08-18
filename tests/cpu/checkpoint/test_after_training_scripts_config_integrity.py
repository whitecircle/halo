#!/usr/bin/env python
"""Config + weight-file integrity of the standalone checkpoint-writing scripts.

Every script covered here loads a model and re-saves a whole checkpoint into its own output
directory, where two failures are silent:

* ``PretrainedConfig.to_dict`` stamps ``model_type`` from the CLASS attribute, which the Bailing/Ling
  vendor configs leave empty — the diff-based write then omits the key entirely and every
  model-type-keyed reader downstream (the sharded merge, the hub key renames) sees no family. The
  tests reproduce that exact shape by emptying a real config class's ``model_type``, so the live
  instance still carries one (loaded from ``config.json``) while the class does not.
* ``save_pretrained`` deletes only the previous shards its own ``-00001-of-00002`` numbering regex
  matches, so a leftover single ``model.safetensors`` or a stale index outlives the save —
  ``from_pretrained`` then prefers that file over the index just written, and the tools that read
  the index (``checkpoint_shard_files``) follow it to shards that are gone.

Run: ``python tests/cpu/checkpoint/test_after_training_scripts_config_integrity.py`` (or ``pytest -m cpu``).
"""

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
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from tests.common.checkpoint_io import weight_files
from tests.common.utils import load_script_module

# Tiny but real: the scripts run the genuine load → transform → save path.
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

# A remote-code repo's shape: the configuration module the auto_map names, plus a modeling stub.
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


cb = load_script_module("scripts/after_training/convert_to_bf16.py")
mpa = load_script_module("scripts/after_training/merge_peft_adapters.py")
mm = load_script_module("scripts/after_training/merge_models.py")
pv = load_script_module("scripts/before_training/patch_vocab.py")


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    """A real fast tokenizer built in-process: the merge scripts refuse a checkpoint without one, and
    downloading a hub tokenizer would make these tests network-bound."""
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")


def _build_tiny_checkpoint(out_dir: Path, seed: int = 0) -> None:
    torch.manual_seed(seed)
    Qwen3ForCausalLM(Qwen3Config(**_TINY_QWEN3)).save_pretrained(out_dir)
    _tiny_tokenizer().save_pretrained(out_dir)


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
    """Leave a previous save's weight files in the output directory.

    The single ``model.safetensors`` is the dangerous one: ``from_pretrained`` prefers it over the
    index, so a sharded save that leaves it behind serves the OLD weights. The index is the mirror
    case for the tools that read it directly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.safetensors").write_bytes(b"stale-single-file")
    (out_dir / "model-00001-of-00002.safetensors").write_bytes(b"stale-shard")
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"gone.weight": "model-00001-of-00002.safetensors"}})
    )


def _weight_files(out_dir: Path) -> list[str]:
    # index included: these assertions pin the whole on-disk artifact set, index and all
    return sorted(weight_files(str(out_dir), include_index=True))


def _saved_model_type(out_dir: Path) -> str:
    with open(out_dir / "config.json") as f:
        return json.load(f).get("model_type", "")


class _VendorConfigWithoutClassModelType:
    """Empty ``Qwen3Config.model_type`` for the duration of a save.

    Reproduces the Bailing/Ling vendor configs on a class the test can otherwise load and save for
    real: the instance keeps the ``model_type`` it was constructed/loaded with, while ``to_dict``
    (which reads the class) writes an empty one and the diff drops the key.
    """

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
            # Below the model size to force the sharded write — the case where a leftover single
            # model.safetensors wins the from_pretrained lookup over the new index.
            mpa.merge_peft_adapter(
                adapter_dir=str(adapter),
                output_dir=str(out),
                dtype=torch.float32,
                max_shard_size="20KB",
                verbose=False,
            )

        written = _weight_files(out)
        assert "model.safetensors.index.json" in written, "expected a sharded save at max_shard_size=20KB"
        # transformers prefers the planted single file over the index, so its survival would serve
        # the stale run's weights.
        assert not (out / "model.safetensors").exists()
        assert _saved_model_type(out) == "qwen3"
        reloaded = Qwen3ForCausalLM.from_pretrained(out)
        assert all(not torch.all(p == 0) for p in reloaded.parameters() if p.numel() > 1), (
            "a reloaded tensor is all-zero — the planted stub shadowed the fresh sharded save"
        )


def test_convert_to_bf16_carries_source_aux_files_without_shadowing_the_fresh_config():
    """A conversion owes its output every non-weight file the source carried, in the right order.

    ``save_pretrained`` writes weights + config only, so a converted checkpoint that skips the aux
    copy lands without the chat template (the model then answers with a raw-text prompt), without the
    remote-code modules its own ``auto_map`` names (unloadable), and without
    ``router_balancing_biases.pt`` (a resume from it restarts every router bias at zero). The copy
    therefore runs BEFORE the save — the live config and tokenizer must overwrite the copied ones,
    not the other way round. ``dtype`` is the discriminator: the source declares the pre-conversion
    one, so a copy that ran last would leave a checkpoint whose config claims float32 over bf16
    weights.
    """
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
        # The copy must still refuse the source's weights, or they would shadow the converted ones.
        assert _weight_files(out) == ["model.safetensors"]
        assert Qwen3ForCausalLM.from_pretrained(out).dtype == torch.bfloat16


def test_convert_to_bf16_unmerged_peft_save_never_sweeps_the_output_dir():
    """--peft without --merge_adapter writes ADAPTER files only: the full-checkpoint sweep must not
    run there, or it deletes whatever full-model weights the directory already holds while the save
    replaces none of them."""
    PartialState()
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter, out = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "out"
        _build_tiny_checkpoint(base)
        _build_lora_adapter(base, adapter)
        _plant_stale_weights(out)
        planted = set(_weight_files(out))

        cb.convert_to_bf16(str(adapter), str(out), "causal_lm", is_peft=True, merge_adapter=False)

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


def test_merge_models_hub_source_ships_the_modules_its_auto_map_names():
    """A Hub ``--tokenizer_source`` must leave a directory that loads.

    Re-emitting the source config through ``save_pretrained`` writes its ``auto_map`` while shipping
    none of the modules named there, so the merged checkpoint raises ``does not appear to have a file
    named modeling_<x>.py`` for every consumer. The snapshot is stubbed to keep the test offline; the
    repo layout it returns is a real remote-code one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo, out = Path(tmp) / "hub-cache", Path(tmp) / "out"
        _build_fake_remote_code_repo(repo)
        out.mkdir()

        with patch.object(mm, "snapshot_download", side_effect=lambda repo_id, **kwargs: str(repo)) as download:
            # allow_missing_tokenizer: this fake repo ships config + modules only, and the merge now
            # REFUSES a tokenizer-less artifact by default — a separate contract from the auto_map one
            # under test here.
            mm._copy_aux_files(
                "org/tiny-remote",
                str(out),
                "bfloat16",
                verbose=False,
                allow_missing_tokenizer=True,
                trust_remote_code=True,
            )
            assert download.call_count == 1, "a Hub id must be resolved to a local snapshot, not re-saved"

        with open(out / "config.json") as f:
            written = json.load(f)
        for reference in written["auto_map"].values():
            module = reference.split(".")[0] + ".py"
            assert (out / module).is_file(), f"auto_map names {reference} but {module} was not shipped"
        assert written["model_type"] == "tiny_remote", "the vendor config's model_type was dropped on re-save"
        # transformers serializes the dtype under ``dtype``; ``torch_dtype`` is the pre-5 spelling.
        assert written.get("dtype", written.get("torch_dtype")) == "bfloat16", (
            "the merged dtype was not stamped into the copied config"
        )


def test_convert_to_bf16_honours_max_shard_size():
    """``save_pretrained`` alone caps shards at transformers' own default (50GB in 5.16), not the
    toolkit's ``DEFAULT_MAX_SHARD_SIZE``: a 120B conversion then lands in 50GB files every
    single-file reader (the sink reset, ``load_file``) has to hold whole. The flag must reach the save."""
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
    """The whole model lands in host RAM before the patch; the shared preflight must warn ahead of it
    (silence means the tool stopped calling the helper) and must not abort the run."""
    from src.checkpoint import tool_io

    monkeypatch.setattr(tool_io, "available_host_ram_bytes", lambda: 1)
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        _build_tiny_checkpoint(src)

        argv = ["patch_vocab.py", "--model_id", str(src), "--output_dir", str(out)]
        with patch.object(sys, "argv", argv):
            pv.main()

        printed = capsys.readouterr().out
        assert "WARNING: patch_vocab" in printed and "RAM" in printed, printed
        assert _weight_files(out) == ["model.safetensors"], "the warn-only preflight aborted the patch"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
