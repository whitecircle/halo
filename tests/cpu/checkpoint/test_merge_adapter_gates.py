#!/usr/bin/env python
"""The shared PEFT merge runs its input gates in order, before anything is loaded or written. The base is re-checked
since under a merge the weights come from it; ``adapter_input_gates`` also front ``convert_to_bf16 --peft``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from halo.checkpoint.adapters import EXPERT_LORA_PEFT_TYPE, reject_expert_lora_adapter
from halo.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_WEIGHT_NAMES,
    MERGED_ADAPTER_CONFIG_DIR,
    SAFETENSORS_INDEX_FILE,
)
from halo.checkpoint.tools import adapter_merge, bf16_conversion

# Each one is the check under test.
_REAL_GATES = (
    "reject_partial_checkpoint",
    "reject_in_place_conversion",
    "reject_expert_lora_adapter",
    "preflight_model_load_resources",
)


class _Merged:
    def merge_and_unload(self):
        return self


class _FakePeftModel:
    @staticmethod
    def from_pretrained(base_model, _adapter_dir):
        return base_model


def _adapter_dir(tmp_path: Path, base: Path, *, peft_type: str = "LORA") -> str:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / ADAPTER_CONFIG_FILE).write_text(
        json.dumps(
            {
                "peft_type": peft_type,
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": str(base),
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
            }
        )
    )
    return str(adapter)


def _names(calls: list[tuple[str, str]]) -> list[str]:
    return [name for name, _target in calls]


def _instrument(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, str]]) -> None:
    """The directory rides along since the two directory gates must see the base as well as the adapter."""
    for name in _REAL_GATES:
        real = getattr(adapter_merge, name)

        def wrapped(*args, _name=name, _real=real, **kwargs):
            calls.append((_name, str(args[0]) if args else ""))
            return _real(*args, **kwargs)

        monkeypatch.setattr(adapter_merge, name, wrapped)

    def stub(name, result):
        def record(*args, **_kwargs):
            calls.append((name, str(args[0]) if args else ""))
            return result

        return record

    monkeypatch.setattr(adapter_merge, "PeftModel", _FakePeftModel)
    monkeypatch.setattr(
        adapter_merge, "resolve_peft_processing_class", stub("resolve_peft_processing_class", object())
    )
    monkeypatch.setattr(adapter_merge, "apply_training_sidecars", stub("apply_training_sidecars", []))
    monkeypatch.setattr(adapter_merge, "save_full_checkpoint", stub("save_full_checkpoint", None))
    monkeypatch.setattr(adapter_merge, "copy_training_sidecars", stub("copy_training_sidecars", None))


def _load_base_model(calls: list[tuple[str, str]]):
    def load(base_model_path, *, excuse_task_head):
        calls.append(("load_base_model", str(base_model_path)))
        return _Merged()

    return load


def test_every_gate_runs_before_the_load_on_both_the_adapter_and_the_base(tmp_path, monkeypatch):
    """Stated as what is observable rather than a transcript of names."""
    base = tmp_path / "base"
    base.mkdir()
    adapter = _adapter_dir(tmp_path, base)
    calls: list[tuple[str, str]] = []
    _instrument(monkeypatch, calls)

    adapter_merge.merge_adapter_into_base(
        adapter,
        str(tmp_path / "out"),
        load_base_model=_load_base_model(calls),
        tool="test",
        verbose=False,
    )

    names = _names(calls)
    load_at = names.index("load_base_model")
    assert set(_REAL_GATES) <= set(names[:load_at]), f"a gate did not run before the load: {names}"
    assert not set(_REAL_GATES) & set(names[load_at:]), f"a gate ran after the load: {names}"
    for gate in ("reject_partial_checkpoint", "reject_in_place_conversion"):
        checked = {target for name, target in calls if name == gate}
        assert checked == {str(adapter), str(base)}, (
            f"{gate} saw {checked}: a merge takes its WEIGHTS from the base, so both inputs owe it"
        )
    assert load_at < names.index("save_full_checkpoint") < names.index("copy_training_sidecars")
    # Left beside the merged weights, every from_pretrained-based tool reads the merged model as an unmerged adapter.
    relocated = tmp_path / "out" / MERGED_ADAPTER_CONFIG_DIR / ADAPTER_CONFIG_FILE
    assert relocated.is_file(), "the adapter config was not relocated beside the merged output"


@pytest.mark.parametrize("side", ["adapter", "base"])
def test_an_output_aimed_at_an_input_is_refused_before_the_load(tmp_path, monkeypatch, side):
    """The save deletes the weight files it does not overwrite."""
    base = tmp_path / "base"
    base.mkdir()
    adapter = _adapter_dir(tmp_path, base)
    calls: list[tuple[str, str]] = []
    _instrument(monkeypatch, calls)

    with pytest.raises(ValueError, match="not in-place"):
        adapter_merge.merge_adapter_into_base(
            adapter,
            adapter if side == "adapter" else str(base),
            load_base_model=_load_base_model(calls),
            tool="test",
            verbose=False,
        )

    assert "load_base_model" not in _names(calls), "the base was loaded despite a refused output directory"


def test_a_per_rank_sharded_base_is_refused_before_the_load(tmp_path, monkeypatch):
    """Its experts would load randomly initialized, a warning only."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "model-00001-of-00002.safetensors").write_bytes(b"")
    (base / SAFETENSORS_INDEX_FILE).write_text(
        json.dumps(
            {
                "metadata": {"format": "ep_sharded", "ep_size": 2},
                "weight_map": {"model.layers.0.mlp.experts.gate_up_proj.shard_0": "model-00001-of-00002.safetensors"},
            }
        )
    )
    adapter = _adapter_dir(tmp_path, base)
    calls: list[tuple[str, str]] = []
    _instrument(monkeypatch, calls)

    with pytest.raises(ValueError, match="per-rank sharded checkpoint"):
        adapter_merge.merge_adapter_into_base(
            adapter, str(tmp_path / "out"), load_base_model=_load_base_model(calls), tool="test", verbose=False
        )

    assert "load_base_model" not in _names(calls)
    assert not os.path.exists(tmp_path / "out"), "a refused merge left an output directory behind"


def test_a_native_expert_lora_adapter_is_refused_before_the_load(tmp_path, monkeypatch):
    """``merge_and_unload`` drops every expert delta and yields a base-quality model that looks merged."""
    base = tmp_path / "base"
    base.mkdir()
    adapter = _adapter_dir(tmp_path, base, peft_type=EXPERT_LORA_PEFT_TYPE)
    calls: list[tuple[str, str]] = []
    _instrument(monkeypatch, calls)

    with pytest.raises(ValueError):
        adapter_merge.merge_adapter_into_base(
            adapter, str(tmp_path / "out"), load_base_model=_load_base_model(calls), tool="test", verbose=False
        )

    assert "load_base_model" not in _names(calls)
    assert not os.path.exists(tmp_path / "out"), "a refused merge left an output directory behind"


@pytest.mark.parametrize("weights_file", ADAPTER_WEIGHT_NAMES)
def test_an_unmarked_expert_adapter_is_caught_in_either_weight_file(tmp_path, weights_file):
    """``adapter_model.bin`` is the saver's fallback; a safetensors-only reader reads such an adapter as plain."""
    base = tmp_path / "base"
    base.mkdir()
    adapter = Path(_adapter_dir(tmp_path, base))  # Unmarked, so only the keys tell.
    expert_tensor = {"base_model.model.layers.0.mlp.experts.gate_up_proj.lora_A": torch.zeros(2, 2)}
    if weights_file.endswith(".safetensors"):
        save_file(expert_tensor, str(adapter / weights_file))
    else:
        torch.save(expert_tensor, adapter / weights_file)

    with pytest.raises(ValueError, match="expert-LoRA adapter"):
        reject_expert_lora_adapter(str(adapter))


def test_the_merge_tool_folds_through_the_shared_sequence(tmp_path, monkeypatch):
    """A tool re-growing its own load→merge→save carries none of these gates."""
    base = tmp_path / "base"
    base.mkdir()
    adapter = _adapter_dir(tmp_path, base)
    tools: list[str] = []

    def capture(_adapter_dir, _output_dir, *, tool, **_kwargs):
        tools.append(tool)
        return _Merged()

    monkeypatch.setattr(adapter_merge, "merge_adapter_into_base", capture)
    adapter_merge.merge_peft_adapter(adapter_dir=adapter, output_dir=str(tmp_path / "out"), verbose=False)

    assert tools == ["merge_peft_adapters"], f"merge_peft_adapters did not reach the shared merge (reached: {tools})"


@pytest.mark.parametrize("side", ["adapter", "base"])
def test_an_unmerged_adapter_conversion_runs_the_shared_gates(tmp_path, monkeypatch, side):
    """A later merge reads the re-saved adapter, so it owes the same gates."""
    assert bf16_conversion.adapter_input_gates is adapter_merge.adapter_input_gates

    base = tmp_path / "base"
    base.mkdir()
    adapter = _adapter_dir(tmp_path, base)
    monkeypatch.setattr(bf16_conversion, "load_model", lambda *a, **k: pytest.fail("loaded past a refused gate"))

    with pytest.raises(ValueError, match="not in-place"):
        bf16_conversion.convert_to_bf16(
            adapter, adapter if side == "adapter" else str(base), "causal_lm", is_peft=True
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
