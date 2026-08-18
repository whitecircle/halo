#!/usr/bin/env python
"""The shared PEFT merge must run its input gates in order, before anything is loaded or written.

``convert_to_bf16 --peft --merge_adapter`` and ``merge_peft_adapters`` both fold a saved adapter into
its base through ``merge_adapter_into_base``. Every gate in front of that fold answers a failure that
is silent without it — a per-rank EP/TP directory whose real expert keys read as MISSING and get
randomly initialized, an ``--output_dir`` aimed at an input whose weight files the save then deletes,
a native expert-LoRA adapter whose deltas ``merge_and_unload`` drops — and each is only worth having
if it runs BEFORE the base model is loaded and the output directory created. The base directory is
re-checked separately because under a merge the WEIGHTS come from the base, so the adapter-side gates
covered the wrong directory.

One sequence serves both tools. These assertions pin that single copy — its gates, their order, and
the point they run at — and fail if a tool grows its own.

Run: ``python tests/cpu/checkpoint/test_merge_adapter_gates.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from accelerate import PartialState

PartialState()  # the merge's helpers log through accelerate's logger

import torch
from safetensors.torch import save_file

from scripts.after_training import convert_to_bf16, merge_peft_adapters
from src.checkpoint import adapters
from src.checkpoint.adapters import EXPERT_LORA_PEFT_TYPE, assert_no_expert_lora_adapter
from src.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_WEIGHT_NAMES, SAFETENSORS_INDEX_FILE

# The gates that must run for real: each one is the check under test, not a stand-in for it.
_REAL_GATES = (
    "reject_sharded_checkpoint",
    "reject_in_place_conversion",
    "assert_no_expert_lora_adapter",
    "preflight_model_load_resources",
)


class _Merged:
    """The merged model, reduced to what the merge itself touches."""

    def merge_and_unload(self):
        return self


class _FakePeftModel:
    @staticmethod
    def from_pretrained(base_model, _adapter_dir):
        return base_model


def _adapter_dir(tmp_path: Path, base: Path, *, peft_type: str = "LORA") -> str:
    """A PEFT adapter directory naming ``base`` as its base model."""
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
    """Record ``(step, the directory it was given)``: real gates wrapped, the heavy edges stubbed out.

    The directory rides along because it is half the contract — the two directory gates must see the
    BASE as well as the adapter, and a transcript of names alone cannot tell the two calls apart.
    """
    for name in _REAL_GATES:
        real = getattr(adapters, name)

        def wrapped(*args, _name=name, _real=real, **kwargs):
            calls.append((_name, str(args[0]) if args else ""))
            return _real(*args, **kwargs)

        monkeypatch.setattr(adapters, name, wrapped)

    def stub(name, result):
        def record(*args, **_kwargs):
            calls.append((name, str(args[0]) if args else ""))
            return result

        return record

    monkeypatch.setattr(adapters, "PeftModel", _FakePeftModel)
    monkeypatch.setattr(adapters, "resolve_peft_processing_class", stub("resolve_peft_processing_class", object()))
    monkeypatch.setattr(adapters, "apply_training_sidecars", stub("apply_training_sidecars", []))
    monkeypatch.setattr(adapters, "save_full_checkpoint", stub("save_full_checkpoint", None))
    monkeypatch.setattr(adapters, "copy_training_sidecars", stub("copy_training_sidecars", None))


def _load_base_model(calls: list[tuple[str, str]]):
    def load(base_model_path, *, excuse_task_head):
        calls.append(("load_base_model", str(base_model_path)))
        return _Merged()

    return load


def test_every_gate_runs_before_the_load_on_both_the_adapter_and_the_base(tmp_path, monkeypatch):
    """The contract the shared merge owes, stated as what is observable rather than as a transcript:
    every gate has run by the time the base is read, each input directory has been through the two
    directory gates, and the write happens after the load with its sidecars last."""
    base = tmp_path / "base"
    base.mkdir()
    adapter = _adapter_dir(tmp_path, base)
    calls: list[tuple[str, str]] = []
    _instrument(monkeypatch, calls)

    adapters.merge_adapter_into_base(
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
    for gate in ("reject_sharded_checkpoint", "reject_in_place_conversion"):
        checked = {target for name, target in calls if name == gate}
        assert checked == {str(adapter), str(base)}, (
            f"{gate} saw {checked}: a merge takes its WEIGHTS from the base, so both inputs owe it"
        )
    # The write follows the load, and the adapter's sidecars follow the write.
    assert load_at < names.index("save_full_checkpoint") < names.index("copy_training_sidecars")
    # The adapter's own config is relocated, never left beside the merged weights where every
    # from_pretrained-based tool downstream would read the merged model as an unmerged adapter.
    relocated = tmp_path / "out" / adapters.MERGED_ADAPTER_CONFIG_DIR / ADAPTER_CONFIG_FILE
    assert relocated.is_file(), "the adapter config was not relocated beside the merged output"


@pytest.mark.parametrize("side", ["adapter", "base"])
def test_an_output_aimed_at_an_input_is_refused_before_the_load(tmp_path, monkeypatch, side):
    """The save deletes the weight files it does not overwrite, so this must fire before anything
    is read — writing into either input destroys that input."""
    base = tmp_path / "base"
    base.mkdir()
    adapter = _adapter_dir(tmp_path, base)
    calls: list[tuple[str, str]] = []
    _instrument(monkeypatch, calls)

    with pytest.raises(ValueError, match="not in-place"):
        adapters.merge_adapter_into_base(
            adapter,
            adapter if side == "adapter" else str(base),
            load_base_model=_load_base_model(calls),
            tool="test",
            verbose=False,
        )

    assert "load_base_model" not in _names(calls), "the base was loaded despite a refused output directory"


def test_a_per_rank_sharded_base_is_refused_before_the_load(tmp_path, monkeypatch):
    """Its experts would load randomly initialized — a warning, never a raise — and the adapter
    would be merged into those."""
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
        adapters.merge_adapter_into_base(
            adapter, str(tmp_path / "out"), load_base_model=_load_base_model(calls), tool="test", verbose=False
        )

    assert "load_base_model" not in _names(calls)
    assert not os.path.exists(tmp_path / "out"), "a refused merge left an output directory behind"


def test_a_native_expert_lora_adapter_is_refused_before_the_load(tmp_path, monkeypatch):
    """``merge_and_unload`` cannot fold grouped expert adapters: it drops every expert delta and
    yields a base-quality model that looks merged."""
    base = tmp_path / "base"
    base.mkdir()
    adapter = _adapter_dir(tmp_path, base, peft_type=EXPERT_LORA_PEFT_TYPE)
    calls: list[tuple[str, str]] = []
    _instrument(monkeypatch, calls)

    with pytest.raises(ValueError):
        adapters.merge_adapter_into_base(
            adapter, str(tmp_path / "out"), load_base_model=_load_base_model(calls), tool="test", verbose=False
        )

    assert "load_base_model" not in _names(calls)
    assert not os.path.exists(tmp_path / "out"), "a refused merge left an output directory behind"


@pytest.mark.parametrize("weights_file", ADAPTER_WEIGHT_NAMES)
def test_an_unmarked_expert_adapter_is_caught_in_either_weight_file(tmp_path, weights_file):
    """The verdict is the SHAPE, so the tensor scan must read whichever file PEFT wrote.

    ``adapter_model.bin`` is the fallback the saver takes when the safetensors write fails, and a
    reader that hand-builds only the safetensors name reads such an adapter as plain — handing
    ``merge_and_unload`` an expert adapter whose every delta it drops, for a base-quality model that
    looks merged.
    """
    base = tmp_path / "base"
    base.mkdir()
    adapter = Path(_adapter_dir(tmp_path, base))  # peft_type LORA: unmarked, so only the keys tell
    expert_tensor = {"base_model.model.layers.0.mlp.experts.gate_up_proj.lora_A": torch.zeros(2, 2)}
    if weights_file.endswith(".safetensors"):
        save_file(expert_tensor, str(adapter / weights_file))
    else:
        torch.save(expert_tensor, adapter / weights_file)

    with pytest.raises(ValueError, match="expert-LoRA adapter"):
        assert_no_expert_lora_adapter(str(adapter))


def _merge_via_convert_to_bf16(adapter: str, output_dir: str) -> None:
    convert_to_bf16.convert_to_bf16(adapter, output_dir, "causal_lm", is_peft=True, merge_adapter=True)


def _merge_via_merge_peft_adapters(adapter: str, output_dir: str) -> None:
    merge_peft_adapters.merge_peft_adapter(adapter_dir=adapter, output_dir=output_dir, verbose=False)


@pytest.mark.parametrize(
    ("script", "run"),
    [
        (convert_to_bf16, _merge_via_convert_to_bf16),
        (merge_peft_adapters, _merge_via_merge_peft_adapters),
    ],
    ids=["convert_to_bf16", "merge_peft_adapters"],
)
def test_both_merge_tools_fold_through_the_shared_sequence(tmp_path, monkeypatch, script, run):
    """A tool that re-grew its own load→merge→save path would carry none of the gates above, so each
    tool's own entry point is driven here and must arrive at the shared merge."""
    assert script.merge_adapter_into_base is adapters.merge_adapter_into_base

    base = tmp_path / "base"
    base.mkdir()
    adapter = _adapter_dir(tmp_path, base)
    tools: list[str] = []

    def capture(_adapter_dir, _output_dir, *, tool, **_kwargs):
        tools.append(tool)
        return _Merged()

    monkeypatch.setattr(script, "merge_adapter_into_base", capture)
    run(adapter, str(tmp_path / "out"))

    assert tools == [script.__name__.rsplit(".", 1)[-1]], (
        f"{script.__name__} did not reach the shared merge (reached: {tools})"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
