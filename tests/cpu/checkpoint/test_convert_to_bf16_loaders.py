#!/usr/bin/env python
"""``halo.checkpoint.tools.bf16_conversion``'s loader dispatch: one table drives both call sites, an unknown type
fails loudly, and each load reaches ``finalize_loaded_model``."""

from __future__ import annotations

import argparse
import json
import types

import pytest
from torch import nn

import halo.checkpoint.tools.bf16_conversion as convert_module
from halo.checkpoint.tools import offline_export
from halo.checkpoint.tools.bf16_conversion import load_model
from halo.checkpoint.tools.offline_export import ADAPTER_TASKS, TASK_MODEL_CLASSES
from scripts.after_training import convert_to_bf16 as convert_script
from scripts.after_training import merge_peft_adapters as merge_script
from tests.common.tiny_models import word_level_tokenizer


@pytest.fixture
def finalized(monkeypatch) -> list:
    """Stubbed because these tests hand the loader sentinels rather than modules."""
    seen: list = []
    monkeypatch.setattr(convert_module, "finalize_loaded_model", seen.append)
    return seen


class _ParserBuilt(Exception):
    """Carries the tool's parser out of ``parse_args`` before it reads ``sys.argv``."""


def _task_choices(monkeypatch, entry) -> set[str]:
    def capture(self, *args, **kwargs):
        raise _ParserBuilt(self)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(_ParserBuilt) as built:
        entry()
    parser = built.value.args[0]
    return set(next(action.choices for action in parser._actions if "--task" in action.option_strings))


def test_table_covers_exactly_the_cli_choices(monkeypatch):
    """A choice without a loader entry must not fall through to a bare AutoModel."""
    assert set(TASK_MODEL_CLASSES) == _task_choices(monkeypatch, convert_script.parse_args)


def test_the_merge_and_the_conversion_spell_one_head_flag(monkeypatch):
    """A command line carries from the conversion to the merge."""
    assert _task_choices(monkeypatch, merge_script.main) == set(ADAPTER_TASKS) < set(TASK_MODEL_CLASSES)


def test_a_base_task_on_an_adapter_is_refused_before_any_write(tmp_path):
    """Through the conversion itself, where an adapter is rewritten with no model load to reach the task gate."""
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="PEFT is only supported"):
        convert_module.convert_to_bf16(str(tmp_path), str(out), "base", is_peft=True)
    assert not out.exists()


class _Generates(nn.Linear):
    def __init__(self):
        super().__init__(1, 1)
        self.generated = False

    def generate(self, *, input_ids, **kwargs):
        self.generated = True
        return input_ids


@pytest.fixture
def adapter_dir(tmp_path):
    """A tokenizer-less adapter directory over a base holding one."""
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    word_level_tokenizer(["Hello"]).save_pretrained(base)
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": str(base), "r": 1})
    )
    return adapter


def test_check_inference_on_a_tokenizerless_adapter_takes_the_bases_tokenizer(monkeypatch, adapter_dir):
    """The conversion is already written by then, and the check returns no verdict: it must not fail the run."""
    model = _Generates()
    monkeypatch.setattr(convert_module, "load_model", lambda *a, **k: model)

    convert_module.run_test_inference(str(adapter_dir), "causal_lm", is_peft=True, trust_remote_code=False)

    assert model.generated


def test_check_inference_skips_with_no_tokenizer_anywhere(monkeypatch, caplog, adapter_dir, tmp_path):
    base = tmp_path / "bare"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": str(base), "r": 1})
    )

    def refuse(*args, **kwargs):
        raise AssertionError("the model loads for a check with no tokenizer to feed it")

    monkeypatch.setattr(convert_module, "load_model", refuse)
    with caplog.at_level("WARNING"):
        convert_module.run_test_inference(str(adapter_dir), "causal_lm", is_peft=True, trust_remote_code=False)
    assert any("skipping the inference check" in rec.getMessage() for rec in caplog.records)


def test_unknown_task_raises_instead_of_falling_back():
    with pytest.raises(ValueError, match="Unknown task"):
        load_model("/nonexistent", "reward_model")


def test_peft_unsupported_type_names_the_supported_ones():
    with pytest.raises(ValueError, match="PEFT is only supported") as exc:
        load_model("/nonexistent", "base", is_peft=True)
    assert all(task in str(exc.value) for task in ADAPTER_TASKS)


def test_known_types_dispatch_to_their_registered_class_through_the_gate(monkeypatch, finalized):
    """``causal_lm`` goes through the config-resolving loader, since a VLM base must keep its vision tower."""
    calls: dict = {}
    monkeypatch.setattr(
        offline_export,
        "from_pretrained_verified",
        lambda cls, path, **kw: calls.setdefault("gated", (cls, path, kw)),
    )
    monkeypatch.setattr(
        offline_export,
        "auto_load_model",
        lambda path, **kw: calls.setdefault("gated", (None, path, kw)),
    )

    for task, model_class in TASK_MODEL_CLASSES.items():
        calls.clear()
        finalized.clear()
        load_model("/some/path", task, dtype="bf16")
        expected = {"dtype": "bf16", "excuse_task_head": False}
        if model_class is None:
            # The config-resolving loader alone takes the class request an adapter's keys can make.
            expected["text_only"] = False
        assert calls["gated"] == (model_class, "/some/path", expected)
        assert finalized == [calls["gated"]], f"{task} skipped the post-load buffer seam"


def test_a_peft_load_gates_the_base_the_adapter_names(monkeypatch, finalized):
    """PEFT's own auto-class would load the base through a raw ``from_pretrained``."""
    seen: dict = {}
    monkeypatch.setattr(
        convert_module.PeftConfig,
        "from_pretrained",
        classmethod(lambda cls, path: types.SimpleNamespace(base_model_name_or_path="/base", modules_to_save=None)),
    )
    base_model = object()

    def fake_load(path, **kwargs):
        seen["load"] = (path, kwargs)
        return base_model

    monkeypatch.setattr(offline_export, "auto_load_model", fake_load)
    monkeypatch.setattr(convert_module.PeftModel, "from_pretrained", lambda base, path: (base, path))

    assert load_model("/adapter", "causal_lm", is_peft=True) == (base_model, "/adapter")
    assert seen["load"] == ("/base", {"excuse_task_head": False, "text_only": False})
    assert finalized == [base_model], "the base the adapter names is the one that must be finalized"


def test_the_task_head_excuse_follows_the_adapters_declaration(monkeypatch, finalized):
    """``modules_to_save`` alone makes an absent seq-cls head legitimate."""
    seen: dict = {}
    monkeypatch.setattr(
        convert_module.PeftConfig,
        "from_pretrained",
        classmethod(
            lambda cls, path: types.SimpleNamespace(base_model_name_or_path="/base", modules_to_save=["score"])
        ),
    )
    monkeypatch.setattr(
        offline_export, "from_pretrained_verified", lambda cls, path, **kw: seen.setdefault("load", kw)
    )
    monkeypatch.setattr(convert_module.PeftModel, "from_pretrained", lambda base, path: base)

    load_model("/adapter", "classification", is_peft=True)
    assert seen["load"]["excuse_task_head"] is True
    assert finalized == [seen["load"]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
