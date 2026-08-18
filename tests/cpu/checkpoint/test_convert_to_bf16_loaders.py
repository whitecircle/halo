#!/usr/bin/env python
"""CPU tests for ``scripts/after_training/convert_to_bf16.py``'s loader dispatch.

One table must drive both call sites that load a model (convert, inference check) instead of a repeated
``model_type`` if-ladder, whose trailing ``else`` silently loads a bare ``AutoModel`` for any unknown
type. An unknown / PEFT-unsupported type must fail loudly.

Which class each type resolves to is asserted at the gate the load actually goes through
(``from_pretrained_verified`` / ``auto_load_model``), because reaching ``from_pretrained`` around it
is the other way this dispatch goes wrong — see
``tests/cpu/checkpoint/test_after_training_load_gates.py``. Every load also reaches
``finalize_loaded_model``: ``--check_inference`` generates from what the loader returns, and
transformers 5 hands back uninitialized non-persistent buffers.

Run: ``python tests/cpu/checkpoint/test_convert_to_bf16_loaders.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import ast
import pathlib
import types

import pytest

import scripts.after_training.convert_to_bf16 as convert_module
from scripts.after_training.convert_to_bf16 import _MODEL_CLASSES, _PEFT_MODEL_TYPES, load_model


@pytest.fixture
def finalized(monkeypatch) -> list:
    """Records what reaches the post-load seam. Stubbed because these tests hand the loader
    sentinels rather than modules, and returned so each can assert the load reached it."""
    seen: list = []
    monkeypatch.setattr(convert_module, "finalize_loaded_model", seen.append)
    return seen


def _model_type_choices() -> set[str]:
    """The ``--model_type`` choices the CLI advertises, read off its own ``add_argument`` call.

    Read statically rather than by running ``parse_args()``, which would need a full argv and
    exit on a missing required flag.
    """
    source = ast.parse(pathlib.Path(convert_module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(source):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--model_type"
        ):
            choices = {kw.arg: kw.value for kw in node.keywords}["choices"]
            return {elt.value for elt in choices.elts}
    raise AssertionError("convert_to_bf16.py no longer declares a --model_type argument")


def test_table_covers_exactly_the_cli_choices():
    """Derived from the parser, not transcribed: a choice added without a loader entry (or a loader
    added without a choice) must fail here rather than silently fall through to a bare AutoModel."""
    assert set(_MODEL_CLASSES) == _model_type_choices()


def test_unknown_model_type_raises_instead_of_falling_back():
    with pytest.raises(ValueError, match="Unknown model_type"):
        load_model("/nonexistent", "reward_model")


def test_peft_unsupported_type_names_the_supported_ones():
    with pytest.raises(ValueError, match="PEFT is only supported") as exc:
        load_model("/nonexistent", "base", is_peft=True)
    assert all(model_type in str(exc.value) for model_type in _PEFT_MODEL_TYPES)
    assert set(_PEFT_MODEL_TYPES) < set(_MODEL_CLASSES), "the PEFT-capable types are a subset of the CLI's"


def test_known_types_dispatch_to_their_registered_class_through_the_gate(monkeypatch, finalized):
    """Each type loads with the class the table registers, and every one of them reaches the
    checkpoint-coverage gate — ``causal_lm`` through the config-resolving loader that is itself
    gated (a VLM base must keep its vision tower), the rest with their pinned class."""
    calls: dict = {}
    monkeypatch.setattr(
        convert_module,
        "from_pretrained_verified",
        lambda cls, path, **kw: calls.setdefault("gated", (cls, path, kw)),
    )
    monkeypatch.setattr(
        convert_module,
        "auto_load_model",
        lambda path, **kw: calls.setdefault("gated", (None, path, kw)),
    )

    for model_type, model_class in _MODEL_CLASSES.items():
        calls.clear()
        finalized.clear()
        load_model("/some/path", model_type, dtype="bf16")
        assert calls["gated"] == (model_class, "/some/path", {"dtype": "bf16", "excuse_task_head": False})
        assert finalized == [calls["gated"]], f"{model_type} skipped the post-load buffer seam"


def test_a_peft_load_gates_the_base_the_adapter_names(monkeypatch, finalized):
    """The weights come from the BASE under ``--peft``, so that is the directory the gate must see —
    PEFT's own auto-class would load it through a raw ``from_pretrained``."""
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

    monkeypatch.setattr(convert_module, "auto_load_model", fake_load)
    monkeypatch.setattr(convert_module.PeftModel, "from_pretrained", lambda base, path: (base, path))

    assert load_model("/adapter", "causal_lm", is_peft=True) == (base_model, "/adapter")
    assert seen["load"] == ("/base", {"excuse_task_head": False})
    assert finalized == [base_model], "the base the adapter names is the one that must be finalized"


def test_the_task_head_excuse_follows_the_adapters_declaration(monkeypatch, finalized):
    """``modules_to_save`` is the only thing that makes an absent seq-cls head legitimate."""
    seen: dict = {}
    monkeypatch.setattr(
        convert_module.PeftConfig,
        "from_pretrained",
        classmethod(
            lambda cls, path: types.SimpleNamespace(base_model_name_or_path="/base", modules_to_save=["score"])
        ),
    )
    monkeypatch.setattr(
        convert_module, "from_pretrained_verified", lambda cls, path, **kw: seen.setdefault("load", kw)
    )
    monkeypatch.setattr(convert_module.PeftModel, "from_pretrained", lambda base, path: base)

    load_model("/adapter", "classifier", is_peft=True)
    assert seen["load"]["excuse_task_head"] is True
    assert finalized == [seen["load"]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
