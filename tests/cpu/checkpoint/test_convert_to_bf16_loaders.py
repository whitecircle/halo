#!/usr/bin/env python
"""CPU tests for ``halo.checkpoint.bf16_conversion``'s loader dispatch.

One table must drive both call sites that load a model (convert, inference check) instead of a repeated
``--task`` if-ladder, whose trailing ``else`` silently loads a bare ``AutoModel`` for any unknown
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

import argparse
import types

import pytest

import halo.checkpoint.bf16_conversion as convert_module
from halo.checkpoint import offline_export
from halo.checkpoint.bf16_conversion import load_model
from halo.checkpoint.offline_export import ADAPTER_TASKS, TASK_MODEL_CLASSES
from scripts.after_training import convert_to_bf16 as convert_script
from scripts.after_training import merge_peft_adapters as merge_script


@pytest.fixture
def finalized(monkeypatch) -> list:
    """Records what reaches the post-load seam. Stubbed because these tests hand the loader
    sentinels rather than modules, and returned so each can assert the load reached it."""
    seen: list = []
    monkeypatch.setattr(convert_module, "finalize_loaded_model", seen.append)
    return seen


class _ParserBuilt(Exception):
    """Carries the tool's parser out of ``parse_args`` before it reads ``sys.argv``."""


def _task_choices(monkeypatch, entry) -> set[str]:
    """The ``--task`` choices a tool's CLI advertises, read off the parser ``entry`` builds."""

    def capture(self, *args, **kwargs):
        raise _ParserBuilt(self)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(_ParserBuilt) as built:
        entry()
    parser = built.value.args[0]
    return set(next(action.choices for action in parser._actions if "--task" in action.option_strings))


def test_table_covers_exactly_the_cli_choices(monkeypatch):
    """Read off the parser, not transcribed: a choice added without a loader entry (or a loader
    added without a choice) must fail here rather than silently fall through to a bare AutoModel."""
    assert set(TASK_MODEL_CLASSES) == _task_choices(monkeypatch, convert_script.parse_args)


def test_the_merge_and_the_conversion_spell_one_head_flag(monkeypatch):
    """A command line carries from the conversion to the merge: both take ``--task``, over the same names,
    the merge over the tasks an adapter is trained for."""
    assert _task_choices(monkeypatch, merge_script.main) == set(ADAPTER_TASKS) < set(TASK_MODEL_CLASSES)


def test_unknown_task_raises_instead_of_falling_back():
    with pytest.raises(ValueError, match="Unknown task"):
        load_model("/nonexistent", "reward_model")


def test_peft_unsupported_type_names_the_supported_ones():
    with pytest.raises(ValueError, match="PEFT is only supported") as exc:
        load_model("/nonexistent", "base", is_peft=True)
    assert all(task in str(exc.value) for task in ADAPTER_TASKS)


def test_known_types_dispatch_to_their_registered_class_through_the_gate(monkeypatch, finalized):
    """Each type loads with the class the table registers, and every one of them reaches the
    checkpoint-coverage gate — ``causal_lm`` through the config-resolving loader that is itself
    gated (a VLM base must keep its vision tower), the rest with their pinned class."""
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

    monkeypatch.setattr(offline_export, "auto_load_model", fake_load)
    monkeypatch.setattr(convert_module.PeftModel, "from_pretrained", lambda base, path: (base, path))

    assert load_model("/adapter", "causal_lm", is_peft=True) == (base_model, "/adapter")
    assert seen["load"] == ("/base", {"excuse_task_head": False, "text_only": False})
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
        offline_export, "from_pretrained_verified", lambda cls, path, **kw: seen.setdefault("load", kw)
    )
    monkeypatch.setattr(convert_module.PeftModel, "from_pretrained", lambda base, path: base)

    load_model("/adapter", "classification", is_peft=True)
    assert seen["load"]["excuse_task_head"] is True
    assert finalized == [seen["load"]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
