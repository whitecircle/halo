#!/usr/bin/env python
"""Every toolkit path that writes a model ``config.json`` must finalize it the same way.

``finalize_exported_config`` is the whole contract: the live ``model_type`` restored for the vendor
classes that declare none (Bailing/Ling), the flat legacy per-layer keys the pinned server reads
(Gemma 4 — transformers 5.16 serializes that geometry only as ``per_layer_config``), and the source
repo's own schema for the families whose serving engines have no config class (Step-3.7). Running
one of the three is not running the contract: the artifact loads and trains, and fails later in the
merge tools or the server on a family the run never mentioned. The trainer's writers reach it
through ``save_model_config`` and the tools through ``save_full_checkpoint``; a bare
``save_pretrained`` on a model or config object owes the call itself. This walks every function
under ``src/`` and ``scripts/`` that calls ``save_pretrained`` on something other than a tokenizer /
processor / adapter config and requires the finalizer (or one of the two writers carrying it) in the
same function.

Run: ``pytest -m cpu tests/cpu/checkpoint/test_config_writers_flatten_legacy_keys.py``
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from tests.common.utils import REPO_ROOT

_ROOTS = ("src", "scripts")
# Receivers whose save_pretrained writes no model config.json: tokenizers, processors, generation
# configs, PEFT adapter configs and adapter-only PeftModel saves.
_EXEMPT_RECEIVER_MARKERS = (
    "tokenizer",
    "processor",
    "processing_class",
    "generation_config",
    "peft_config",
    "peft_model",
    "adapter",
)
# Functions whose save produces no artifact: adapter-only by construction (no config.json), or the
# throwaway staging re-serialization the source-schema carry differences against — which runs the
# finalizer's first two steps and cannot call the finalizer itself without recursing into it.
_EXEMPT_FUNCTIONS = {
    ("scripts/after_training/convert_to_bf16.py", "convert_to_bf16"),
    ("src/checkpoint/config_export.py", "_serialized_config"),
}
_FINALIZERS = {"finalize_exported_config", "save_full_checkpoint", "save_model_config"}


def _receiver_text(call: ast.Call) -> str:
    return ast.unparse(call.func.value) if isinstance(call.func, ast.Attribute) else ""


def _config_writers(path: pathlib.Path) -> list[tuple[str, bool]]:
    """``(function name, finalizes)`` for every function calling a non-exempt ``save_pretrained``."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        calls = [c for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)]
        saves = [
            c
            for c in calls
            if c.func.attr == "save_pretrained"
            and not any(marker in _receiver_text(c).lower() for marker in _EXEMPT_RECEIVER_MARKERS)
        ]
        if not saves:
            continue
        finalizes = any(
            (isinstance(c.func, ast.Name) and c.func.id in _FINALIZERS)
            or (isinstance(c.func, ast.Attribute) and c.func.attr in _FINALIZERS)
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
        )
        found.append((node.name, finalizes))
    return found


def _python_files():
    for root in _ROOTS:
        yield from sorted((REPO_ROOT / root).rglob("*.py"))


@pytest.mark.parametrize("path", list(_python_files()), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_every_model_config_writer_finalizes_the_config(path):
    relative = str(path.relative_to(REPO_ROOT))
    unfinalized = [
        name
        for name, finalizes in _config_writers(path)
        if not finalizes and (relative, name) not in _EXEMPT_FUNCTIONS and name not in _FINALIZERS
    ]
    assert not unfinalized, (
        f"{relative}: {unfinalized} write a model config.json without finalize_exported_config "
        f"(or save_full_checkpoint / save_model_config) — a Gemma 4 export from here is unservable, "
        f"a Bailing one is family-less and a Step-3.7 one carries a schema no server parses"
    )


def test_the_roster_is_not_vacuous():
    writers = {(str(p.relative_to(REPO_ROOT)), name) for p in _python_files() for name, _ in _config_writers(p)}
    assert ("src/checkpoint/tool_io.py", "save_full_checkpoint") in writers
    assert ("scripts/after_training/reattach_vision_tower.py", "main") in writers or any(
        path.endswith("reattach_vision_tower.py") for path, _ in writers
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
