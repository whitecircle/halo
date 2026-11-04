#!/usr/bin/env python
"""Every path writing a model ``config.json`` runs ``finalize_exported_config``, or the artifact fails later on a
family the run never mentioned."""

from __future__ import annotations

import ast
import pathlib

import pytest

from tests.common.repo import REPO_ROOT

_ROOTS = ("halo", "scripts")
# Receivers whose save_pretrained writes no model config.json.
_EXEMPT_RECEIVER_MARKERS = (
    "tokenizer",
    "processor",
    "processing_class",
    "generation_config",
    "peft_config",
    "peft_model",
    "adapter",
)
# The throwaway staging re-serialization runs the finalizer's first step and cannot call it without recursing.
_EXEMPT_FUNCTIONS = {
    ("halo/checkpoint/config_export.py", "_serialized_config"),
}
_FINALIZERS = {"finalize_exported_config", "save_full_checkpoint", "save_model_config"}


def _receiver_text(call: ast.Call) -> str:
    return ast.unparse(call.func.value) if isinstance(call.func, ast.Attribute) else ""


def _config_writers(path: pathlib.Path) -> list[tuple[str, bool]]:
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
        assert (REPO_ROOT / root).is_dir(), f"the sweep lost its {root}/ root"
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
        f"(or save_full_checkpoint / save_model_config) — a Bailing export from here is family-less, "
        f"a Step-3.7 one carries a schema no server parses and a Laguna one names remote code it lacks"
    )


@pytest.mark.parametrize(("relative", "name"), sorted(_EXEMPT_FUNCTIONS))
def test_every_exemption_names_an_unfinalized_writer(relative, name):
    """A stale exemption silently covers whatever unfinalized save lands in that function next."""
    assert (name, False) in _config_writers(REPO_ROOT / relative), (
        f"{relative}:{name} is exempt but is no unfinalized config writer — drop the exemption"
    )


def test_the_roster_is_not_vacuous():
    writers = {(str(p.relative_to(REPO_ROOT)), name) for p in _python_files() for name, _ in _config_writers(p)}
    assert ("halo/checkpoint/tools/offline_export.py", "save_full_checkpoint") in writers
    assert ("halo/checkpoint/tools/vision_reattach.py", "reattach_vision_tower") in writers


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
