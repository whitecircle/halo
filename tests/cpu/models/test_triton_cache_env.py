#!/usr/bin/env python
"""CPU tests for the persistent Triton cache anchor set at training setup.

``TRITON_CACHE_DIR`` defaults to ``~/.triton`` — ephemeral inside a ``--rm`` container — and fla's
autotuners persist shape-keyed measured configs there, so losing it re-benchmarks kernels per fresh
sequence length on every run (a per-rank straggler stall, tens of seconds each). The anchor must
derive from ``HF_HOME`` exactly like the FA4 kernel cache, must never override an explicit operator
choice, and must actually be applied by the run's setup — an anchor nothing calls is no anchor.

    pytest -m cpu tests/cpu/models/test_triton_cache_env.py
"""

from __future__ import annotations

import ast
import inspect
import os

import pytest

from src.models.patches.attention import anchor_jit_cache_dir
from src.training.environment import setup_training_environment

_TRITON_VAR = "TRITON_CACHE_DIR"
_TRITON_SUBDIR = "triton_cache"


def test_derives_from_hf_home(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/data/hf")
    monkeypatch.delenv(_TRITON_VAR, raising=False)
    anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == os.path.join("/data/hf", _TRITON_SUBDIR), (
        "the Triton cache must land on the same volume as every other kernel cache"
    )


def test_explicit_setting_is_respected(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/data/hf")
    monkeypatch.setenv(_TRITON_VAR, "/elsewhere/triton")
    anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == "/elsewhere/triton", "an operator's explicit choice was overridden"


def test_falls_back_to_tempdir_without_hf_home(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv(_TRITON_VAR, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    import tempfile

    tempfile.tempdir = None  # gettempdir caches; force a re-read of TMPDIR
    try:
        anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
        assert os.environ[_TRITON_VAR] == os.path.join(str(tmp_path), _TRITON_SUBDIR)
    finally:
        tempfile.tempdir = None


def test_the_run_setup_anchors_it_before_anything_can_compile():
    """Read off the setup's own source: the anchor has to be applied there, and this is the only
    assertion that survives the call being dropped. Triton reads the variable lazily at first
    compile, so a setup that stops anchoring costs a re-benchmark per run and fails nothing."""
    tree = ast.parse(inspect.getsource(setup_training_environment))
    anchored = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "anchor_jit_cache_dir"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert _TRITON_VAR in anchored, f"setup_training_environment anchors {sorted(anchored)}, not {_TRITON_VAR}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
