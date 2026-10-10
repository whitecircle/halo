#!/usr/bin/env python
"""CPU tests for the persistent Triton cache anchor set at training setup.

``TRITON_CACHE_DIR`` defaults to ``~/.triton`` — ephemeral inside a ``--rm`` container — and fla's
autotuners persist shape-keyed measured configs there, so losing it re-benchmarks kernels per fresh
sequence length on every run (a per-rank straggler stall, tens of seconds each). The anchor must
derive from ``HF_HOME`` exactly like the FA4 kernel cache, must never override an explicit operator
choice, must actually be applied by the run's setup — an anchor nothing calls is no anchor — and must
not point at a location it cannot write: a shared ``HF_HOME`` mounted read-only fails the first
compile with EROFS, so that case falls back to the temp dir and says so.

    pytest -m cpu tests/cpu/models/test_triton_cache_env.py
"""

from __future__ import annotations

import ast
import inspect
import logging
import os
import tempfile

import pytest

from src.models.patches import attention as attention_mod
from src.models.patches.attention import anchor_jit_cache_dir
from src.training.environment import setup_training_environment

_TRITON_VAR = "TRITON_CACHE_DIR"
_TRITON_SUBDIR = "triton_cache"


@pytest.fixture
def temp_root(monkeypatch, tmp_path):
    """A fresh temp dir that ``tempfile.gettempdir`` resolves to (it caches, so force a re-read)."""
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setenv("TMPDIR", str(root))
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.delenv(_TRITON_VAR, raising=False)
    return root


def _deny_writes_under(monkeypatch, root) -> None:
    """Make every path under ``root`` read as unwritable, as on a read-only mount.

    Simulated at the access check because the image runs tests as root, which permission bits do
    not stop; a read-only filesystem stops root too.
    """
    real_access = os.access
    root = os.path.abspath(root)

    def access(path, mode, *args, **kwargs):
        target = os.path.abspath(path)
        if mode & os.W_OK and (target == root or target.startswith(root + os.sep)):
            return False
        return real_access(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "access", access)


def test_derives_from_hf_home(monkeypatch, temp_root, tmp_path):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == os.path.join(str(hf_home), _TRITON_SUBDIR), (
        "the Triton cache must land on the same volume as every other kernel cache"
    )


def test_explicit_setting_is_respected(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/data/hf")
    monkeypatch.setenv(_TRITON_VAR, "/elsewhere/triton")
    anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == "/elsewhere/triton", "an operator's explicit choice was overridden"


def test_falls_back_to_tempdir_without_hf_home(monkeypatch, temp_root):
    monkeypatch.delenv("HF_HOME", raising=False)
    anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == os.path.join(str(temp_root), _TRITON_SUBDIR)


def test_read_only_hf_home_falls_back_to_tempdir_with_a_warning(monkeypatch, temp_root, tmp_path, caplog):
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    _deny_writes_under(monkeypatch, hf_home)
    with caplog.at_level(logging.WARNING, logger=attention_mod.__name__):
        anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == os.path.join(str(temp_root), _TRITON_SUBDIR), (
        "a read-only HF_HOME must not receive the cache: the first compile would fail with EROFS"
    )
    assert any(_TRITON_VAR in record.getMessage() for record in caplog.records), (
        "the fallback costs a recompile in every new container and must be reported"
    )


def test_read_only_cache_dir_under_a_writable_hf_home_falls_back(monkeypatch, temp_root, tmp_path):
    """A cache directory mounted read-only on its own is judged by itself, not by its parent."""
    hf_home = tmp_path / "hf"
    (hf_home / _TRITON_SUBDIR).mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    _deny_writes_under(monkeypatch, hf_home / _TRITON_SUBDIR)
    anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == os.path.join(str(temp_root), _TRITON_SUBDIR)


def test_hf_home_not_created_yet_keeps_the_anchor(monkeypatch, temp_root, tmp_path):
    """Triton creates its own directory, so a writable volume whose HF_HOME is not made yet keeps it."""
    hf_home = tmp_path / "fresh" / "hf"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == os.path.join(str(hf_home), _TRITON_SUBDIR)


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
