#!/usr/bin/env python
"""The persistent Triton cache anchor: ``~/.triton`` is ephemeral in a ``--rm`` container, re-benchmarking fla's
autotuners per sequence length. It falls back where ``HF_HOME`` is read-only."""

from __future__ import annotations

import ast
import inspect
import logging
import os
import tempfile

import pytest

from halo.entrypoints.run_setup import configure_training_run
from halo.kernels import jit_cache
from halo.kernels.jit_cache import anchor_jit_cache_dir

_TRITON_VAR = "TRITON_CACHE_DIR"
_TRITON_SUBDIR = "triton_cache"


@pytest.fixture
def temp_root(monkeypatch, tmp_path):
    """``tempfile.gettempdir`` caches, so force a re-read."""
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setenv("TMPDIR", str(root))
    monkeypatch.setattr(tempfile, "tempdir", None)
    # Set first so monkeypatch also removes what the anchor writes.
    monkeypatch.setenv(_TRITON_VAR, "")
    monkeypatch.delenv(_TRITON_VAR)
    return root


def _deny_writes_under(monkeypatch, root) -> None:
    """At the access check, since the image runs tests as root, which permission bits do not stop."""
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
    with caplog.at_level(logging.WARNING, logger=jit_cache.__name__):
        anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == os.path.join(str(temp_root), _TRITON_SUBDIR), (
        "a read-only HF_HOME must not receive the cache: the first compile would fail with EROFS"
    )
    assert any(_TRITON_VAR in record.getMessage() for record in caplog.records), "the fallback must be reported"


def test_no_writable_location_leaves_the_library_default(monkeypatch, temp_root, tmp_path, caplog):
    """A temp dir another user owns is no fallback."""
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    _deny_writes_under(monkeypatch, hf_home)
    _deny_writes_under(monkeypatch, temp_root)
    with caplog.at_level(logging.WARNING, logger=jit_cache.__name__):
        anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert _TRITON_VAR not in os.environ, "an unwritable fallback was set, so the first compile would still fail"
    assert any(_TRITON_VAR in record.getMessage() for record in caplog.records), "leaving it unset must be reported"


def test_a_host_with_no_temp_dir_still_anchors_on_hf_home_and_otherwise_leaves_the_default(
    monkeypatch, temp_root, tmp_path, caplog
):
    """``tempfile.gettempdir`` raises when no candidate is usable, which must not cost a writable ``HF_HOME``."""

    def no_temp_dir():
        raise FileNotFoundError("No usable temporary directory found")

    monkeypatch.setattr(tempfile, "gettempdir", no_temp_dir)
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == os.path.join(str(hf_home), _TRITON_SUBDIR)
    monkeypatch.delenv(_TRITON_VAR)
    _deny_writes_under(monkeypatch, hf_home)
    with caplog.at_level(logging.WARNING, logger=jit_cache.__name__):
        anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert _TRITON_VAR not in os.environ
    assert any("no usable temp dir" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("kind", ["file", "dangling link"])
def test_a_non_directory_at_the_cache_path_falls_back(monkeypatch, temp_root, tmp_path, kind):
    """Triton cannot create its cache under a file or a broken link."""
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    target = hf_home / _TRITON_SUBDIR
    if kind == "file":
        target.write_text("")
        target.chmod(0o755)
    else:
        target.symlink_to(tmp_path / "missing")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == os.path.join(str(temp_root), _TRITON_SUBDIR)


def test_read_only_cache_dir_under_a_writable_hf_home_falls_back(monkeypatch, temp_root, tmp_path):
    """Judged by itself, not by its parent."""
    hf_home = tmp_path / "hf"
    (hf_home / _TRITON_SUBDIR).mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    _deny_writes_under(monkeypatch, hf_home / _TRITON_SUBDIR)
    anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == os.path.join(str(temp_root), _TRITON_SUBDIR)


def test_hf_home_not_created_yet_keeps_the_anchor(monkeypatch, temp_root, tmp_path):
    """Triton creates its own directory."""
    hf_home = tmp_path / "fresh" / "hf"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    anchor_jit_cache_dir(_TRITON_VAR, _TRITON_SUBDIR)
    assert os.environ[_TRITON_VAR] == os.path.join(str(hf_home), _TRITON_SUBDIR)


def test_the_run_setup_anchors_it_before_anything_can_compile():
    """Triton reads the variable at first compile, so a setup that stops anchoring fails nothing and costs a
    re-benchmark per run."""
    tree = ast.parse(inspect.getsource(configure_training_run))
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
    assert _TRITON_VAR in anchored, f"configure_training_run anchors {sorted(anchored)}, not {_TRITON_VAR}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
