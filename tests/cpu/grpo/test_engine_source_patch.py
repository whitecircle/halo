#!/usr/bin/env python
"""``docker/engine_source_patch.py`` rewrites exactly what its table names and refuses anything else;
the vLLM table's residual harmony-gate scan refuses a gate the table missed.

    python tests/cpu/grpo/test_engine_source_patch.py
"""

import importlib.util
import pathlib
import py_compile
import sys

import pytest

from tests.common.repo import load_script_module

# Registered so the table's own ``from engine_source_patch import ...`` resolves, as in the image.
patcher = load_script_module("docker/engine_source_patch.py", register=True)
vllm_table = load_script_module("docker/vllm/patches/patch_vllm_source.py")

VALUE = patcher.Edit("mod.py", 'VALUE = "old"', 'VALUE = "new"', "module value")


def _module(root, text):
    path = root / "mod.py"
    path.write_bytes(text.encode("utf-8"))
    return path


def test_an_edit_rewrites_byte_for_byte_and_replaces_only_its_modules_bytecode(tmp_path):
    path = _module(tmp_path, 'VALUE = "old"\r\nOTHER = 1\r\n')
    (tmp_path / "__pycache__").mkdir()
    stale = tmp_path / "__pycache__/mod.cpython-312.opt-1.pyc"
    neighbour = tmp_path / "__pycache__/mod_other.cpython-312.pyc"
    stale.write_bytes(b"stale")
    neighbour.write_bytes(b"unpatched")
    patcher.apply_edits(tmp_path, [VALUE])
    assert path.read_bytes() == b'VALUE = "new"\r\nOTHER = 1\r\n'
    assert not stale.exists() and neighbour.read_bytes() == b"unpatched"
    assert pathlib.Path(importlib.util.cache_from_source(str(path))).is_file()


@pytest.mark.parametrize(
    "text, error",
    [
        (None, "upstream moved the file"),
        ("OTHER = 1\n", "module value: expected the pre-image exactly once, found 0 — upstream changed"),
        ('VALUE = "old"\nVALUE = "old"\n', "found 2"),
    ],
)
def test_a_missing_file_or_a_missing_or_doubled_pre_image_fails(tmp_path, text, error):
    path = _module(tmp_path, text) if text is not None else None
    with pytest.raises(SystemExit, match=error):
        patcher.apply_edits(tmp_path, [VALUE])
    assert path is None or path.read_text() == text


def test_a_re_run_fails_before_writing_so_an_insertion_is_not_doubled(tmp_path):
    anchor = "def f():\n    pass\n"
    insertion = patcher.Edit("mod.py", anchor, anchor + "\n\ndef g():\n    pass\n", "insertion")
    first = _module(tmp_path, anchor)
    (tmp_path / "later.py").write_text('VALUE = "old"\n')
    table = [insertion, patcher.Edit("later.py", 'VALUE = "old"', 'VALUE = "new"', "later value")]
    patcher.apply_edits(tmp_path, table)
    patched = first.read_bytes()
    with pytest.raises(SystemExit, match="later value: expected the pre-image exactly once, found 0"):
        patcher.apply_edits(tmp_path, table)
    assert first.read_bytes() == patched


def test_a_later_edit_that_undoes_an_earlier_one_fails_the_post_image_check(tmp_path):
    _module(tmp_path, 'VALUE = "old"\n')
    undo = patcher.Edit("mod.py", 'VALUE = "new"', 'VALUE = "newer"', "undo")
    with pytest.raises(SystemExit, match="module value: the post-image is not in place"):
        patcher.apply_edits(tmp_path, [VALUE, undo])


def test_verify_refuses_an_unpatched_file_and_accepts_an_insertion(tmp_path):
    anchor = "def f():\n    pass\n"
    insertion = patcher.Edit("mod.py", anchor, anchor + "\n\ndef g():\n    pass\n", "insertion")
    path = _module(tmp_path, anchor)
    with pytest.raises(SystemExit, match="insertion: the post-image is not in place"):
        patcher.verify_edits(tmp_path, [insertion])
    patcher.apply_edits(tmp_path, [insertion])
    patcher.verify_edits(tmp_path, [insertion])
    path.write_text(path.read_text() + "\n\n" + anchor)
    with pytest.raises(SystemExit, match="insertion: the post-image is not in place"):
        patcher.verify_edits(tmp_path, [insertion])


def test_an_edit_that_breaks_the_syntax_fails_the_build(tmp_path):
    _module(tmp_path, 'VALUE = "old"\n')
    with pytest.raises(py_compile.PyCompileError):
        patcher.apply_edits(tmp_path, [patcher.Edit("mod.py", 'VALUE = "old"', 'VALUE = "old', "broken")])


def test_main_verify_mode_checks_without_rewriting(tmp_path, monkeypatch):
    (tmp_path / "halo_scratch_engine").mkdir()
    (tmp_path / "halo_scratch_engine/__init__.py").write_text("")
    path = _module(tmp_path / "halo_scratch_engine", 'VALUE = "old"\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    checked = []
    monkeypatch.setattr(sys, "argv", ["patch", "--verify"])
    with pytest.raises(SystemExit, match="post-image is not in place"):
        patcher.main("halo_scratch_engine", [VALUE], checked.append)
    assert path.read_text() == 'VALUE = "old"\n' and not checked
    monkeypatch.setattr(sys, "argv", ["patch"])
    patcher.main("halo_scratch_engine", [VALUE], checked.append)
    assert path.read_text() == 'VALUE = "new"\n' and checked == [path.parent]


@pytest.mark.parametrize(
    "gate",
    ['self.use_harmony = self.model_config.hf_config.model_type == "gpt_oss"', 'is_harmony=t in ("gpt_oss",),'],
)
def test_the_vllm_table_refuses_a_harmony_gate_it_left_behind(tmp_path, gate):
    (tmp_path / "patched.py").write_text('self.use_harmony = False  # model_type == "gpt_oss"\n')
    vllm_table.reject_harmony_gates(tmp_path)
    (tmp_path / "moved.py").write_text(f"        {gate}\n")
    with pytest.raises(SystemExit, match="moved.py"):
        vllm_table.reject_harmony_gates(tmp_path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
