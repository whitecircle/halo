#!/usr/bin/env python
"""Conventions a GPU test script must honour, pinned so a new script cannot quietly break them.

Both are about scratch: a GPU test allocates its dirs through the launcher and hands them back.

``tests.common.distributed.setup_cache_dirs`` hands back two ``tempfile.mkdtemp`` dirs — an
output dir and an HF datasets cache. Nothing reclaims them on its own, so a test that allocates
and never frees leaks two directories per rank per run; across a nightly suite that is thousands
of stale trees on the scratch volume, and the HF cache half is not small.

Four spellings discharge the obligation, and the scan accepts any of them: a ``cleanup_dirs``
call, the ``gpu_test_main`` harness (whose ``finally`` calls ``cleanup_dirs`` for the body),
``ctx.on_teardown(...)``, or a ``finally:`` that ``shutil.rmtree``s. Anything else is a leak.

AST, not grep: the check is whether the call is really made, so a mention inside a docstring,
a comment or a disabled code path must not satisfy it.

The second half is the path itself. ``tests/gpu/conftest.py`` points ``TMPDIR`` at a per-run dir
under pytest's basetemp and ``tests/common/distributed.py`` allocates through it, so a literal
``/mnt/...`` in a test escapes basetemp, survives the run, and assumes a volume layout the rulebook
does not guarantee on this host.

Run: ``pytest -m cpu tests/cpu/conventions/test_gpu_harness_conventions.py``
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.common.utils import REPO_ROOT, imports_name

_GPU_ROOT = Path(REPO_ROOT) / "tests" / "gpu"

_ALLOCATOR = "setup_cache_dirs"
_RECLAIMERS = ("cleanup_dirs", "on_teardown")
_HARNESS = "gpu_test_main"
_RMTREE = "rmtree"
_FORBIDDEN_PATH_PREFIX = "/mnt/"


def _called_names(tree: ast.AST) -> set[str]:
    """Every name invoked as a call in ``tree`` — ``f()`` and ``obj.f()`` alike.

    The attribute form is folded in under its bare attribute so ``ctx.on_teardown(...)`` and
    ``shutil.rmtree(...)`` are found without pinning the receiver, which tests spell differently.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def _allocator_linenos(tree: ast.AST) -> list[int]:
    """Line of every ``setup_cache_dirs`` call, so an offender points at its own allocation."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _ALLOCATOR
    ]


def _has_finally_rmtree(tree: ast.AST) -> bool:
    """Whether some ``try`` block's ``finally`` reclaims a tree.

    Only ``finalbody`` counts: an ``rmtree`` on the success path alone leaks whenever the body
    raises, which is exactly the run that leaves the dirs behind.
    """
    return any(
        isinstance(node, ast.Try) and node.finalbody and any(_RMTREE in _called_names(stmt) for stmt in node.finalbody)
        for node in ast.walk(tree)
    )


def _reclaims_its_dirs(path: Path, tree: ast.AST) -> bool:
    """Whether ``path`` discharges the cleanup obligation by any of the four accepted spellings."""
    called = _called_names(tree)
    return any(name in called for name in _RECLAIMERS) or imports_name(path, _HARNESS) or _has_finally_rmtree(tree)


def _allocating_scripts() -> list[tuple[Path, ast.AST]]:
    """Every GPU test script that calls ``setup_cache_dirs``, parsed."""
    found = []
    for script in sorted(_GPU_ROOT.rglob("test_*.py")):
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        if _ALLOCATOR in _called_names(tree):
            found.append((script, tree))
    return found


def test_every_gpu_test_that_allocates_cache_dirs_also_reclaims_them():
    """``setup_cache_dirs`` without a matching cleanup leaks two temp dirs per rank per run."""
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{_allocator_linenos(tree)[0]}"
        for path, tree in _allocating_scripts()
        if not _reclaims_its_dirs(path, tree)
    ]
    assert not offenders, (
        "these GPU tests call setup_cache_dirs and never free the dirs — add cleanup_dirs(output_dir, "
        "cache_dir) in a finally: block, or move the test onto the gpu_test_main harness, which owns "
        "the whole lifecycle:\n  " + "\n  ".join(offenders)
    )


def test_the_scan_is_not_vacuous(tmp_path):
    """Guard the guard: the sweep must parse the suite and actually catch a leaker.

    A count of allocating scripts is the wrong pin — every migration onto ``gpu_test_main`` legally
    lowers it. What must hold is that the sweep reads the whole tree and that its detector still
    fires: a script calling the allocator with no reclaim is an offender, and the same script with a
    ``cleanup_dirs`` in a ``finally`` is not.
    """
    assert len(list(_GPU_ROOT.rglob("test_*.py"))) > 100, "the sweep lost the tests/gpu tree"

    leaker = tmp_path / "test_leaker.py"
    leaker.write_text("def main():\n    out, cache = setup_cache_dirs('x', 0)\n", encoding="utf-8")
    leaker_tree = ast.parse(leaker.read_text(encoding="utf-8"))
    assert _ALLOCATOR in _called_names(leaker_tree), f"the sweep no longer sees a {_ALLOCATOR!r} call"
    assert not _reclaims_its_dirs(leaker, leaker_tree), "a script that never frees its dirs must be an offender"

    reclaimer = tmp_path / "test_reclaimer.py"
    reclaimer.write_text(
        "def main():\n"
        "    out, cache = setup_cache_dirs('x', 0)\n"
        "    try:\n        pass\n    finally:\n        cleanup_dirs(out, cache)\n",
        encoding="utf-8",
    )
    reclaimer_tree = ast.parse(reclaimer.read_text(encoding="utf-8"))
    assert _reclaims_its_dirs(reclaimer, reclaimer_tree), "a cleanup_dirs call must discharge the obligation"


def test_no_gpu_test_names_a_host_scratch_path():
    """Scratch comes from the launcher's ``TMPDIR``, checkpoint locations from a shared constant."""
    offenders = []
    for script in sorted(_GPU_ROOT.rglob("*.py")):
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        offenders += [
            f"{script.relative_to(REPO_ROOT)}:{node.lineno}: {node.value!r}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith(_FORBIDDEN_PATH_PREFIX)
        ]
    assert not offenders, (
        "hardcoded host paths in tests/gpu — use tests.common.distributed.shared_scratch_dir / "
        "setup_cache_dirs for scratch, and a constant in tests/common/models.py for a checkpoint "
        "location:\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
