#!/usr/bin/env python
"""Conventions a GPU test script must honour, pinned so a new script cannot quietly break them.

Every manifest script runs under ``gpu_test_main`` unless its lifecycle is one the harness cannot
express, named in ``_OWN_LIFECYCLE`` with the reason. Every row launches under its own node id, with only
markers from ``ALL_MARKERS``: ``gpu``, one tier and the GPU-count tag of its own ``nproc``.

The rest are about scratch: a GPU test allocates its dirs through the launcher and hands them back.

``tests.common.distributed.setup_cache_dirs`` hands back two ``tempfile.mkdtemp`` dirs — an
output dir and an HF datasets cache. Nothing reclaims them on its own, so a test that allocates
and never frees leaks two directories per rank per run; across a full GPU suite run that is thousands
of stale trees on the scratch volume, and the HF cache half is not small.

Four spellings discharge the obligation, and the scan accepts any of them: a ``cleanup_dirs``
call, a ``gpu_test_main`` call (the harness's ``finally`` calls ``cleanup_dirs`` for the body),
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
from collections import Counter
from pathlib import Path

import pytest

from tests.common.repo import REPO_ROOT
from tests.gpu.manifest import ALL_MARKERS, MANIFEST, Row, TestSpec, script_path

_GPU_ROOT = Path(REPO_ROOT) / "tests" / "gpu"

_ALLOCATOR = "setup_cache_dirs"
_RECLAIMERS = ("cleanup_dirs", "on_teardown")
_HARNESS = "gpu_test_main"
_RMTREE = "rmtree"
_FORBIDDEN_PATH_PREFIX = "/mnt/"
_GPU_COUNT_MARKERS = frozenset(marker for marker in ALL_MARKERS if marker[0].isdigit() and marker.endswith("gpu"))
# Manifest scripts that keep their own lifecycle, each for a reason the harness cannot express.
_OWN_LIFECYCLE = {
    "kernels/test_deepgemm.py": "prints SKIP: when deep_gemm is absent; the harness has no skip channel",
}


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


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


def _calls_the_harness(tree: ast.AST) -> bool:
    """Whether ``tree`` calls ``gpu_test_main``, by its imported name or through its module
    (``harness.gpu_test_main(...)``). The entry is a decorator factory, so every use is a call."""
    return _HARNESS in _called_names(tree)


def _reclaims_its_dirs(tree: ast.AST) -> bool:
    """Whether ``tree`` discharges the cleanup obligation by any of the four accepted spellings."""
    called = _called_names(tree)
    return any(name in called for name in _RECLAIMERS) or _calls_the_harness(tree) or _has_finally_rmtree(tree)


def _allocating_scripts() -> list[tuple[Path, ast.AST]]:
    """Every GPU test script that calls ``setup_cache_dirs``, parsed."""
    found = []
    for script in sorted(_GPU_ROOT.rglob("test_*.py")):
        tree = _parse(script)
        if _ALLOCATOR in _called_names(tree):
            found.append((script, tree))
    return found


def test_every_gpu_test_that_allocates_cache_dirs_also_reclaims_them():
    """``setup_cache_dirs`` without a matching cleanup leaks two temp dirs per rank per run."""
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{_allocator_linenos(tree)[0]}"
        for path, tree in _allocating_scripts()
        if not _reclaims_its_dirs(tree)
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
    leaker_tree = _parse(leaker)
    assert _ALLOCATOR in _called_names(leaker_tree), f"the sweep no longer sees a {_ALLOCATOR!r} call"
    assert not _reclaims_its_dirs(leaker_tree), "a script that never frees its dirs must be an offender"

    reclaimer = tmp_path / "test_reclaimer.py"
    reclaimer.write_text(
        "def main():\n"
        "    out, cache = setup_cache_dirs('x', 0)\n"
        "    try:\n        pass\n    finally:\n        cleanup_dirs(out, cache)\n",
        encoding="utf-8",
    )
    reclaimer_tree = _parse(reclaimer)
    assert _reclaims_its_dirs(reclaimer_tree), "a cleanup_dirs call must discharge the obligation"


def _runs_under_the_harness(path: Path) -> bool:
    """Whether the script calls ``gpu_test_main``, or re-launches another suite's entry that does."""
    tree = _parse(path)
    return _calls_the_harness(tree) or any(
        isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("tests.gpu.")
        and _calls_the_harness(_parse(REPO_ROOT / (node.module.replace(".", "/") + ".py")))
        for node in ast.walk(tree)
    )


def test_every_manifest_script_runs_under_the_harness():
    """A hand-rolled lifecycle hides a non-zero rank's error, blocks teardown until the NCCL watchdog
    on a single-rank failure, and emits no result line, so the launcher cannot tell FAIL from ERROR."""
    offenders = sorted(
        rel for rel in MANIFEST if rel not in _OWN_LIFECYCLE and not _runs_under_the_harness(script_path(rel))
    )
    assert not offenders, (
        "move these onto tests.common.harness.gpu_test_main (or, for a lifecycle the harness cannot "
        "express, add an _OWN_LIFECYCLE entry naming why):\n  " + "\n  ".join(offenders)
    )


def _launches() -> list[tuple[str, str, TestSpec]]:
    """``(script, args, spec)`` for every manifest row, the spec carrying that row's own launch values."""
    return [(rel, args, spec) for rel, script_spec in MANIFEST.items() for args, spec in script_spec.launches()]


def test_every_manifest_marker_is_registered():
    """``tests/conftest.py`` registers ``ALL_MARKERS`` and nothing else, and pytest only warns on an
    unregistered mark, so a typo would silently drop its row from every selection by that marker."""
    assert MANIFEST, "the manifest is empty, so the marker scan checks nothing"
    unregistered = sorted(
        f"{rel} [{args}]: {marker}"
        for rel, args, spec in _launches()
        for marker in spec.markers
        if marker not in ALL_MARKERS
    )
    assert not unregistered, (
        "add these markers to ALL_MARKERS in tests/gpu/manifest.py, or fix the typo:\n  " + "\n  ".join(unregistered)
    )


def _misfit(spec: TestSpec) -> bool:
    """Whether a launch lacks ``gpu``, exactly one tier, or the ``Ngpu`` tag of its own ``nproc``."""
    markers = set(spec.markers)
    return (
        "gpu" not in markers
        or len({"core", "full"} & markers) != 1
        or markers & _GPU_COUNT_MARKERS != {f"{spec.nproc}gpu"}
    )


def test_every_row_launches_under_its_own_id_tier_and_width():
    """Rows of one script share its node-id prefix, so two with the same args would collapse into one pytest
    id; and a row's own ``nproc`` or tier must keep its markers selecting it, since ``-m "gpu and core"`` and
    the ``Ngpu`` tags are how the tiers and the GPU budget are chosen."""
    launches = _launches()
    duplicated = sorted(node for node, count in Counter((rel, args) for rel, args, _ in launches).items() if count > 1)
    assert not duplicated, f"rows sharing a node id: {duplicated}"
    misfit = sorted(
        f"{rel} [{args}]: nproc={spec.nproc} {spec.markers}" for rel, args, spec in launches if _misfit(spec)
    )
    assert not misfit, (
        "each row needs 'gpu', exactly one of 'core'/'full' and the Ngpu tag of its nproc:\n  " + "\n  ".join(misfit)
    )


def test_a_row_launches_with_its_own_values_over_the_spec():
    """A row's set fields replace the spec's and the rest carry over, so a wider row that keeps the spec's
    markers is caught by the width check above."""
    spec = TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), args_matrix=("--a", Row("--b", nproc=4)), timeout=60)
    (a, a_spec), (b, b_spec) = spec.launches()
    assert (a, a_spec.nproc, a_spec.markers, a_spec.timeout) == ("--a", 2, ("gpu", "core", "2gpu"), 60)
    assert (b, b_spec.nproc, b_spec.markers, b_spec.timeout) == ("--b", 4, ("gpu", "core", "2gpu"), 60)
    assert not _misfit(a_spec) and _misfit(b_spec)


def test_a_module_qualified_harness_call_counts(tmp_path):
    """``harness.gpu_test_main(...)`` is the same entry as the imported name, so neither pin may flag
    it, while importing the harness module without calling the entry still runs no lifecycle."""
    qualified = tmp_path / "test_qualified.py"
    qualified.write_text(
        "from tests.common import harness\n\n\n"
        "@harness.gpu_test_main(min_world_size=2)\n"
        "def run(ctx):\n    out, cache = setup_cache_dirs('x', 0)\n",
        encoding="utf-8",
    )
    assert _runs_under_the_harness(qualified), "a module-qualified gpu_test_main call must count as the harness"
    assert _reclaims_its_dirs(_parse(qualified)), "a module-qualified gpu_test_main call must reclaim the dirs"

    uncalled = tmp_path / "test_uncalled.py"
    uncalled.write_text(
        "from tests.common import harness\n\n\ndef run():\n    out, cache = setup_cache_dirs('x', 0)\n",
        encoding="utf-8",
    )
    assert not _runs_under_the_harness(uncalled), "an import alone must not count as running under the harness"
    assert not _reclaims_its_dirs(_parse(uncalled)), "an import alone must not discharge the cleanup obligation"


def test_own_lifecycle_exemptions_are_live():
    """An exemption that outlives its reason blesses a script nothing checks."""
    for rel in _OWN_LIFECYCLE:
        assert rel in MANIFEST, f"stale exemption: {rel} is not a manifest script"
        assert not _runs_under_the_harness(script_path(rel)), f"{rel} now runs under the harness — drop its exemption"


def test_no_gpu_test_names_a_host_scratch_path():
    """Scratch comes from the launcher's ``TMPDIR``, checkpoint locations from a shared constant."""
    offenders = []
    for script in sorted(_GPU_ROOT.rglob("*.py")):
        tree = _parse(script)
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
