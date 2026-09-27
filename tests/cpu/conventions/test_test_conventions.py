#!/usr/bin/env python
"""Drift pins: one mechanism each for collection, tier marking, and import resolution.

* No test file may hand-roll pytest collection. Every suite runs under pytest, and a standalone run
  goes through ``pytest.main`` from the module's ``__main__``. A runner class that collects by hand —
  a literal list of ``(label, fn)`` pairs, or a ``globals()`` sweep — drops any test missing from its
  list, and reports failures only as a printed summary pytest never parses.
  ``tests.common.harness.record_check(checks, name, fn)`` is not that — the banned mechanism is the
  printed summary, not recording a verdict the harness then returns.
* A collection of the CPU tier from the repo root never imports a GPU script: everything under
  ``tests/gpu/`` except the two launcher entry points is a torchrun script, and one that acts at
  import takes the whole session down.
* No CPU test file re-declares the ``cpu`` marker: ``tests/conftest.py`` applies it by path to
  everything under ``tests/cpu/``, so a per-file ``pytestmark`` is a second mechanism for the same
  selection that only rots when the collector's rule changes.
* No test file bootstraps ``sys.path`` to the repo root: the images bake ``PYTHONPATH=/workspace``
  and ``tests/conftest.py`` covers a pytest run, so the insert is dead weight. A ``scripts/`` entry
  point is loaded through ``tests.common.utils.load_script_module`` instead of a path insert.

The marker and ``sys.path`` pins read the AST, so every spelling of the binding or the call is caught
and a mention in a docstring or a string is not. The two text-matched spellings are assembled from
fragments below so these pins do not match themselves and no path has to be exempted.

Run: python tests/cpu/conventions/test_test_conventions.py
"""

import ast
import functools
import pathlib
import textwrap

import pytest

from tests.common.utils import REPO_ROOT, probe_findings

CPU_MAIN_ENTRY = 'if __name__ == "__main__":\n    raise SystemExit(pytest.main([__file__, "-v"]))'
BANNED_RUNNER = "Test" + "Runner"
BANNED_SUMMARY = "print_" + "summary("
# The ``sys.path`` methods that put a directory on the import path.
SYS_PATH_GROWERS = {"insert", "append", "extend"}
# The two legitimate inserts: the root conftest owns the pytest-run path, and one test writes a
# module into `tmp_path` and imports it back.
BOOTSTRAP_EXEMPT = {"tests/conftest.py", "tests/cpu/checkpoint/test_parallel_config_save.py"}
# The profiling benchmarks are not suites: they report a perf table to stdout by design, and nothing
# selects them by verdict.
SUMMARY_EXEMPT = {"tests/gpu/profiling/benchmark_collators.py", "tests/gpu/profiling/benchmark_torch_compile.py"}


# Modules a CPU-tier collection may load from tests/gpu/: the launcher side, never a torchrun script.
GPU_LAUNCHER_MODULES = {"__init__.py", "conftest.py", "manifest.py", "test_suite.py", "test_launcher_contract.py"}
COLLECTION_MARKER = "GPU_SCRIPTS_IMPORTED:"


def _test_files():
    return sorted((REPO_ROOT / "tests").rglob("*.py"))


@functools.cache
def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _assignment_targets(node: ast.AST) -> list[ast.expr]:
    """What a plain, annotated or augmented assignment binds; nothing for any other node."""
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, ast.AnnAssign | ast.AugAssign):
        return [node.target]
    return []


def _names_the_cpu_marker(node: ast.AST) -> bool:
    """``pytest.mark.cpu``, or ``mark.cpu`` after ``from pytest import mark``."""
    if not (isinstance(node, ast.Attribute) and node.attr == "cpu"):
        return False
    owner = node.value
    return (isinstance(owner, ast.Attribute) and owner.attr == "mark") or (
        isinstance(owner, ast.Name) and owner.id == "mark"
    )


def _redeclared_cpu_marker_lines(tree: ast.Module) -> list[int]:
    """Lines binding ``pytestmark`` to a value that carries the ``cpu`` marker, bare or among other marks."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in _assignment_targets(node))
        and node.value is not None
        and any(_names_the_cpu_marker(part) for part in ast.walk(node.value))
    ]


def _is_sys_path(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "path"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


def _sys_path_bootstrap_lines(tree: ast.Module) -> list[int]:
    """Lines that grow ``sys.path``: a :data:`SYS_PATH_GROWERS` call on it, or an assignment to it or to a
    slice of it. Reading it or removing an entry is not a bootstrap."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in SYS_PATH_GROWERS and _is_sys_path(node.func.value):
                lines.append(node.lineno)
        elif any(
            _is_sys_path(target.value if isinstance(target, ast.Subscript) else target)
            for target in _assignment_targets(node)
        ):
            lines.append(node.lineno)
    return lines


def _hand_listed_test_functions(tree: ast.Module) -> list[int]:
    """Lines of literal collections (list, tuple, set, dict values) naming two or more of the module's
    own ``test*`` functions, bare or inside ``(label, fn)`` pairs: the shape of a hand-rolled runner."""
    tests = {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test")}
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.List | ast.Tuple | ast.Set):
            elements = node.elts
        elif isinstance(node, ast.Dict):
            elements = node.values
        else:
            continue
        named = set()
        for element in elements:
            parts = element.elts if isinstance(element, ast.Tuple | ast.List) else [element]
            named.update(part.id for part in parts if isinstance(part, ast.Name) and part.id in tests)
        if len(named) >= 2:
            lines.append(node.lineno)
    return lines


def test_every_cpu_test_file_ends_in_its_pytest_main_entry():
    """A CPU test file runs standalone through ``pytest.main`` from its ``__main__``, spelled one way
    and placed last: a file without it drops out of the documented ``python tests/cpu/...``
    invocation, a bare ``pytest.main(...)`` exits 0 on failure, and code after the block is skipped
    by the standalone run it serves."""
    offenders = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in _test_files()
        if path.name.startswith("test_")
        and "tests/cpu/" in path.as_posix()
        and not path.read_text(encoding="utf-8").rstrip().endswith(CPU_MAIN_ENTRY)
    )
    assert not offenders, f"CPU test files not ending in\n{CPU_MAIN_ENTRY}\n  " + "\n  ".join(offenders)


def test_no_test_file_reimplements_pytest_collection():
    offenders = sorted(
        str(path.relative_to(REPO_ROOT)) for path in _test_files() if BANNED_RUNNER in path.read_text(encoding="utf-8")
    )
    assert not offenders, (
        f"{BANNED_RUNNER} is retired: a hand-listed runner hides tests it forgets to register, and its "
        f'printed summary is invisible to pytest. Use `if __name__ == "__main__": '
        f'raise SystemExit(pytest.main([__file__, "-v"]))` instead. Offenders:\n  ' + "\n  ".join(offenders)
    )


def test_no_test_file_hand_lists_its_test_functions():
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in _test_files()
        for line in _hand_listed_test_functions(_tree(path))
    ]
    assert not offenders, (
        "a literal collection of test functions is a hand-rolled runner: pytest already collects a CPU "
        "test's functions, and a GPU script records each property with "
        "`tests.common.harness.record_check(checks, name, fn)`. Offenders:\n  " + "\n  ".join(offenders)
    )


def test_the_runner_list_detector_fires_on_the_banned_shapes():
    """Anti-vacuity for the pin above: each banned shape is caught, explicit calls are not."""
    banned = textwrap.dedent(
        """
        def test_a(): ...
        def test_b(): ...
        ALL_TESTS = [("a", test_a), ("b", test_b)]
        for fn in (test_a, test_b):
            fn()
        FNS = {"a": test_a, "b": test_b}
        """
    )
    allowed = textwrap.dedent(
        """
        def test_a(): ...
        def test_b(): ...
        def run(checks):
            record_check(checks, "a", test_a)
            record_check(checks, "b", test_b)
        """
    )
    assert len(_hand_listed_test_functions(ast.parse(banned))) == 3
    assert _hand_listed_test_functions(ast.parse(allowed)) == []


def test_a_cpu_tier_collection_never_imports_a_gpu_script():
    """``pytest -m cpu`` from the repo root walks ``tests/gpu/``; only the launcher side may load there.

    Out of process, since the collection under test is a whole pytest session. Everything is
    deselected by ``-m cpu``, so a clean run ends in ``NO_TESTS_COLLECTED``; anything else (a script
    exiting at import is an ``INTERNALERROR``, a failed import a collection error) is reported.
    """
    script = textwrap.dedent(
        f"""
        import pathlib, sys
        import pytest
        gpu = pathlib.Path("tests/gpu").resolve()
        code = pytest.main(["--collect-only", "-q", "-m", "cpu", "-p", "no:cacheprovider", str(gpu)])
        found = [] if code == pytest.ExitCode.NO_TESTS_COLLECTED else [f"exit {{int(code)}}"]
        for module in list(sys.modules.values()):
            path = pathlib.Path(getattr(module, "__file__", None) or "/").resolve()
            if path.is_relative_to(gpu) and path.name not in {sorted(GPU_LAUNCHER_MODULES)!r}:
                found.append(str(path.relative_to(gpu)))
        print({COLLECTION_MARKER!r} + "|".join(sorted(found)))
        """
    )
    imported = probe_findings(script, COLLECTION_MARKER)
    assert not imported, f"a CPU-tier collection imported GPU scripts or failed: {imported}"


def test_no_test_file_reports_its_result_as_a_printed_summary():
    offenders = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in _test_files()
        if BANNED_SUMMARY in path.read_text(encoding="utf-8")
        and str(path.relative_to(REPO_ROOT)) not in SUMMARY_EXEMPT
    )
    assert not offenders, (
        f"`{BANNED_SUMMARY}` reports pass/fail only to stdout, where pytest and the GPU launcher "
        "cannot tell a FAIL from an ERROR. Record verdicts with tests.common.harness.record_check and "
        "return them as the run's `checks` dict. Offenders:\n  " + "\n  ".join(offenders)
    )


def test_no_cpu_test_redeclares_the_path_applied_marker():
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in _test_files()
        for line in _redeclared_cpu_marker_lines(_tree(path))
    ]
    assert not offenders, (
        "a `pytestmark` carrying `pytest.mark.cpu` duplicates the marker tests/conftest.py already applies to "
        "everything under tests/cpu/. One mechanism only — delete the per-file marker. Offenders:\n  "
        + "\n  ".join(offenders)
    )


def test_no_test_bootstraps_the_repo_root_onto_sys_path():
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in _test_files()
        if str(path.relative_to(REPO_ROOT)) not in BOOTSTRAP_EXEMPT
        for line in _sys_path_bootstrap_lines(_tree(path))
    ]
    assert not offenders, (
        "growing `sys.path` is dead weight: the images bake PYTHONPATH=/workspace and tests/conftest.py "
        "covers a pytest run. Load a scripts/ entry point with tests.common.utils.load_script_module. "
        "Offenders:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "source",
    [
        "pytestmark = pytest.mark.cpu",
        "pytestmark = [pytest.mark.cpu]",
        "pytestmark = (pytest.mark.slow, pytest.mark.cpu)",
        "pytestmark: list = [pytest.mark.cpu]",
        "pytestmark += [pytest.mark.cpu]",
        "from pytest import mark\npytestmark = [mark.cpu]",
    ],
)
def test_the_marker_detector_fires_on_every_spelling(source):
    assert len(_redeclared_cpu_marker_lines(ast.parse(source))) == 1


@pytest.mark.parametrize(
    "source",
    [
        "pytestmark = [pytest.mark.gpu, pytest.mark.core]",
        '"""Never write pytestmark = pytest.mark.cpu here."""',
    ],
)
def test_the_marker_detector_passes_other_marks_and_mentions(source):
    assert _redeclared_cpu_marker_lines(ast.parse(source)) == []


@pytest.mark.parametrize(
    "source",
    [
        "sys.path.insert(0, ROOT)",
        "sys.path.append(ROOT)",
        "sys.path.extend([ROOT])",
        "sys.path[:0] = [ROOT]",
        "sys.path += [ROOT]",
        "sys.path = [ROOT, *sys.path]",
    ],
)
def test_the_bootstrap_detector_fires_on_every_spelling(source):
    assert len(_sys_path_bootstrap_lines(ast.parse(f"import sys\n{source}"))) == 1


@pytest.mark.parametrize(
    "source",
    [
        "found = ROOT in sys.path",
        "sys.path.remove(ROOT)",
        '"""Never call sys.path.insert here."""',
    ],
)
def test_the_bootstrap_detector_passes_reads_removals_and_mentions(source):
    assert _sys_path_bootstrap_lines(ast.parse(f"import sys\n{source}")) == []


def test_the_conventions_scan_reads_the_whole_suite():
    """Anti-vacuity: the four pins above pass trivially if the file sweep finds nothing."""
    files = _test_files()
    assert len(files) > 400, f"only {len(files)} test files scanned — the sweep lost its root"
    assert any(path.name == "conftest.py" for path in files)
    # Every exemption must still name a live file that still uses what it exempts, or it silently
    # outlives the file it was written for and blesses a path nothing checks.
    for rel in BOOTSTRAP_EXEMPT | SUMMARY_EXEMPT:
        assert (REPO_ROOT / rel).is_file(), f"stale exemption: {rel}"
    for rel in BOOTSTRAP_EXEMPT:
        assert _sys_path_bootstrap_lines(_tree(REPO_ROOT / rel)), (
            f"{rel} no longer grows sys.path — drop its exemption"
        )
    for rel in SUMMARY_EXEMPT:
        assert BANNED_SUMMARY in (REPO_ROOT / rel).read_text(encoding="utf-8"), (
            f"{rel} no longer uses `{BANNED_SUMMARY}` — drop its exemption"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
