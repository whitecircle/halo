#!/usr/bin/env python
"""Drift pins: one mechanism each for collection, tier marking, and import resolution.

* No test file may hand-roll pytest collection. Every suite runs under pytest, and a standalone run
  goes through ``pytest.main`` from the module's ``__main__``. A runner class that collects by hand —
  a literal list of ``(label, fn)`` pairs, or a ``globals()`` sweep — drops any test missing from its
  list, and reports failures only as a printed summary pytest never parses.
  ``tests.common.harness.record_check(checks, name, fn)`` is not that — the banned mechanism is the
  printed summary, not recording a verdict the harness then returns.
* No CPU test file re-declares the ``cpu`` marker: ``tests/conftest.py`` applies it by path to
  everything under ``tests/cpu/``, so a per-file ``pytestmark`` is a second mechanism for the same
  selection that only rots when the collector's rule changes.
* No test file bootstraps ``sys.path`` to the repo root: the images bake ``PYTHONPATH=/workspace``
  and ``tests/conftest.py`` covers a pytest run, so the insert is dead weight. A ``scripts/`` entry
  point is loaded through ``tests.common.utils.load_script_module`` instead of a path insert.

The banned spellings are assembled from fragments below so these pins do not match themselves and no
path has to be exempted.

Run: python tests/cpu/conventions/test_test_conventions.py
"""

import pytest

from tests.common.utils import REPO_ROOT

BANNED_RUNNER = "Test" + "Runner"
BANNED_SUMMARY = "print_" + "summary("
BANNED_MARKER = "pytestmark = pytest" + ".mark.cpu"
BANNED_BOOTSTRAP = "sys.path" + ".insert"
# The two legitimate inserts: the root conftest owns the pytest-run path, and one test writes a
# module into `tmp_path` and imports it back.
BOOTSTRAP_EXEMPT = {"tests/conftest.py", "tests/cpu/checkpoint/test_parallel_config_save.py"}
# The profiling benchmarks are not suites: they report a perf table to stdout by design, and nothing
# selects them by verdict.
SUMMARY_EXEMPT = {"tests/gpu/profiling/benchmark_collators.py", "tests/gpu/profiling/benchmark_torch_compile.py"}


def _test_files():
    return sorted((REPO_ROOT / "tests").rglob("*.py"))


def test_every_cpu_test_file_has_its_pytest_main_entry():
    """A CPU test file runs standalone through ``pytest.main([__file__, ...])`` from its ``__main__``;
    a file without one silently drops out of the documented ``python tests/cpu/...`` invocation."""
    offenders = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in _test_files()
        if path.name.startswith("test_")
        and "tests/cpu/" in path.as_posix()
        and "pytest.main([__file__" not in path.read_text(encoding="utf-8")
    )
    assert not offenders, "CPU test files without a `pytest.main([__file__, ...])` entry:\n  " + "\n  ".join(offenders)


def test_no_test_file_reimplements_pytest_collection():
    offenders = sorted(
        str(path.relative_to(REPO_ROOT)) for path in _test_files() if BANNED_RUNNER in path.read_text(encoding="utf-8")
    )
    assert not offenders, (
        f"{BANNED_RUNNER} is retired: a hand-listed runner hides tests it forgets to register, and its "
        f'printed summary is invisible to pytest. Use `if __name__ == "__main__": '
        f'raise SystemExit(pytest.main([__file__, "-v"]))` instead. Offenders:\n  ' + "\n  ".join(offenders)
    )


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
    offenders = sorted(
        str(path.relative_to(REPO_ROOT)) for path in _test_files() if BANNED_MARKER in path.read_text(encoding="utf-8")
    )
    assert not offenders, (
        f"`{BANNED_MARKER}` duplicates the marker tests/conftest.py already applies to everything under "
        "tests/cpu/. One mechanism only — delete the per-file marker. Offenders:\n  " + "\n  ".join(offenders)
    )


def test_no_test_bootstraps_the_repo_root_onto_sys_path():
    offenders = sorted(
        str(path.relative_to(REPO_ROOT))
        for path in _test_files()
        if BANNED_BOOTSTRAP in path.read_text(encoding="utf-8")
        and str(path.relative_to(REPO_ROOT)) not in BOOTSTRAP_EXEMPT
    )
    assert not offenders, (
        f"`{BANNED_BOOTSTRAP}` is dead weight: the images bake PYTHONPATH=/workspace and tests/conftest.py "
        "covers a pytest run. Load a scripts/ entry point with tests.common.utils.load_script_module. "
        "Offenders:\n  " + "\n  ".join(offenders)
    )


def test_the_conventions_scan_reads_the_whole_suite():
    """Anti-vacuity: the four pins above pass trivially if the file sweep finds nothing."""
    files = _test_files()
    assert len(files) > 400, f"only {len(files)} test files scanned — the sweep lost its root"
    assert any(path.name == "conftest.py" for path in files)
    # Every exemption must still name a live file that still uses what it exempts, or it silently
    # outlives the file it was written for and blesses a path nothing checks.
    for rel, banned in [(rel, BANNED_BOOTSTRAP) for rel in BOOTSTRAP_EXEMPT] + [
        (rel, BANNED_SUMMARY) for rel in SUMMARY_EXEMPT
    ]:
        assert (REPO_ROOT / rel).is_file(), f"stale exemption: {rel}"
        assert banned in (REPO_ROOT / rel).read_text(encoding="utf-8"), (
            f"{rel} no longer uses `{banned}` — drop its exemption"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
