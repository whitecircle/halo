#!/usr/bin/env python3
"""``pytest.importorskip`` may only stand in front of a dependency that can legitimately be absent.

On a pinned, required dependency the skip inverts the failure direction: an image built without it
(or with a broken install) reports a green run with the tests that would have caught it silently
skipped. This scans ``tests/cpu`` and fails on every such guard, identified from two facts rather
than from a hand-maintained name list:

* the package is declared in ``[project] dependencies`` — an install without it is broken, not a
  supported configuration;
* another CPU test already imports it plainly at module level, so the suite cannot survive its
  absence anyway and the skip protects nothing.

A dotted THIRD-PARTY target (``transformers.models.<family>.modeling_<family>``) is left alone: the
pin guarantees the distribution, not that a given release ships that submodule. A dotted FIRST-PARTY
target (``src.*``, ``scripts.*``, ``tests.*``) is never exempt — it ships in this repository, so a
skip there hides a broken module instead of an absent one.

Usage:
    python tests/cpu/config/test_dependency_skips.py
"""

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CPU_TESTS = REPO_ROOT / "tests" / "cpu"
PYPROJECT = REPO_ROOT / "pyproject.toml"
# A requirement string is the distribution name up to its extras bracket / version specifier / marker.
_REQUIREMENT_TAIL = re.compile(r"[\[<>=!~;\s]")
# Packages that ship in this repository: always importable, so a guard in front of one is never a
# supported-configuration check. Load a scripts/ entry point with tests.common.utils.load_script_module.
FIRST_PARTY_ROOTS = {"src", "scripts", "tests"}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).strip().lower()


def _required_import_roots() -> set[str]:
    """Top-level import names of every ``[project] dependencies`` entry.

    ``packages_distributions()`` maps an installed distribution back to the names it is imported
    under, which resolves the renames (``faiss-cpu`` → ``faiss``, ``flash-linear-attention`` →
    ``fla``) without a table to maintain; a distribution absent from this environment falls back to
    its own normalized name.
    """
    declared = {
        _normalize(_REQUIREMENT_TAIL.split(dep, maxsplit=1)[0])
        for dep in tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    }
    installed = {root for root, dists in packages_distributions().items() if declared & {_normalize(d) for d in dists}}
    return declared | installed


def _cpu_test_files() -> list[Path]:
    return sorted(CPU_TESTS.rglob("test_*.py"))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _is_importorskip(func: ast.expr) -> bool:
    return (isinstance(func, ast.Attribute) and func.attr == "importorskip") or (
        isinstance(func, ast.Name) and func.id == "importorskip"
    )


def _importorskip_sites(paths: list[Path]) -> list[tuple[Path, int, str]]:
    """``(file, line, target)`` for every ``importorskip`` call on a literal module name."""
    sites = []
    for path in paths:
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Call) and _is_importorskip(node.func) and node.args:
                target = node.args[0]
                if isinstance(target, ast.Constant) and isinstance(target.value, str):
                    sites.append((path, node.lineno, target.value))
    return sites


def _plainly_imported_roots(paths: list[Path]) -> set[str]:
    """Top-level import names the files import unguarded at module level."""
    roots = set()
    for path in paths:
        for node in _parse(path).body:
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _is_banned_guard(target: str, plain_roots: set[str], required_roots: set[str]) -> bool:
    """A guard is banned in front of a first-party module, or a required, plainly-imported package."""
    if target.split(".")[0] in FIRST_PARTY_ROOTS:
        return True
    return "." not in target and _normalize(target) in required_roots and target in plain_roots


def _offenders(sites: list[tuple[Path, int, str]], plain_roots: set[str], required_roots: set[str]) -> list[str]:
    return [
        f"{path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}:{line} — importorskip({target!r})"
        for path, line, target in sites
        if _is_banned_guard(target, plain_roots, required_roots)
    ]


def test_no_importorskip_on_a_required_dependency():
    files = _cpu_test_files()
    sites = _importorskip_sites(files)
    assert sites, "scanner found no importorskip call at all — it cannot be detecting anything"

    offenders = _offenders(sites, _plainly_imported_roots(files), _required_import_roots())
    assert not offenders, (
        "pytest.importorskip guards a first-party module, or a dependency pyproject declares required "
        "and other CPU tests import plainly — the guard hides a broken image instead of failing:\n  "
        + "\n  ".join(offenders)
    )


def test_scan_flags_a_reinstated_transformers_skip(tmp_path):
    """Sensitivity: the scan above must reject the exact guard it exists to keep out."""
    reinstated = tmp_path / "test_reinstated.py"
    reinstated.write_text('import pytest\n\n\ndef test_x():\n    pytest.importorskip("transformers")\n')

    offenders = _offenders(
        _importorskip_sites([reinstated]), _plainly_imported_roots(_cpu_test_files()), _required_import_roots()
    )
    assert len(offenders) == 1 and "transformers" in offenders[0], offenders


def test_scan_flags_a_first_party_skip_but_spares_a_third_party_submodule(tmp_path):
    """The dotted carve-out covers a third-party submodule only — ``scripts.*`` is always present."""
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "import pytest\n\n\ndef test_x():\n"
        '    pytest.importorskip("scripts.inference.policies.gradio_policies_demo")\n'
        '    pytest.importorskip("transformers.models.qwen3.modeling_qwen3")\n'
    )

    offenders = _offenders(
        _importorskip_sites([probe]), _plainly_imported_roots(_cpu_test_files()), _required_import_roots()
    )
    assert len(offenders) == 1 and "scripts.inference" in offenders[0], offenders


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
