#!/usr/bin/env python3
"""``pytest.importorskip`` may only stand in front of a dependency that can legitimately be absent; on a required
one, a broken image reports a green run. Flags a ``[project] dependencies`` package another CPU test imports
plainly, and any first-party target, but no dotted third-party submodule, which the pin does not promise."""

import ast
import re
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

from tests.common.repo import REPO_ROOT

CPU_TESTS = REPO_ROOT / "tests" / "cpu"
PYPROJECT = REPO_ROOT / "pyproject.toml"
_REQUIREMENT_TAIL = re.compile(r"[\[<>=!~;\s]")
# Always importable, so a guard in front of one is never a supported-configuration check. Load a scripts/ entry point
# with tests.common.repo.load_script_module.
FIRST_PARTY_ROOTS = {"halo", "scripts", "tests"}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).strip().lower()


def _required_import_roots() -> set[str]:
    """Top-level import names of every ``[project] dependencies`` entry. ``packages_distributions()`` resolves the
    renames (``faiss-cpu`` -> ``faiss``); a distribution absent here falls back to its normalized name."""
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
    reinstated = tmp_path / "test_reinstated.py"
    reinstated.write_text('import pytest\n\n\ndef test_x():\n    pytest.importorskip("transformers")\n')

    offenders = _offenders(
        _importorskip_sites([reinstated]), _plainly_imported_roots(_cpu_test_files()), _required_import_roots()
    )
    assert len(offenders) == 1 and "transformers" in offenders[0], offenders


def test_scan_flags_a_first_party_skip_but_spares_a_third_party_submodule(tmp_path):
    """The dotted carve-out covers a third-party submodule only; ``scripts.*`` is always present."""
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "import pytest\n\n\ndef test_x():\n"
        '    pytest.importorskip("scripts.inference.playground.gradio_openai_chatbot")\n'
        '    pytest.importorskip("transformers.models.qwen3.modeling_qwen3")\n'
    )

    offenders = _offenders(
        _importorskip_sites([probe]), _plainly_imported_roots(_cpu_test_files()), _required_import_roots()
    )
    assert len(offenders) == 1 and "scripts.inference" in offenders[0], offenders


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
