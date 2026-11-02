"""Every module under ``halo/`` keeps its docstring to what it owns, in five lines or fewer; the mechanism
belongs in the function that implements it or on the module's owning doc page."""

import ast

import pytest

from tests.common.repo import REPO_ROOT

_PACKAGE_ROOT = REPO_ROOT / "halo"
_MAX_LINES = 5
_MODULES = sorted(_PACKAGE_ROOT.rglob("*.py"))


def _docstring_lines(path) -> int:
    docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    return len(docstring.splitlines()) if docstring else 0


def test_the_sweep_sees_the_package_tree():
    assert len(_MODULES) > 400, "the sweep lost the halo/ package tree"


def test_every_module_docstring_fits_five_lines():
    long = {
        str(path.relative_to(REPO_ROOT)): lines for path in _MODULES if (lines := _docstring_lines(path)) > _MAX_LINES
    }
    assert not long, (
        f"module docstrings over {_MAX_LINES} lines: {long}. Say what the module owns; move the mechanism to the "
        "function that implements it or to the owning doc page (skills/docs/docs-ownership.md)."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
