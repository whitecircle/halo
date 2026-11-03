"""Every package init under ``halo/`` re-exports nothing: a symbol has one import path, and importing a leaf never
pays for a package's whole tree. ``halo/__init__.py`` is the process bootstrap and the one exception."""

import ast

import pytest

from tests.common.repo import REPO_ROOT

_PACKAGE_ROOT = REPO_ROOT / "halo"
_BOOTSTRAP = _PACKAGE_ROOT / "__init__.py"
_INITS = sorted(p for p in _PACKAGE_ROOT.rglob("__init__.py") if p != _BOOTSTRAP)


def test_the_sweep_sees_the_package_tree():
    assert len(_INITS) > 45, "the sweep lost the halo/ package tree"


@pytest.mark.parametrize("init", _INITS, ids=lambda p: str(p.relative_to(_PACKAGE_ROOT.parent)))
def test_package_init_re_exports_nothing(init):
    body = ast.parse(init.read_text(encoding="utf-8")).body
    docstring = (
        body[:1]
        if body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
        else []
    )
    offending = [type(node).__name__ for node in body[len(docstring) :]]
    assert not offending, (
        f"{init.relative_to(_PACKAGE_ROOT.parent)} holds {offending}: package inits carry a docstring only — "
        "import a symbol from the module that defines it."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
