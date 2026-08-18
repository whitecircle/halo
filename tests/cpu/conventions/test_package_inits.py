"""Every package init under ``src/`` re-exports nothing: a symbol has one import path, the module that
defines it, and importing a leaf never pays for a package's whole tree. ``src/__init__.py`` is the
process bootstrap and the one exception."""

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3] / "src"
_BOOTSTRAP = _SRC / "__init__.py"
_INITS = sorted(p for p in _SRC.rglob("__init__.py") if p != _BOOTSTRAP)


def test_the_sweep_sees_the_package_tree():
    assert len(_INITS) > 45, "the sweep lost the src/ package tree"


@pytest.mark.parametrize("init", _INITS, ids=lambda p: str(p.relative_to(_SRC.parent)))
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
        f"{init.relative_to(_SRC.parent)} holds {offending}: package inits carry a docstring only — "
        "import a symbol from the module that defines it."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
