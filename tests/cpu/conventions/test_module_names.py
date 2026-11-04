"""No module or package under ``halo/`` or ``tests/common/`` carries a grab-bag or mechanism word as a whole
``_``-separated token of its name (CLAUDE.md, Code style)."""

from pathlib import Path

import pytest

from tests.common.repo import REPO_ROOT

_BANNED_TOKENS = frozenset(
    {
        "utils",
        "helpers",
        "common",
        "misc",
        "core",
        "mixins",
        "patches",
        "patching",
        "wrapper",
        "manager",
        "shared",
        "wiring",
        "autograd",
        "extension",
    }
)
# An MoE term that only looks like a banned word: read as one token.
_COMPOUND_TERMS = ("shared_experts",)
_ROOTS = (REPO_ROOT / "halo", REPO_ROOT / "tests" / "common")


def _banned_tokens(name: str) -> set[str]:
    """The banned words ``name`` (a module stem or a package directory name) carries as whole tokens."""
    for term in _COMPOUND_TERMS:
        name = name.replace(term, term.replace("_", ""))
    return set(name.split("_")) & _BANNED_TOKENS


def _named_units(root: Path) -> list[Path]:
    """Every package directory and module under ``root``, ``root`` itself and package inits excluded."""
    units = [path for path in root.rglob("*") if path.is_dir() and (path / "__init__.py").is_file()]
    units += [path for path in root.rglob("*.py") if path.name != "__init__.py"]
    return sorted(path for path in units if "__pycache__" not in path.parts)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("utils", {"utils"}),
        ("io_helpers", {"helpers"}),
        ("model_wrapper", {"wrapper"}),
        ("shared_experts_layer", set()),
    ],
)
def test_banned_tokens_reads_whole_tokens(name, expected):
    assert _banned_tokens(name) == expected


@pytest.mark.parametrize("root", _ROOTS, ids=lambda root: str(root.relative_to(REPO_ROOT)))
def test_no_unit_name_carries_a_banned_word(root):
    units = _named_units(root)
    assert len(units) > 20, f"the sweep lost the {root.relative_to(REPO_ROOT)} tree"
    offending = {
        str(unit.relative_to(REPO_ROOT)): sorted(banned)
        for unit in units
        if (banned := _banned_tokens(unit.stem if unit.suffix == ".py" else unit.name))
    }
    assert not offending, (
        f"{offending}: a module or package name says the concept it owns, never a grab-bag or mechanism word. "
        "Name it for what it holds, or move its contents to the modules that own them."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
