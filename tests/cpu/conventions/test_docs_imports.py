"""Every ``import`` line a reader can copy out of the docs, skills or rulebook resolves against the tree."""

import importlib
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LINE = re.compile(
    r"^(?:from (src(?:\.[a-z_0-9]+)*) import ([A-Za-z_][\w, ]*)|import (src(?:\.[a-z_0-9]+)+))\s*$", re.M
)


def _doc_import_lines():
    files = [
        *(_REPO_ROOT / "agent-docs").rglob("*.md"),
        *(_REPO_ROOT / "skills").rglob("*.md"),
        _REPO_ROOT / "CLAUDE.md",
        _REPO_ROOT / "README.md",
    ]
    for path in sorted(files):
        for match in _LINE.finditer(path.read_text(encoding="utf-8")):
            yield path.relative_to(_REPO_ROOT), match.group(0).strip()


_LINES = list(_doc_import_lines())


def test_the_sweep_sees_the_docs():
    assert len(_LINES) > 15, "the sweep found almost no import lines in the docs"


@pytest.mark.parametrize(("page", "line"), _LINES, ids=[f"{p}:{l}" for p, l in _LINES])
def test_doc_import_line_resolves(page, line):
    match = _LINE.match(line)
    module_name = match.group(1) or match.group(3)
    module = importlib.import_module(module_name)
    for name in (match.group(2) or "").split(","):
        name = name.strip()
        if name and not hasattr(module, name):
            importlib.import_module(f"{module_name}.{name}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
