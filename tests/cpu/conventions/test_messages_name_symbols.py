"""No string in the package or in a ``scripts/`` tool names a file of the package by path: a message names the
symbol its reader acts on, which a move leaves valid. Docstrings stay out of the sweep, and a ``scripts/`` tool
is named by its path on purpose."""

import ast
import re
from functools import cache
from pathlib import Path

import pytest

from scripts.docs.references import (
    PACKAGE,
    RepoIndex,
    is_string_literal,
    literal_text,
    string_constants,
    whole_string_literals,
)
from tests.common.repo import REPO_ROOT
from tests.common.sample_tree import spelled, write_sample_tree

_PACKAGE_MODULES = sorted((REPO_ROOT / PACKAGE).rglob("*.py"))
_SCRIPT_MODULES = sorted((REPO_ROOT / "scripts").rglob("*.py"))
_MODULES = (*_PACKAGE_MODULES, *_SCRIPT_MODULES)
_INDEX = RepoIndex(REPO_ROOT)
# A file path that is no tail of a longer name or path, read from the package root; a bare ``__init__.py``
# names every package's, not the root's.
_PACKAGE_RELATIVE_FILE = re.compile(r"(?<![\w./-])(?!__init__\.py\b)\w[\w/]*\.py\b")


@cache
def _messages(module: Path) -> list[tuple[int, str]]:
    """``(line, text)`` of each string of ``module`` but its docstrings; an f-string reads whole, with each
    field formatting a module-level string constant spelled out."""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    constants = string_constants(tree)
    # A string that is a statement of its own documents the code around it.
    docstrings = {
        id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Expr) and is_string_literal(node.value)
    }
    return [
        (node.lineno, literal_text(node, constants))
        for node in whole_string_literals(tree)
        if id(node) not in docstrings
    ]


def _package_paths(module: Path, index: RepoIndex) -> list[tuple[int, str]]:
    """``(line, path)`` for each file of the package a message of ``module`` names by path."""
    package = index.root / PACKAGE
    found = []
    for line, text in _messages(module):
        found += [
            (line, reference.text)
            for reference in index.references(text)
            if reference.kind == "path" and reference.text.startswith(f"{PACKAGE}/")
        ]
        found += [
            (line, f"{PACKAGE}/{path}") for path in _PACKAGE_RELATIVE_FILE.findall(text) if (package / path).is_file()
        ]
    return found


def test_the_sweep_reads_the_messages():
    assert len(_PACKAGE_MODULES) > 300, f"the sweep sees only {len(_PACKAGE_MODULES)} modules of {PACKAGE}/"
    assert len(_SCRIPT_MODULES) > 50, f"the sweep sees only {len(_SCRIPT_MODULES)} modules of scripts/"
    messages = sum(len(_messages(module)) for module in _MODULES)
    assert messages > 10_000, f"the sweep reads only {messages} strings"


def test_no_message_names_a_package_file_by_path():
    found = [
        f"{module.relative_to(REPO_ROOT)}:{line}: {path}"
        for module in _MODULES
        for line, path in _package_paths(module, _INDEX)
    ]
    assert not found, (
        f"these strings name a file of {PACKAGE}/ by path, which a move leaves stale; name the function, "
        "class or constant the reader acts on instead:\n  " + "\n  ".join(found)
    )


def test_a_message_path_is_found_wherever_the_string_is_built(tmp_path):
    """A path in a message, an f-string or a formatted constant is reported once per string, from either root; a
    docstring, a comment, a ``scripts/`` path, a non-file, a longer path's tail and a bare ``__init__.py`` are
    not."""
    write_sample_tree(
        tmp_path,
        {
            "$pkg/__init__.py": "",
            "$pkg/models/table.py": "",
            "$pkg/module.py": '''
                """Owns what $pkg/table.py lists."""

                _HOME = "$pkg/table.py"


                class Spec:
                    """Read off $pkg/table.py."""

                    name: str = "spec"
                    """The key in $pkg/table.py."""


                def check(found):
                    """Raise unless $pkg/table.py lists ``found``."""
                    # The table in $pkg/table.py.
                    if not found:
                        raise ValueError(f"no {found!r} entry: add it to $pkg/table.py")
                    if found == "home":
                        raise ValueError(f"see {_HOME}")
                    if found == "models":
                        raise ValueError("see models/table.py, not models/gone.py, lib/models/table.py or __init__.py")
                    raise ValueError("listed in $pkg/table.py; run $scripts/tool.py")
            ''',
        },
    )

    found = _package_paths(tmp_path / spelled("$pkg/module.py"), RepoIndex(tmp_path))

    table, models_table = spelled("$pkg/table.py"), spelled("$pkg/models/table.py")
    assert sorted(found) == [(3, table), (17, table), (19, table), (21, models_table), (22, table)], found


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
