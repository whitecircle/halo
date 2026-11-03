#!/usr/bin/env python
"""Every script ``halo launch`` and ``halo run`` index is in ``agent-docs/reference/scripts-reference.md``, and vice
versa."""

from __future__ import annotations

import re

import pytest

from halo.cli import tool_commands, training_methods
from tests.common.repo import REPO_ROOT

_DOC = REPO_ROOT / "agent-docs" / "reference" / "scripts-reference.md"


def _scripts_on_disk() -> set[str]:
    indexed = {*tool_commands(REPO_ROOT).values(), *training_methods(REPO_ROOT).values()}
    return {str(path.relative_to(REPO_ROOT)) for path in indexed}


def _scripts_in_roster() -> set[str]:
    """The flag-helper table is not one."""
    rows: set[str] = set()
    in_script_table = False
    for line in _DOC.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            in_script_table = False
        elif line.startswith("| Script "):
            in_script_table = True
        elif in_script_table and (row := re.match(r"\| `(scripts/[a-z0-9_/]+\.py)` \|", line)):
            rows.add(row[1])
    return rows


def test_every_indexed_script_has_a_roster_row():
    missing = _scripts_on_disk() - _scripts_in_roster()
    assert not missing, f"{sorted(missing)} are runnable through the halo CLI but have no row in {_DOC.name}"


def test_every_roster_row_names_an_indexed_script():
    stale = _scripts_in_roster() - _scripts_on_disk()
    assert not stale, f"{_DOC.name} documents {sorted(stale)}, which the halo CLI indexes no longer"


def test_the_roster_is_found():
    """Anti-vacuity: the two pins above would pass on empty sets."""
    on_disk, in_roster = _scripts_on_disk(), _scripts_in_roster()
    assert {"scripts/training/sft.py", "scripts/after_training/merge_pp_stages.py"} <= on_disk & in_roster
    assert len(on_disk) >= 40, sorted(on_disk)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
