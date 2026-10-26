#!/usr/bin/env python
"""Every runnable script ``halo launch`` and ``halo run`` index is in the scripts-reference roster, and vice versa.

``agent-docs/reference/scripts-reference.md`` is where an operator finds a tool and its input guards, and
the ``--help`` sweep covers a new script's flag surface but not its absence from that page — a tool nobody
can find is one nobody runs. The reverse pin catches a documented script that was retired or renamed
without its row.

    python tests/cpu/checkpoint/test_scripts_reference_roster.py
"""

from __future__ import annotations

import re

import pytest

from halo.cli import tool_commands, training_methods
from tests.common.repo import REPO_ROOT

_DOC = REPO_ROOT / "agent-docs" / "reference" / "scripts-reference.md"


def _scripts_on_disk() -> set[str]:
    """Every script the CLI indexes, repo-relative."""
    indexed = {*tool_commands(REPO_ROOT).values(), *training_methods(REPO_ROOT).values()}
    return {str(path.relative_to(REPO_ROOT)) for path in indexed}


def _scripts_in_roster() -> set[str]:
    """The scripts the page's ``| Script |`` tables name in their first column (the flag-helper table is not one)."""
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
    """Anti-vacuity: both sides must see the scripts, or the two pins above pass on empty sets."""
    on_disk, in_roster = _scripts_on_disk(), _scripts_in_roster()
    assert {"scripts/training/sft.py", "scripts/after_training/merge_pp_stages.py"} <= on_disk & in_roster
    assert len(on_disk) >= 40, sorted(on_disk)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
