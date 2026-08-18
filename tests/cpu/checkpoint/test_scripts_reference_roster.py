#!/usr/bin/env python
"""Every before/after-training tool on disk is in the scripts-reference roster, and vice versa.

``agent-docs/reference/scripts-reference.md`` is where an operator finds a tool and its input guards, and
the `--help` smoke test covers a new script's importability but not its absence from that page — a
tool nobody can find is one nobody runs. The reverse pin catches a documented tool that was retired
or renamed without its row.

    python tests/cpu/checkpoint/test_scripts_reference_roster.py
"""

from __future__ import annotations

import re

import pytest

from tests.common.utils import REPO_ROOT

_DOC = REPO_ROOT / "agent-docs" / "reference" / "scripts-reference.md"
_SUBTREES = ("after_training", "before_training")


def _tools_on_disk(subtree: str) -> set[str]:
    return {
        f"scripts/{subtree}/{path.name}"
        for path in (REPO_ROOT / "scripts" / subtree).glob("*.py")
        if not path.name.startswith("_")
    }


def _tools_in_roster(subtree: str) -> set[str]:
    """The tools the page's tables name in their first column."""
    return set(
        re.findall(rf"^\| `(scripts/{subtree}/[a-z0-9_]+\.py)` \|", _DOC.read_text(encoding="utf-8"), re.MULTILINE)
    )


@pytest.mark.parametrize("subtree", _SUBTREES)
def test_every_tool_on_disk_has_a_roster_row(subtree: str):
    missing = _tools_on_disk(subtree) - _tools_in_roster(subtree)
    assert not missing, f"{sorted(missing)} exist under scripts/{subtree}/ but have no row in {_DOC.name}"


@pytest.mark.parametrize("subtree", _SUBTREES)
def test_every_roster_row_names_a_tool_on_disk(subtree: str):
    stale = _tools_in_roster(subtree) - _tools_on_disk(subtree)
    assert not stale, f"{_DOC.name} documents {sorted(stale)}, which no longer exist under scripts/{subtree}/"


def test_the_roster_is_found():
    """Anti-vacuity: both parses must see the tools, or the two pins above pass on empty sets."""
    assert _tools_on_disk("after_training") and _tools_in_roster("after_training")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
