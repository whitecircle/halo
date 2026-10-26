"""Every toolkit registry holds what the registry-order golden pins, in the golden's order, rendered in a fresh
interpreter so no module a test imported first can reorder it.

Run: python tests/cpu/conventions/test_registry_order.py
"""

import json

import pytest

import src.distributed.expert_parallel.families.roster as ep_roster
from tests.common.equivalence.revisions import probe_in_tree
from tests.common.goldens.registry_order import GOLDEN_PATH, render
from tests.common.repo import REPO_ROOT

GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_every_registry_holds_the_golden_order():
    check = probe_in_tree(REPO_ROOT, render.__module__)
    assert check.returncode == 0, check.stderr


def test_every_registry_holds_entries():
    """Anti-vacuity: a registry that stopped filling, or a filter that dropped every class, would match an
    empty golden."""
    assert [name for name, entries in GOLDEN.items() if not entries] == []


def test_a_reversed_subclass_walk_reorders_the_ep_roster(monkeypatch):
    """The EP roster is read off the subclass walk; walked backwards it holds every class in another order."""
    walk = ep_roster.iter_subclasses
    roster = json.loads(render())["ep layer classes"]
    monkeypatch.setattr(ep_roster, "iter_subclasses", lambda base: walk(base)[::-1])

    reordered = json.loads(render())["ep layer classes"]

    assert sorted(reordered) == sorted(roster)
    assert reordered == roster[::-1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
