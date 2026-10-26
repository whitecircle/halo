#!/usr/bin/env python
"""What configuring MoE balancing and building the run's callbacks issue, pinned against the balancing
collective-sequence golden: the collectives in call order, the records between them, the raise, the
roster and the router config left behind, per mode, loss kind and weight-sync flag.

Run: python tests/cpu/callbacks/test_balancing_collectives.py
"""

import json

import pytest

from tests.common.goldens.balancing_collectives import CASES, GOLDEN_PATH, case_key, trace_case

GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_the_cases_are_the_golden_cases():
    assert [case_key(*case) for case in CASES] == list(GOLDEN)


def test_the_golden_holds_every_collective_seam():
    """Anti-vacuity: the cases reach the mode agreement, the nothing-balanced consensus and the export
    contract, and some raise."""
    events = [event for trace in GOLDEN.values() for event in trace]
    assert any(event.startswith("all_gather_object") for event in events)
    assert any(event.startswith("all_reduce SUM") for event in events)
    assert any(event.startswith("all_reduce MAX") for event in events)
    assert any(event.startswith("raised") for event in events)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case_key(*case))
def test_balancing_issues_the_golden_trace(case):
    assert trace_case(*case) == GOLDEN[case_key(*case)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
