"""CPU tests for EPSpanTimer: timing stats and the torch.profiler range bridge.

The EP layers label their dispatch / expert-compute / combine spans with
``torch.profiler.record_function``, and the EP benchmark's ``--comm_profile`` swaps in
``time_operation`` — these tests pin the contract both rely on: the operation name
must show up as a profiler event, and the monitor must aggregate timings.

Run: python tests/cpu/conventions/test_ep_span_timer.py
"""

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tests.common.ep_span_timer import EPSpanTimer


def _profiled_event_names() -> tuple[set[str], EPSpanTimer]:
    """Run one timed op under a CPU profiler; return the recorded event names + monitor."""
    monitor = EPSpanTimer()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with monitor.time_operation("ep.dispatch"):
            torch.randn(8, 8) @ torch.randn(8, 8)
    return {e.name for e in prof.events() or []}, monitor


def test_time_operation_emits_profiler_range():
    """The monitor's span must be visible to torch.profiler under the operation name."""
    names, _ = _profiled_event_names()
    assert "ep.dispatch" in names, f"record_function bridge missing; got {sorted(names)[:10]}"


def test_time_operation_aggregates_stats():
    _, monitor = _profiled_event_names()
    stats = monitor.stats["ep.dispatch"]
    assert stats.count == 1
    assert stats.total_time > 0.0
    assert stats.avg_time == stats.total_time


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
