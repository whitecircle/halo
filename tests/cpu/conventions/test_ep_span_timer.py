"""``EPSpanTimer`` aggregates timings and surfaces each operation's name as a torch.profiler event."""

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tests.common.ep_span_timer import EPSpanTimer


def _profiled_event_names() -> tuple[set[str], EPSpanTimer]:
    monitor = EPSpanTimer()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with monitor.time_operation("ep.dispatch"):
            torch.randn(8, 8) @ torch.randn(8, 8)
    return {e.name for e in prof.events() or []}, monitor


def test_time_operation_emits_profiler_range():
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
