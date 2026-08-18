"""Per-operation timing for EP training (opt-in via HALO_EP_PERF_PROFILE).

The EP layers wrap their dispatch/compute/combine spans in :meth:`PerformanceMonitor.time_operation`
to attribute the all-to-all communication fraction; the benchmark scripts read the resulting
``.stats`` map. The opt-in lives at the call site and nowhere else — the EP layer consults
``HALO_EP_PERF_PROFILE`` and routes its spans to ``torch.profiler.record_function`` when it is off,
so a monitor that exists is a monitor that times.
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class TimingStats:
    """Accumulated timings for one operation; its name is the ``PerformanceMonitor.stats`` key."""

    count: int = 0
    total_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0

    def update(self, time_taken: float):
        self.count += 1
        self.total_time += time_taken


class PerformanceMonitor:
    """Times named operations during training. Read results from ``.stats`` (name -> TimingStats)."""

    def __init__(self):
        self.stats: dict[str, TimingStats] = defaultdict(TimingStats)

    @contextmanager
    def time_operation(self, operation_name: str):
        """Time a span on CUDA events where available — the wall clock would miss async kernels."""
        cuda = torch.cuda.is_available()
        start_event = end_event = None
        if cuda:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        wall_start = time.perf_counter()

        try:
            with torch.profiler.record_function(operation_name):
                yield
        finally:
            if start_event is not None and end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event) / 1000.0  # ms -> s
            else:
                elapsed = time.perf_counter() - wall_start

            self.stats[operation_name].update(elapsed)


_global_monitor: PerformanceMonitor | None = None


def get_performance_monitor() -> PerformanceMonitor:
    """The process-wide monitor (the EP layers gate on HALO_EP_PERF_PROFILE before reaching it)."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor
