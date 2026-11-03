"""Timing of named spans on CUDA events, swapped in for the EP layers' ``record_function`` labels by
``benchmark_sft_ep.py --comm_profile``."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import torch

_global_timer: EPSpanTimer | None = None


@dataclass
class TimingStats:
    """Accumulated timings for one operation; its name is the ``EPSpanTimer.stats`` key."""

    count: int = 0
    total_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0

    def update(self, time_taken: float):
        self.count += 1
        self.total_time += time_taken


class EPSpanTimer:
    """Times named spans. Read results from ``.stats`` (name -> TimingStats)."""

    def __init__(self):
        self.stats: dict[str, TimingStats] = defaultdict(TimingStats)

    @contextmanager
    def time_operation(self, operation_name: str):
        """Time a span on CUDA events where available; a wall clock would miss async kernels."""
        cuda = torch.cuda.is_available()
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
            if cuda:
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event) / 1000.0
            else:
                elapsed = time.perf_counter() - wall_start
            self.stats[operation_name].update(elapsed)


def get_ep_span_timer() -> EPSpanTimer:
    """The process-wide timer, created on first use."""
    global _global_timer
    if _global_timer is None:
        _global_timer = EPSpanTimer()
    return _global_timer
