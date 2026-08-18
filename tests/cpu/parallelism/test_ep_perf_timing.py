#!/usr/bin/env python
"""CPU test: the EP phase-timing opt-in must stay opt-in.

Every EP layer wraps its dispatch / expert-compute / combine phases in ``self._perf``, resolved once
in ``EPMoELayerBase._init_ep_state``: ``HALO_EP_PERF_PROFILE`` on → :meth:`PerformanceMonitor.time_operation`
(two ``torch.cuda.synchronize()`` per phase per forward), off → ``torch.profiler.record_function``, a
label with no device work. ``agent-docs/reference/configuration-reference.md`` promises exactly that: with the
knob off the ``ep.*`` spans still appear in a profiler trace **at no cost**.

Nothing else pins that branch. Wiring the monitor unconditionally — or resolving it before reading the
flag — costs a per-phase full-device sync on every training step of every EP run, a throughput
regression with no failing test and no error message. So: the off path must be the profiler label AND
must not even construct a monitor; the on path must be the process-wide monitor's bound
``time_operation`` and must record what it times.

Run: ``pytest -m cpu tests/cpu/parallelism/test_ep_perf_timing.py``
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.diagnostics import performance_monitor
from src.diagnostics.performance_monitor import get_performance_monitor
from src.distributed.expert_parallel import base_layer
from src.distributed.expert_parallel.base_layer import EPMoELayerBase


class _PerfProbeLayer(EPMoELayerBase):
    """Concrete, and built through the state constructor alone: the ``__init__`` template above it
    wants a real HF MoE block, while the ``_perf`` branch under test lives entirely in the state."""

    def __init__(self, ep_config, hidden_dim):
        self._init_ep_state(ep_config, hidden_dim)

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} probes construction only; its forward is never exercised")


def _ep_layer(monkeypatch: pytest.MonkeyPatch) -> EPMoELayerBase:
    """An EP layer built through the REAL state constructor, with only its EP plumbing stubbed.

    The dispatcher is the one constructor dependency needing live EP process groups; the ``_perf``
    branch under test sits after it and reads nothing from it.
    """
    monkeypatch.setattr(base_layer, "DeepEPDispatcher", lambda *args, **kwargs: object())
    ep_config = SimpleNamespace(
        num_experts=8,
        experts_per_rank=8,
        ep_rank=0,
        ep_size=1,
        expert_start_idx=0,
        expert_end_idx=8,
        expert_tp_size=1,
        expert_tp_rank=0,
        expert_tp_group=None,
        fp32_router=False,
        fp32_experts=False,
        use_grouped_gemm=False,
    )
    return _PerfProbeLayer(ep_config, hidden_dim=8)


@pytest.fixture(autouse=True)
def _fresh_global_monitor(monkeypatch: pytest.MonkeyPatch):
    """Reset the process-wide monitor so "was one created?" is this test's own answer."""
    monkeypatch.setattr(performance_monitor, "_global_monitor", None)


def test_the_spans_are_profiler_labels_and_build_no_monitor_when_the_knob_is_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HALO_EP_PERF_PROFILE", raising=False)
    layer = _ep_layer(monkeypatch)

    assert layer._perf is torch.profiler.record_function, (
        f"an EP layer built with HALO_EP_PERF_PROFILE unset labels its phases with {layer._perf!r} "
        f"instead of torch.profiler.record_function: the default path now pays the monitor's "
        f"per-phase torch.cuda.synchronize() on every forward of every EP run"
    )
    with layer._perf("ep.dispatch"):
        pass
    assert performance_monitor._global_monitor is None, (
        "the default path constructed a PerformanceMonitor — the opt-in must be read before the "
        "monitor is reached, not after"
    )


def test_the_spans_are_timed_by_the_process_monitor_when_the_knob_is_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HALO_EP_PERF_PROFILE", "1")
    layer = _ep_layer(monkeypatch)

    monitor = get_performance_monitor()
    assert getattr(layer._perf, "__self__", None) is monitor, (
        f"HALO_EP_PERF_PROFILE=1 must route the phase spans to the process-wide monitor's "
        f"time_operation; got {layer._perf!r}"
    )
    with layer._perf("ep.dispatch"):
        pass
    assert monitor.stats["ep.dispatch"].count == 1, (
        "the opt-in path timed nothing — the benchmark scripts read these stats to attribute the "
        "all-to-all fraction, so an untimed span reports 0 s of communication"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
