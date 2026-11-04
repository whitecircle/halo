#!/usr/bin/env python
"""An EP layer labels its phases and times nothing unless ``--comm_profile``: the timer by default costs a per-phase
full-device sync on every step, a regression with no error message."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import halo.distributed.expert_parallel.moe_layer.layer as layer_module
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from tests.common.ep_span_timer import DropWarmupSpans, EPSpanTimer


class _PerfProbeLayer(EPMoELayerBase):
    """Built through the state constructor alone, since the ``__init__`` template wants a real HF MoE block."""

    def __init__(self, ep_config, hidden_dim):
        self._init_ep_state(ep_config, hidden_dim)

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} probes construction only; its forward is never exercised")


def _ep_layer(monkeypatch: pytest.MonkeyPatch) -> EPMoELayerBase:
    """The dispatcher needs live EP groups; ``_perf`` is set after it and reads nothing from it."""
    monkeypatch.setattr(layer_module, "DeepEPDispatcher", lambda *args, **kwargs: object())
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
        grouped_gemm_enabled=False,
    )
    return _PerfProbeLayer(ep_config, hidden_dim=8)


def test_the_spans_are_profiler_labels(monkeypatch: pytest.MonkeyPatch):
    layer = _ep_layer(monkeypatch)

    assert layer._perf is torch.profiler.record_function, (
        f"an EP layer labels its phases with {layer._perf!r} instead of torch.profiler.record_function: "
        f"every EP run now pays the timer's per-phase torch.cuda.synchronize() on every forward"
    )


def test_warmup_spans_do_not_reach_the_reported_average():
    """Step 1 builds DeepEP's buffers inside its dispatch; averaged in, a 10 s span reads as communication."""
    timer = EPSpanTimer()
    drop = DropWarmupSpans(2, timer)
    timer.stats["ep.dispatch"].update(10.0)
    drop.on_step_end(None, SimpleNamespace(global_step=1), None)
    assert timer.stats["ep.dispatch"].count == 1
    drop.on_step_end(None, SimpleNamespace(global_step=2), None)
    assert "ep.dispatch" not in timer.stats
    timer.stats["ep.dispatch"].update(0.04)
    drop.on_step_end(None, SimpleNamespace(global_step=3), None)
    assert timer.stats["ep.dispatch"].avg_time == 0.04


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
