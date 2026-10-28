#!/usr/bin/env python
"""CPU test: an EP layer labels its phases and times nothing.

Every EP layer wraps its dispatch / expert-compute / combine phases in ``self._perf``, set in
``EPMoELayerBase._init_ep_state`` to ``torch.profiler.record_function``: a label with no device work,
so the ``ep.*`` spans appear in a profiler trace at no cost. Only ``benchmark_sft_ep.py --comm_profile``
swaps in :meth:`EPSpanTimer.time_operation` (two ``torch.cuda.synchronize()`` per phase per forward).

Wiring the timer by default costs a per-phase full-device sync on every training step of every EP run,
a throughput regression with no failing test and no error message. So the layer's spans must be the
profiler label.

Run: ``pytest -m cpu tests/cpu/parallelism/test_ep_perf_timing.py``
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import halo.distributed.expert_parallel.moe_layer.layer as layer_module
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase


class _PerfProbeLayer(EPMoELayerBase):
    """Concrete, and built through the state constructor alone: the ``__init__`` template above it
    wants a real HF MoE block, while the ``_perf`` attribute under test lives entirely in the state."""

    def __init__(self, ep_config, hidden_dim):
        self._init_ep_state(ep_config, hidden_dim)

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} probes construction only; its forward is never exercised")


def _ep_layer(monkeypatch: pytest.MonkeyPatch) -> EPMoELayerBase:
    """An EP layer built through the REAL state constructor, with only its EP plumbing stubbed.

    The dispatcher is the one constructor dependency needing live EP process groups; the ``_perf``
    attribute under test is set after it and reads nothing from it.
    """
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
