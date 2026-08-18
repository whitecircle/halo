#!/usr/bin/env python
"""CPU tests for :class:`RouterBiasBalancingCallback` observability seams.

All-zero expert-load counters make the DeepSeek-V3 sign update ``sign(0-0)=0`` — a silent no-op
every step. The callback must WARN ONCE at the first bias update when the (all-reduced) counters
are entirely zero, and stay silent when real load was recorded.

Run: ``python tests/cpu/callbacks/test_router_bias_balancing_callback.py``
"""

from __future__ import annotations

import pytest
import torch

import src.callbacks.router_bias_balancing as rbb
from src.callbacks.router_bias_balancing import RouterBiasBalancingCallback


class _Router(torch.nn.Module):
    def __init__(self, counts=None, num_experts=4):
        super().__init__()
        self.balancing_biases = torch.zeros(num_experts)
        self.expert_load_counter = None if counts is None else torch.tensor(counts, dtype=torch.float32)


class _Model(torch.nn.Module):
    def __init__(self, routers):
        super().__init__()
        self.routers = torch.nn.ModuleList(routers)


def _capture_warnings(fn):
    calls: list[str] = []
    orig = rbb.logger.warning
    rbb.logger.warning = lambda msg, *a, **k: calls.append(str(msg))
    try:
        fn()
    finally:
        rbb.logger.warning = orig
    return calls


def _step(callback, model):
    callback.on_step_end(args=None, state=None, control=None, model=model)


def test_warns_once_on_all_zero_counters_at_first_update():
    model = _Model([_Router(counts=[0, 0, 0, 0])])
    callback = RouterBiasBalancingCallback()
    warnings = _capture_warnings(lambda: _step(callback, model))
    assert any("ZERO" in w for w in warnings), f"no zero-counter warning emitted: {warnings}"
    # Sign update with zero counters is a no-op — biases must stay zero.
    assert torch.equal(model.routers[0].balancing_biases, torch.zeros(4))
    # Warn-once: a second all-zero step stays silent (tied to the first update).
    model.routers[0].expert_load_counter = torch.zeros(4)
    warnings2 = _capture_warnings(lambda: _step(callback, model))
    assert not any("ZERO" in w for w in warnings2), f"zero-counter warning repeated: {warnings2}"


def test_no_warning_with_real_load_and_bias_moves():
    model = _Model([_Router(counts=[10, 0, 0, 0])])
    callback = RouterBiasBalancingCallback(update_rate=1e-3)
    warnings = _capture_warnings(lambda: _step(callback, model))
    assert not any("ZERO" in w for w in warnings), f"spurious zero-counter warning: {warnings}"
    biases = model.routers[0].balancing_biases
    assert biases[0] < 0 < biases[1], "sign update did not move the biases toward balance"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
