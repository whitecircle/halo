#!/usr/bin/env python
""":class:`BalancingBiasCallback`: zero-load warning, summary gate, lazy counter.
All-zero counters make the DeepSeek-V3 sign update ``sign(0-0)=0``, a silent no-op every step."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import halo.callbacks.balancing as balancing
from halo.callbacks.balancing import BalancingBiasCallback


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
    orig = balancing.logger.warning
    balancing.logger.warning = lambda msg, *a, **k: calls.append(str(msg))
    try:
        fn()
    finally:
        balancing.logger.warning = orig
    return calls


def _step(callback, model):
    callback.on_step_end(args=None, state=SimpleNamespace(global_step=1), control=None, model=model)


def test_warns_once_on_all_zero_counters_at_first_update():
    model = _Model([_Router(counts=[0, 0, 0, 0])])
    callback = BalancingBiasCallback()
    warnings = _capture_warnings(lambda: _step(callback, model))
    assert any("ZERO" in w for w in warnings), f"no zero-counter warning emitted: {warnings}"
    assert torch.equal(model.routers[0].balancing_biases, torch.zeros(4))
    model.routers[0].expert_load_counter = torch.zeros(4)
    warnings2 = _capture_warnings(lambda: _step(callback, model))
    assert not any("ZERO" in w for w in warnings2), f"zero-counter warning repeated: {warnings2}"


def test_no_warning_with_real_load_and_bias_moves():
    model = _Model([_Router(counts=[10, 0, 0, 0])])
    callback = BalancingBiasCallback(update_rate=1e-3)
    warnings = _capture_warnings(lambda: _step(callback, model))
    assert not any("ZERO" in w for w in warnings), f"spurious zero-counter warning: {warnings}"
    biases = model.routers[0].balancing_biases
    assert biases[0] < 0 < biases[1], "sign update did not move the biases toward balance"


def test_metric_summary_derives_on_the_first_step_and_every_nth():
    """Slot 0 carries the lowest load, so each update raises its bias by one ``update_rate`` step."""
    model = _Model([_Router(counts=[1, 2, 3, 4])])
    callback = BalancingBiasCallback(log_every_n_steps=3)
    derived, slot0_bias = [], []
    for step in range(1, 8):
        model.routers[0].expert_load_counter = torch.tensor([1.0, 2.0, 3.0, 4.0])
        callback.on_step_end(args=None, state=SimpleNamespace(global_step=step), control=None, model=model)
        slot0_bias.append(model.routers[0].balancing_biases[0].item())
        logs: dict = {}
        callback.on_log(args=None, state=None, control=None, logs=logs)
        if "moe/load_cv" in logs:
            derived.append(step)
    assert derived == [1, 3, 6]
    assert slot0_bias == pytest.approx([1e-3 * k for k in range(1, 8)])


def test_missing_counter_is_created_at_the_bias_width():
    router = _Router(counts=None, num_experts=5)
    _step(BalancingBiasCallback(), _Model([router]))
    assert router.expert_load_counter.dtype == torch.float32
    assert torch.equal(router.expert_load_counter, torch.zeros(5))
    assert torch.equal(router.balancing_biases, torch.zeros(5))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
