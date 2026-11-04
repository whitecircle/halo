#!/usr/bin/env python
"""Callback scoping under pipeline parallelism, where each stage holds different layers. A world all-reduce is a shape
mismatch or a cross-stage blend; a stage forward returns a bare tensor, so MoE metrics would record nothing."""

from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from halo.callbacks.balancing import BalancingBiasCallback
from halo.callbacks.factory import build_run_callbacks

_BALANCING_MOD = "halo.callbacks.balancing"


class _Router(torch.nn.Module):
    def __init__(self, counts, num_experts=4):
        super().__init__()
        self.balancing_biases = torch.zeros(num_experts)
        self.expert_load_counter = torch.tensor(counts, dtype=torch.float32)


class _RouterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _Router(counts=[10.0, 0.0, 0.0, 0.0])


def _run_step(callback) -> dict:
    """``dist`` stubbed as a 2-rank job; captures the all-reduce kwargs."""
    captured: dict = {}

    def _fake_all_reduce(tensor, op=None, group=None):
        captured["op"] = op
        captured["group"] = group

    fake_dist = types.SimpleNamespace(all_reduce=_fake_all_reduce, ReduceOp=torch.distributed.ReduceOp)
    with (
        patch(f"{_BALANCING_MOD}.dist", fake_dist),
        patch(f"{_BALANCING_MOD}.get_global_world_size", return_value=2),
    ):
        callback.on_step_end(args=None, state=SimpleNamespace(global_step=1), control=None, model=_RouterModel())
    assert "group" in captured, "counter all-reduce was never issued"
    return captured


def test_bias_balancing_all_reduce_honors_injected_reduce_group():
    sentinel = object()
    callback = BalancingBiasCallback()
    callback.reduce_group = sentinel
    captured = _run_step(callback)
    assert captured["group"] is sentinel, captured
    assert captured["op"] == torch.distributed.ReduceOp.SUM, captured


def test_bias_balancing_defaults_to_world_group():
    callback = BalancingBiasCallback()
    assert callback.reduce_group is None  # The trainer injects it; default = world group.
    captured = _run_step(callback)
    assert captured["group"] is None, captured


class _MoEConfig:
    def __init__(self):
        self.model_type = "test_moe"
        self.num_experts = 4
        self.num_experts_per_tok = 2
        self.output_router_logits = False
        self.router_aux_loss_coef = 0.0


class _MoEModel:
    def __init__(self):
        self.config = _MoEConfig()

    def modules(self):
        return iter([self])


def _perf_args():
    return SimpleNamespace(
        moe_balancing="none",
        enable_efficiency_metrics=False,
        enable_moe_metrics=True,
        enable_torch_profiler=False,
        router_balancing_rate=1e-3,
        num_full_model_params=None,
        report_mfu_diagnostics=False,
    )


def _parallelism(pp_size: int):
    return SimpleNamespace(tp_size=1, ep_size=1, cp_size=1, expert_tp_size=1, pp_size=pp_size)


def _moe_metric_callbacks(pp_size: int) -> list:
    callbacks = build_run_callbacks(
        _perf_args(), SimpleNamespace(), _MoEModel(), _parallelism(pp_size), balancing_mode="none"
    )
    return [cb for cb in callbacks if type(cb).__name__ == "MoEMetricsCallback"]


def test_moe_metrics_installed_without_pp_but_skipped_under_pp():
    assert len(_moe_metric_callbacks(pp_size=1)) == 1, "MoEMetricsCallback must install at pp_size=1"
    assert _moe_metric_callbacks(pp_size=2) == [], "MoEMetricsCallback must be skipped under PP"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
