#!/usr/bin/env python
"""Callback scoping under pipeline parallelism (CPU).

Under PP each stage holds different layers, so world-wide callback collectives and world-based
throughput math are wrong:

* ``RouterBiasBalancingCallback`` must all-reduce its stacked per-layer expert counters over the
  trainer-injected ``reduce_group`` (the stage group) — a world all-reduce is a shape mismatch or a
  silent cross-stage blend.
* ``build_perf_callbacks`` must not install ``MoEMetricsCallback`` at ``pp_size > 1`` (a stage
  forward returns a bare tensor, so its router-logits hook would silently record nothing).

Run: python tests/cpu/parallelism/test_pp_callback_scoping.py
"""

from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.callbacks.router_bias_balancing import RouterBiasBalancingCallback
from src.callbacks.wiring import build_perf_callbacks

_RBB_MOD = "src.callbacks.router_bias_balancing"


# RouterBiasBalancingCallback.reduce_group


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
    """Drive ``on_step_end`` with ``dist`` stubbed as a 2-rank job; capture the all-reduce kwargs."""
    captured: dict = {}

    def _fake_all_reduce(tensor, op=None, group=None):
        captured["op"] = op
        captured["group"] = group

    fake_dist = types.SimpleNamespace(all_reduce=_fake_all_reduce, ReduceOp=torch.distributed.ReduceOp)
    with (
        patch(f"{_RBB_MOD}.dist", fake_dist),
        patch(f"{_RBB_MOD}.get_global_world_size", return_value=2),
    ):
        callback.on_step_end(args=None, state=None, control=None, model=_RouterModel())
    assert "group" in captured, "counter all-reduce was never issued"
    return captured


def test_bias_balancing_all_reduce_honors_injected_reduce_group():
    sentinel = object()
    callback = RouterBiasBalancingCallback()
    callback.reduce_group = sentinel
    captured = _run_step(callback)
    assert captured["group"] is sentinel, captured
    assert captured["op"] == torch.distributed.ReduceOp.SUM, captured


def test_bias_balancing_defaults_to_world_group():
    callback = RouterBiasBalancingCallback()
    assert callback.reduce_group is None  # trainer injects; default = world group
    captured = _run_step(callback)
    assert captured["group"] is None, captured


# build_perf_callbacks: MoEMetricsCallback skipped under PP


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
    callbacks = build_perf_callbacks(_perf_args(), SimpleNamespace(), _MoEModel(), _parallelism(pp_size))
    return [cb for cb in callbacks if type(cb).__name__ == "MoEMetricsCallback"]


def test_moe_metrics_installed_without_pp_but_skipped_under_pp():
    assert len(_moe_metric_callbacks(pp_size=1)) == 1, "MoEMetricsCallback must install at pp_size=1"
    assert _moe_metric_callbacks(pp_size=2) == [], "MoEMetricsCallback must be skipped under PP"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
