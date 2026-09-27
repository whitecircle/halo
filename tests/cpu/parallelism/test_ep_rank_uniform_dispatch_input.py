#!/usr/bin/env python
"""CPU tests for the rank-uniform EP autograd graph under expert-only training.

With only the experts trainable, nothing upstream of the first MoE layer requires grad. A rank whose
local experts receive no token computes a constant there, so its combine carries no autograd node
unless the dispatch has a grad-requiring input, and every later layer whose input loses grad through
it goes the same way. Its backward then takes fewer DeepEP collectives than its peers'. The layer
therefore makes such an input a grad-requiring leaf (``_rank_uniform_dispatch_input``).

A recording transport stands in for DeepEP here with the same autograd shape (dispatch backward = a
collective, combine backward = a collective, rows ``[:received]`` reach the local experts), under the
real ``_dispatch_compute_combine`` and per-expert compute. Asserted: an idle rank and a busy rank run
the same backward collectives in the same order; the leaf changes no parameter gradient; and it is
made only for a grad-enabled training forward over a transport whose experts train and whose dispatch
has no grad-requiring operand.

Run: python tests/cpu/parallelism/test_ep_rank_uniform_dispatch_input.py
"""

import contextlib

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.common.ep_stubs import StubEPLayerBase

HIDDEN, INTER, LOCAL_EXPERTS, TOKENS = 6, 4, 2, 5


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, topk_idx, topk_weights, transport):
        ctx.transport, ctx.rows = transport, x.shape[0]
        received = transport.received
        return x[:received].clone(), topk_idx[:received].clone(), topk_weights[:received].clone()

    @staticmethod
    def backward(ctx, grad_x, _grad_idx, grad_weights):
        ctx.transport.log.append(("dispatch.backward", ctx.transport.name))

        def pad(grad):
            return torch.cat([grad, grad.new_zeros(ctx.rows - grad.shape[0], *grad.shape[1:])])

        return pad(grad_x), None, pad(grad_weights), None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, _topk_weights, transport, rows):
        ctx.transport, ctx.received = transport, x.shape[0]
        out = x.new_zeros(rows, x.shape[1])
        out[: x.shape[0]] = x
        return out

    @staticmethod
    def backward(ctx, grad):
        ctx.transport.log.append(("combine.backward", ctx.transport.name))
        return grad[: ctx.received], None, None, None


class _RecordingTransport:
    """DeepEP's autograd shape: ``received`` rows reach this rank's experts; both backwards log."""

    _noop = False

    def __init__(self, name: str, received: int, log: list):
        self.name, self.received, self.log = name, received, log

    def dispatch(self, x, topk_idx, topk_weights):
        self.rows = x.shape[0]
        recv_x, recv_idx, recv_weights = _Dispatch.apply(x, topk_idx, topk_weights, self)
        return recv_x, recv_idx, recv_weights, None

    def combine(self, x, topk_weights, _handle):
        return _Combine.apply(x, topk_weights, self, self.rows)


class _Layer(StubEPLayerBase):
    """Just enough EP state for the real dispatch → per-expert compute → combine path."""

    def __init__(self, transport: _RecordingTransport, seed: int, ep_size: int = 2):
        super().__init__()
        self.dispatcher = transport
        self.ep_size = ep_size
        self.expert_tp_size = 1
        self.experts_per_rank = LOCAL_EXPERTS
        self.fp32_experts = False
        self._use_grouped_mm = False
        self._activation_warmed = True
        self._capture_routing = False
        self._expert_adapters_enabled = True
        self._hook_synced_expert_params = []
        self._perf = lambda _label: contextlib.nullcontext()
        self.act_fn = F.silu
        generator = torch.Generator().manual_seed(seed)
        self.gate_up_proj = nn.Parameter(torch.randn(LOCAL_EXPERTS, HIDDEN, 2 * INTER, generator=generator))
        self.down_proj = nn.Parameter(torch.randn(LOCAL_EXPERTS, INTER, HIDDEN, generator=generator))

    def route(self, flat: torch.Tensor) -> torch.Tensor:
        experts = torch.zeros(flat.shape[0], 1, dtype=torch.long)
        return self._dispatch_compute_combine(flat, experts, torch.full((flat.shape[0], 1), 0.5), flat.dtype)


def _two_layer_step(received: tuple[int, int], *, bypass_leaf: bool = False):
    """One forward+backward through two stacked layers; returns (backward log, param grads)."""
    log: list = []
    layers = [_Layer(_RecordingTransport(f"l{i}", received[i], log), seed=i) for i in range(2)]
    if bypass_leaf:
        for layer in layers:
            layer._rank_uniform_dispatch_input = lambda flat, _weights: flat
    hidden = torch.randn(TOKENS, HIDDEN, generator=torch.Generator().manual_seed(7))
    for layer in layers:
        hidden = hidden + layer.route(hidden)
    loss = hidden.square().mean()
    if loss.requires_grad:
        loss.backward()
    grads = [p.grad for layer in layers for p in (layer.gate_up_proj, layer.down_proj)]
    return log, grads


@pytest.mark.parametrize("idle", [(0, TOKENS), (0, 0), (TOKENS, 0)], ids=["first", "all", "second"])
def test_idle_rank_runs_the_busy_ranks_backward_collectives(idle):
    """Every collective a busy rank's backward enters, an idle rank's enters too, in the same order."""
    busy_log, _ = _two_layer_step((TOKENS, TOKENS))
    idle_log, _ = _two_layer_step(idle)
    assert busy_log == idle_log, f"busy rank {busy_log} vs idle rank {idle_log}"
    assert sorted(busy_log) == sorted(
        (op, layer) for op in ("combine.backward", "dispatch.backward") for layer in ("l0", "l1")
    )


def test_leaf_changes_no_parameter_gradient():
    """The leaf's own gradient is discarded; every expert gradient is what it was without it."""
    _, with_leaf = _two_layer_step((TOKENS, TOKENS))
    _, without = _two_layer_step((TOKENS, TOKENS), bypass_leaf=True)
    assert all(g is not None for g in with_leaf)
    for got, expected in zip(with_leaf, without, strict=True):
        assert torch.equal(got, expected)


def _dispatch_input(layer: _Layer, *, flat_grad=False, weights_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    flat = torch.randn(TOKENS, HIDDEN, requires_grad=flat_grad)
    weights = torch.rand(TOKENS, 1, requires_grad=weights_grad)
    return flat, layer._rank_uniform_dispatch_input(flat, weights)


def test_leaf_shares_the_input_and_requires_grad():
    flat, out = _dispatch_input(_Layer(_RecordingTransport("l0", TOKENS, []), seed=0))
    assert out is not flat and out.requires_grad and out.grad_fn is None
    assert out.data_ptr() == flat.data_ptr()


@pytest.mark.parametrize(
    "case", ["input_requires_grad", "weights_require_grad", "eval", "no_grad", "ep1", "experts_frozen"]
)
def test_no_leaf_where_the_dispatch_needs_none(case):
    """A grad-requiring operand already gives the dispatch its node; a reference/eval pass and ep1
    build no backward collective; frozen experts leave nothing to differ across ranks."""
    layer = _Layer(_RecordingTransport("l0", TOKENS, []), seed=0, ep_size=1 if case == "ep1" else 2)
    if case == "eval":
        layer.eval()
    if case == "experts_frozen":
        layer.requires_grad_(False)
    grad_mode = torch.no_grad() if case == "no_grad" else contextlib.nullcontext()
    with grad_mode:
        flat, out = _dispatch_input(
            layer, flat_grad=case == "input_requires_grad", weights_grad=case == "weights_require_grad"
        )
    assert out is flat


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
