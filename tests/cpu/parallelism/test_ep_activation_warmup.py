#!/usr/bin/env python
"""CPU tests for the expert-activation warmup that precedes the first DeepEP dispatch.

The expert activation is the one lazily-compiled callable inside the dispatch→combine span: every
expert combine on the roster is a Triton kernel (the fused GLUs, the clamped SwiGLUs, GptOss), and
Triton compiles one on its first call. Left cold it compiles BETWEEN the two collectives, while every
peer of the EP group spins in DeepEP's barrier — whose budget bounds rank SKEW, not idle time.

Asserted here: the warmup runs to completion before the first dispatch on a real dispatch group
(``ep_size > 1``; at ``ep_size == 1`` the dispatcher is a no-op with no barrier to stall), exactly
once per layer, in both grad modes AND with a real backward (the backward kernel is a separate
compilation, built only when a grad is first requested); and each family's own warm hook reaches the
callable its compute path actually calls, with the operands that path produces.

Run: pytest tests/cpu/parallelism/test_ep_activation_warmup.py
"""

import contextlib
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.distributed.expert_parallel.base_layer import _ACTIVATION_WARMUP_TOKENS
from src.distributed.expert_parallel.gc_scope import active_checkpoint_scope, scoped_checkpoint_func
from src.distributed.expert_parallel.layers import gpt_oss as gpt_oss_layer
from tests.common.ep_stubs import StubEPLayerBase

HIDDEN, INTER, LOCAL_EXPERTS = 6, 4, 2


class _RecordingLayer(StubEPLayerBase):
    """EP layer whose dispatch/compute/combine and activation warmup append to one event log, so the
    ORDER between them is the assertion."""

    def __init__(self, ep_size: int = 2):
        super().__init__()
        self.ep_size = ep_size
        self.expert_tp_size = 1
        self._capture_routing = False
        self._activation_warmed = False
        self._perf = lambda _label: contextlib.nullcontext()
        self.gate_up_proj = nn.Parameter(torch.randn(LOCAL_EXPERTS, HIDDEN, 2 * INTER))
        self.down_proj = nn.Parameter(torch.randn(LOCAL_EXPERTS, INTER, HIDDEN))
        self.events: list[tuple] = []

    def _warm_expert_activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        self.events.append(("warm", tuple(gate_up.shape), torch.is_grad_enabled()))
        if gate_up.requires_grad:
            # The backward kernel is a separate compilation, built only when a grad is asked for —
            # so record whether the warmup actually asked, not just that it ran under enable_grad.
            gate_up.register_hook(lambda _grad: self.events.append(("warm_backward",)))
        return super()._warm_expert_activation(gate_up)

    def _glu_combine(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return gate * up

    def _gc_dispatch(self, flat, experts, weights):
        self.events.append(("dispatch",))
        return flat, experts, weights, None

    def _compute_experts(self, tokens, experts, weights, output_dtype):
        self.events.append(("compute",))
        return tokens.to(output_dtype)

    def _gc_combine(self, output, recv_topk_weights, handle):
        self.events.append(("combine",))
        return output


def _run_forward(layer: _RecordingLayer, tokens: int = 3) -> None:
    flat = torch.randn(tokens, HIDDEN)
    experts = torch.zeros(tokens, 1, dtype=torch.long)
    layer._dispatch_compute_combine(flat, experts, torch.ones(tokens, 1), torch.float32)


def test_activation_is_warmed_before_the_first_dispatch():
    """Every warm call must land ahead of the dispatch: a trace after it stalls peers in the barrier."""
    layer = _RecordingLayer()
    _run_forward(layer)
    names = [event[0] for event in layer.events]
    dispatch = names.index("dispatch")
    warms = [index for index, name in enumerate(names) if name.startswith("warm")]
    assert warms and max(warms) < dispatch, "the warmup must finish before the dispatch"
    assert names[-3:] == ["dispatch", "compute", "combine"]


def test_warmup_covers_both_grad_modes_and_runs_a_real_backward():
    """The forward and backward kernels are separate compilations, and the backward is only built
    when a grad is first requested — so warming under ``enable_grad`` is not enough on its own, the
    warmup has to run the backward too."""
    layer = _RecordingLayer()
    _run_forward(layer)
    warms = [event for event in layer.events if event[0] == "warm"]
    assert {grad for _, _, grad in warms} == {True, False}
    assert {shape for _, shape, _ in warms} == {(tokens, 2 * INTER) for tokens in _ACTIVATION_WARMUP_TOKENS}
    assert ("warm_backward",) in layer.events, (
        "the grad-enabled warm ran no backward, so the backward kernel still compiles inside the span"
    )
    assert layer.events.index(("warm_backward",)) < [e[0] for e in layer.events].index("dispatch")


def test_warmup_runs_once_per_layer():
    """Latched: repeating it would put a synthetic forward+backward in front of every dispatch."""
    layer = _RecordingLayer()
    _run_forward(layer)
    warmed = sum(1 for event in layer.events if event[0] == "warm" and event[2])
    layer.events.clear()
    _run_forward(layer)
    assert warmed == len(_ACTIVATION_WARMUP_TOKENS)
    assert [event[0] for event in layer.events] == ["dispatch", "compute", "combine"]


def test_warmup_backward_leaves_no_gradient_on_expert_params():
    """The warm inputs are leaves; a graph reaching the expert weights would accumulate a gradient
    from synthetic data into the trained parameters."""
    layer = _RecordingLayer()
    _run_forward(layer)
    assert layer.gate_up_proj.grad is None
    assert layer.down_proj.grad is None


def test_no_dispatch_group_means_no_warmup():
    """``ep_size == 1`` runs the no-op dispatcher: no barrier to stall, so the warmup would only
    move the same compilation earlier."""
    layer = _RecordingLayer(ep_size=1)
    _run_forward(layer)
    assert [event[0] for event in layer.events] == ["dispatch", "compute", "combine"]


def test_the_warmup_draws_no_random_numbers():
    """The warm-up runs inside the gradient-checkpointed region, and the checkpoint restores the RNG
    to region entry before recomputing: a draw here is replayed by the forward and skipped by the
    recompute (it is latched), so every RNG consumer downstream of the MoE in that block — expert-LoRA
    dropout — differentiates activations the loss was never computed from."""
    layer = _RecordingLayer()
    before = torch.random.get_rng_state()
    layer._warm_activation_graphs(torch.device("cpu"), torch.float32)

    assert torch.equal(torch.random.get_rng_state(), before)
    assert [event[0] for event in layer.events].count("warm") == 2 * len(_ACTIVATION_WARMUP_TOKENS)


@pytest.mark.parametrize("inter", (INTER, 16), ids=("width_not_multiple_of_16", "width_multiple_of_16"))
def test_the_warmup_covers_every_element_count_class_a_run_can_present(inter):
    """These kernels take the element count (tokens x local intermediate) as a RUNTIME argument, and
    Triton compiles a separate binary per divisibility-by-16 class of it. Warming one class leaves the
    other to compile between the dispatch and the combine, with every peer already in DeepEP's barrier.
    """
    layer = _RecordingLayer()
    layer.down_proj = nn.Parameter(torch.zeros(LOCAL_EXPERTS, inter, HIDDEN))
    layer._warm_activation_graphs(torch.device("cpu"), torch.float32)

    warm_shapes = [event[1] for event in layer.events if event[0] == "warm"]
    warmed = {(tokens * inter) % 16 == 0 for tokens, _width in warm_shapes}
    # A width that is itself a multiple of 16 makes every token count divisible: the other class is
    # unreachable at runtime too, so warming it would be warming nothing.
    assert warmed == ({True} if inter % 16 == 0 else {True, False})


class _ScopeCachingLayer(_RecordingLayer):
    """A layer whose dispatch takes a replay slot, the way ``EPMoELayerBase._gc_dispatch`` does."""

    def _gc_dispatch(self, flat, experts, weights):
        active_checkpoint_scope().slot(self, "dispatch").setdefault("flat", flat.detach())
        return super()._gc_dispatch(flat, experts, weights)

    def _compute_experts(self, tokens, experts, weights, output_dtype):
        # Saves its own output, so the non-reentrant checkpoint really has something to unpack —
        # without a saved tensor it never recomputes and the replay path is never entered.
        return super()._compute_experts(tokens, experts, weights, output_dtype).sigmoid()


@pytest.mark.parametrize("use_reentrant", (True, False))
def test_the_warmup_does_not_recompute_the_block_it_runs_inside(use_reentrant):
    """The warm-up runs a real backward INSIDE the checkpointed block. A non-reentrant checkpoint packs
    everything saved there through its own hooks, so unpacking one drives that checkpoint's recompute
    mid-forward — and the recompute (the warm-up already latched) replays a dispatch the original pass
    has not made yet: the replay guard raises, and without it a second all-to-all into the live
    ElasticBuffer corrupts every gradient in the stage."""
    layer = _ScopeCachingLayer()
    tokens = 3

    def body(flat):
        return layer._dispatch_compute_combine(
            flat, torch.zeros(tokens, 1, dtype=torch.long), torch.ones(tokens, 1), torch.float32
        )

    def inner(function, *args, **kwargs):
        return checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)

    flat = torch.randn(tokens, HIDDEN, requires_grad=True)
    scoped_checkpoint_func(inner)(body, flat).sum().backward()

    assert [event[0] for event in layer.events].count("dispatch") == 2, "one original pass, one recompute"
    assert flat.grad is not None


class _SeparateStorageLayer(_RecordingLayer):
    """Separate/ETP storage: no fused ``gate_up_proj`` attribute."""

    def __init__(self):
        super().__init__()
        del self.gate_up_proj
        self.gate_proj = nn.Parameter(torch.randn(LOCAL_EXPERTS, HIDDEN, INTER))
        self.up_proj = nn.Parameter(torch.randn(LOCAL_EXPERTS, HIDDEN, INTER))


@pytest.mark.parametrize(
    ("layer_cls", "expect_contiguous"),
    ((_RecordingLayer, False), (_SeparateStorageLayer, True)),
)
def test_warm_halves_carry_the_stride_the_compute_path_produces(layer_cls, expect_contiguous):
    """The warm must hand the kernel the operands the compute path produces — fused storage chunks a
    strided view, separate storage does not — or it warms a different amount of work than the run does
    (a strided operand takes the kernel's materializing copy)."""
    seen = []
    layer = layer_cls()
    layer._glu_combine = lambda gate, up: seen.append((gate.is_contiguous(), up.is_contiguous())) or gate
    layer._warm_expert_activation(torch.randn(4, 2 * INTER))
    assert seen == [(expect_contiguous, expect_contiguous)]


def _gpt_oss_stub(*, grouped_mm: bool, expert_tp_size: int) -> gpt_oss_layer.EPGptOssMoELayer:
    """A GptOss layer carrying only what its warm hook reads — its compute-path selection and constants."""
    layer = object.__new__(gpt_oss_layer.EPGptOssMoELayer)
    layer._use_grouped_mm = grouped_mm
    layer.expert_tp_size = expert_tp_size
    layer.alpha = 1.702
    layer.limit = 7.0
    return layer


@pytest.mark.parametrize(
    ("grouped_mm", "expert_tp_size", "contiguous_halves"),
    (
        (False, 1, False),  # per-expert loop: halves split straight out of the interleaved pair
        (True, 1, True),  # grouped GEMM: de-interleaved contiguous parameters
        (False, 2, True),  # ETP loop: the same de-interleaved storage
    ),
)
def test_gptoss_warms_the_activation_its_compute_path_calls(
    monkeypatch, grouped_mm, expert_tp_size, contiguous_halves
):
    """GptOss's activation is not behind the base GLU seam, and its paths reach the same kernel with
    different operands — the loop path splits a strided view out of one interleaved projection output,
    the grouped and ETP paths hold the halves de-interleaved. Warming the wrong layout warms a
    different amount of work than the run does."""
    called = []
    monkeypatch.setattr(
        gpt_oss_layer,
        "fused_gptoss_glu",
        lambda gate, up, alpha, limit: called.append(("fused_gptoss_glu", gate.is_contiguous())) or gate,
    )
    _gpt_oss_stub(grouped_mm=grouped_mm, expert_tp_size=expert_tp_size)._warm_expert_activation(
        torch.randn(4, 2 * INTER)
    )
    assert called == [("fused_gptoss_glu", contiguous_halves)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
