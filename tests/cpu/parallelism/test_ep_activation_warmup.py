#!/usr/bin/env python
"""CPU tests for the expert-kernel warmup that precedes the first DeepEP dispatch.

The expert activation, the routing sort and the fused permute/unpermute are the lazily-compiled callables
inside the dispatch→combine span: every expert combine on the roster is a Triton kernel (the fused GLUs, the
clamped SwiGLUs, GptOss), as are the routing-sort and permute kernels, and Triton compiles one on its first
call. Left cold it compiles BETWEEN the two collectives, while every peer of the EP group spins in DeepEP's
barrier — whose budget bounds rank SKEW, not idle time.

Asserted here: the warmup runs to completion before the first dispatch on a real dispatch group (``ep_size > 1``),
exactly once per layer, in both grad modes AND with a real backward (the backward kernel is a separate compilation,
built only when a grad is first requested), even when that first forward runs under inference mode; it warms the
routing-sort and permute kernels wherever the compute path runs them, the grouped path at every ``ep_size``; the kernels
take their row count unspecialized, so one warm count covers every dispatch size; every gather of the grouped compute,
at any receive size, off a padded receive or a plain one, launches an index_select kernel the warm-up launched
(index_select picks one of two by index length, and CUDA loads a kernel on its first launch); and each family's own warm
hook reaches the callable its compute path actually calls, with the operands that path produces.

The grouped path warms at every ``ep_size``: without a barrier at ``ep_size == 1`` or under pure expert TP, a rank
first loading a kernel in a later step still holds up its peers' next collective. Torch's bf16 grouped GEMM picks one
of two CUTLASS tiles by the rows per expert, so every grouped GEMM the expert compute of a real layer launches,
forward and backward, over receives of one row per expert to past the switch, is a kernel class the warm-up launched,
for GptOss's de-interleaved pair with biases, Qwen3's separate halves and Qwen3.5's fused pair, at ep1, ep2, pure
expert TP and ep4 x etp2, with fp32 expert masters, expert LoRA and fake-quantized mxfp8, and after a first forward
with the expert adapters off. The warm-up runs each of those GEMMs at one row per expert and at one past the switch,
leaves the RNG its LoRA dropout draws from where it was, and leaves no gradient. Every permute and bias-gather
backward the expert compute runs, the warm-up ran first, at every one of those rank layouts: the weighted unpermute's,
and GptOss's expert-bias gather's, which its backward reaches by taking the expert parameters' gradients.

Run: pytest tests/cpu/parallelism/test_ep_activation_warmup.py
"""

import contextlib
from collections import defaultdict

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from halo.distributed.expert_parallel.deepep.capacity import padded_wire_hidden
from halo.distributed.expert_parallel.families import gpt_oss as gpt_oss_layer
from halo.distributed.expert_parallel.moe_layer import compute
from halo.distributed.expert_parallel.moe_layer.compute import _WARMUP_TOKENS
from halo.kernels import fused_glu, moe_permute, moe_route_sort
from halo.models.gradient_checkpointing import active_recompute_scope, scoped_checkpoint_func
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.grouped_gemm_layers import (
    GEMM_FAMILIES,
    GEMM_LAYOUTS,
    SWEPT_ROWS_PER_EXPERT,
    gemm_layer,
    routed_receive,
)
from tests.common.kernel_launches import GROUPED_MM_SMALL_TILE_DIM, INDEX_SELECT_SMALL_INDEX_ROWS, KernelLaunches

# Widths whose fp32 rows span a multiple of 16 bytes, which the grouped GEMM requires of every operand.
HIDDEN, INTER, LOCAL_EXPERTS = 8, 4, 2
# Every expert layout in every rank layout, and each variant with a dispatch group.
GEMM_CASES = (
    *((family, layout, "defaults") for family in GEMM_FAMILIES for layout in GEMM_LAYOUTS),
    *((family, "ep2", variant) for family in GEMM_FAMILIES for variant in ("fp32_experts", "expert_lora", "mxfp8")),
)
# (family, rank layout) of the grouped-GEMM check's negative controls.
NEGATIVE_CONTROL_CASES = (("gpt_oss", "ep2"), ("qwen3_moe", "ep1"), ("qwen3_5_moe", "ep4_etp2"))


class _RecordingDispatcher:
    """Identity dispatch and combine that log into the layer's event list."""

    def __init__(self, events: list):
        self.events = events

    def dispatch(self, flat, experts, weights, *, owner):
        self.events.append(("dispatch",))
        return flat, experts, weights, None, None

    def combine(self, output, recv_topk_weights, handle, *, owner):
        self.events.append(("combine",))
        return output


class _RecordingLayer(StubEPLayerBase):
    """EP layer whose dispatch/compute/combine and activation warmup append to one event log, so the
    ORDER between them is the assertion."""

    def __init__(self, ep_size: int = 2):
        super().__init__()
        self.ep_size = ep_size
        self.experts_per_rank = LOCAL_EXPERTS
        self.expert_tp_size = 1
        self._capture_routing = False
        self._activation_warmed = False
        self._use_grouped_mm = True
        self._lowp_quant_format = None
        self._lowp_weight_cacheable = True
        self._expert_adapters_enabled = True
        self._perf = lambda _label: contextlib.nullcontext()
        self.gate_up_proj = nn.Parameter(torch.randn(LOCAL_EXPERTS, HIDDEN, 2 * INTER))
        self.down_proj = nn.Parameter(torch.randn(LOCAL_EXPERTS, INTER, HIDDEN))
        self.events: list[tuple] = []
        self.dispatcher = _RecordingDispatcher(self.events)

    def _grouped_expert_ffn(self, sorted_tokens, offs, sorted_expert_ids, output_dtype):
        """The real grouped FFN, logged with its rows per expert: only the warm-up calls it, the compute being stubbed."""
        self.events.append(("warm_gemm", sorted_tokens.shape[0] // self.experts_per_rank))
        return super()._grouped_expert_ffn(sorted_tokens, offs, sorted_expert_ids, output_dtype)

    def _warm_expert_activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        self.events.append(("warm", tuple(gate_up.shape), torch.is_grad_enabled()))
        if gate_up.requires_grad:
            # The backward kernel is a separate compilation, built only when a grad is asked for —
            # so record whether the warmup actually asked, not just that it ran under enable_grad.
            gate_up.register_hook(lambda _grad: self.events.append(("warm_backward",)))
        return super()._warm_expert_activation(gate_up)

    def _glu_combine(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return gate * up

    def _compute_experts(self, tokens, experts, weights, output_dtype, *, recv_counts):
        self.events.append(("compute",))
        return tokens.to(output_dtype)


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
    assert "warm_gemm" in names, "the grouped GEMM must warm too"
    assert names[-3:] == ["dispatch", "compute", "combine"]


def test_warmup_covers_both_grad_modes_and_runs_a_real_backward():
    """The forward and backward kernels are separate compilations, and the backward is only built
    when a grad is first requested — so warming under ``enable_grad`` is not enough on its own, the
    warmup has to run the backward too."""
    layer = _RecordingLayer()
    _run_forward(layer)
    warms = [event for event in layer.events if event[0] == "warm"]
    assert {grad for _, _, grad in warms} == {True, False}
    assert {shape for _, shape, _ in warms} == {(_WARMUP_TOKENS, 2 * INTER)}
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
    assert warmed == 1
    assert [event[0] for event in layer.events] == ["dispatch", "compute", "combine"]


def test_warmup_backward_leaves_no_gradient_on_expert_params():
    """The warm inputs are leaves; a graph reaching the expert weights would accumulate a gradient
    from synthetic data into the trained parameters."""
    layer = _RecordingLayer()
    _run_forward(layer)
    assert layer.gate_up_proj.grad is None
    assert layer.down_proj.grad is None


def test_without_a_dispatch_group_the_activation_warms_within_the_gemm_warm_up():
    """``ep_size == 1`` runs the no-op dispatcher and no activation pass of its own: the grouped GEMM's warm-up runs the
    expert FFN, its activation with it, at one row per expert and at one past the tile switch, since which tile a step
    runs follows its rows per expert, and a rank loading one in a later step holds up the next collective."""
    layer = _RecordingLayer(ep_size=1)
    _run_forward(layer)
    assert layer.events == [
        ("warm_gemm", 1),
        ("warm_gemm", GROUPED_MM_SMALL_TILE_DIM + 1),
        ("dispatch",),
        ("compute",),
        ("combine",),
    ]


def test_the_per_expert_loop_without_a_dispatch_group_warms_nothing():
    """At ``ep_size == 1`` the per-expert loop runs no grouped GEMM and has no barrier: nothing to warm."""
    layer = _RecordingLayer(ep_size=1)
    layer._use_grouped_mm = False
    _run_forward(layer)
    assert [event[0] for event in layer.events] == ["dispatch", "compute", "combine"]


def test_the_warmup_draws_no_random_numbers():
    """The warm-up runs inside the gradient-checkpointed region, and the checkpoint restores the RNG
    to region entry before recomputing: a draw here is replayed by the forward and skipped by the
    recompute (it is latched), so every RNG consumer downstream of the MoE in that block — expert-LoRA
    dropout — differentiates activations the loss was never computed from."""
    layer = _RecordingLayer()
    before = torch.random.get_rng_state()
    layer._warm_activation_graphs(torch.device("cpu"), torch.float32, HIDDEN, top_k=2)

    assert torch.equal(torch.random.get_rng_state(), before)
    assert [event[0] for event in layer.events].count("warm") == 2


@pytest.mark.parametrize(
    ("kernel", "row_count"),
    (
        (fused_glu._glu_fwd_kernel, "n_rows"),
        (fused_glu._glu_bwd_kernel, "n_rows"),
        (moe_permute._gather_reduce_kernel, "n_src"),
        (moe_route_sort._histogram_kernel, "n_rows"),
        (moe_route_sort._scan_kernel, "n_blocks"),
        (moe_route_sort._scan_kernel, "n_valid"),
        (moe_route_sort._place_kernel, "n_rows"),
        (moe_route_sort._place_kernel, "n_valid"),
    ),
)
def test_the_kernels_do_not_specialize_on_the_dispatch_row_count(kernel, row_count):
    """The row count changes every dispatch. Triton compiles a separate binary per class of a
    specializable integer (1, a multiple of 16, neither), so a class the warmup did not run would compile
    between the dispatch and the combine. Unspecialized, the one warm count covers every size."""
    params = {param.name: param for param in kernel.params}
    assert params[row_count].do_not_specialize


_PERMUTE_BACKWARDS = {("MoEGatherPermute", "backward"), ("MoEWeightedUnpermute", "backward")}


def _record_permute_backwards(monkeypatch) -> list[tuple[str, str]]:
    """Wrap the warm-up's two permute Functions so each backward they run appends to the returned list."""
    calls = []
    for name in ("MoEGatherPermute", "MoEWeightedUnpermute"):
        function = getattr(moe_permute, name)

        class _Recording(function):
            @staticmethod
            def backward(ctx, *grads, _name=name, _function=function):
                calls.append((_name, "backward"))
                return _function.backward(ctx, *grads)

        monkeypatch.setattr(moe_permute, name, _Recording)
    return calls


@pytest.mark.parametrize(
    ("top_k", "ep_size", "grouped_mm", "warmed"),
    ((2, 2, True, True), (1, 2, True, True), (2, 1, True, True), (2, 2, False, False), (2, 1, False, False)),
    ids=(
        "fused_permute",
        "fused_permute_top_k_below_ep_size",
        "fused_permute_without_a_dispatch_group",
        "per_expert_loop",
        "per_expert_loop_without_a_dispatch_group",
    ),
)
def test_the_routing_sort_and_permute_kernels_are_warmed_where_the_compute_path_runs_them(
    monkeypatch, top_k, ep_size, grouped_mm, warmed
):
    """The grouped path always takes the routing sort and the fused permute, whatever ``top_k`` and ``ep_size``:
    the sort's kernels, its gather's backward and the weighted unpermute's forward and backward run inside the
    span, so the warmup runs them first, at the layer's ``top_k`` and local expert count, which the sort's kernels
    compile on. ``use_grouped_gemm: false`` runs the per-expert loop, the documented way around these kernels:
    there is nothing to warm, and a warm-up that ran them anyway would compile the kernels the loop exists to
    avoid."""
    calls = _record_permute_backwards(monkeypatch)
    sorts = []
    real_warm = compute.warm_route_sort_kernels
    monkeypatch.setattr(compute, "warm_route_sort_kernels", lambda *args: sorts.append(args[:2]) or real_warm(*args))
    layer = _RecordingLayer(ep_size=ep_size)
    layer._use_grouped_mm = grouped_mm
    flat = torch.randn(3, HIDDEN)
    experts = torch.zeros(3, top_k, dtype=torch.long)
    layer._dispatch_compute_combine(flat, experts, torch.ones(3, top_k), torch.float32)
    assert set(calls) == (_PERMUTE_BACKWARDS if warmed else set())
    assert sorts == ([(top_k, LOCAL_EXPERTS)] if warmed else [])


def test_a_first_forward_under_inference_mode_warms_every_kernel(monkeypatch):
    """The warm-up latches on a layer's first forward whatever mode it runs in, so a first forward under
    ``torch.inference_mode()`` must warm what a later training forward runs: the grad-enabled activation
    with its backward, and the permute kernels' backward. Latched without them, those compile between that
    later forward's dispatch and combine."""
    calls = _record_permute_backwards(monkeypatch)
    layer = _RecordingLayer()
    flat = torch.randn(3, HIDDEN)
    experts = torch.zeros(3, 2, dtype=torch.long)
    with torch.inference_mode():
        layer._dispatch_compute_combine(flat, experts, torch.ones(3, 2), torch.float32)

    names = [event[0] for event in layer.events]
    assert ("warm_backward",) in layer.events, "the inference-mode forward latched with the backward kernel cold"
    assert names.index("warm_backward") < names.index("dispatch")
    assert set(calls) == _PERMUTE_BACKWARDS, "the inference-mode forward latched with the permute kernels cold"
    layer.events.clear()
    _run_forward(layer)
    assert [event[0] for event in layer.events] == ["dispatch", "compute", "combine"]


def _gather_kernels(top_k: int, local_slots: int, padded: bool) -> set[tuple]:
    """The kernels the grouped compute's gathers launch for a receive of ``local_slots`` local slots over rows of
    ``top_k``: the permute's at bf16, off the padded receive's row-strided view under a dispatch group (``padded``) or
    off the plain tokens without one, and the expert-bias gather's at bf16 and at fp32 masters."""
    rows = -(-local_slots // top_k)
    width = padded_wire_hidden(HIDDEN) if padded else HIDDEN
    tokens = torch.zeros(rows, width, dtype=torch.bfloat16)[:, :HIDDEN]
    token_idx = torch.arange(local_slots) % rows
    inv_map = torch.zeros(rows, top_k, dtype=torch.long)
    expert_ids = torch.arange(local_slots) % LOCAL_EXPERTS
    biases = [torch.zeros(LOCAL_EXPERTS, HIDDEN, dtype=dtype) for dtype in (torch.bfloat16, torch.float32)]
    with KernelLaunches() as launches:
        moe_permute.MoEGatherPermute.apply(tokens, token_idx, inv_map)
        for bias in biases:
            moe_permute.MoEExpertBiasGather.apply(bias, expert_ids)
    return launches.kernels


def _gathers_left_cold(top_k: int, padded: bool = True) -> set[tuple]:
    """The kernels the grouped compute's gathers launch, over receives of one local slot to past the index_select
    cut-off, that the permute warm-up at ``top_k`` did not."""
    with KernelLaunches() as warm:
        moe_permute.warm_permute_kernels(_WARMUP_TOKENS, HIDDEN, top_k, torch.device("cpu"), torch.bfloat16)
    slots = range(1, 2 * INDEX_SELECT_SMALL_INDEX_ROWS + 2)
    return set().union(*(_gather_kernels(top_k, local_slots, padded) for local_slots in slots)) - warm.kernels


@pytest.mark.parametrize("padded", (True, False), ids=("dispatched", "without_a_dispatch_group"))
@pytest.mark.parametrize("top_k", (1, 4, 8))
def test_every_gather_of_the_grouped_compute_launches_a_kernel_the_warm_up_launched(top_k, padded):
    """index_select reads an index of up to 16 rows with a kernel of its own, and a gather's index is as long as the
    receive's local slots, which the routing sets: from one slot to past the cut-off, the permute's gather of bf16
    tokens and GptOss's expert-bias gather (bf16, or fp32 masters under ``fp32_experts``) launch only kernels the
    permute warm-up launched, at ``top_k`` 1 too, where the warm-up's own permute reads exactly 16 rows, and off the
    plain tokens of a layer without a dispatch group as off a padded receive."""
    assert _gathers_left_cold(top_k, padded) == set()


@pytest.mark.parametrize("top_k", (1, 4))
def test_the_gather_check_fails_its_negative_control(monkeypatch, top_k):
    """The warm-up without its index_select launches: at ``top_k`` 4 its permute reads 64 rows and leaves the
    small-index kernel cold, at ``top_k`` 1 it reads 16 and leaves the long-index one cold, and fp32 cold at both."""
    monkeypatch.setattr(moe_permute, "_warm_index_select", lambda *args: None)
    cold = _gathers_left_cold(top_k)
    path = "small, rank 2" if top_k > 1 else "gather"
    assert ("index_select", torch.bfloat16, torch.int64, path) in cold, cold
    assert any(kernel[1] == torch.float32 for kernel in cold), cold


def _warm_up_launches(layer) -> KernelLaunches:
    """The kernels ``layer``'s warm-up launches."""
    with KernelLaunches() as launches:
        layer._warm_activation_graphs(torch.device("cpu"), torch.bfloat16, layer.hidden_dim, layer.top_k)
    return launches


def _compute_gemm_kernels(layer) -> set[tuple]:
    """The grouped-GEMM kernel classes ``layer``'s expert compute launches, forward and backward, over receives of
    :data:`SWEPT_ROWS_PER_EXPERT` local slots per local expert."""
    with KernelLaunches() as launches:
        for rows_per_expert in SWEPT_ROWS_PER_EXPERT:
            tokens, experts, weights, counts = routed_receive(layer, rows_per_expert)
            out = layer._compute_experts(tokens, experts, weights, torch.bfloat16, recv_counts=counts)
            out.float().sum().backward()
    layer.zero_grad(set_to_none=True)
    return {kernel for kernel in launches.kernels if kernel[0] == "grouped_mm"}


@pytest.mark.parametrize(("family", "layout", "variant"), GEMM_CASES, ids="-".join)
def test_every_grouped_gemm_of_the_expert_compute_launches_a_kernel_the_warm_up_launched(family, layout, variant):
    """torch's bf16 grouped GEMM runs one of two CUTLASS tiles per operand layout, picked for the forward and the input
    gradient by the mean rows per expert, which change with every receive, and CUDA loads a kernel on its first launch:
    over receives from one row per expert to past the switch, the expert compute's grouped GEMMs, forward, input and
    weight gradient, launch only kernels the warm-up launched."""
    layer = gemm_layer(family, layout, variant)
    warm = _warm_up_launches(layer).kernels
    compute_kernels = _compute_gemm_kernels(layer)
    assert {kernel[3] for kernel in compute_kernels} == {"small", "large"}, "the receives must cross the tile switch"
    assert compute_kernels - warm == set()


@pytest.mark.parametrize(("family", "layout", "variant"), GEMM_CASES, ids="-".join)
def test_the_warm_up_runs_every_grouped_gemm_on_both_sides_of_the_tile_switch(family, layout, variant):
    """The warm-up's own grouped GEMMs, by operand layout (the forward, the input gradient and the weight gradient), run
    at one row per expert and at one past the switch."""
    rows_by_layout = defaultdict(set)
    for kernel, rows_per_expert in _warm_up_launches(gemm_layer(family, layout, variant)).grouped_mm_calls:
        rows_by_layout[kernel[2]].add(rows_per_expert)
    both = {1, GROUPED_MM_SMALL_TILE_DIM + 1}
    assert rows_by_layout == {
        "a row-major, b row-major": both,
        "a row-major, b column-major": both,
        "a column-major, b row-major": both,
    }


@pytest.mark.parametrize(("family", "layout"), NEGATIVE_CONTROL_CASES, ids="-".join)
def test_the_grouped_gemm_check_fails_without_the_gemm_warm_up(monkeypatch, family, layout):
    """The rest of the warm-up runs no grouped GEMM: without :meth:`_warm_expert_gemm` every grouped-GEMM kernel the
    expert compute launches is one the warm-up did not."""
    compute_kernels = _compute_gemm_kernels(gemm_layer(family, layout, "defaults"))
    monkeypatch.setattr(compute.EPExpertComputeMixin, "_warm_expert_gemm", lambda *args: None)
    assert compute_kernels - _warm_up_launches(gemm_layer(family, layout, "defaults")).kernels == compute_kernels


@pytest.mark.parametrize(("family", "layout"), NEGATIVE_CONTROL_CASES, ids="-".join)
def test_the_grouped_gemm_check_fails_with_one_row_per_expert_only(monkeypatch, family, layout):
    """A warm-up at one row per expert only leaves every large-tile kernel the expert compute launches cold."""
    compute_kernels = _compute_gemm_kernels(gemm_layer(family, layout, "defaults"))
    monkeypatch.setattr(compute, "GROUPED_MM_SMALL_TILE_DIM", 0)
    cold = compute_kernels - _warm_up_launches(gemm_layer(family, layout, "defaults")).kernels
    assert cold and {kernel[3] for kernel in cold} == {"large"}, cold


# The autograd Functions of the grouped compute whose backward runs a kernel of its own (the weighted unpermute's, the
# permute's gather-reduce, the expert-bias one-hot GEMM).
_GROUPED_COMPUTE_FUNCTIONS = (
    moe_permute.MoEGatherPermute,
    moe_permute.MoEWeightedUnpermute,
    moe_permute.MoEExpertBiasGather,
)


def _record_backwards(monkeypatch) -> list[str]:
    """Wrap each :data:`_GROUPED_COMPUTE_FUNCTIONS` backward, on the class every call site shares, so each run appends
    the Function's name to the returned list."""
    calls = []
    for function in _GROUPED_COMPUTE_FUNCTIONS:
        backward = function.backward

        def recording(ctx, *grads, _name=function.__name__, _backward=backward):
            calls.append(_name)
            return _backward(ctx, *grads)

        monkeypatch.setattr(function, "backward", staticmethod(recording))
    return calls


def _cold_backwards(monkeypatch, layer, warm_up=None) -> set[str]:
    """The :data:`_GROUPED_COMPUTE_FUNCTIONS` whose backward ``layer``'s expert compute runs, over receives of
    :data:`SWEPT_ROWS_PER_EXPERT` local slots per local expert, that its warm-up (``warm_up``, the layer's own by
    default) did not run."""
    calls = _record_backwards(monkeypatch)
    (warm_up or _warm_up_launches)(layer)
    warm = set(calls)
    calls.clear()
    _compute_gemm_kernels(layer)
    assert calls, "the expert compute ran none of the grouped compute's backwards"
    return set(calls) - warm


@pytest.mark.parametrize("layout", tuple(GEMM_LAYOUTS))
@pytest.mark.parametrize("family", tuple(GEMM_FAMILIES))
def test_every_backward_of_the_grouped_compute_runs_in_the_warm_up(monkeypatch, family, layout):
    """At every rank layout, with a dispatch group or without one, the permute's gather-reduce, the weighted
    unpermute's and GptOss's expert-bias gather's backward run in the warm-up before the expert compute runs them: the
    bias gather sits off the tokens' path, and the warm-up's backward reaches it by taking the expert parameters'
    gradients."""
    assert _cold_backwards(monkeypatch, gemm_layer(family, layout, "defaults")) == set()


@pytest.mark.parametrize("layout", ("ep1", "etp2"))
def test_the_backward_check_fails_without_the_permute_warm_up(monkeypatch, layout):
    """Without the permute warm-up a layer with no dispatch group warms only its GEMM, whose expert FFN runs no permute:
    the gather-reduce and the weighted unpermute's backward are left to the first step."""
    monkeypatch.setattr(compute, "warm_permute_kernels", lambda *args: None)
    cold = _cold_backwards(monkeypatch, gemm_layer("qwen3_5_moe", layout, "defaults"))
    assert cold == {"MoEGatherPermute", "MoEWeightedUnpermute"}, cold


def test_the_backward_check_fails_with_the_tokens_gradient_only(monkeypatch):
    """A warm-up backward that takes the tokens' gradient alone never reaches GptOss's expert-bias gather, whose input
    is the bias parameter."""
    layer = gemm_layer("gpt_oss", "ep2", "defaults")

    def without_parameters(warmed):
        warmed.expert_named_params = lambda: []
        try:
            _warm_up_launches(warmed)
        finally:
            del warmed.expert_named_params

    assert _cold_backwards(monkeypatch, layer, without_parameters) == {"MoEExpertBiasGather"}


def _adapters_off_warm_up_kernels(layer) -> set[tuple]:
    """The kernels the warm-up launches latching on a first forward with ``layer``'s expert adapters off, as a reference
    pass runs it; the adapters are back on after, as training runs them."""
    layer._expert_adapters_enabled = False
    warm = _warm_up_launches(layer).kernels
    assert layer._expert_adapters_enabled is False, "the warm-up must leave the adapters as it found them"
    layer._expert_adapters_enabled = True
    return warm


@pytest.mark.parametrize("family", tuple(GEMM_FAMILIES))
def test_a_first_forward_with_the_adapters_off_warms_the_adapter_gemms(family):
    """A reference pass runs the expert adapters off, and the warm-up latches on it: the adapter GEMMs training runs
    still warm, the adapters' weight gradient among them, which takes a kernel of its own with the base frozen."""
    layer = gemm_layer(family, "ep2", "expert_lora")
    warm = _adapters_off_warm_up_kernels(layer)
    assert _compute_gemm_kernels(layer) - warm == set()


def test_the_adapter_check_fails_its_negative_control(monkeypatch):
    """An expert FFN run with the adapters off leaves the adapters' weight-gradient kernel cold."""
    layer = gemm_layer("qwen3_moe", "ep2", "expert_lora")
    ffn = type(layer)._grouped_expert_ffn

    def adapters_off(self, *args):
        self._expert_adapters_enabled = False
        return ffn(self, *args)

    with monkeypatch.context() as patched:
        patched.setattr(type(layer), "_grouped_expert_ffn", adapters_off)
        warm = _adapters_off_warm_up_kernels(layer)
    cold = _compute_gemm_kernels(layer) - warm
    assert ("grouped_mm", torch.bfloat16, "a column-major, b row-major", "small") in cold, cold


@pytest.mark.parametrize("layout", ("ep1", "ep2"))
def test_the_grouped_gemm_warm_up_leaves_the_rng_and_no_gradient(layout):
    """Expert-LoRA dropout draws inside the warm-up's FFN, which runs inside the checkpointed block: the generator is
    restored after it, so the latched warm-up shifts no later draw the recompute replays. Nor does its backward leave a
    gradient on a weight, base or adapter."""
    layer = gemm_layer("qwen3_moe", layout, "expert_lora")
    before = torch.random.get_rng_state()
    _warm_up_launches(layer)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert [name for name, parameter in layer.named_parameters() if parameter.grad is not None] == []


def test_the_rng_check_fails_its_negative_control(monkeypatch):
    """Without the generator restored, the warm-up's LoRA dropout moves it."""
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kwargs: contextlib.nullcontext())
    layer = gemm_layer("qwen3_moe", "ep2", "expert_lora")
    before = torch.random.get_rng_state()
    _warm_up_launches(layer)
    assert not torch.equal(torch.random.get_rng_state(), before)


class _ScopeCachingDispatcher(_RecordingDispatcher):
    """A dispatch that takes its owner's replay slot, the way ``DeepEPDispatcher.dispatch`` does in a recompute
    scope."""

    def dispatch(self, flat, experts, weights, *, owner):
        active_recompute_scope().slot(owner, "dispatch").setdefault("flat", flat.detach())
        return super().dispatch(flat, experts, weights, owner=owner)


class _ScopeCachingLayer(_RecordingLayer):
    """A layer whose dispatcher takes a replay slot."""

    def __init__(self):
        super().__init__()
        self.dispatcher = _ScopeCachingDispatcher(self.events)

    def _compute_experts(self, tokens, experts, weights, output_dtype, *, recv_counts):
        # Saves its own output, so the non-reentrant checkpoint really has something to unpack —
        # without a saved tensor it never recomputes and the replay path is never entered.
        return super()._compute_experts(tokens, experts, weights, output_dtype, recv_counts=recv_counts).sigmoid()


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
    """A GptOss layer carrying only its compute-path selection; each test latches its own combine."""
    layer = object.__new__(gpt_oss_layer.EPGptOssMoELayer)
    layer._use_grouped_mm = grouped_mm
    layer.expert_tp_size = expert_tp_size
    return layer


@pytest.mark.parametrize(
    ("grouped_mm", "expert_tp_size", "contiguous_halves"),
    (
        (False, 1, False),  # per-expert loop: halves split straight out of the interleaved pair
        (True, 1, True),  # grouped GEMM: de-interleaved contiguous parameters
        (False, 2, True),  # ETP loop: the same de-interleaved storage
        (True, 2, True),  # ETP grouped GEMM: the same storage again
    ),
)
def test_gptoss_warms_the_activation_its_compute_path_calls(grouped_mm, expert_tp_size, contiguous_halves):
    """GptOss's activation is not behind the base GLU seam, and its paths reach the same kernel with
    different operands — the loop path splits a strided view out of one interleaved projection output,
    the grouped and ETP paths hold the halves de-interleaved. Warming the wrong layout warms a
    different amount of work than the run does."""
    called = []
    layer = _gpt_oss_stub(grouped_mm=grouped_mm, expert_tp_size=expert_tp_size)
    layer._gptoss_glu = lambda gate, up: called.append(gate.is_contiguous()) or gate
    layer._warm_expert_activation(torch.randn(4, 2 * INTER))
    assert called == [contiguous_halves]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
