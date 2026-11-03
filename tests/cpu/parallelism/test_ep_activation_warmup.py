#!/usr/bin/env python
"""Expert-kernel warmup before the first DeepEP dispatch. A cold Triton or CUTLASS kernel compiles between dispatch and
combine while peers spin in DeepEP's barrier, whose budget bounds rank skew."""

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
GEMM_CASES = (
    *((family, layout, "defaults") for family in GEMM_FAMILIES for layout in GEMM_LAYOUTS),
    *((family, "ep2", variant) for family in GEMM_FAMILIES for variant in ("fp32_experts", "expert_lora", "mxfp8")),
)
NEGATIVE_CONTROL_CASES = (("gpt_oss", "ep2"), ("qwen3_moe", "ep1"), ("qwen3_5_moe", "ep4_etp2"))


class _RecordingDispatcher:
    def __init__(self, events: list):
        self.events = events

    def dispatch(self, flat, experts, weights, *, owner):
        self.events.append(("dispatch",))
        return flat, experts, weights, None, None

    def combine(self, output, recv_topk_weights, handle, *, owner):
        self.events.append(("combine",))
        return output


class _RecordingLayer(StubEPLayerBase):
    """Dispatch, compute, combine and warmup log to one list, so their order is the assertion."""

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
        """Only the warm-up calls the real FFN, the compute being stubbed."""
        self.events.append(("warm_gemm", sorted_tokens.shape[0] // self.experts_per_rank))
        return super()._grouped_expert_ffn(sorted_tokens, offs, sorted_expert_ids, output_dtype)

    def _warm_expert_activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        self.events.append(("warm", tuple(gate_up.shape), torch.is_grad_enabled()))
        if gate_up.requires_grad:
            # The backward kernel is built only when a grad is asked for, so record whether the warmup asked.
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
    """A trace after the dispatch stalls peers in the barrier."""
    layer = _RecordingLayer()
    _run_forward(layer)
    names = [event[0] for event in layer.events]
    dispatch = names.index("dispatch")
    warms = [index for index, name in enumerate(names) if name.startswith("warm")]
    assert warms and max(warms) < dispatch, "the warmup must finish before the dispatch"
    assert "warm_gemm" in names, "the grouped GEMM must warm too"
    assert names[-3:] == ["dispatch", "compute", "combine"]


def test_warmup_covers_both_grad_modes_and_runs_a_real_backward():
    """The backward kernel is built only when a grad is first requested, so ``enable_grad`` alone is not enough."""
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
    """A graph reaching the expert weights would accumulate a gradient from synthetic data."""
    layer = _RecordingLayer()
    _run_forward(layer)
    assert layer.gate_up_proj.grad is None
    assert layer.down_proj.grad is None


def test_without_a_dispatch_group_the_activation_warms_within_the_gemm_warm_up():
    """Which tile a step runs follows its rows per expert, and a rank loading one in a later step holds up the next
    collective."""
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
    layer = _RecordingLayer(ep_size=1)
    layer._use_grouped_mm = False
    _run_forward(layer)
    assert [event[0] for event in layer.events] == ["dispatch", "compute", "combine"]


def test_the_warmup_draws_no_random_numbers():
    """The checkpoint restores the RNG to region entry, so a draw here is replayed by the forward and skipped by the
    recompute, and downstream LoRA dropout diverges."""
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
    """Triton compiles per class of a specializable integer (1, a multiple of 16, neither); unspecialized, one warm
    count covers every dispatch size."""
    params = {param.name: param for param in kernel.params}
    assert params[row_count].do_not_specialize


_PERMUTE_BACKWARDS = {("MoEGatherPermute", "backward"), ("MoEWeightedUnpermute", "backward")}


def _record_permute_backwards(monkeypatch) -> list[tuple[str, str]]:
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
    """``use_grouped_gemm: false`` runs the per-expert loop, the documented way around these kernels, so warming
    there would compile what it avoids."""
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
    """The warm-up latches on the first forward in any mode; latched without grad, the kernels compile inside a
    later dispatch-combine span."""
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
    with KernelLaunches() as warm:
        moe_permute.warm_permute_kernels(_WARMUP_TOKENS, HIDDEN, top_k, torch.device("cpu"), torch.bfloat16)
    slots = range(1, 2 * INDEX_SELECT_SMALL_INDEX_ROWS + 2)
    return set().union(*(_gather_kernels(top_k, local_slots, padded) for local_slots in slots)) - warm.kernels


@pytest.mark.parametrize("padded", (True, False), ids=("dispatched", "without_a_dispatch_group"))
@pytest.mark.parametrize("top_k", (1, 4, 8))
def test_every_gather_of_the_grouped_compute_launches_a_kernel_the_warm_up_launched(top_k, padded):
    """index_select reads an index of up to 16 rows with its own kernel, and a gather's index is as long as the
    receive's local slots."""
    assert _gathers_left_cold(top_k, padded) == set()


@pytest.mark.parametrize("top_k", (1, 4))
def test_the_gather_check_fails_its_negative_control(monkeypatch, top_k):
    """At ``top_k`` 4 the permute reads 64 rows and leaves the small-index kernel cold; at 1 it reads 16 and leaves
    the long-index one cold."""
    monkeypatch.setattr(moe_permute, "_warm_index_select", lambda *args: None)
    cold = _gathers_left_cold(top_k)
    path = "small, rank 2" if top_k > 1 else "gather"
    assert ("index_select", torch.bfloat16, torch.int64, path) in cold, cold
    assert any(kernel[1] == torch.float32 for kernel in cold), cold


def _warm_up_launches(layer) -> KernelLaunches:
    with KernelLaunches() as launches:
        layer._warm_activation_graphs(torch.device("cpu"), torch.bfloat16, layer.hidden_dim, layer.top_k)
    return launches


def _compute_gemm_kernels(layer) -> set[tuple]:
    with KernelLaunches() as launches:
        for rows_per_expert in SWEPT_ROWS_PER_EXPERT:
            tokens, experts, weights, counts = routed_receive(layer, rows_per_expert)
            out = layer._compute_experts(tokens, experts, weights, torch.bfloat16, recv_counts=counts)
            out.float().sum().backward()
    layer.zero_grad(set_to_none=True)
    return {kernel for kernel in launches.kernels if kernel[0] == "grouped_mm"}


@pytest.mark.parametrize(("family", "layout", "variant"), GEMM_CASES, ids="-".join)
def test_every_grouped_gemm_of_the_expert_compute_launches_a_kernel_the_warm_up_launched(family, layout, variant):
    """torch's bf16 grouped GEMM picks one of two CUTLASS tiles by mean rows per expert, and CUDA loads a kernel on
    its first launch."""
    layer = gemm_layer(family, layout, variant)
    warm = _warm_up_launches(layer).kernels
    compute_kernels = _compute_gemm_kernels(layer)
    assert {kernel[3] for kernel in compute_kernels} == {"small", "large"}, "the receives must cross the tile switch"
    assert compute_kernels - warm == set()


@pytest.mark.parametrize(("family", "layout", "variant"), GEMM_CASES, ids="-".join)
def test_the_warm_up_runs_every_grouped_gemm_on_both_sides_of_the_tile_switch(family, layout, variant):
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
    compute_kernels = _compute_gemm_kernels(gemm_layer(family, layout, "defaults"))
    monkeypatch.setattr(compute.EPExpertComputeMixin, "_warm_expert_gemm", lambda *args: None)
    assert compute_kernels - _warm_up_launches(gemm_layer(family, layout, "defaults")).kernels == compute_kernels


@pytest.mark.parametrize(("family", "layout"), NEGATIVE_CONTROL_CASES, ids="-".join)
def test_the_grouped_gemm_check_fails_with_one_row_per_expert_only(monkeypatch, family, layout):
    """One row per expert only leaves every large-tile kernel cold."""
    compute_kernels = _compute_gemm_kernels(gemm_layer(family, layout, "defaults"))
    monkeypatch.setattr(compute, "GROUPED_MM_SMALL_TILE_DIM", 0)
    cold = compute_kernels - _warm_up_launches(gemm_layer(family, layout, "defaults")).kernels
    assert cold and {kernel[3] for kernel in cold} == {"large"}, cold


# The Functions whose backward runs a kernel of its own.
_GROUPED_COMPUTE_FUNCTIONS = (
    moe_permute.MoEGatherPermute,
    moe_permute.MoEWeightedUnpermute,
    moe_permute.MoEExpertBiasGather,
)


def _record_backwards(monkeypatch) -> list[str]:
    calls = []
    for function in _GROUPED_COMPUTE_FUNCTIONS:
        backward = function.backward

        def recording(ctx, *grads, _name=function.__name__, _backward=backward):
            calls.append(_name)
            return _backward(ctx, *grads)

        monkeypatch.setattr(function, "backward", staticmethod(recording))
    return calls


def _cold_backwards(monkeypatch, layer, warm_up=None) -> set[str]:
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
    """The bias gather sits off the tokens' path, so the warm-up's backward reaches it through the expert
    parameters' gradients."""
    assert _cold_backwards(monkeypatch, gemm_layer(family, layout, "defaults")) == set()


@pytest.mark.parametrize("layout", ("ep1", "etp2"))
def test_the_backward_check_fails_without_the_permute_warm_up(monkeypatch, layout):
    """A layer with no dispatch group warms only its GEMM, whose FFN runs no permute."""
    monkeypatch.setattr(compute, "warm_permute_kernels", lambda *args: None)
    cold = _cold_backwards(monkeypatch, gemm_layer("qwen3_5_moe", layout, "defaults"))
    assert cold == {"MoEGatherPermute", "MoEWeightedUnpermute"}, cold


def test_the_backward_check_fails_with_the_tokens_gradient_only(monkeypatch):
    """GptOss's expert-bias gather takes the bias parameter as input, so the tokens' gradient never reaches it."""
    layer = gemm_layer("gpt_oss", "ep2", "defaults")

    def without_parameters(warmed):
        warmed.expert_named_params = lambda: []
        try:
            _warm_up_launches(warmed)
        finally:
            del warmed.expert_named_params

    assert _cold_backwards(monkeypatch, layer, without_parameters) == {"MoEExpertBiasGather"}


def _adapters_off_warm_up_kernels(layer) -> set[tuple]:
    """As a reference pass runs it; the adapters are back on after."""
    layer._expert_adapters_enabled = False
    warm = _warm_up_launches(layer).kernels
    assert layer._expert_adapters_enabled is False, "the warm-up must leave the adapters as it found them"
    layer._expert_adapters_enabled = True
    return warm


@pytest.mark.parametrize("family", tuple(GEMM_FAMILIES))
def test_a_first_forward_with_the_adapters_off_warms_the_adapter_gemms(family):
    """A reference pass runs the adapters off and the warm-up latches on it; their weight gradient takes a kernel of
    its own with the base frozen."""
    layer = gemm_layer(family, "ep2", "expert_lora")
    warm = _adapters_off_warm_up_kernels(layer)
    assert _compute_gemm_kernels(layer) - warm == set()


def test_the_adapter_check_fails_its_negative_control(monkeypatch):
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
    """Expert-LoRA dropout draws inside the checkpointed block, so a moved generator would shift draws the recompute
    replays."""
    layer = gemm_layer("qwen3_moe", layout, "expert_lora")
    before = torch.random.get_rng_state()
    _warm_up_launches(layer)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert [name for name, parameter in layer.named_parameters() if parameter.grad is not None] == []


def test_the_rng_check_fails_its_negative_control(monkeypatch):
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kwargs: contextlib.nullcontext())
    layer = gemm_layer("qwen3_moe", "ep2", "expert_lora")
    before = torch.random.get_rng_state()
    _warm_up_launches(layer)
    assert not torch.equal(torch.random.get_rng_state(), before)


class _ScopeCachingDispatcher(_RecordingDispatcher):
    """Takes its owner's replay slot, the way ``DeepEPDispatcher.dispatch`` does in a recompute scope."""

    def dispatch(self, flat, experts, weights, *, owner):
        active_recompute_scope().slot(owner, "dispatch").setdefault("flat", flat.detach())
        return super().dispatch(flat, experts, weights, owner=owner)


class _ScopeCachingLayer(_RecordingLayer):
    def __init__(self):
        super().__init__()
        self.dispatcher = _ScopeCachingDispatcher(self.events)

    def _compute_experts(self, tokens, experts, weights, output_dtype, *, recv_counts):
        # Without a saved tensor the non-reentrant checkpoint never recomputes and the replay path is never entered.
        return super()._compute_experts(tokens, experts, weights, output_dtype, recv_counts=recv_counts).sigmoid()


@pytest.mark.parametrize("use_reentrant", (True, False))
def test_the_warmup_does_not_recompute_the_block_it_runs_inside(use_reentrant):
    """Unpacking a saved tensor drives the recompute mid-forward, replaying a dispatch not yet made; a second
    all-to-all into the live ElasticBuffer corrupts gradients."""
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
    """Separate/ETP storage has no fused ``gate_up_proj``."""

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
    """A strided operand takes the kernel's materializing copy, so the wrong stride warms a different amount of work
    than the run does."""
    seen = []
    layer = layer_cls()
    layer._glu_combine = lambda gate, up: seen.append((gate.is_contiguous(), up.is_contiguous())) or gate
    layer._warm_expert_activation(torch.randn(4, 2 * INTER))
    assert seen == [(expect_contiguous, expect_contiguous)]


def _gpt_oss_stub(*, grouped_mm: bool, expert_tp_size: int) -> gpt_oss_layer.EPGptOssMoELayer:
    layer = object.__new__(gpt_oss_layer.EPGptOssMoELayer)
    layer._use_grouped_mm = grouped_mm
    layer.expert_tp_size = expert_tp_size
    return layer


@pytest.mark.parametrize(
    ("grouped_mm", "expert_tp_size", "contiguous_halves"),
    (
        (False, 1, False),  # Per-expert loop: halves split straight out of the interleaved pair.
        (True, 1, True),  # Grouped GEMM: de-interleaved contiguous parameters.
        (False, 2, True),
        (True, 2, True),
    ),
)
def test_gptoss_warms_the_activation_its_compute_path_calls(grouped_mm, expert_tp_size, contiguous_halves):
    """GptOss's paths reach the same kernel with different operands, a strided view on the loop path and
    de-interleaved halves elsewhere."""
    called = []
    layer = _gpt_oss_stub(grouped_mm=grouped_mm, expert_tp_size=expert_tp_size)
    layer._gptoss_glu = lambda gate, up: called.append(gate.is_contiguous()) or gate
    layer._warm_expert_activation(torch.randn(4, 2 * INTER))
    assert called == [contiguous_halves]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
