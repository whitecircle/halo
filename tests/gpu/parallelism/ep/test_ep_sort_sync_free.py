#!/usr/bin/env python
"""The grouped-GEMM token sort must not synchronize the host.

A host sync per MoE layer per forward drains the launch queue and stops the CPU running ahead of the
GPU; at 60 MoE layers that is 60 stalls per forward. The offending ops are the ones whose OUTPUT SHAPE
is data-dependent (``bincount``, ``unique_consecutive(return_counts=True)``, ``nonzero``) — CUDA can
only size their output by reading the device back.

``torch.cuda.set_sync_debug_mode("error")`` turns any such read into an exception, so this asserts the
property directly rather than trusting a claim no check enforces.

Scope: the whole sort, a counting sort in three Triton launches (``halo/kernels/moe_route_sort.py``) sized by the
dispatch's host-side receive counts at ``ep_size > 1`` and by every slot at ``ep_size == 1``; and the per-expert
loop (``use_grouped_gemm: false``, GptOss under expert TP) at ``ep_size > 1``, which sizes each expert's slot
read off its count. Each must also equal its device-sized reference (the ``nonzero`` sort; the ``torch.where``
loop, under deterministic algorithms) bit for bit, and that reference must trip the same detector, so a
sync-free pass is not a detector that sees nothing. The counting sort also runs every case of
``tests/common/route_sort_reference.py`` (top_k 1 to 10, 1 to 512 local experts, idle experts, rows with no local
slot, a zero-token rank, no counts at ``ep_size`` 1) against the torch construction, gate-weight gradient included,
compiles no binary past its warm-up and takes no pinned host block. In a fresh process, its forward and backward and
the grouped compute's gathers after the layer warm-up, over receives of one local slot to many, launch no kernel CUDA
still has to load (lazy loading loads a kernel on its first launch), and a control first launch shows the profiler
reports such a load.

Likewise, in a fresh process per case and order, a real layer's expert compute after its warm-up, forward and
backward over receives of one row per local expert to past the grouped GEMM's tile switch, in ascending and in
descending order, for each operand layout the expert FFN hands it (GptOss's de-interleaved pair with biases at
``ep_size`` 2, Qwen3's halves under pure expert TP, Qwen3.5's fused pair at ``ep_size`` 1, expert LoRA's adapters):
loads no grouped-GEMM kernel, and no kernel past its first receive outside GptOss's expert-bias backward. A kernel
whose variant follows the receive size loads past the first receive in one order or the other, and a rank first
needing it in a later step holds up its peers' next collective; one the first receive loads in both orders is a
step-1 load, logged. The expert-bias backward is the known limit, its loads logged too: a one-hot GEMM whose kernel
cuBLAS picks by the receive's length, which the warm-up runs at its own receive sizes only. The region ran the GEMMs
of both tiles, read off the CUTLASS kernels' names. Two controls: with the GEMM warm-up stubbed out, every CUTLASS
grouped-GEMM kernel the region runs is reported loaded under its own name; with the warm-up at one row per expert
only, the region loads large-tile kernels, and only those.

Run with 1 GPU:
    torchrun --nproc_per_node=1 \
        tests/gpu/parallelism/ep/test_ep_sort_sync_free.py
"""

import contextlib
import json
import os
import tempfile
from functools import partial

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from halo.distributed.expert_parallel.deepep.capacity import padded_wire_hidden
from halo.distributed.expert_parallel.moe_layer.compute import _WARMUP_TOKENS
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase, named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.kernels import moe_route_sort
from halo.kernels.moe_permute import MoEExpertBiasGather, MoEGatherPermute, warm_permute_kernels
from halo.kernels.moe_route_sort import sort_routing_by_expert, warm_route_sort_kernels
from tests.common.ep_reference import (
    GroupedSortLayer,
    dispatched_receive_counts,
    grouped_sort_mismatches,
    nonzero_grouped_sort,
    received_expert_ids,
    where_expert_loop,
)
from tests.common.grouped_gemm_layers import SWEPT_ROWS_PER_EXPERT, gemm_layer, routed_receive
from tests.common.harness import cuda_syncs_are_errors, gpu_test_main
from tests.common.parallelism import single_process_ep_config
from tests.common.rank_log import log
from tests.common.repo import probe_findings
from tests.common.route_sort_reference import (
    CASES,
    case_findings,
    compiled_sort_binaries,
    pinned_host_requests,
    receive,
    receive_counts,
    result_mismatches,
    sort_with_grad,
    swept_receives,
)
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

EXPERTS_PER_RANK = 8
TOP_K = 4
NUM_TOKENS = 512
HIDDEN = 256
# The family whose real EP layer the loop check runs; any family's loop reads its slots the same way.
LOOP_FAMILY = "qwen3_moe"
# (top_k, local experts) of the lazy-load check: gpt-oss at ep4 under expert TP 2, Qwen3.5 at ep8, and top_k 1.
LAZY_LOAD_SHAPES = ((4, 8), (8, 32), (1, 3))
# gpt-oss's hidden size; the expert compute reads it off DeepEP's padded receive.
LAZY_LOAD_HIDDEN = 2880
_LAZY_LOAD_MARKER = "LAZY_LOADS:"
_LAZY_LOAD_PROBE = """
import torch
from tests.gpu.parallelism.ep.test_ep_sort_sync_free import lazy_load_findings
print({marker!r} + "|".join(lazy_load_findings(torch.device("cuda", {index}))))
"""
# (family, layout, variant) of the grouped-GEMM lazy-load check (``tests/common/grouped_gemm_layers.py``), each in a
# fresh process, so none relies on a kernel another case loaded.
GEMM_LAZY_LOAD_CASES = (
    ("gpt_oss", "ep2", "defaults"),
    ("qwen3_moe", "etp2", "defaults"),
    ("qwen3_5_moe", "ep1", "defaults"),
    ("qwen3_5_moe", "ep2", "expert_lora"),
)
# The orders the sweep runs its receives in, each in its own process: a kernel whose variant follows the receive size
# loads past the first receive in at least one of them.
SWEEP_ORDERS = ("ascending", "descending")
# The profiler range of each receive's compute: its label, then its index in the sweep.
RECEIVE_RANGE = "receive "
# The autograd node of GptOss's expert-bias gradient, the one-hot GEMM ``onehot(eids)ᵀ @ grad_out``, in the profiler.
EXPERT_BIAS_BACKWARD = "MoEExpertBiasGatherBackward"
# In the profiler's names of torch's CUTLASS grouped-GEMM kernels: their problem shape and pointer-array schedule; and
# the name of the kernel that prepares their arguments.
CUTLASS_GROUPED_GEMM_MARKERS = ("GroupProblemShape", "PtrArray")
GROUPED_GEMM_PREPARE_KERNEL = "prepare_grouped_gemm_data"
# A CUTLASS grouped GEMM's tile in its name (``Schedule`` in ``aten/src/ATen/native/cuda/GroupMM.cu``; the cluster is
# 2x1x1 at both): on SM10x the large tile runs the 2-SM UMMA atom (``SM100_MMA_F16BF16_2x1SM_SS``) and the small one
# the 1-SM atom, on SM90 the large tile is the cooperative schedule and the small one ping-pong.
LARGE_TILE_MARKERS = ("2x1SM", "Cooperative")
SMALL_TILE_MARKERS = ("SM100_MMA", "Pingpong")
# The CUTLASS grouped-GEMM kernels a sweep across the tile switch runs at least: the forward and the input gradient at
# both tiles, the weight gradient at its one.
SWEPT_GROUPED_GEMM_KERNELS = 5
# Each control's probe setup and the findings function that reads its region: the GEMM warm-up stubbed out, or run at
# one row per expert only.
GEMM_CONTROLS = {
    "no_gemm_warm_up": (
        "compute.EPExpertComputeMixin._warm_expert_gemm = lambda *args: None",
        "grouped_gemm_control_findings",
    ),
    "one_row_warm_up": ("compute.GROUPED_MM_SMALL_TILE_DIM = 0", "grouped_gemm_tile_control_findings"),
}
_GEMM_LAZY_LOAD_PROBE = """
import torch
from halo.distributed.expert_parallel.moe_layer import compute
from tests.gpu.parallelism.ep import test_ep_sort_sync_free as check
{setup}
print({marker!r} + "|".join(check.{findings}(torch.device("cuda", {index}), {case}, {order!r})))
"""


def make_layer(ep_size):
    return GroupedSortLayer(ep_size=ep_size, experts_per_rank=EXPERTS_PER_RANK)


def make_inputs(ep_size, device):
    torch.manual_seed(0)
    tokens = torch.randn(NUM_TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)
    experts = torch.randint(0, EXPERTS_PER_RANK, (NUM_TOKENS, TOP_K), device=device)
    if ep_size > 1:
        # DeepEP marks slots routed to another rank with -1.
        experts = torch.where(torch.rand_like(experts, dtype=torch.float) < 0.5, experts, -1)
    # The dispatch's gate weights carry the router's gradient.
    weights = torch.rand(NUM_TOKENS, TOP_K, device=device, dtype=torch.float32, requires_grad=True)
    return tokens, experts, weights


@contextlib.contextmanager
def deterministic_algorithms():
    """Torch's deterministic-algorithms mode, restored to the caller's setting on exit."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def run_sort(ep_size, device):
    stub = make_layer(ep_size)
    tokens, experts, weights = make_inputs(ep_size, device)
    counts = dispatched_receive_counts(stub, experts)
    torch.cuda.synchronize()
    with cuda_syncs_are_errors():
        out = EPMoELayerBase._sort_tokens_for_grouped_mm(stub, tokens, experts, weights, recv_counts=counts)
    torch.cuda.synchronize()
    return out, (stub, tokens, experts, weights)


@gpu_test_main(exact_world_size=1, prefix="ep_sort_sync_free", partial_state=False)
def run(ctx):
    checks: dict[str, bool] = {}
    device = ctx.device

    # ep_size == 1: no DeepEP padding to compact, so nothing in the sort may read the device back
    # (a sync raises out of run_sort and fails the run).
    got, (stub, tokens, experts, weights) = run_sort(1, device)
    sorted_tokens, offs, sorted_token_idx, sorted_weights, sorted_expert_ids, _ = got
    mismatched = grouped_sort_mismatches(got, nonzero_grouped_sort(stub, tokens, experts, weights))
    log(f"[ep1] sort ran sync-free: {sorted_tokens.shape[0]} rows, offs={offs.tolist()}; mismatched={mismatched}")
    # Every slot sizes it, so the counting sort serves ep_size 1 too, whose kernels compile on no size.
    checks["ep1_runs_the_counting_sort"] = type(sorted_weights.grad_fn).__name__ == "_CountingSortBackward"
    checks["ep1_sort_equals_torch_construction_bitwise"] = not mismatched

    # The counts must still be right — a sync-free histogram that miscounts is worse than a sync.
    reference = torch.zeros(EXPERTS_PER_RANK, device=device, dtype=torch.long)
    unique, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
    reference[unique.long()] = counts.long()
    expected_offs = torch.cumsum(reference, 0).to(torch.int32)
    checks["ep1_offsets_match_reference"] = torch.equal(offs, expected_offs)
    # Every (token, slot) pair must land in some expert at ep1.
    checks["ep1_every_slot_lands_in_an_expert"] = int(offs[-1]) == NUM_TOKENS * TOP_K
    checks["ep1_sorted_index_covers_every_slot"] = sorted_token_idx.numel() == NUM_TOKENS * TOP_K
    checks["ep1_sorted_weights_cover_every_slot"] = sorted_weights.numel() == NUM_TOKENS * TOP_K

    # ep_size > 1: the receive counts size the compaction, so this sort raises on any read-back too.
    got, (stub, tokens, experts, weights) = run_sort(4, device)
    mismatched = grouped_sort_mismatches(got, nonzero_grouped_sort(stub, tokens, experts, weights))
    log(f"[ep4] sort ran sync-free: {got[2].numel()} of {experts.numel()} slots local; mismatched={mismatched}")
    checks["ep4_padding_present"] = bool((experts < 0).any())
    checks["ep4_sort_equals_nonzero_sort_bitwise"] = not mismatched
    checks["ep4_runs_the_counting_sort"] = type(got[3].grad_fn).__name__ == "_CountingSortBackward"

    # The detector sees the device-sized nonzero sort.
    checks["ep4_nonzero_sort_trips_the_detector"] = trips_the_detector(
        lambda: nonzero_grouped_sort(stub, tokens, experts, weights)
    )

    checks |= check_counting_sort(device)
    checks |= check_lazy_loads()
    checks |= check_grouped_gemm_lazy_loads()
    checks |= check_loop(device)
    return {"checks": checks}


def trips_the_detector(fn) -> bool:
    torch.cuda.synchronize()
    try:
        with cuda_syncs_are_errors():
            fn()
    except RuntimeError as exc:
        return "synchroniz" in str(exc).lower()
    finally:
        torch.cuda.synchronize()
    return False


def check_counting_sort(device) -> dict[str, bool]:
    """Each reference case through :func:`sort_routing_by_expert` after the layer warm-up for its ``top_k`` and local
    expert count: sync-free, bit-identical to the torch construction (gate-weight gradient included), served by
    the warmed binaries and handed no pinned host block. The receive and the reference are built outside the
    detector, whose host copies sync."""
    checks = {}
    warm_up_counted = False
    pinned = 0
    for index, case in enumerate(CASES):
        inputs = receive(case, device, seed=index)
        want = sort_with_grad(moe_route_sort._argsort_routing, case, inputs)
        cold = compiled_sort_binaries()
        warm_route_sort_kernels(case.top_k, case.local_experts, device)
        warm = compiled_sort_binaries()
        warm_up_counted |= warm > cold
        torch.cuda.synchronize()
        requests = pinned_host_requests()
        try:
            with cuda_syncs_are_errors():
                got = sort_with_grad(sort_routing_by_expert, case, inputs)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            log(f"[counting sort] {case.name} raised {exc}")
            checks[f"{case.name}_sync_free"] = False
            continue
        pinned += pinned_host_requests() - requests
        mismatched = case_findings(case, inputs) + result_mismatches(case, got, want)
        log(f"[counting sort] {case.name}: {inputs.upstream.numel()} local slots; mismatched={mismatched}")
        checks[f"{case.name}_sync_free"] = True
        checks[f"{case.name}_equals_torch_construction_bitwise"] = not mismatched
        checks[f"{case.name}_compiles_nothing_past_the_warm_up"] = compiled_sort_binaries() == warm
    # The count is not blind: some warm-up compiled a shape no earlier sort had.
    checks["warm_up_compiles_are_counted"] = warm_up_counted
    checks["counting_sort_takes_no_pinned_host_memory"] = pinned == 0
    # Nor is the pinned count: one block handed out moves it.
    requests = pinned_host_requests()
    torch.empty(1, pin_memory=True)
    checks["pinned_host_requests_are_counted"] = pinned_host_requests() > requests
    return checks


def profiled_launches(fn) -> tuple[list[tuple[str, int | None, bool]], set[str]]:
    """The kernel of each launch in ``fn`` that CUDA loaded first, with the index of the :data:`RECEIVE_RANGE` profiler
    range the load fell in (None outside every one) and whether it fell inside :data:`EXPERT_BIAS_BACKWARD`, and the
    name of every kernel ``fn`` ran. A load is a ``Lazy Function Loading`` event of the profiler's trace, named by the
    launch call it falls inside (most ATen launches load there) or else by the next launch (CUTLASS loads in its
    attribute setup, Triton when it loads its binary, both before their launch); ``?`` where neither exists."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as trace:
            events = json.load(trace)["traceEvents"]
    launches = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver") and "Launch" in e.get("name", "")]
    kernels = {e.get("args", {}).get("correlation"): e["name"] for e in events if e.get("cat") == "kernel"}
    receives = [
        (int(e["name"].removeprefix(RECEIVE_RANGE)), e["ts"], e["ts"] + e.get("dur", 0))
        for e in events
        if e.get("cat") == "user_annotation" and e.get("name", "").startswith(RECEIVE_RANGE)
    ]
    bias_backwards = [
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if "dur" in e and e.get("name", "").endswith(EXPERT_BIAS_BACKWARD)
    ]
    loads = []
    for load in (e for e in events if e.get("name") == "Lazy Function Loading"):
        inside = (e for e in launches if e["ts"] <= load["ts"] <= e["ts"] + e.get("dur", 0))
        after = (e for e in launches if e["ts"] > load["ts"])
        launch = next(inside, None) or min(after, key=lambda e: e["ts"], default=None)
        name = kernels.get(launch.get("args", {}).get("correlation"), "?") if launch else "?"
        receive = next((index for index, start, end in receives if start <= load["ts"] <= end), None)
        loads.append((name, receive, any(start <= load["ts"] <= end for start, end in bias_backwards)))
    return loads, set(kernels.values())


def profiled_lazy_loads(fn) -> list[str]:
    """The lazily loaded kernels of :func:`profiled_launches`."""
    return [name for name, *_ in profiled_launches(fn)[0]]


def lazy_load_findings(device) -> list[str]:
    """In a process that has launched no kernel yet: at each of :data:`LAZY_LOAD_SHAPES`, the layer warm-up's sort and
    permute kernels, then over :func:`swept_receives` (one local slot to many, with counts and, as at ``ep_size`` 1,
    without) the sort, forward and backward, and the grouped compute's gathers (the permute's of bf16 tokens off a
    padded receive's view, the expert-bias gather's at bf16 and fp32), profiled. Each kernel CUDA loads in that region
    is a finding, and so is a control first launch (index_select over fp64, which nothing else runs) whose load the
    profiler does not report: the check would be blind."""
    torch.cuda.set_device(device)
    rows = torch.zeros((1, 4), dtype=torch.float64, device=device)
    index = torch.zeros(1, dtype=torch.long, device=device)
    findings = [] if profiled_lazy_loads(lambda: rows.index_select(0, index)) else ["control: no lazy load reported"]
    for top_k, local_experts in LAZY_LOAD_SHAPES:
        warm_route_sort_kernels(top_k, local_experts, device)
        warm_permute_kernels(_WARMUP_TOKENS, LAZY_LOAD_HIDDEN, top_k, device, torch.bfloat16)
        biases = [
            torch.zeros((local_experts, LAZY_LOAD_HIDDEN), device=device, dtype=dtype)
            for dtype in (torch.bfloat16, torch.float32)
        ]
        receives = []
        for n_rows, _, inputs in swept_receives(top_k, local_experts):
            weights = inputs.weights.to(device).requires_grad_(True)
            padded = torch.zeros((n_rows, padded_wire_hidden(LAZY_LOAD_HIDDEN)), device=device, dtype=torch.bfloat16)
            tokens = padded[:, :LAZY_LOAD_HIDDEN]
            receives.append((inputs.experts.to(device), weights, inputs.counts, inputs.upstream.to(device), tokens))

        def sort_and_gather_every_receive(receives=receives, local_experts=local_experts, biases=biases):
            for experts, weights, counts, upstream, tokens in receives:
                token_idx, sorted_weights, expert_ids, _, inv_map = sort_routing_by_expert(
                    experts, weights, local_experts, counts
                )
                torch.autograd.grad(sorted_weights, weights, upstream)
                MoEGatherPermute.apply(tokens, token_idx, inv_map)
                for bias in biases:
                    MoEExpertBiasGather.apply(bias, expert_ids)

        findings += [
            f"top_k {top_k}, {local_experts} local experts: CUDA loaded {name} past the warm-up"
            for name in profiled_lazy_loads(sort_and_gather_every_receive)
        ]
    return findings


def check_lazy_loads() -> dict[str, bool]:
    """:func:`lazy_load_findings` in a fresh interpreter, since this one has launched every kernel the sort runs."""
    probe = _LAZY_LOAD_PROBE.format(marker=_LAZY_LOAD_MARKER, index=torch.cuda.current_device())
    findings = probe_findings(probe, _LAZY_LOAD_MARKER)
    log(f"[lazy loads] {findings}")
    return {
        "lazy_loads_are_reported": not any(finding.startswith("control") for finding in findings),
        "sort_and_gathers_load_no_kernel_past_the_warm_up": not findings,
    }


def is_cutlass_grouped_gemm(name: str) -> bool:
    return any(marker in name for marker in CUTLASS_GROUPED_GEMM_MARKERS)


def is_grouped_gemm_kernel(name: str) -> bool:
    return is_cutlass_grouped_gemm(name) or GROUPED_GEMM_PREPARE_KERNEL in name


def grouped_gemm_tile(name: str) -> str | None:
    """The tile, ``large`` or ``small``, of the CUTLASS grouped-GEMM kernel ``name`` by :data:`LARGE_TILE_MARKERS` and
    :data:`SMALL_TILE_MARKERS` (the large tile's 2-SM atom is an ``SM100_MMA`` too, so it is read first); None where
    the name carries neither."""
    if any(marker in name for marker in LARGE_TILE_MARKERS):
        return "large"
    if any(marker in name for marker in SMALL_TILE_MARKERS):
        return "small"
    return None


def profiled_gemm_region(device, case: int, order: str) -> tuple[str, list[tuple[str, int | None, bool]], set[str]]:
    """In a process that has launched no kernel yet: the layer warm-up of ``GEMM_LAZY_LOAD_CASES[case]``, then its
    expert compute, forward and backward, over receives of :data:`SWEPT_ROWS_PER_EXPERT` local slots per local expert
    in ``order`` (``ascending`` or ``descending``), each receive in its own :data:`RECEIVE_RANGE` profiler range. The
    case's label, the kernels CUDA loaded in that region, each with the receive it loaded in and whether it loaded in
    the expert-bias backward, and the CUTLASS grouped-GEMM kernels the region ran."""
    torch.cuda.set_device(device)
    family, layout, variant = GEMM_LAZY_LOAD_CASES[case]
    layer = gemm_layer(family, layout, variant, device=device)
    swept = SWEPT_ROWS_PER_EXPERT if order == "ascending" else SWEPT_ROWS_PER_EXPERT[::-1]
    receives = [routed_receive(layer, rows_per_expert) for rows_per_expert in swept]
    layer._warm_activation_graphs(device, torch.bfloat16, layer.hidden_dim, layer.top_k)
    upstream = [torch.ones(tokens.shape, device=device, dtype=torch.bfloat16) for tokens, *_ in receives]

    def compute_every_receive():
        for index, ((tokens, experts, weights, counts), grad) in enumerate(zip(receives, upstream, strict=True)):
            with record_function(f"{RECEIVE_RANGE}{index}"):
                out = layer._compute_experts(tokens, experts, weights, torch.bfloat16, recv_counts=counts)
                torch.autograd.backward(out, grad)

    loads, kernels = profiled_launches(compute_every_receive)
    return f"{family} {layout} {variant}", loads, {name for name in kernels if is_cutlass_grouped_gemm(name)}


def grouped_gemm_lazy_load_findings(device, case: int, order: str) -> list[str]:
    """:func:`profiled_gemm_region` of ``case`` in ``order``. A grouped-GEMM kernel CUDA loads in the region is a
    ``grouped GEMM`` finding; any other kernel is an ``expert-bias GEMM`` one where it loads inside
    :data:`EXPERT_BIAS_BACKWARD`, a ``receive size`` one where it loads past the first receive (its variant follows the
    receive) and a ``first receive`` one where that receive loads it (a step-1 load). A region that ran fewer than
    :data:`SWEPT_GROUPED_GEMM_KERNELS` CUTLASS grouped-GEMM kernels, or not both tiles by name, is a ``sweep``
    finding."""
    label, loads, ran = profiled_gemm_region(device, case, order)
    findings = []
    for name, index, in_bias_backward in loads:
        if is_grouped_gemm_kernel(name):
            kind = "grouped GEMM"
        elif in_bias_backward:
            kind = "expert-bias GEMM"
        elif index == 0:
            kind = "first receive"
        else:
            kind = "receive size"
        where = "outside every receive" if index is None else f"in receive {index}"
        findings.append(f"{kind}: {label}, {order}: CUDA loaded {name} {where} past the warm-up")
    tiles = {grouped_gemm_tile(name) for name in ran}
    if len(ran) < SWEPT_GROUPED_GEMM_KERNELS or tiles != {"small", "large"}:
        findings.append(
            f"sweep: {label}, {order}: the region ran {len(ran)} grouped-GEMM kernels, tiles {tiles}: {sorted(ran)}"
        )
    return findings


def grouped_gemm_control_findings(device, case: int, order: str) -> list[str]:
    """:func:`profiled_gemm_region` of ``case`` with the GEMM warm-up stubbed out (by the caller), so every CUTLASS
    grouped-GEMM kernel the region runs is cold: each one no load is named after is a finding, and so is a region that
    ran none. CUTLASS loads its kernel before launching it, so this checks the naming of such a load."""
    label, loads, ran = profiled_gemm_region(device, case, order)
    if not ran:
        return [f"sweep: {label}: the region ran no CUTLASS grouped-GEMM kernel"]
    loaded = {name for name, *_ in loads}
    return [f"unreported: {label}: {name} ran cold, and no load is named after it" for name in sorted(ran - loaded)]


def grouped_gemm_tile_control_findings(device, case: int, order: str) -> list[str]:
    """:func:`profiled_gemm_region` of ``case`` with the GEMM warm-up at one row per expert only (by the caller), whose
    tiles are all small: each CUTLASS grouped-GEMM kernel loaded in the region that :func:`grouped_gemm_tile` does not
    read as large is a finding, and so is a region that loaded none."""
    label, loads, _ = profiled_gemm_region(device, case, order)
    cutlass = [name for name, *_ in loads if is_cutlass_grouped_gemm(name)]
    if not cutlass:
        return [f"unreported: {label}: no CUTLASS grouped-GEMM kernel loaded past a one-row warm-up"]
    return [
        f"tile: {label}: {name} reads as {grouped_gemm_tile(name)}"
        for name in cutlass
        if grouped_gemm_tile(name) != "large"
    ]


def grouped_gemm_probe(case: int, order: str, *, control: str | None = None) -> list[str]:
    """:func:`grouped_gemm_lazy_load_findings` of ``case`` in ``order``, or under a :data:`GEMM_CONTROLS` ``control``
    its findings function, in a fresh interpreter."""
    setup, findings = GEMM_CONTROLS[control] if control else ("", "grouped_gemm_lazy_load_findings")
    probe = _GEMM_LAZY_LOAD_PROBE.format(
        setup=setup,
        marker=_LAZY_LOAD_MARKER,
        findings=findings,
        index=torch.cuda.current_device(),
        case=case,
        order=order,
    )
    return probe_findings(probe, _LAZY_LOAD_MARKER)


def check_grouped_gemm_lazy_loads() -> dict[str, bool]:
    """:func:`grouped_gemm_probe` of every case in every order, and both controls on the first. The ``first receive``
    findings are logged, not checked: whatever its size, the first receive loads them, so every rank loads them in its
    first step. So are the ``expert-bias GEMM`` ones, the known limit: cuBLAS picks that GEMM's kernel by the receive's
    length."""
    findings = [
        finding
        for case in range(len(GEMM_LAZY_LOAD_CASES))
        for order in SWEEP_ORDERS
        for finding in grouped_gemm_probe(case, order)
    ]
    no_warm_up = grouped_gemm_probe(0, SWEEP_ORDERS[0], control="no_gemm_warm_up")
    one_row = grouped_gemm_probe(0, SWEEP_ORDERS[0], control="one_row_warm_up")
    step_one = [finding for finding in findings if finding.startswith("first receive")]
    expert_bias = [finding for finding in findings if finding.startswith("expert-bias GEMM")]
    log(f"[expert compute lazy loads] {[finding for finding in findings if finding not in step_one + expert_bias]}")
    log(f"[expert compute lazy loads in the first receive, step-1 loads] {step_one}")
    log(f"[expert-bias one-hot GEMM lazy loads, the known limit] {expert_bias}")
    log(f"[grouped GEMM lazy loads, GEMM warm-up stubbed out] {no_warm_up}")
    log(f"[grouped GEMM lazy loads, GEMM warm-up at one row per expert] {one_row}")
    return {
        "grouped_gemm_lazy_loads_are_reported": not no_warm_up,
        "grouped_gemm_tile_switch_loads_read_as_large": not one_row,
        "grouped_gemm_sweeps_run_both_tiles": not any(finding.startswith("sweep") for finding in findings),
        "grouped_gemm_loads_no_kernel_past_the_warm_up": not any(
            finding.startswith("grouped GEMM") for finding in findings
        ),
        "expert_compute_loads_no_kernel_past_its_first_receive": not any(
            finding.startswith("receive size") for finding in findings
        ),
    }


def check_loop(device) -> dict[str, bool]:
    """A real EP layer's per-expert loop, marked ``ep_size=2``: sync-free off the counts, bit-identical to the
    ``torch.where`` loop, which trips the detector.

    Each token routes to distinct experts, as a router's top-k picks them. The sync check runs in the default mode;
    the bitwise comparison runs under deterministic algorithms, outside which CUDA's ``index_add_`` adds by atomics in
    a run-dependent order, so it compares the slot reads and not that order. The reference must reproduce itself
    there first."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[LOOP_FAMILY]).to(device, torch.bfloat16)
    apply_ep_layers(model, single_process_ep_config(use_grouped_gemm=False))
    layer = named_ep_layers(model)[0][1]
    layer.ep_size = 2
    # Rank 0 of two: slots routed to the other half of the experts arrive as -1.
    experts = received_expert_ids(NUM_TOKENS, TOP_K, layer.experts_per_rank, layer.ep_size, seed=0).to(device)
    tokens = torch.randn(experts.shape[0], layer.hidden_dim, device=device, dtype=torch.bfloat16)
    weights = torch.rand(experts.shape, device=device, dtype=torch.float32)
    counts = receive_counts(experts, layer.experts_per_rank)
    counted = partial(layer._compute_experts, tokens, experts, weights, torch.bfloat16, recv_counts=counts)
    reference = partial(where_expert_loop, layer, tokens, experts, weights, torch.bfloat16)

    checks = {"loop_runs_the_loop": not layer._use_grouped_mm, "loop_every_expert_received": 0 not in counts}
    with torch.no_grad():
        torch.cuda.synchronize()
        try:
            with cuda_syncs_are_errors():
                counted()
            torch.cuda.synchronize()
            checks["loop_sync_free"] = True
        except RuntimeError as exc:
            log(f"[loop] raised {exc}")
            checks["loop_sync_free"] = False
            return checks
        with deterministic_algorithms():
            want = reference()
            checks["loop_where_loop_is_deterministic"] = torch.equal(want, reference())
            checks["loop_equals_where_loop_bitwise"] = torch.equal(counted(), want)
        checks["loop_where_loop_trips_the_detector"] = trips_the_detector(reference)
    log(f"[loop] {LOOP_FAMILY}: counts={counts}; {checks}")
    return checks


if __name__ == "__main__":
    run()
