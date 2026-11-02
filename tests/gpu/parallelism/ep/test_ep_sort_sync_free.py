#!/usr/bin/env python
"""The grouped-GEMM token sort must not synchronize the host.

A host sync per MoE layer per forward drains the launch queue and stops the CPU running ahead of the
GPU; at 60 MoE layers that is 60 stalls per forward. The offending ops are the ones whose OUTPUT SHAPE
is data-dependent (``bincount``, ``unique_consecutive(return_counts=True)``, ``nonzero``) — CUDA can
only size their output by reading the device back.

``torch.cuda.set_sync_debug_mode("error")`` turns any such read into an exception, so this asserts the
property directly rather than trusting a claim no check enforces.

Scope: the whole sort, at ``ep_size == 1`` and at ``ep_size > 1``, where the dispatch's host-side receive
counts size a counting sort in three Triton launches (``halo/kernels/moe_route_sort.py``); and the per-expert
loop (``use_grouped_gemm: false``, GptOss under expert TP) at ``ep_size > 1``, which sizes each expert's slot
read off its count. Each must also equal its device-sized reference (the ``nonzero`` sort; the ``torch.where``
loop, under deterministic algorithms) bit for bit, and that reference must trip the same detector, so a
sync-free pass is not a detector that sees nothing. The counting sort also runs every case of
``tests/common/route_sort_reference.py`` (top_k 1 to 10, 1 to 128 local experts, idle experts, rows with no local
slot, a zero-token rank) against the torch construction, gate-weight gradient included, compiles no binary past
its warm-up and takes no pinned host block.

Run with 1 GPU:
    torchrun --nproc_per_node=1 \
        tests/gpu/parallelism/ep/test_ep_sort_sync_free.py
"""

import contextlib
from functools import partial

import torch

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase, named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.kernels import moe_route_sort
from halo.kernels.moe_route_sort import sort_routing_by_expert, warm_route_sort_kernels
from tests.common.ep_reference import (
    GroupedSortLayer,
    grouped_sort_mismatches,
    nonzero_grouped_sort,
    received_expert_ids,
    where_expert_loop,
)
from tests.common.harness import gpu_test_main
from tests.common.parallelism import single_process_ep_config
from tests.common.rank_log import log
from tests.common.route_sort_reference import (
    CASES,
    case_findings,
    compiled_sort_binaries,
    pinned_host_requests,
    receive,
    receive_counts,
    result_mismatches,
    sort_with_grad,
)
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

EXPERTS_PER_RANK = 8
TOP_K = 4
NUM_TOKENS = 512
HIDDEN = 256
# The family whose real EP layer the loop check runs; any family's loop reads its slots the same way.
LOOP_FAMILY = "qwen3_moe"


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
def sync_is_an_error():
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


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
    counts = receive_counts(experts, EXPERTS_PER_RANK) if ep_size > 1 else None
    torch.cuda.synchronize()
    with sync_is_an_error():
        out = EPMoELayerBase._sort_tokens_for_grouped_mm(stub, tokens, experts, weights, recv_counts=counts)
    torch.cuda.synchronize()
    return out, (stub, tokens, experts, weights)


@gpu_test_main(exact_world_size=1, prefix="ep_sort_sync_free", partial_state=False)
def run(ctx):
    checks: dict[str, bool] = {}
    device = ctx.device

    # ep_size == 1: no DeepEP padding to compact, so nothing in the sort may read the device back
    # (a sync raises out of run_sort and fails the run).
    (sorted_tokens, offs, sorted_token_idx, sorted_weights, sorted_expert_ids, _), _ = run_sort(1, device)
    log(f"[ep1] sort ran sync-free: {sorted_tokens.shape[0]} rows, offs={offs.tolist()}")

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
    checks |= check_loop(device)
    return {"checks": checks}


def trips_the_detector(fn) -> bool:
    torch.cuda.synchronize()
    try:
        with sync_is_an_error():
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
            with sync_is_an_error():
                got = sort_with_grad(sort_routing_by_expert, case, inputs)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            log(f"[counting sort] {case.name} raised {exc}")
            checks[f"{case.name}_sync_free"] = False
            continue
        pinned += pinned_host_requests() - requests
        mismatched = case_findings(case, inputs) + result_mismatches(case, got, want)
        log(f"[counting sort] {case.name}: {sum(inputs.counts)} local slots; mismatched={mismatched}")
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
            with sync_is_an_error():
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
