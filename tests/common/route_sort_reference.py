"""The routing sort's oracle: receives a dispatch hands expert compute, and the counting sort's kernels vs the torch
construction. Controls are CPU only: on CUDA the count check is a device-side assert that ends the process's CUDA
context. Launch-key and backward comparisons against the warm-up run under ``TRITON_INTERPRET=1``."""

from __future__ import annotations

from typing import NamedTuple
from unittest import mock

import torch
import triton
from triton.runtime.interpreter import InterpretedFunction

from halo.kernels import moe_route_sort
from halo.kernels.moe_route_sort import (
    NONLOCAL_EXPERT_ID_MESSAGE,
    RECEIVE_COUNTS_DISAGREE,
    _argsort_routing,
    _counting_sort,
    _histogram_kernel,
    _place_kernel,
    _scan_kernel,
    warm_route_sort_kernels,
)
from tests.common.kernel_launches import KernelLaunches
from tests.common.triton_launch_keys import moved_arguments, recorded_launch_keys

OUTPUTS = ("sorted_token_idx", "sorted_weights", "sorted_expert_ids", "offs", "inv_map")
SORT_KERNELS = (_histogram_kernel, _scan_kernel, _place_kernel)


class Case(NamedTuple):
    name: str
    rows: int
    top_k: int
    local_experts: int
    # 1: no dispatch, every id local and no receive counts.
    ep_size: int
    # Rows whose every slot is another rank's; DeepEP sends none, but the kernels must pass them through.
    empty_rows: int = 0
    idle_expert: int | None = None


# The roster's EP shapes (top_k 4 to 10) and edges past them, over one to several sort blocks (1024 slots each). At
# ep_size 1, the full expert sets (256 and 512).
CASES = (
    Case("gpt_oss_ep8", 600, 4, 4, 8),
    Case("qwen3_5_ep8", 300, 8, 32, 8, empty_rows=17),
    Case("top6_ep4", 400, 6, 16, 4, idle_expert=5),
    Case("qwen3_next_ep8", 260, 10, 64, 8),
    Case("ep2_128_local", 200, 8, 128, 2, idle_expert=127),
    Case("one_local_expert", 300, 4, 1, 32),
    Case("top1_ep2", 50, 1, 3, 2),
    Case("no_local_slot", 40, 4, 4, 8, empty_rows=40),
    Case("zero_tokens", 0, 8, 32, 8),
    Case("qwen3_5_ep1", 300, 8, 256, 1),
    Case("qwen3_next_ep1", 120, 10, 512, 1),
    Case("top1_ep1", 50, 1, 3, 1, idle_expert=2),
)
# Every case's (top_k, local experts), and gpt-oss at ep4 under expert TP 2 and at ep2.
LAUNCH_SHAPES = tuple(sorted({(case.top_k, case.local_experts) for case in CASES} | {(4, 8), (4, 16)}))


class Receive(NamedTuple):
    experts: torch.Tensor
    weights: torch.Tensor
    # None at ep_size 1, where nothing is dispatched.
    counts: tuple[int, ...] | None
    upstream: torch.Tensor


def receive_counts(experts: torch.Tensor, experts_per_rank: int) -> tuple[int, ...]:
    """The per-local-expert slot counts a DeepEP dispatch reports for ``experts`` (``-1`` = not this rank's), read back
    to the host."""
    local = experts[(experts >= 0) & (experts < experts_per_rank)]
    return tuple(torch.bincount(local, minlength=experts_per_rank).tolist())


def receive(case: Case, device, seed: int = 0) -> Receive:
    """``case``'s ``[rows, top_k]`` local ids, fp32 gate weights, host-side counts and an upstream gradient, built on
    the host and moved. One slot of each fourth row holds an id past the local range, which the sort drops too."""
    g = torch.Generator().manual_seed(seed)
    picks = torch.rand(case.rows, case.local_experts * case.ep_size, generator=g).argsort(dim=1)[:, : case.top_k]
    if case.idle_expert is not None:
        spare = case.local_experts if case.ep_size > 1 else 0
        picks = torch.where(picks == case.idle_expert, spare, picks)
    experts = torch.where(picks < case.local_experts, picks, -1)
    experts[: case.empty_rows] = -1
    experts[1::4, 0] = torch.where(experts[1::4, 0] < 0, case.local_experts, experts[1::4, 0])
    weights = torch.rand(experts.shape, generator=g)
    counts = receive_counts(experts, case.local_experts) if case.ep_size > 1 else None
    upstream = torch.randn(experts.numel() if counts is None else sum(counts), generator=g)
    return Receive(experts.to(device), weights.to(device), counts, upstream.to(device))


def sort_with_grad(sort, case: Case, inputs: Receive) -> tuple[list[torch.Tensor], torch.Tensor]:
    """``sort``'s five outputs on ``inputs`` and the gate-weight gradient ``inputs.upstream`` gives."""
    leaf = inputs.weights.clone().requires_grad_(True)
    outputs = sort(inputs.experts, leaf, case.local_experts, inputs.counts)
    (grad,) = torch.autograd.grad(outputs[1], leaf, inputs.upstream)
    return [t.detach() for t in outputs], grad


def _differs(got: torch.Tensor, want: torch.Tensor) -> bool:
    if got.dtype != want.dtype or got.shape != want.shape:
        return True
    if got.dtype == torch.float32:
        got, want = got.view(torch.int32), want.view(torch.int32)
    return not torch.equal(got, want)


def result_mismatches(case: Case, got, want) -> list[str]:
    """The outputs and gradient of two :func:`sort_with_grad` results that differ in any bit."""
    (got_outputs, got_grad), (want_outputs, want_grad) = got, want
    names = [name for name, g, w in zip(OUTPUTS, got_outputs, want_outputs, strict=True) if _differs(g, w)]
    if _differs(got_grad, want_grad):
        names.append("gate-weight gradient")
    return [f"{case.name}: {name}" for name in names]


def case_findings(case: Case, inputs: Receive) -> list[str]:
    """What ``case`` claims to exercise and its receive does not."""
    findings = []
    if case.idle_expert is not None and bool((inputs.experts == case.idle_expert).any()):
        findings.append(f"{case.name}: expert {case.idle_expert} is not idle")
    local = (inputs.experts >= 0) & (inputs.experts < case.local_experts)
    if case.ep_size == 1 and not bool(local.all()):
        findings.append(f"{case.name}: an id is not local at ep_size 1")
    if case.ep_size > 1 and case.rows and not bool((inputs.experts >= case.local_experts).any()):
        findings.append(f"{case.name}: no out-of-range id")
    empty = inputs.experts[: case.empty_rows]
    if bool(((empty >= 0) & (empty < case.local_experts)).any()):
        findings.append(f"{case.name}: a row meant empty holds a local slot")
    return findings


def counting_sort_mismatches(device) -> list[str]:
    """Every :data:`CASES` output and gradient where the counting sort differs from torch's, and each case that
    exercised less than it claims."""
    findings = []
    for index, case in enumerate(CASES):
        inputs = receive(case, device, seed=index)
        want = sort_with_grad(_argsort_routing, case, inputs)
        findings += case_findings(case, inputs) + result_mismatches(
            case, sort_with_grad(_counting_sort, case, inputs), want
        )
    return findings


def control_findings() -> list[str]:
    """The CPU controls, each named when it fails: counts a slot off raise, a count moved between two experts passes
    (the sort reads only the sum). At ``ep_size`` 1 a stray id raises in both constructions; a swapped entry and a
    one-bit gradient change are both reported."""
    findings = []
    case = CASES[1]
    inputs = receive(case, "cpu")
    moved = (inputs.counts[0] - 1, inputs.counts[1] + 1, *inputs.counts[2:])
    moved_inputs = inputs._replace(counts=moved)
    findings += [
        f"moved count: {finding}"
        for finding in result_mismatches(
            case, sort_with_grad(_counting_sort, case, moved_inputs), sort_with_grad(_argsort_routing, case, inputs)
        )
    ]
    short = [inputs.counts[0] - 1, *inputs.counts[1:]]
    over = [inputs.counts[0] + 1, *inputs.counts[1:]]
    ep1_case = next(case for case in CASES if case.ep_size == 1)
    ep1_experts = receive(ep1_case, "cpu").experts
    past, padded = ep1_experts.clone(), ep1_experts.clone()
    past[0, 0], padded[0, 0] = ep1_case.local_experts, -1
    refused = [
        (f"count one too {change}", _counting_sort, inputs.experts, case.local_experts, counts)
        for change, counts in (("few", tuple(short)), ("many", tuple(over)))
    ]
    refused += [
        (f"{name} at ep_size 1, {sort.__name__}", sort, experts, ep1_case.local_experts, None)
        for name, experts in (("an id past the local range", past), ("a -1", padded))
        for sort in (_counting_sort, _argsort_routing)
    ]
    for name, sort, experts, local_experts, counts in refused:
        message = NONLOCAL_EXPERT_ID_MESSAGE if counts is None else RECEIVE_COUNTS_DISAGREE
        try:
            sort(experts, torch.zeros(experts.shape), local_experts, counts)
            findings.append(f"{name} was not refused")
        except RuntimeError as exc:
            if message not in str(exc):
                findings.append(f"{name} raised something else: {exc}")
    outputs, grad = sort_with_grad(_counting_sort, case, inputs)
    same_expert = (outputs[2] == outputs[2][0]).nonzero().flatten()[:2]
    swapped = [t.clone() for t in outputs]
    swapped[0][same_expert] = outputs[0][same_expert.flip(0)]
    if not result_mismatches(case, (swapped, grad), (outputs, grad)):
        findings.append("a swap within one expert went unreported")
    nudged = grad.clone()
    flat = nudged.view(-1).view(torch.int32)
    flat[flat.nonzero()[0]] ^= 1
    if not result_mismatches(case, (outputs, nudged), (outputs, grad)):
        findings.append("a one-bit gradient change went unreported")
    return findings


def swept_receives(top_k: int, local_experts: int) -> list[tuple[int, int, Receive]]:
    """``(rows, local slots, receive)`` at the row, block and slot counts Triton would class apart if it specialized on
    them."""
    rows_per_block = moe_route_sort._BLOCK_SLOTS // triton.next_power_of_2(top_k)
    many = 16 * rows_per_block
    receives = []
    for rows, n_valid in ((0, 0), (1, 1), (17, 1), (16, 16), (40, 0), (many, many), (many + 1, 17)):
        experts = torch.full((rows, top_k), -1, dtype=torch.long)
        slot = torch.arange(n_valid)
        experts[slot % max(rows, 1), slot // max(rows, 1)] = slot % local_experts
        counts = receive_counts(experts, local_experts)
        receives.append((rows, n_valid, Receive(experts, torch.zeros(experts.shape), counts, torch.zeros(n_valid))))
    for rows in (1, 16, 17, many + 1):
        experts = torch.arange(rows * top_k).view(rows, top_k) % local_experts
        receives.append(
            (rows, rows * top_k, Receive(experts, torch.zeros(experts.shape), None, torch.zeros(experts.numel())))
        )
    return receives


def launch_key_findings(shapes: tuple[tuple[int, int], ...] = LAUNCH_SHAPES) -> list[str]:
    """Each kernel launch whose Triton specialization differs from the warm-up's at the same shape, which would compile
    mid-step between dispatch and combine. Under ``TRITON_INTERPRET=1``. Kernels are read off the module, so a probe
    may rebind one first."""
    kernels = [getattr(moe_route_sort, kernel.__name__) for kernel in SORT_KERNELS]
    findings = []
    with (
        recorded_launch_keys(kernels) as launches,
        mock.patch.object(moe_route_sort, "sort_routing_by_expert", _counting_sort),
    ):
        for top_k, local_experts in shapes:
            launches.clear()
            warm_route_sort_kernels(top_k, local_experts, torch.device("cpu"))
            warm = {name: key for name, _, key in launches}
            for rows, n_valid, inputs in swept_receives(top_k, local_experts):
                launches.clear()
                _counting_sort(inputs.experts, inputs.weights, local_experts, inputs.counts)
                for name, names, key in launches:
                    moved = moved_arguments(names, key, warm[name])
                    if moved:
                        findings.append(
                            f"{name} at top_k {top_k}, {local_experts} local experts, {rows} rows, {n_valid} local "
                            f"slots: specialized apart from the warm-up on {', '.join(moved)}"
                        )
    return findings


def backward_kernel_findings(shapes: tuple[tuple[int, int], ...] = LAUNCH_SHAPES) -> list[str]:
    """Each kernel the backward launches that the warm-up did not, which CUDA would load mid-step between dispatch and
    combine. Under ``TRITON_INTERPRET=1``."""
    if not isinstance(moe_route_sort._place_kernel, InterpretedFunction):
        raise RuntimeError("backward_kernel_findings runs the kernels under TRITON_INTERPRET=1")
    findings = []
    with mock.patch.object(moe_route_sort, "sort_routing_by_expert", _counting_sort):
        for top_k, local_experts in shapes:
            with KernelLaunches() as warm:
                warm_route_sort_kernels(top_k, local_experts, torch.device("cpu"))
            for rows, n_valid, inputs in swept_receives(top_k, local_experts):
                leaf = inputs.weights.clone().requires_grad_(True)
                sorted_weights = _counting_sort(inputs.experts, leaf, local_experts, inputs.counts)[1]
                with KernelLaunches() as backward:
                    torch.autograd.grad(sorted_weights, leaf, inputs.upstream)
                findings += [
                    f"top_k {top_k}, {local_experts} local experts, {rows} rows, {n_valid} local slots: the backward "
                    f"launches {kernel}, which the warm-up did not"
                    for kernel in sorted(backward.kernels - warm.kernels, key=str)
                ]
    return findings


def compiled_sort_binaries() -> int:
    return sum(len(kernel.device_caches[torch.cuda.current_device()][0]) for kernel in SORT_KERNELS)


def pinned_host_requests() -> int:
    """The pinned host blocks this process has been handed so far, from the cache or newly page-locked."""
    return torch.cuda.host_memory_stats()["active_requests.allocated"]
