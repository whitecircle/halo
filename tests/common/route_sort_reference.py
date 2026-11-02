"""The routing sort's oracle: the receives a dispatch hands the expert compute, the counting sort's kernels against the
torch construction they replace (:func:`counting_sort_mismatches`, on CUDA or under ``TRITON_INTERPRET=1`` on CPU),
that comparison's and the count check's negative controls (:func:`control_findings`, CPU only: on CUDA the check is a
device-side assert, which ends the process's CUDA context), the Triton specialization of every launch against the
warm-up's (:func:`launch_key_findings`, under ``TRITON_INTERPRET=1``), and the counts of the kernels' compiled binaries
and of the pinned host blocks handed out.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import NamedTuple
from unittest import mock

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import make_backend
from triton.runtime.interpreter import InterpretedFunction
from triton.runtime.jit import JITFunction, create_function_from_signature

from halo.kernels import moe_route_sort
from halo.kernels.moe_route_sort import (
    RECEIVE_COUNTS_DISAGREE,
    _argsort_routing,
    _counting_sort,
    _histogram_kernel,
    _place_kernel,
    _scan_kernel,
    warm_route_sort_kernels,
)

OUTPUTS = ("sorted_token_idx", "sorted_weights", "sorted_expert_ids", "offs", "inv_map")
SORT_KERNELS = (_histogram_kernel, _scan_kernel, _place_kernel)
# Specialization is the backend's rule, the same on every CUDA arch.
_KEY_TARGET = GPUTarget("cuda", 100, 32)


class Case(NamedTuple):
    name: str
    rows: int
    top_k: int
    local_experts: int
    ep_size: int
    # Rows whose every slot is another rank's; DeepEP sends none, the kernels must still pass them through.
    empty_rows: int = 0
    # A local expert no slot routes to.
    idle_expert: int | None = None


# The roster's EP shapes (top_k 4 to 10) and the edges past them (top_k 1, one local expert, no local slot, no
# token), over one to several sort blocks (1024 slots each).
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
)
# Every case's (top_k, local experts), and gpt-oss at ep4 under expert TP 2 and at ep2.
LAUNCH_SHAPES = tuple(sorted({(case.top_k, case.local_experts) for case in CASES} | {(4, 8), (4, 16)}))


class Receive(NamedTuple):
    experts: torch.Tensor
    weights: torch.Tensor
    counts: tuple[int, ...]
    # The gradient arriving at ``sorted_weights``.
    upstream: torch.Tensor


def receive_counts(experts: torch.Tensor, experts_per_rank: int) -> tuple[int, ...]:
    """The per-local-expert slot counts a DeepEP dispatch reports for the received ids ``experts`` (``-1`` = not
    this rank's), read back to the host."""
    local = experts[(experts >= 0) & (experts < experts_per_rank)]
    return tuple(torch.bincount(local, minlength=experts_per_rank).tolist())


def receive(case: Case, device, seed: int = 0) -> Receive:
    """``case``'s ``[rows, top_k]`` local ids (``-1`` = another rank's; one slot of each fourth row holds an id past
    the local range, which the sort drops too), fp32 gate weights, host-side receive counts and an upstream
    gradient, built on the host and moved to ``device``."""
    g = torch.Generator().manual_seed(seed)
    picks = torch.rand(case.rows, case.local_experts * case.ep_size, generator=g).argsort(dim=1)[:, : case.top_k]
    if case.idle_expert is not None:
        picks = torch.where(picks == case.idle_expert, case.local_experts, picks)
    experts = torch.where(picks < case.local_experts, picks, -1)
    experts[: case.empty_rows] = -1
    experts[1::4, 0] = torch.where(experts[1::4, 0] < 0, case.local_experts, experts[1::4, 0])
    weights = torch.rand(experts.shape, generator=g)
    counts = receive_counts(experts, case.local_experts)
    upstream = torch.randn(sum(counts), generator=g)
    return Receive(experts.to(device), weights.to(device), counts, upstream.to(device))


def sort_with_grad(sort, case: Case, inputs: Receive) -> tuple[list[torch.Tensor], torch.Tensor]:
    """``sort``'s five outputs on ``inputs`` and the gate-weight gradient that ``inputs.upstream`` gives."""
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
    """The outputs, and the gradient, of two :func:`sort_with_grad` results that differ in any bit."""
    (got_outputs, got_grad), (want_outputs, want_grad) = got, want
    names = [name for name, g, w in zip(OUTPUTS, got_outputs, want_outputs, strict=True) if _differs(g, w)]
    if _differs(got_grad, want_grad):
        names.append("gate-weight gradient")
    return [f"{case.name}: {name}" for name in names]


def case_findings(case: Case, inputs: Receive) -> list[str]:
    """What ``case`` claims to exercise and its receive does not."""
    findings = []
    if case.idle_expert is not None and inputs.counts[case.idle_expert] != 0:
        findings.append(f"{case.name}: expert {case.idle_expert} is not idle")
    if case.rows and not bool((inputs.experts >= case.local_experts).any()):
        findings.append(f"{case.name}: no out-of-range id")
    empty = inputs.experts[: case.empty_rows]
    if bool(((empty >= 0) & (empty < case.local_experts)).any()):
        findings.append(f"{case.name}: a row meant empty holds a local slot")
    return findings


def counting_sort_mismatches(device) -> list[str]:
    """Every :data:`CASES` output and gate-weight gradient where the counting sort differs from the torch
    construction, and each case that exercised less than it claims."""
    findings = []
    for index, case in enumerate(CASES):
        inputs = receive(case, device, seed=index)
        want = sort_with_grad(_argsort_routing, case, inputs)
        findings += case_findings(case, inputs) + result_mismatches(
            case, sort_with_grad(_counting_sort, case, inputs), want
        )
    return findings


def control_findings() -> list[str]:
    """The controls on CPU, each named when it does not hold: counts one slot short and one slot over raise the count
    check, while a count moved between two experts passes it and sorts as the torch construction does, since the sort
    reads only their sum; a swap of two of one expert's sorted entries, and a one-bit change of one gradient element,
    are both reported by the comparison."""
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
    for name, bad in (("count one too few", short), ("count one too many", over)):
        try:
            _counting_sort(inputs.experts, inputs.weights, case.local_experts, tuple(bad))
            findings.append(f"{name} was not refused")
        except RuntimeError as exc:
            if RECEIVE_COUNTS_DISAGREE not in str(exc):
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
    """``(rows, local slots, receive)`` at the row, block and slot counts Triton would class apart if it specialized
    on them (0, 1, a multiple of 16, neither): one block, sixteen and seventeen, ``local slots`` dealt row by row,
    expert after expert, the rest another rank's."""
    rows_per_block = moe_route_sort._BLOCK_SLOTS // triton.next_power_of_2(top_k)
    many = 16 * rows_per_block
    receives = []
    for rows, n_valid in ((0, 0), (1, 1), (17, 1), (16, 16), (40, 0), (many, many), (many + 1, 17)):
        experts = torch.full((rows, top_k), -1, dtype=torch.long)
        slot = torch.arange(n_valid)
        experts[slot % max(rows, 1), slot // max(rows, 1)] = slot % local_experts
        counts = receive_counts(experts, local_experts)
        receives.append((rows, n_valid, Receive(experts, torch.zeros(experts.shape), counts, torch.zeros(n_valid))))
    return receives


def launch_key_findings(shapes: tuple[tuple[int, int], ...] = LAUNCH_SHAPES) -> list[str]:
    """Each launch of the counting sort's kernels, over :func:`swept_receives` at every ``(top_k, local experts)`` of
    ``shapes``, whose Triton specialization differs from the layer warm-up's at the same shape, named by the arguments
    that moved: such a launch would compile its binary mid-step, between a dispatch and its combine.

    Under ``TRITON_INTERPRET=1`` the kernels run on CPU, the warm-up's routed through the counting sort, and each
    launch's key is what the compiled kernel's binder makes of the same arguments. The kernels are read off the module,
    so a probe may rebind one first."""
    kernels = [getattr(moe_route_sort, kernel.__name__) for kernel in SORT_KERNELS]
    if not all(isinstance(kernel, InterpretedFunction) for kernel in kernels):
        raise RuntimeError("launch_key_findings runs the kernels under TRITON_INTERPRET=1")
    backend = make_backend(_KEY_TARGET)
    launches = []

    def recorder(kernel):
        compiled = JITFunction(kernel.fn, **kernel.kwargs)
        bind = create_function_from_signature(compiled.signature, compiled.params, backend)
        names = [param.name for param in compiled.params]
        run = kernel.run

        def record(*args, grid, warmup, **kwargs):
            launches.append((kernel.__name__, names, bind(*args, **kwargs)[1]))
            return run(*args, grid=grid, warmup=warmup, **kwargs)

        return record

    findings = []
    with ExitStack() as stack:
        for kernel in kernels:
            stack.enter_context(mock.patch.object(kernel, "run", recorder(kernel)))
        stack.enter_context(mock.patch.object(moe_route_sort, "sort_routing_by_expert", _counting_sort))
        for top_k, local_experts in shapes:
            launches.clear()
            warm_route_sort_kernels(top_k, local_experts, torch.device("cpu"))
            warm = {name: key for name, _, key in launches}
            for rows, n_valid, inputs in swept_receives(top_k, local_experts):
                launches.clear()
                _counting_sort(inputs.experts, inputs.weights, local_experts, inputs.counts)
                for name, names, key in launches:
                    moved = [arg for arg, a, b in zip(names, key, warm[name], strict=True) if a != b]
                    if moved:
                        findings.append(
                            f"{name} at top_k {top_k}, {local_experts} local experts, {rows} rows, {n_valid} local "
                            f"slots: specialized apart from the warm-up on {', '.join(moved)}"
                        )
    return findings


def compiled_sort_binaries() -> int:
    """The binaries :data:`SORT_KERNELS` hold compiled for the current CUDA device."""
    return sum(len(kernel.device_caches[torch.cuda.current_device()][0]) for kernel in SORT_KERNELS)


def pinned_host_requests() -> int:
    """The pinned host blocks this process has been handed so far, from the cache or newly page-locked."""
    return torch.cuda.host_memory_stats()["active_requests.allocated"]
