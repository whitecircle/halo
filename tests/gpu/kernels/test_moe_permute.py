#!/usr/bin/env python
"""Fused MoE permute/unpermute kernels match an fp64 ``index_add_`` reference, forward and backward.

The kernels walk ``inv_map`` (sentinel-padded sorted positions per token) instead of padding and
materializing a ``[recv_N, top_k, H]`` gather, so the cases cover tokens routed to fewer than ``top_k``
local experts (sentinel slots), tokens routed to none, a hidden size that is not a multiple of the
column tile, strided inputs, routing weights that need no gradient, and an empty dispatch.

Run: torchrun --nproc_per_node=1 tests/gpu/kernels/test_moe_permute.py
"""

import torch

from src.kernels import moe_permute
from src.kernels.moe_permute import MoEWeightedUnpermute, build_inv_map, gather_reduce_rows
from tests.common.harness import gpu_test_main, record_check
from tests.common.utils import fro_rel_err, max_abs_rel_err

DTYPE_TOLS = ((torch.float32, 1e-5), (torch.bfloat16, 2e-2))
# A bf16 result accumulated in fp32 is the exact result rounded once; a partial sum or a weighted row rounded to
# bf16 on the way lands 1.4x-1.7x that floor at the dense routings (less at 10% fill, where most tokens sum one row).
BF16_FLOOR_RATIO_MAX = 1.1
# (tokens, top_k, hidden, fraction of the top_k slots that hold a row)
ROUTINGS = ((517, 8, 2816, 0.5), (64, 4, 100, 1.0), (33, 8, 1025, 0.1))


def _routing(n_tokens: int, top_k: int, n_sorted: int, generator: torch.Generator):
    """Sorted-row → token map with at most ``top_k`` rows per token, and its ``inv_map``."""
    slots = torch.arange(n_tokens, device="cuda").repeat_interleave(top_k)
    token_idx = slots[torch.randperm(slots.numel(), generator=generator, device="cuda")[:n_sorted]]
    return token_idx, build_inv_map(token_idx, n_tokens, top_k)


def _check_against_index_add(expert_out, weights, token_idx, inv_map, n_tokens, grad, tol):
    out = MoEWeightedUnpermute.apply(expert_out, weights, token_idx, inv_map)
    out.backward(grad)
    ref_out = expert_out.detach().double().requires_grad_(True)
    ref_w = weights.detach().double().requires_grad_(True)
    expected = torch.zeros(n_tokens, expert_out.shape[1], dtype=torch.float64, device="cuda")
    expected = expected.index_add(0, token_idx, ref_out * ref_w.unsqueeze(-1))
    expected.backward(grad.double())
    for got, want in ((out, expected.detach()), (expert_out.grad, ref_out.grad), (weights.grad, ref_w.grad)):
        assert max_abs_rel_err(got, want) < tol
        if got.dtype == torch.bfloat16:
            floor = fro_rel_err(want.to(torch.bfloat16), want)
            assert fro_rel_err(got, want) <= BF16_FLOOR_RATIO_MAX * floor, (fro_rel_err(got, want), floor)


def test_weighted_unpermute_matches_index_add(dtype, tol, n_tokens, top_k, hidden, fill):
    generator = torch.Generator(device="cuda").manual_seed(n_tokens)
    n_sorted = int(n_tokens * top_k * fill)
    token_idx, inv_map = _routing(n_tokens, top_k, n_sorted, generator)
    expert_out = torch.randn(n_sorted, hidden, generator=generator, device="cuda", dtype=dtype).requires_grad_(True)
    weights = torch.rand(n_sorted, generator=generator, device="cuda", dtype=dtype).requires_grad_(True)
    grad = torch.randn(n_tokens, hidden, generator=generator, device="cuda", dtype=dtype)
    _check_against_index_add(expert_out, weights, token_idx, inv_map, n_tokens, grad, tol)


def test_weighted_unpermute_takes_strided_inputs():
    """Column slices of wider buffers (a row stride larger than the width): the backward kernel indexes
    the saved tensors with unit strides, so it must receive contiguous copies."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    n_tokens, top_k, hidden = 257, 8, 704
    n_sorted = n_tokens * top_k // 2
    token_idx, inv_map = _routing(n_tokens, top_k, n_sorted, generator)
    wide_out = torch.randn(n_sorted, 2 * hidden, generator=generator, device="cuda").requires_grad_(True)
    wide_w = torch.rand(n_sorted, 2, generator=generator, device="cuda").requires_grad_(True)
    expert_out, weights = wide_out[:, :hidden], wide_w[:, 0]
    expert_out.retain_grad()
    weights.retain_grad()
    assert not expert_out.is_contiguous() and not weights.is_contiguous()
    grad = torch.randn(n_tokens, hidden, generator=generator, device="cuda")
    _check_against_index_add(expert_out, weights, token_idx, inv_map, n_tokens, grad, 1e-5)


def test_gather_reduce_sums_only_real_rows():
    """Sentinel slots contribute nothing and a token with no rows comes back zero."""
    src = torch.arange(1, 7, device="cuda", dtype=torch.float32).unsqueeze(-1).expand(6, 3).contiguous()
    inv_map = torch.tensor([[0, 3], [5, 6], [6, 6]], device="cuda")  # 6 == len(src): sentinel
    out = gather_reduce_rows(src, inv_map)
    torch.testing.assert_close(out[:, 0], torch.tensor([1.0 + 4.0, 6.0, 0.0], device="cuda"))


def test_gather_reduce_takes_a_strided_weight():
    """``gather_reduce_rows`` is public and its kernel indexes the weight with a unit stride, so a strided
    weight (a column of a wider buffer) must reach it as a contiguous copy, not be read at the wrong rows."""
    generator = torch.Generator(device="cuda").manual_seed(2)
    n_tokens, top_k, hidden = 65, 4, 300
    n_sorted = n_tokens * top_k // 2
    token_idx, inv_map = _routing(n_tokens, top_k, n_sorted, generator)
    src = torch.randn(n_sorted, hidden, generator=generator, device="cuda")
    weight = torch.rand(n_sorted, 3, generator=generator, device="cuda")[:, 1]
    assert not weight.is_contiguous()
    expected = torch.zeros(n_tokens, hidden, dtype=torch.float64, device="cuda")
    expected = expected.index_add(0, token_idx, src.double() * weight.double().unsqueeze(-1))
    assert max_abs_rel_err(gather_reduce_rows(src, inv_map, weight), expected) < 1e-5


def test_empty_dispatch():
    inv_map = torch.zeros(0, 8, dtype=torch.long, device="cuda")
    out = MoEWeightedUnpermute.apply(
        torch.zeros(0, 16, device="cuda"),
        torch.zeros(0, device="cuda"),
        torch.zeros(0, dtype=torch.long, device="cuda"),
        inv_map,
    )
    assert out.shape == (0, 16)


def test_weights_needing_no_grad_keep_no_expert_outputs(dtype):
    """With weights that need no gradient the backward neither keeps the expert outputs nor reads them,
    writes the same expert-output gradient bit for bit, and runs the binary the weight-gradient case
    compiled (the EP warm-up compiles that one alone)."""
    kernel = moe_permute._weighted_unpermute_bwd_kernel
    generator = torch.Generator(device="cuda").manual_seed(1)
    n_tokens, top_k, hidden = 129, 8, 1100
    n_sorted = n_tokens * top_k // 2
    token_idx, inv_map = _routing(n_tokens, top_k, n_sorted, generator)
    expert_out = torch.randn(n_sorted, hidden, generator=generator, device="cuda", dtype=dtype)
    weights = torch.rand(n_sorted, generator=generator, device="cuda", dtype=dtype)
    grad = torch.randn(n_tokens, hidden, generator=generator, device="cuda", dtype=dtype)
    runs = {}
    for weights_need_grad in (True, False):
        eo, w = expert_out.clone().requires_grad_(True), weights.clone().requires_grad_(weights_need_grad)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            out = MoEWeightedUnpermute.apply(eo, w, token_idx, inv_map)
        out.backward(grad)
        runs[weights_need_grad] = (sum(t.numel() * t.element_size() for t in saved), eo.grad)
        if weights_need_grad:
            compiled = len(kernel.device_caches[torch.cuda.current_device()][0])
    (full_bytes, full_grad), (saved_bytes, expert_grad) = runs[True], runs[False]
    assert saved_bytes == full_bytes - expert_out.numel() * expert_out.element_size()
    assert torch.equal(expert_grad, full_grad)
    assert len(kernel.device_caches[torch.cuda.current_device()][0]) == compiled, (
        "the frozen-weight backward compiled a binary of its own"
    )


def test_one_compile_serves_every_row_count():
    """The dispatch's sorted-row count changes every step; its class (1, a multiple of 16, neither) must
    not select a new gather-reduce binary, so one warm call compiles what every later count runs."""
    kernel = moe_permute._gather_reduce_kernel
    generator = torch.Generator(device="cuda").manual_seed(0)

    def run(n_tokens: int) -> None:
        token_idx, inv_map = _routing(n_tokens, 2, n_tokens * 2, generator)
        expert_out = torch.randn(n_tokens * 2, 256, device="cuda", dtype=torch.bfloat16)
        weights = torch.rand(n_tokens * 2, device="cuda", dtype=torch.bfloat16)
        MoEWeightedUnpermute.apply(expert_out, weights, token_idx, inv_map)

    kernel.device_caches.clear()  # start from an empty cache, whatever earlier checks compiled
    run(8)
    warm = len(kernel.device_caches[torch.cuda.current_device()][0])
    for n_tokens in (1, 3, 17, 24, 513):
        run(n_tokens)
    assert len(kernel.device_caches[torch.cuda.current_device()][0]) == warm, "a row count compiled a new binary"


@gpu_test_main(exact_world_size=1, prefix="moe_permute", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    for dtype, tol in DTYPE_TOLS:
        for n_tokens, top_k, hidden, fill in ROUTINGS:
            record_check(
                checks,
                f"weighted_unpermute_matches_index_add[{dtype}-{n_tokens}x{top_k}x{hidden}]",
                lambda: test_weighted_unpermute_matches_index_add(dtype, tol, n_tokens, top_k, hidden, fill),
            )
    record_check(checks, "weighted_unpermute_takes_strided_inputs", test_weighted_unpermute_takes_strided_inputs)
    for dtype, _ in DTYPE_TOLS:
        record_check(
            checks,
            f"weights_needing_no_grad_keep_no_expert_outputs[{dtype}]",
            lambda: test_weights_needing_no_grad_keep_no_expert_outputs(dtype),
        )
    record_check(checks, "gather_reduce_sums_only_real_rows", test_gather_reduce_sums_only_real_rows)
    record_check(checks, "gather_reduce_takes_a_strided_weight", test_gather_reduce_takes_a_strided_weight)
    record_check(checks, "empty_dispatch", test_empty_dispatch)
    record_check(checks, "one_compile_serves_every_row_count", test_one_compile_serves_every_row_count)
    return {"checks": checks}


if __name__ == "__main__":
    run()
