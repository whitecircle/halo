#!/usr/bin/env python
"""Fused MoE permute/unpermute kernels match an fp64 ``index_add_`` reference, forward and backward.

Kernels walk sentinel-padded ``inv_map`` instead of materializing ``[recv_N, top_k, H]``, so cases cover sentinel
slots, token-less rows and empty dispatch. Expert groups padded to whole GEMM segments add rows of token -1: the gather
gives them zeros and copies every other row bit for bit off a row-strided receive, and the unpermute backward gives
them no gradient, whatever the upstream gradient's first row holds.
"""

import torch

from halo.kernels import moe_permute
from halo.kernels.moe_permute import (
    MoEGatherPermute,
    MoEWeightedUnpermute,
    build_inv_map,
    gather_padded_rows,
    gather_reduce_rows,
)
from tests.common.harness import gpu_test_main, record_check
from tests.common.numerics import fro_rel_err, max_abs_rel_err

DTYPE_TOLS = ((torch.float32, 1e-5), (torch.bfloat16, 2e-2))
# A bf16 result accumulated in fp32 is the exact result rounded once; rounding a partial sum to bf16 lands 1.4x-1.7x
# that floor at dense routings.
BF16_FLOOR_RATIO_MAX = 1.1
# (tokens, top_k, hidden, fraction of the top_k slots that hold a row)
ROUTINGS = ((517, 8, 2816, 0.5), (64, 4, 100, 1.0), (33, 8, 1025, 0.1))


def _routing(n_tokens: int, top_k: int, n_sorted: int, generator: torch.Generator):
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
    """The backward kernel indexes saved tensors with unit strides, so it needs contiguous copies, and reads the
    incoming gradient at its own row stride."""
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
    grad = torch.randn(n_tokens, hidden + 64, generator=generator, device="cuda")[:, :hidden]
    assert grad.stride() == (hidden + 64, 1)
    _check_against_index_add(expert_out, weights, token_idx, inv_map, n_tokens, grad, 1e-5)


def test_one_backward_compile_serves_every_gradient_row_stride():
    """Training hands the backward a row-strided view of the padded transport buffer; a new binary would compile inside
    the dispatch-combine span."""
    kernel = moe_permute._weighted_unpermute_bwd_kernel
    generator = torch.Generator(device="cuda").manual_seed(0)
    token_idx, inv_map = _routing(64, 2, 128, generator)

    def run(row_stride: int) -> None:
        expert_out = torch.randn(128, 360, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weights = torch.rand(128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn(64, row_stride, device="cuda", dtype=torch.bfloat16)[:, :360]
        MoEWeightedUnpermute.apply(expert_out, weights, token_idx, inv_map).backward(grad)

    kernel.device_caches.clear()
    run(360)
    warm = len(kernel.device_caches[torch.cuda.current_device()][0])
    for row_stride in (384, 361, 3072):
        run(row_stride)
    assert len(kernel.device_caches[torch.cuda.current_device()][0]) == warm, "a row stride compiled a new binary"


def test_gather_reduce_sums_only_real_rows():
    src = torch.arange(1, 7, device="cuda", dtype=torch.float32).unsqueeze(-1).expand(6, 3).contiguous()
    inv_map = torch.tensor([[0, 3], [5, 6], [6, 6]], device="cuda")  # 6 == len(src): sentinel.
    out = gather_reduce_rows(src, inv_map)
    torch.testing.assert_close(out[:, 0], torch.tensor([1.0 + 4.0, 6.0, 0.0], device="cuda"))


def test_gather_reduce_takes_a_strided_weight():
    """The kernel indexes the weight with a unit stride, so a strided weight must reach it as a contiguous copy."""
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
    """The backward neither keeps nor reads expert outputs, writes the same gradient bit for bit, and runs the
    weight-gradient case's binary."""
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
    """The sorted-row count changes every step; its class (1, a multiple of 16, neither) must not select a new
    binary."""
    kernel = moe_permute._gather_reduce_kernel
    generator = torch.Generator(device="cuda").manual_seed(0)

    def run(n_tokens: int) -> None:
        token_idx, inv_map = _routing(n_tokens, 2, n_tokens * 2, generator)
        expert_out = torch.randn(n_tokens * 2, 256, device="cuda", dtype=torch.bfloat16)
        weights = torch.rand(n_tokens * 2, device="cuda", dtype=torch.bfloat16)
        MoEWeightedUnpermute.apply(expert_out, weights, token_idx, inv_map)

    kernel.device_caches.clear()  # Start from an empty cache, whatever earlier checks compiled.
    run(8)
    warm = len(kernel.device_caches[torch.cuda.current_device()][0])
    for n_tokens in (1, 3, 17, 24, 513):
        run(n_tokens)
    assert len(kernel.device_caches[torch.cuda.current_device()][0]) == warm, "a row count compiled a new binary"


def _padded_index(n_tokens: int, generator: torch.Generator) -> torch.Tensor:
    """Every token once, in a shuffled order, with a -1 padding row after every seventh."""
    order = torch.randperm(n_tokens, generator=generator, device="cuda")
    padded = torch.cat([order.view(-1, 7), order.new_full((n_tokens // 7, 1), -1)], dim=1).reshape(-1)
    assert (padded < 0).sum() == n_tokens // 7
    return padded


def test_the_padded_gather_copies_rows_bit_for_bit_and_zeros_the_padding():
    """Off a contiguous receive and off the row-strided view DeepEP's padded wire width leaves, with -0.0, inf and NaN
    among the values: a copy keeps each, where a sum with zero would lose the sign of -0.0."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    n_tokens, hidden = 7 * 300, 1100
    index = _padded_index(n_tokens, generator)
    for dtype in (torch.bfloat16, torch.float32):
        for wire in (hidden, hidden + 180):
            receive = torch.randn(n_tokens, wire, generator=generator, device="cuda", dtype=dtype)
            receive[0, :4] = torch.tensor(
                [-0.0, float("inf"), float("-inf"), float("nan")], device="cuda", dtype=dtype
            )
            tokens = receive[:, :hidden]
            out = gather_padded_rows(tokens, index)
            real = index >= 0
            bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
            assert torch.equal(out[real].view(bits), tokens[index[real]].contiguous().view(bits)), f"{dtype} {wire}"
            assert out[~real].view(bits).count_nonzero() == 0, f"{dtype} wire {wire}: a padding row is not +0.0"


def test_the_padded_permute_backward_sums_each_tokens_rows_and_skips_the_padding():
    generator = torch.Generator(device="cuda").manual_seed(0)
    n_tokens, hidden = 7 * 40, 96
    index = _padded_index(n_tokens, generator)
    tokens = torch.randn(n_tokens, hidden, generator=generator, device="cuda", requires_grad=True)
    inv_map = torch.full((n_tokens, 1), index.numel(), device="cuda")
    inv_map[index[index >= 0], 0] = (index >= 0).nonzero().squeeze(1)
    grad = torch.randn(index.numel(), hidden, generator=generator, device="cuda")
    MoEGatherPermute.apply(tokens, index, inv_map, True).backward(grad)
    expected = torch.zeros_like(tokens).index_add_(0, index[index >= 0], grad[index >= 0])
    assert torch.equal(tokens.grad, expected)


def test_the_unpermute_backward_gives_a_padding_row_no_gradient():
    """A padding row's token is -1: read as an index it is the row before the upstream gradient's first, here an inf
    row of the buffer the gradient is a view into. Its expert-output and gate-weight gradients are zero, not the NaN
    of inf times its zero weight, and every real row's equal the unpadded backward's."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    n_tokens, hidden = 7 * 40, 96
    index = _padded_index(n_tokens, generator)
    real = index >= 0
    inv_map = torch.full((n_tokens, 1), index.numel(), device="cuda")
    inv_map[index[real], 0] = real.nonzero().squeeze(1)
    dense_inv_map = torch.empty_like(inv_map)
    dense_inv_map[index[real], 0] = torch.arange(n_tokens, device="cuda")
    for dtype in (torch.bfloat16, torch.float32):
        expert_out = torch.randn(index.numel(), hidden, generator=generator, device="cuda", dtype=dtype)
        weights = torch.rand(index.numel(), generator=generator, device="cuda", dtype=dtype)
        # As the padded layout has them: a zero output and a zero gate weight on every padding row.
        expert_out[~real], weights[~real] = 0, 0
        buffer = torch.randn(n_tokens + 1, hidden, generator=generator, device="cuda", dtype=dtype)
        buffer[0] = float("inf")
        grad = buffer[1:]

        def backward(out_rows, weight_rows, token_idx, token_map, upstream=grad):
            out_rows, weight_rows = out_rows.clone().requires_grad_(True), weight_rows.clone().requires_grad_(True)
            MoEWeightedUnpermute.apply(out_rows, weight_rows, token_idx, token_map).backward(upstream)
            return out_rows.grad, weight_rows.grad

        want_out, want_w = backward(expert_out[real], weights[real], index[real], dense_inv_map)
        got_out, got_w = backward(expert_out, weights, index, inv_map)
        assert torch.equal(got_out[real], want_out) and torch.equal(got_w[real], want_w), f"{dtype}: a real row moved"
        assert not got_out[~real].any() and not got_w[~real].any(), f"{dtype}: a padding row took a gradient"


def test_one_padded_gather_compile_serves_every_row_stride_and_count():
    kernel = moe_permute._gather_padded_rows_kernel
    generator = torch.Generator(device="cuda").manual_seed(0)

    def run(n_tokens: int, row_stride: int) -> None:
        receive = torch.randn(n_tokens, row_stride, generator=generator, device="cuda", dtype=torch.bfloat16)
        gather_padded_rows(receive[:, :360], _padded_index(n_tokens, generator))

    kernel.device_caches.clear()
    run(14, 360)
    warm = len(kernel.device_caches[torch.cuda.current_device()][0])
    for n_tokens, row_stride in ((7, 360), (7 * 16, 384), (7 * 73, 361), (7 * 3, 3072)):
        run(n_tokens, row_stride)
    assert len(kernel.device_caches[torch.cuda.current_device()][0]) == warm, "a stride or a count compiled a binary"


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
    record_check(
        checks,
        "padded_gather_copies_rows_bit_for_bit_and_zeros_the_padding",
        test_the_padded_gather_copies_rows_bit_for_bit_and_zeros_the_padding,
    )
    record_check(
        checks,
        "padded_permute_backward_sums_each_tokens_rows_and_skips_the_padding",
        test_the_padded_permute_backward_sums_each_tokens_rows_and_skips_the_padding,
    )
    record_check(
        checks,
        "unpermute_backward_gives_a_padding_row_no_gradient",
        test_the_unpermute_backward_gives_a_padding_row_no_gradient,
    )
    record_check(
        checks,
        "one_padded_gather_compile_serves_every_row_stride_and_count",
        test_one_padded_gather_compile_serves_every_row_stride_and_count,
    )
    record_check(checks, "one_compile_serves_every_row_count", test_one_compile_serves_every_row_count)
    record_check(
        checks,
        "one_backward_compile_serves_every_gradient_row_stride",
        test_one_backward_compile_serves_every_gradient_row_stride,
    )
    return {"checks": checks}


if __name__ == "__main__":
    run()
