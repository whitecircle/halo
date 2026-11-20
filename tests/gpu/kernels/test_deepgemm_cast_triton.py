#!/usr/bin/env python
"""Triton FP8 casts of the native DeepGEMM path, verifying that
- each is bitwise DeepGEMM's own reference cast (``per_token_cast_to_fp8`` 1x128 rows, ``per_channel_cast_to_fp8``
  128x1 tiles, UE8M0 scales) of the gathered, zero-padded operand, on random data, a width not a multiple of 128, a
  zero-padded contraction, all-zero and all-padding tiles, subnormals, powers of two and bf16's largest values,
- the weight casts are bitwise ``per_block_cast_to_fp8`` (128x128) of each expert's weight and ``per_token_cast_to_fp4``
  (1x32 MXFP4) of its transposed, zero-padded weight, on widths off the block grid, one expert, a strided half of a
  fused bank, an fp32 master, and for MXFP4 every e2m1 rounding midpoint, each tie going to the even code,
- the gradient casts, which drop the reference's 1e-4 floor on a group's largest element, keep a gradient of a
  mean-reduced loss's scale to fp8's rounding, where the floor would zero most of it,
- a new token count never compiles a new kernel variant, which under EP would stall the other ranks.

Run: torchrun --nproc_per_node=1 tests/gpu/kernels/test_deepgemm_cast_triton.py
"""

import sys

import torch
import torch.nn.functional as F
import triton

from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.deepgemm_cast_triton import (
    GRADIENT_AMAX_FLOOR,
    TILE,
    _rows_kernel,
    _tiles_kernel,
    cast_rows_to_fp8,
    cast_tiles_to_fp8,
    cast_weight_blocks_to_fp8,
    cast_weight_rows_to_mxfp4,
)
from tests.common.harness import gpu_test_main, record_check
from tests.common.rank_log import log

DEV = "cuda"


def _row_map(counts: list[int]) -> torch.Tensor:
    """Each group's tokens at the start of a 128-row segment, -1 on the padding, as the DeepGEMM layout has them."""
    src, token = [], 0
    for count in counts:
        src += list(range(token, token + count)) + [-1] * (-count % TILE)
        token += count
    return torch.tensor(src, device=DEV, dtype=torch.int32)


def _gathered(x: torch.Tensor, src: torch.Tensor, width: int) -> torch.Tensor:
    out = x.new_zeros(src.numel(), width)
    live = src >= 0
    out[live, : x.shape[1]] = x[src[live].long()]
    return out


def _cases():
    g = torch.Generator(device=DEV).manual_seed(0)
    bf = torch.bfloat16
    uneven = [100, 0, 245, 1, 130]
    t = sum(uneven)
    yield "rand_2880", torch.randn(t, 2880, device=DEV, generator=g, dtype=bf), uneven
    yield "rand_768", torch.randn(t, 768, device=DEV, generator=g, dtype=bf), uneven
    yield "pow2", (2.0 ** torch.randint(-30, 30, (t, 512), device=DEV, generator=g)).to(bf), uneven
    yield "zeros", torch.zeros(t, 256, device=DEV, dtype=bf), uneven
    yield "subnormal", torch.full((t, 256), 1e-40, device=DEV, dtype=bf), uneven
    yield "huge", torch.full((t, 256), 3e38, device=DEV, dtype=bf), uneven
    yield "aligned", torch.randn(4 * TILE, 1024, device=DEV, generator=g, dtype=bf), [TILE] * 4
    yield "fp32", torch.randn(t, 384, device=DEV, generator=g), uneven


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor


def _assert_bits_equal(out: tuple, ref: tuple, name: str) -> None:
    for got, expected, part in zip(out, ref, ("data", "scales"), strict=True):
        assert got.shape == expected.shape and got.dtype == expected.dtype, f"{name} {part}: {got.shape} {got.dtype}"
        assert torch.equal(_bits(got), _bits(expected)), f"{name} {part} differs"


def test_bitwise_deepgemm_reference():
    _, utils = deepgemm._deep_gemm()
    per_token_cast_to_fp8, per_channel_cast_to_fp8 = utils.per_token_cast_to_fp8, utils.per_channel_cast_to_fp8
    for name, x, counts in _cases():
        src = _row_map(counts)
        k = x.shape[1]
        for width in (k, -(-k // 256) * 256):
            ref = per_token_cast_to_fp8(_gathered(x, src, width), use_ue8m0=True, gran_k=TILE)
            _assert_bits_equal(cast_rows_to_fp8(x, src, width), ref, f"{name} rows width={width}")
        ref = per_channel_cast_to_fp8(_gathered(x, src, k), use_ue8m0=True, gran_k=TILE)
        _assert_bits_equal(cast_tiles_to_fp8(x, src), ref, f"{name} tiles")
        print(f"  {name:10s} rows and tiles bitwise DeepGEMM's reference — OK")


def _weight_cases():
    g = torch.Generator(device=DEV).manual_seed(0)
    bf = torch.bfloat16
    yield "gpt_oss", torch.randn(4, 2880, 2880, device=DEV, generator=g, dtype=bf)
    yield "off_grid", torch.randn(3, 200, 100, device=DEV, generator=g, dtype=bf) * 40
    yield "one_expert", torch.randn(1, 256, 384, device=DEV, generator=g, dtype=bf)
    yield "fused_half", torch.randn(2, 512, 2 * 640, device=DEV, generator=g, dtype=bf)[..., :640]
    yield "fp32_master", torch.randn(2, 384, 256, device=DEV, generator=g)
    yield "zeros", torch.zeros(2, 256, 256, device=DEV, dtype=bf)


def test_weight_cast_bitwise_deepgemm_reference():
    _, utils = deepgemm._deep_gemm()
    for name, w in _weight_cases():
        reference = [
            utils.per_block_cast_to_fp8(w[e].contiguous(), use_ue8m0=True, gran_k=TILE) for e in range(len(w))
        ]
        ref = tuple(torch.stack(parts) for parts in zip(*reference, strict=True))
        _assert_bits_equal(cast_weight_blocks_to_fp8(w), ref, f"{name} weight blocks")
        print(f"  {name:11s} weight blocks bitwise DeepGEMM's reference — OK")


def _midpoints(experts: int, k: int, n: int) -> torch.Tensor:
    """Each 32-group's amax is 6 (scale 1), the rest are e2m1 midpoints of either sign, and zeros of either sign."""
    values = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 0.0, -0.0]
    )
    w = values[torch.arange(experts * k * n) % len(values)].view(experts, k, n)
    w[:, ::32, :] = 6.0
    return w.to(DEV, torch.bfloat16)


def test_mxfp4_weight_cast_bitwise_deepgemm_reference():
    _, utils = deepgemm._deep_gemm()
    cases = [*_weight_cases(), ("midpoints", _midpoints(2, 256, 192))]
    for name, w in cases:
        if name == "zeros":
            continue
        k_pad = -(-w.shape[1] // 256) * 256
        reference = [
            utils.per_token_cast_to_fp4(
                F.pad(w[e].t().contiguous(), (0, k_pad - w.shape[1])), use_ue8m0=True, gran_k=32
            )
            for e in range(len(w))
        ]
        ref = tuple(torch.stack(parts) for parts in zip(*reference, strict=True))
        flag = torch.zeros(1, device=DEV, dtype=torch.int32)
        _assert_bits_equal(cast_weight_rows_to_mxfp4(w, k_pad, flag), ref, f"{name} mxfp4 rows")
        assert not flag.item(), f"{name}: a finite weight set the non-finite flag"
        print(f"  {name:11s} mxfp4 weight rows bitwise DeepGEMM's reference — OK")


def test_mxfp4_weight_cast_flags_a_non_finite_weight():
    """e2m1 has no NaN code: a NaN or inf anywhere in the bank, the last row and column of an off-grid one included,
    sets the cast's flag, with no separate pass over the weight. The flag is the only guard: a NaN element among
    finite ones keeps its group's scale and takes a finite non-zero code."""
    for poison in (float("nan"), float("inf"), float("-inf")):
        for e, k, n in ((0, 0, 0), (2, 199, 99), (1, 37, 64)):
            w = torch.zeros(3, 200, 100, device=DEV, dtype=torch.bfloat16)
            w[e, k, n] = poison
            flag = torch.zeros(1, device=DEV, dtype=torch.int32)
            cast_weight_rows_to_mxfp4(w, 256, flag)
            assert flag.item() == 1, f"{poison} at {(e, k, n)} was cast without setting the flag"
    w = torch.full((1, 32, 1), 0.5, device=DEV, dtype=torch.bfloat16)
    clean = cast_weight_rows_to_mxfp4(w, 32, flag)
    w[0, 0, 0] = float("nan")
    data, scale = cast_weight_rows_to_mxfp4(w, 32, flag)
    assert torch.equal(scale, clean[1]) and data[0, 0, 0] & 0xF, "premise: the cast alone hides a NaN element"
    assert torch.equal(data[0, 0, 1:], clean[0][0, 0, 1:])
    print("  mxfp4 weight cast flags NaN / +-inf anywhere in the bank — OK")


def test_gradient_casts_scale_by_the_groups_own_amax():
    """Unit-scale data casts as the reference does, in every group the floor leaves alone. At 1e-9 every group's scale
    is the power of two just holding its own largest element over 448 and next to no nonzero element rounds to zero;
    the reference's floor, the control, zeroes over a tenth of them. An all-zero group keeps a finite scale."""
    g = torch.Generator(device=DEV).manual_seed(0)
    counts = [100, 0, 245, 1, 130]
    src = _row_map(counts)
    x = torch.randn(sum(counts), 512, device=DEV, generator=g, dtype=torch.bfloat16)
    casts = {
        "rows": lambda t, floor: cast_rows_to_fp8(t, src, t.shape[1], floor),
        "tiles": lambda t, floor: cast_tiles_to_fp8(t, src, floor),
    }

    def group_amax(t: torch.Tensor, name: str) -> torch.Tensor:
        gathered = _gathered(t, src, 512).float().abs()
        return (
            gathered.view(-1, 512 // TILE, TILE).amax(-1) if name == "rows" else gathered.view(-1, TILE, 512).amax(1)
        )

    for name, cast in casts.items():
        (data, scales), (ref_data, ref_scales) = cast(x, GRADIENT_AMAX_FLOOR), cast(x, 1e-4)
        above = group_amax(x, name) >= 1e-4  # a padding row's all-zero group is the one the floor moves
        assert torch.equal(_bits(data), _bits(ref_data)), f"unit-scale {name} data differs from the reference"
        assert torch.equal(scales[above], ref_scales[above]), f"unit-scale {name} scales differ from the reference"
        small = x * 1e-9
        gathered = _gathered(small, src, 512).float()
        data, scales = cast(small, GRADIENT_AMAX_FLOOR)
        groups = group_amax(small, name)
        expected = torch.exp2(torch.ceil(torch.log2(groups / 448)))
        assert torch.equal(scales[groups > 0], expected[groups > 0]), f"{name}: a scale is not its group's own"
        assert torch.isfinite(scales).all() and (scales > 0).all(), f"{name}: an all-zero group's scale"
        nonzero = gathered != 0
        zeroed = ((data.float() == 0) & nonzero).sum() / nonzero.sum()
        # e4m3 spans 2^-9..448 of a group's scale, so only an element some 2^-18 of its group's largest rounds away.
        assert zeroed < 1e-4, f"{name}: {zeroed:.2%} of the nonzero elements rounded to zero"
        floored = ((cast(small, 1e-4)[0].float() == 0) & nonzero).sum() / nonzero.sum()
        assert floored > 0.1, f"{name}: the reference floor zeroes {floored:.2%} at this scale, so the control is void"
    print("  gradient casts keep a 1e-9-scale gradient the reference floor would zero — OK")


def test_no_recompile_across_token_counts():
    """A new token count, padded to whole 128-row tiles, specializes alike: a new receive never compiles a new variant
    mid-run."""
    compiled = []

    def hook(*, fn, **_):
        if fn.jit_function in (_rows_kernel, _tiles_kernel):
            compiled.append(fn.name)
        return False  # a truthy return would skip the compile

    def cast(counts: list[int]) -> None:
        x = torch.randn(sum(counts), 2880, device=DEV, dtype=torch.bfloat16)
        src = _row_map(counts)
        for floor in (1e-4, GRADIENT_AMAX_FLOOR):
            cast_rows_to_fp8(x, src, 2880, floor)
            cast_tiles_to_fp8(x, src, floor)

    cast([256, 300])
    prev = triton.knobs.runtime.jit_cache_hook
    triton.knobs.runtime.jit_cache_hook = hook
    try:
        cast([1, 1])
        cast([1000, 7, 0, 129])
    finally:
        triton.knobs.runtime.jit_cache_hook = prev
    assert not compiled, f"new token counts compiled {len(compiled)} new kernel variant(s)"
    print("  no recompile across token counts — OK")


@gpu_test_main(exact_world_size=1, prefix="deepgemm_cast_triton", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(checks, "bitwise_deepgemm_reference", test_bitwise_deepgemm_reference)
    record_check(checks, "weight_cast_bitwise_deepgemm_reference", test_weight_cast_bitwise_deepgemm_reference)
    record_check(
        checks, "mxfp4_weight_cast_bitwise_deepgemm_reference", test_mxfp4_weight_cast_bitwise_deepgemm_reference
    )
    record_check(
        checks, "mxfp4_weight_cast_flags_a_non_finite_weight", test_mxfp4_weight_cast_flags_a_non_finite_weight
    )
    record_check(
        checks, "gradient_casts_scale_by_the_groups_own_amax", test_gradient_casts_scale_by_the_groups_own_amax
    )
    record_check(checks, "no_recompile_across_token_counts", test_no_recompile_across_token_counts)
    return {"checks": checks}


if __name__ == "__main__":
    if not deepgemm.deepgemm_available():
        # The Hopper image ships no deep_gemm, whose reference casts these are checked against.
        log("SKIP: deep_gemm absent; its reference casts are the oracle")
        sys.exit(0)
    run()
