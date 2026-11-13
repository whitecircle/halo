#!/usr/bin/env python
"""Triton FP8 casts of the native DeepGEMM path, verifying that
- each is bitwise DeepGEMM's own reference cast (``per_token_cast_to_fp8`` 1x128 rows, ``per_channel_cast_to_fp8``
  128x1 tiles, UE8M0 scales) of the gathered, zero-padded operand, on random data, a width not a multiple of 128, a
  zero-padded contraction, all-zero and all-padding tiles, subnormals, powers of two and bf16's largest values,
- the weight casts are bitwise ``per_block_cast_to_fp8`` (128x128) and ``per_token_cast_to_fp4`` (1x32 MXFP4) of each
  expert's transposed, zero-padded weight, on widths off the block grid, one expert, a strided half of a fused bank, an
  fp32 master, and for MXFP4 every e2m1 rounding midpoint, where DeepGEMM rounds ties down,
- a new token count never compiles a new kernel variant, which under EP would stall the other ranks.

Run: torchrun --nproc_per_node=1 tests/gpu/kernels/test_deepgemm_cast_triton.py
"""

import sys

import torch
import torch.nn.functional as F
import triton

from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.deepgemm_cast_triton import (
    TILE,
    _blocks_kernel,
    _fp4_rows_kernel,
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
            utils.per_block_cast_to_fp8(w[e].t().contiguous(), use_ue8m0=True, gran_k=TILE) for e in range(len(w))
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
        _assert_bits_equal(cast_weight_rows_to_mxfp4(w, k_pad), ref, f"{name} mxfp4 rows")
        print(f"  {name:11s} mxfp4 weight rows bitwise DeepGEMM's reference — OK")


def test_no_recompile_across_token_counts():
    """Token counts and row strides are not specialized, so a new receive never compiles a new variant mid-run."""
    compiled = []

    def hook(*, fn, **_):
        if fn.jit_function in (_rows_kernel, _tiles_kernel, _blocks_kernel, _fp4_rows_kernel):
            compiled.append(fn.name)
        return False  # a truthy return would skip the compile

    def cast(counts: list[int]) -> None:
        x = torch.randn(sum(counts), 2880, device=DEV, dtype=torch.bfloat16)
        src = _row_map(counts)
        cast_rows_to_fp8(x, src, 2880)
        cast_tiles_to_fp8(x, src)

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
    record_check(checks, "no_recompile_across_token_counts", test_no_recompile_across_token_counts)
    return {"checks": checks}


if __name__ == "__main__":
    if not deepgemm.deepgemm_available():
        # The Hopper image ships no deep_gemm, whose reference casts these are checked against.
        log("SKIP: deep_gemm absent; its reference casts are the oracle")
        sys.exit(0)
    run()
