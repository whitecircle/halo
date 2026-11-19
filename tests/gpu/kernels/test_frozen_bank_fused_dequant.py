#!/usr/bin/env python
"""The fused Triton dequant of a frozen expert bank equals the eager oracle bit for bit, for NVFP4 and MXFP8 banks at
Inkling 2's expert widths, on every e2m1 code, the NaN and smallest e8m0 scales and subnormal fp8 scales; a bank's
expert slices take the same kernel with their own scales, on the binaries the warm-up compiles; and it times both paths
on a real bank."""

import torch

from halo.kernels.lowp.dequant_triton import (
    _mxfp8_bank_kernel,
    _nvfp4_bank_kernel,
    dequantize_bank_triton,
    supports_fused_dequant,
)
from halo.kernels.lowp.frozen_bank import FrozenExpertBank, warm_fused_dequant_kernels
from halo.kernels.lowp.quantization import FORMAT_BLOCK_SIZE, BlockScaledTensor, dequantize
from tests.common.harness import gpu_test_main, record_check
from tests.common.rank_log import log

DEV = "cuda"
# Inkling-2-Small's expert projections: gate/up [2I, H] and down [H, I], H 4608, I 2304.
SHAPES = ((4608, 4608), (4608, 2304))
# An odd row count at one block of columns: a one-expert slice's MXFP8 scales start off 16-byte alignment.
ODD_ROWS_SHAPE = (3, 256)


def _eager(data, scales, global_scale, quant_format) -> torch.Tensor:
    packed = quant_format == "nvfp4"
    return dequantize(
        BlockScaledTensor(
            data=data,
            scales=scales,
            block_size=FORMAT_BLOCK_SIZE[quant_format],
            axis=-1,
            packed=packed,
            pow2_scale=not packed,
            global_scale=None if global_scale is None else global_scale.view(-1, 1, 1),
        )
    )


def _bank(quant_format: str, experts: int, n: int, k: int, generator: torch.Generator):
    """Random bank whose first rows hold the edge values: every e2m1 code, e8m0 codes 0 and 255, subnormal and zero fp8
    scales."""
    block = FORMAT_BLOCK_SIZE[quant_format]
    if quant_format == "nvfp4":
        data = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device=DEV, generator=generator)
        data[0, 0, :8] = torch.arange(0, 256, 32, dtype=torch.uint8) | torch.arange(8, dtype=torch.uint8)
        data[0, 1, :16] = torch.arange(16, dtype=torch.uint8) * 17
        scale_values = torch.rand((experts, n, k // block), device=DEV, generator=generator) * 448
        scale_values[0, 0, :4] = torch.tensor([0.0, 2.0**-9, 2.0**-7, 448.0])
        scales = scale_values.to(torch.float8_e4m3fn)
        global_scale = torch.rand(experts, device=DEV, generator=generator) * 1e-3 + 1e-6
        return data, scales, global_scale
    values = (torch.randn((experts, n, k), device=DEV, generator=generator) * 100).clamp(-448, 448)
    data = values.to(torch.float8_e4m3fn)
    scales = torch.randint(100, 140, (experts, n, k // block), dtype=torch.uint8, device=DEV, generator=generator)
    scales[0, 0, :4] = torch.tensor([0, 1, 254, 255], dtype=torch.uint8)
    return data, scales, None


def check_bitwise() -> None:
    generator = torch.Generator(device=DEV).manual_seed(0)
    for quant_format in ("nvfp4", "mxfp8"):
        for n, k in SHAPES:
            data, scales, global_scale = _bank(quant_format, 3, n, k, generator)
            fused = dequantize_bank_triton(data, scales, global_scale, quant_format)
            eager = _eager(data, scales, global_scale, quant_format)
            same = torch.equal(fused.view(torch.int16), eager.view(torch.int16))
            assert same, (
                f"{quant_format} [{n}, {k}]: {int((fused.view(torch.int16) != eager.view(torch.int16)).sum())} bits differ"
            )
            assert torch.isnan(fused).any() == (quant_format == "mxfp8"), f"{quant_format}: the NaN scale code is lost"
            log(f"  {quant_format} [{n}, {k}]: bit-identical to eager")


def _bitwise_difference(actual: torch.Tensor, expected: torch.Tensor) -> int:
    assert actual.shape == expected.shape, f"shape {list(actual.shape)} against {list(expected.shape)}"
    return int((actual.view(torch.int16) != expected.view(torch.int16)).sum())


def check_bank_slices() -> None:
    """``FrozenExpertBank.dequantize_linear(start, end)``, the call the expert GEMM and the per-expert loop make: each
    expert range equals eager on its own rows of data, block scales and global scale."""
    generator = torch.Generator(device=DEV).manual_seed(2)
    experts = 5
    for quant_format in ("nvfp4", "mxfp8"):
        for n, k in SHAPES:
            data, scales, global_scale = _bank(quant_format, experts, n, k, generator)
            bank = FrozenExpertBank(quant_format, data, scales, global_scale)
            assert supports_fused_dequant(bank.data, bank.in_features), f"{quant_format} K={k} is off the fused path"
            if global_scale is not None:
                assert global_scale.unique().numel() == experts, "premise: every expert has its own global scale"
            for start, end in ((0, experts), (1, 4), (3, experts), (2, 3), (experts - 1, experts)):
                fused = bank.dequantize_linear(start, end)
                eager = _eager(
                    data[start:end],
                    scales[start:end],
                    None if global_scale is None else global_scale[start:end],
                    quant_format,
                )
                differing = _bitwise_difference(fused, eager)
                assert differing == 0, f"{quant_format} [{n}, {k}] experts [{start}, {end}): {differing} bits differ"
            log(f"  {quant_format} [{n}, {k}]: every expert range is bit-identical to eager")


def _compiled_binaries() -> int:
    device = torch.cuda.current_device()
    return sum(len(kernel.device_caches[device][0]) for kernel in (_nvfp4_bank_kernel, _mxfp8_bank_kernel))


def check_warm_up_leaves_no_compile() -> None:
    """The load-time warm-up launches one expert per format and ``K``; a whole bank and every one-expert slice, whose
    global scale (NVFP4) or block scales (MXFP8 at an odd row count) sit off 16-byte alignment, run on those
    binaries."""
    generator = torch.Generator(device=DEV).manual_seed(3)
    experts = 5
    banks = [
        FrozenExpertBank(quant_format, *_bank(quant_format, experts, n, k, generator))
        for quant_format in ("nvfp4", "mxfp8")
        for n, k in (*SHAPES, ODD_ROWS_SHAPE)
    ]
    for kernel in (_nvfp4_bank_kernel, _mxfp8_bank_kernel):
        kernel.device_caches.clear()
    warm_fused_dequant_kernels(banks)
    warmed = _compiled_binaries()
    assert warmed == len(banks), f"{warmed} binaries for {len(banks)} format and K pairs"
    for bank in banks:
        bank.dequantize_linear()
        for expert in range(experts):
            bank.dequantize_linear(expert, expert + 1)
    after = _compiled_binaries()
    log(f"  [warm-up] {warmed} binaries, {after} after every bank and one-expert slice")
    assert after == warmed, f"{after - warmed} binaries compiled after the warm-up"


def check_speed() -> None:
    """A 16-expert NVFP4 gate/up bank, the eager path in its 4-expert chunks as the bank runs it."""
    generator = torch.Generator(device=DEV).manual_seed(1)
    data, scales, global_scale = _bank("nvfp4", 16, *SHAPES[0], generator)

    def eager_bank():
        for lo in range(0, 16, 4):
            _eager(data[lo : lo + 4], scales[lo : lo + 4], global_scale[lo : lo + 4], "nvfp4")

    def fused_bank():
        dequantize_bank_triton(data, scales, global_scale, "nvfp4")

    times = {}
    for name, fn in (("eager", eager_bank), ("fused", fused_bank)):
        fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            fn()
        end.record()
        end.synchronize()
        times[name] = start.elapsed_time(end) / 5
    gbytes = 16 * SHAPES[0][0] * SHAPES[0][1] * 2.5 / 1e9  # half a byte read and two written per element
    log(
        f"  16-expert [4608, 4608] NVFP4 bank: eager {times['eager']:.1f} ms, fused {times['fused']:.2f} ms "
        f"({gbytes / times['fused'] * 1e3:.0f} GB/s), {times['eager'] / times['fused']:.0f}x"
    )
    assert times["fused"] < times["eager"], "the fused dequant is slower than eager"


@gpu_test_main(exact_world_size=1, prefix="frozen_bank_fused_dequant", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(checks, "bitwise", check_bitwise)
    record_check(checks, "bank_slices", check_bank_slices)
    record_check(checks, "warm_up_leaves_no_compile", check_warm_up_leaves_no_compile)
    record_check(checks, "speed", check_speed)
    return {"checks": checks}


if __name__ == "__main__":
    run()
