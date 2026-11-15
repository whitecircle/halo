#!/usr/bin/env python
"""The fused Triton dequant of a frozen expert bank equals the eager oracle bit for bit, for NVFP4 and MXFP8 banks at
Inkling 2's expert widths, on every e2m1 code, the NaN and smallest e8m0 scales and subnormal fp8 scales; and it times
both paths on a real bank."""

import torch

from halo.kernels.lowp.dequant_triton import dequantize_bank_triton
from halo.kernels.lowp.quantization import FORMAT_BLOCK_SIZE, BlockScaledTensor, dequantize
from tests.common.harness import gpu_test_main, record_check
from tests.common.rank_log import log

DEV = "cuda"
# Inkling-2-Small's expert projections: gate/up [2I, H] and down [H, I], H 4608, I 2304.
SHAPES = ((4608, 4608), (4608, 2304))


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
    record_check(checks, "speed", check_speed)
    return {"checks": checks}


if __name__ == "__main__":
    run()
