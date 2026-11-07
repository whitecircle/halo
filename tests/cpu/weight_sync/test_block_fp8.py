#!/usr/bin/env python
"""The block-FP8 quantizer the weight push and the export tool share: the diagnostic's math bit for bit, a bounded
dequantization error, and a scale the engine can read back."""

import pytest
import torch

from halo.checkpoint.tools.fp8_dequant import dequantize_block_fp8
from halo.kernels.lowp.block_fp8 import BLOCK, FP8_E4M3_MAX, quantize_block_fp8


def _reference(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The validated diagnostic quantizer, verbatim (GPU row: bit-exact against a fresh boot)."""
    n, k = w.shape[-2:]
    x = torch.nn.functional.pad(w.float(), (0, (-k) % BLOCK, 0, (-n) % BLOCK))
    xb = x.reshape(x.shape[-2] // BLOCK, BLOCK, x.shape[-1] // BLOCK, BLOCK)
    amax = xb.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (xb / scale).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(x.shape)[:n, :k].contiguous()
    return q, scale[:, 0, :, 0].float().contiguous()


@pytest.mark.parametrize("shape", [(128, 128), (256, 384), (200, 130), (5, 7), (512, 64)])
def test_the_quantizer_equals_the_diagnostics_math_bit_for_bit(shape):
    torch.manual_seed(0)
    weight = (torch.randn(shape) * torch.rand(shape).mul(4)).bfloat16()
    q, scale = quantize_block_fp8(weight)
    ref_q, ref_scale = _reference(weight)
    assert q.dtype == torch.float8_e4m3fn and scale.dtype == torch.float32
    assert torch.equal(scale, ref_scale)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))


def test_scales_are_exact_powers_of_two_including_at_the_boundary():
    """A block whose maximum is exactly 448 * 2**e keeps scale 2**e: the next power would halve the resolution."""
    weight = torch.zeros(BLOCK, 2 * BLOCK)
    weight[0, 0] = FP8_E4M3_MAX * 4
    weight[0, BLOCK] = FP8_E4M3_MAX * 4 * 1.0001
    _, scale = quantize_block_fp8(weight)
    assert scale.tolist() == [[4.0, 8.0]]
    assert torch.equal(torch.log2(scale), torch.log2(scale).round())


def test_a_dequantized_weight_is_within_the_formats_half_step_of_the_original():
    torch.manual_seed(1)
    weight = torch.randn(300, 260, dtype=torch.float32)
    q, scale = quantize_block_fp8(weight)
    restored = dequantize_block_fp8(q, scale, (BLOCK, BLOCK), "w").float()
    # e4m3 keeps 3 mantissa bits: a relative step of 1/8 at most, half of it as rounding error, plus the block scale's
    # slack when the maximum sits just above a power of two.
    assert ((restored - weight).abs() <= weight.abs() * 0.0625 + scale.max() * 2**-9).all()
    assert (restored - weight).norm() / weight.norm() < 0.04


def test_the_grid_covers_a_padded_edge_and_leading_dimensions_quantize_alike():
    torch.manual_seed(2)
    bank = torch.randn(3, 130, 129)
    q, scale = quantize_block_fp8(bank)
    assert q.shape == bank.shape and scale.shape == (3, 2, 2)
    for expert in range(3):
        single_q, single_scale = quantize_block_fp8(bank[expert])
        assert torch.equal(q[expert].view(torch.uint8), single_q.view(torch.uint8))
        assert torch.equal(scale[expert], single_scale)


def test_an_all_zero_block_gets_a_finite_scale_and_zero_values():
    q, scale = quantize_block_fp8(torch.zeros(BLOCK, BLOCK))
    assert torch.isfinite(scale).all() and (scale > 0).all()
    assert not q.float().any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
