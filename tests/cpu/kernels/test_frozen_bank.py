"""Frozen NVFP4/MXFP8 expert banks: dequantization against the release's own arithmetic, and the grouped GEMM against
the same GEMM over the dequantized bf16 weight, forward and input gradient, bitwise."""

import pytest
import torch
import torch.nn.functional as F

from halo.kernels.lowp.frozen_bank import FrozenExpertBank, frozen_bank_grouped_mm

# e2m1 codes 0..15, low nibble first: what the Inkling 2 release, and every engine serving it, decodes.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_NVFP4_DENOMINATOR = 448.0 * 6.0


def _random_nvfp4_bank(experts: int, n: int, k: int, seed: int = 0) -> FrozenExpertBank:
    generator = torch.Generator().manual_seed(seed)
    data = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, generator=generator)
    # Positive, finite e4m3 block scales spread over several binades.
    scales = (torch.rand((experts, n, k // 16), generator=generator) * 4 + 0.25).to(torch.float8_e4m3fn)
    amax = torch.rand((experts,), generator=generator) * 3 + 0.5
    return FrozenExpertBank("nvfp4", data, scales, amax / _NVFP4_DENOMINATOR)


def _random_mxfp8_bank(experts: int, n: int, k: int, seed: int = 0) -> FrozenExpertBank:
    generator = torch.Generator().manual_seed(seed)
    data = (torch.randn((experts, n, k), generator=generator) * 50).to(torch.float8_e4m3fn)
    scales = torch.randint(118, 130, (experts, n, k // 32), dtype=torch.uint8, generator=generator)
    return FrozenExpertBank("mxfp8", data, scales)


def _reference_nvfp4(bank: FrozenExpertBank) -> torch.Tensor:
    """The release arithmetic written out: code x e4m3 block scale x amax / denominator, rounded once to bf16."""
    lut = torch.tensor(_E2M1)
    codes = torch.stack([bank.data & 15, bank.data >> 4], dim=-1).long().flatten(-2)
    values = lut[codes]
    block = bank.scales.float().repeat_interleave(16, dim=-1)
    return (values * block * bank.global_scale.view(-1, 1, 1)).to(torch.bfloat16)


def _reference_mxfp8(bank: FrozenExpertBank) -> torch.Tensor:
    block = torch.exp2(bank.scales.float() - 127).repeat_interleave(32, dim=-1)
    return (bank.data.float() * block).to(torch.bfloat16)


def test_nvfp4_dequantization_matches_the_release_arithmetic():
    bank = _random_nvfp4_bank(experts=6, n=8, k=64)
    actual = bank.dequantize_linear()
    expected = _reference_nvfp4(bank)
    assert actual.shape == (6, 8, 64)
    assert expected.abs().max() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nvfp4_low_nibble_is_the_first_element():
    """0x71 holds code 1 (0.5) low and code 7 (6.0) high; a swapped decode reads 6.0 first."""
    data = torch.full((1, 1, 8), 0x71, dtype=torch.uint8)
    scales = torch.ones((1, 1, 1), dtype=torch.float8_e4m3fn)
    bank = FrozenExpertBank("nvfp4", data, scales, torch.ones(1))
    row = bank.dequantize_linear()[0, 0, :2].float()
    assert row.tolist() == [0.5, 6.0]


def test_nvfp4_global_scale_is_per_expert():
    bank = _random_nvfp4_bank(experts=2, n=4, k=32)
    doubled = FrozenExpertBank("nvfp4", bank.data, bank.scales, bank.global_scale * torch.tensor([1.0, 2.0]))
    base, scaled = bank.dequantize_linear().float(), doubled.dequantize_linear().float()
    torch.testing.assert_close(scaled[0], base[0], rtol=0, atol=0)
    torch.testing.assert_close(scaled[1], base[1] * 2, rtol=0, atol=0)


def test_mxfp8_dequantization_matches_the_release_arithmetic():
    bank = _random_mxfp8_bank(experts=5, n=8, k=96)
    torch.testing.assert_close(bank.dequantize_linear(), _reference_mxfp8(bank), rtol=0, atol=0)


def test_chunked_dequantization_equals_one_pass():
    """The chunk loop must cover every expert exactly once, including a short tail chunk."""
    bank = _random_nvfp4_bank(experts=7, n=4, k=32)
    torch.testing.assert_close(bank.dequantize_linear(), _reference_nvfp4(bank), rtol=0, atol=0)
    torch.testing.assert_close(bank.dequantize_linear(2, 7), _reference_nvfp4(bank)[2:7], rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad",
    [
        lambda: FrozenExpertBank("nvfp4", torch.zeros(2, 4, 16, dtype=torch.uint8), torch.zeros(2, 4, 2), None),
        lambda: FrozenExpertBank("mxfp8", torch.zeros(2, 4, 32), torch.zeros(2, 4, 1), torch.ones(2)),
        lambda: FrozenExpertBank(
            "nvfp4", torch.zeros(2, 4, 16, dtype=torch.uint8), torch.zeros(2, 4, 3), torch.ones(2)
        ),
        lambda: FrozenExpertBank(
            "nvfp4", torch.zeros(2, 4, 16, dtype=torch.uint8), torch.zeros(2, 4, 2), torch.ones(3)
        ),
        lambda: FrozenExpertBank("mxfp4", torch.zeros(2, 4, 16, dtype=torch.uint8), torch.zeros(2, 4, 1), None),
    ],
    ids=["nvfp4-no-global", "mxfp8-with-global", "scale-grid", "global-count", "unknown-format"],
)
def test_malformed_banks_are_refused(bad):
    with pytest.raises(ValueError):
        bad()


def _grouped_inputs(bank: FrozenExpertBank, rows_per_expert: tuple[int, ...]):
    generator = torch.Generator().manual_seed(1)
    x = torch.randn((sum(rows_per_expert), bank.in_features), generator=generator).to(torch.bfloat16)
    offs = torch.tensor(rows_per_expert).cumsum(0).to(torch.int32)
    return x, offs


@pytest.mark.parametrize("make_bank", [_random_nvfp4_bank, _random_mxfp8_bank], ids=["nvfp4", "mxfp8"])
def test_grouped_mm_matches_the_dequantized_weight_forward_and_input_grad(make_bank):
    """Same kernel, same bf16 weight: the bank path must be bitwise the GEMM over the dequantized tensor, and give
    the weight no gradient."""
    bank = make_bank(4, 16, 64)
    rows = (3, 0, 5, 2)  # an expert routed nothing
    x, offs = _grouped_inputs(bank, rows)
    weight = bank.dequantize_linear().transpose(1, 2)

    x_bank = x.clone().requires_grad_(True)
    x_ref = x.clone().requires_grad_(True)
    out_bank = frozen_bank_grouped_mm(x_bank, bank, offs=offs)
    out_ref = F.grouped_mm(x_ref, weight, offs=offs)
    torch.testing.assert_close(out_bank, out_ref, rtol=0, atol=0)

    cotangent = torch.randn(out_ref.shape, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    out_bank.backward(cotangent)
    out_ref.backward(cotangent)
    assert x_ref.grad.abs().max() > 0
    torch.testing.assert_close(x_bank.grad, x_ref.grad, rtol=0, atol=0)
    assert not bank.data.requires_grad and bank.data.grad is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
