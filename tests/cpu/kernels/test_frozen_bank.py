"""Frozen NVFP4/MXFP8 expert banks: dequantization against the release's own arithmetic, and the grouped GEMM against
the same GEMM over the dequantized bf16 weight, forward and input gradient, bitwise."""

import logging

import pytest
import torch
import torch.nn.functional as F

from halo.kernels.lowp import dequant_triton, frozen_bank
from halo.kernels.lowp.dequant_triton import FUSED_DEQUANT_BLOCK, dequantize_bank_triton, supports_fused_dequant
from halo.kernels.lowp.frozen_bank import FrozenExpertBank, frozen_bank_grouped_mm, warm_fused_dequant_kernels

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


class _CudaTensor(torch.Tensor):
    """A CPU tensor that reports CUDA, so the bank's CUDA branches run without a GPU; slices keep the class."""

    is_cuda = property(lambda self: True)


def _on_fake_cuda(bank: FrozenExpertBank) -> FrozenExpertBank:
    return FrozenExpertBank(bank.quant_format, bank.data.as_subclass(_CudaTensor), bank.scales, bank.global_scale)


@pytest.fixture
def fused_launches(monkeypatch):
    """The fused kernel replaced by a recorder of its arguments, which answers with the eager dequantization."""
    launches = []

    def record(data, scales, global_scale, quant_format):
        launches.append((data, scales, global_scale, quant_format))
        plain = FrozenExpertBank(quant_format, data.as_subclass(torch.Tensor), scales, global_scale)
        return plain.dequantize_linear()

    monkeypatch.setattr(frozen_bank, "dequantize_bank_triton", record)
    return launches


@pytest.mark.parametrize(
    ("on_cuda", "in_features", "expected"),
    [
        (True, FUSED_DEQUANT_BLOCK, True),
        (True, 18 * FUSED_DEQUANT_BLOCK, True),
        (True, FUSED_DEQUANT_BLOCK // 2, False),
        (True, FUSED_DEQUANT_BLOCK + 32, False),
        (False, FUSED_DEQUANT_BLOCK, False),
    ],
)
def test_the_fused_dequant_takes_cuda_banks_at_a_multiple_of_its_block(on_cuda, in_features, expected):
    data = torch.zeros((1, 1, in_features // 2), dtype=torch.uint8)
    assert supports_fused_dequant(data.as_subclass(_CudaTensor) if on_cuda else data, in_features) is expected


@pytest.mark.parametrize("in_features", [FUSED_DEQUANT_BLOCK, FUSED_DEQUANT_BLOCK // 2])
def test_the_fused_launcher_refuses_a_bank_its_predicate_rejects(in_features):
    """A CPU bank at the fused width is refused by the same rule the bank dispatches on, before any launch."""
    bank = _random_nvfp4_bank(experts=2, n=2, k=in_features)
    assert not supports_fused_dequant(bank.data, bank.in_features)
    with pytest.raises(ValueError, match=f"multiple of {FUSED_DEQUANT_BLOCK}"):
        dequantize_bank_triton(bank.data, bank.scales, bank.global_scale, "nvfp4")


@pytest.mark.parametrize("make_bank", [_random_nvfp4_bank, _random_mxfp8_bank], ids=["nvfp4", "mxfp8"])
def test_a_cuda_bank_hands_the_fused_kernel_its_expert_slices(make_bank, fused_launches):
    """Experts ``[2, 5)`` reach the kernel with their own rows of data, block scales and global scale."""
    plain = make_bank(6, 4, FUSED_DEQUANT_BLOCK)
    bank = _on_fake_cuda(plain)
    out = bank.dequantize_linear(2, 5)
    ((data, scales, global_scale, quant_format),) = fused_launches
    assert quant_format == plain.quant_format
    for launched, stored in ((data.as_subclass(torch.Tensor), plain.data), (scales, plain.scales)):
        assert torch.equal(launched.view(torch.uint8), stored[2:5].view(torch.uint8))
        assert not torch.equal(stored[2:5].view(torch.uint8), stored[:3].view(torch.uint8))
    if plain.global_scale is None:
        assert global_scale is None
    else:
        assert torch.equal(global_scale, plain.global_scale[2:5])
        assert not torch.equal(plain.global_scale[2:5], plain.global_scale[:3])
    torch.testing.assert_close(out, plain.dequantize_linear()[2:5], rtol=0, atol=0)


def test_a_cuda_bank_off_the_fused_width_warns_once_and_dequantizes_eagerly(fused_launches, caplog, monkeypatch):
    monkeypatch.setattr(frozen_bank, "_eager_cuda_banks_warned", set())
    plain = _random_nvfp4_bank(experts=3, n=4, k=FUSED_DEQUANT_BLOCK // 2)
    bank = _on_fake_cuda(plain)
    with caplog.at_level(logging.WARNING, logger=frozen_bank.__name__):
        first = bank.dequantize_linear()
        _on_fake_cuda(plain).dequantize_linear(1, 2)
    assert not fused_launches
    torch.testing.assert_close(first, _reference_nvfp4(plain), rtol=0, atol=0)
    (record,) = caplog.records
    assert f"K={FUSED_DEQUANT_BLOCK // 2}" in record.getMessage()
    assert f"multiple of {FUSED_DEQUANT_BLOCK}" in record.getMessage()


def test_a_cpu_bank_takes_the_eager_path_without_a_warning(fused_launches, caplog, monkeypatch):
    monkeypatch.setattr(frozen_bank, "_eager_cuda_banks_warned", set())
    bank = _random_nvfp4_bank(experts=2, n=4, k=FUSED_DEQUANT_BLOCK // 2)
    with caplog.at_level(logging.WARNING, logger=frozen_bank.__name__):
        bank.dequantize_linear()
    assert not fused_launches and not caplog.records


def test_the_warm_up_launches_one_expert_per_format_and_width(fused_launches):
    """Two layers' banks of one format and width share a binary, so the second launches nothing; an eager bank (CPU,
    or off the fused width) launches nothing either."""
    wide, narrow = 2 * FUSED_DEQUANT_BLOCK, FUSED_DEQUANT_BLOCK
    banks = [
        _on_fake_cuda(_random_nvfp4_bank(3, 2, wide)),
        _on_fake_cuda(_random_nvfp4_bank(3, 2, narrow)),
        _on_fake_cuda(_random_nvfp4_bank(3, 2, wide, seed=1)),
        _on_fake_cuda(_random_mxfp8_bank(3, 2, narrow)),
        _random_mxfp8_bank(3, 2, wide),
        _on_fake_cuda(_random_mxfp8_bank(3, 2, FUSED_DEQUANT_BLOCK // 2)),
    ]
    warm_fused_dequant_kernels(iter(banks))
    launched = [
        (quant_format, data.shape[0], data.shape[2] * (2 if quant_format == "nvfp4" else 1))
        for data, _scales, _global_scale, quant_format in fused_launches
    ]
    assert launched == [("nvfp4", 1, wide), ("nvfp4", 1, narrow), ("mxfp8", 1, narrow)]


@pytest.mark.parametrize(
    ("kernel", "argument", "setting"),
    (
        (dequant_triton._mxfp8_bank_kernel, "scales_ptr", "do_not_specialize_on_alignment"),
        (dequant_triton._nvfp4_bank_kernel, "global_ptr", "do_not_specialize_on_alignment"),
        (dequant_triton._nvfp4_bank_kernel, "rows_per_expert", "do_not_specialize"),
    ),
)
def test_a_one_expert_slice_runs_the_binary_its_bank_warmed(kernel, argument, setting):
    """Triton compiles per 16-byte alignment of a pointer and per class of an integer (1, a multiple of 16, neither).
    A slice's MXFP8 scales start a multiple of 8 bytes into the bank and its NVFP4 global scale a multiple of 4, and
    a one-row expert hands in a count of 1; unspecialized, each runs the binary the warm-up compiled."""
    params = {param.name: param for param in kernel.params}
    assert getattr(params[argument], setting)


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
