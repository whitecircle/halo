#!/usr/bin/env python
"""The fp32-result matmuls: half-precision CUDA operands keep their dtype so the GEMM stays on tensor cores.
``out_dtype`` has no CPU kernel, so ``tests.common.fake_cuda.CudaLike`` takes the CUDA branch with a stand-in."""

import pytest
import torch
import torch.nn.functional as F

from halo.kernels.fp32_matmul import bmm_fp32, mm_fp32, to_autocast_dtype
from halo.models.attention import tree_sparse
from halo.models.attention.sdpa_flex_sliding import _Fp32ScoreMatmul
from halo.models.attention.tree_sparse import _SparseLatentAttention
from tests.common.fake_cuda import CudaBmm, CudaLike

HALF = (torch.bfloat16, torch.float16)


@pytest.fixture
def bmm(monkeypatch):
    recorder = CudaBmm(torch.bmm)
    monkeypatch.setattr(torch, "bmm", recorder)
    monkeypatch.setattr(torch, "baddbmm", recorder.baddbmm)
    return recorder


def _operands(dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(3, 17, 40, generator=generator).to(dtype), torch.randn(3, 40, 9, generator=generator).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, *HALF])
def test_cpu_inputs_are_upcast(bmm, dtype):
    a, b = _operands(dtype)
    wide = torch.promote_types(dtype, torch.float32)
    got = bmm_fp32(a, b)
    assert bmm.calls == [(wide, wide, None)]
    assert got.dtype == wide
    assert torch.equal(got, bmm._bmm(a.to(wide), b.to(wide)))


@pytest.mark.parametrize("dtype", HALF)
def test_half_cuda_inputs_keep_their_dtype_with_an_fp32_result(bmm, dtype):
    a, b = (t.as_subclass(CudaLike) for t in _operands(dtype))
    got = bmm_fp32(a, b)
    assert bmm.calls == [(dtype, dtype, torch.float32)], "half CUDA operands must reach the GEMM unconverted"
    assert got.dtype == torch.float32


@pytest.mark.parametrize("dtype", HALF)
def test_half_cuda_inputs_accumulate_in_place_in_the_gemm(bmm, dtype):
    a, b = (t.as_subclass(CudaLike) for t in _operands(dtype))
    accumulator = torch.ones(3, 17, 9)
    want = accumulator + bmm._bmm(a.as_subclass(torch.Tensor).float(), b.as_subclass(torch.Tensor).float())
    assert bmm_fp32(a, b, out=accumulator) is accumulator
    assert bmm.calls == [(dtype, dtype, torch.float32)]
    assert torch.equal(accumulator, want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_inputs_accumulate_in_fp32(bmm, dtype):
    a, b = _operands(dtype)
    accumulator = torch.ones(3, 17, 9)
    want = accumulator + bmm._bmm(a.float(), b.float())
    assert bmm_fp32(a, b, out=accumulator) is accumulator
    assert torch.equal(accumulator, want)


def test_fp32_cuda_inputs_take_the_plain_gemm(bmm):
    a, b = (t.as_subclass(CudaLike) for t in _operands(torch.float32))
    bmm_fp32(a, b)
    assert bmm.calls == [(torch.float32, torch.float32, None)]


def _sparse_problem(dtype):
    """A query with no key among them."""
    generator = torch.Generator().manual_seed(1)
    query = torch.randn(6, 2, 8, generator=generator).to(dtype)
    latent = torch.randn(6, 8, generator=generator).to(dtype)
    keys = torch.tensor([[0, -1], [0, 1], [1, 2], [-1, -1], [2, 3], [4, 5]])
    return query, latent, keys


@pytest.mark.parametrize("dtype", HALF)
def test_sparse_latent_attention_keeps_half_cuda_operands(bmm, dtype):
    query, latent, keys = _sparse_problem(dtype)
    out = _SparseLatentAttention.apply(query.as_subclass(CudaLike), latent.as_subclass(CudaLike), keys, 0.3)
    assert bmm.calls == [(dtype, dtype, torch.float32)] * 2, "the score and value products"
    assert out.dtype == dtype


def test_sparse_latent_attention_runs_every_product_through_bmm_fp32(monkeypatch):
    calls = []

    def counted(a, b):
        calls.append((a.dtype, b.dtype))
        return bmm_fp32(a, b)

    monkeypatch.setattr(tree_sparse, "bmm_fp32", counted)
    query, latent, keys = _sparse_problem(torch.bfloat16)
    query.requires_grad_(True)
    latent.requires_grad_(True)
    _SparseLatentAttention.apply(query, latent, keys, 0.3).sum().backward()
    assert calls == [(torch.bfloat16, torch.bfloat16)] * 7, "two forward products and five backward products"


def test_score_matmul_runs_on_cpu_and_matches_fp32_math():
    generator = torch.Generator().manual_seed(2)
    query = torch.randn(4, 10, 16, generator=generator).to(torch.bfloat16).requires_grad_(True)
    key_t = torch.randn(4, 16, 12, generator=generator).to(torch.bfloat16).requires_grad_(True)
    scores = _Fp32ScoreMatmul.apply(query, key_t)
    assert scores.dtype == torch.float32
    assert torch.equal(scores, torch.bmm(query.detach().float(), key_t.detach().float()))
    scores.sum().backward()
    ones = torch.ones_like(scores).to(torch.bfloat16)
    assert torch.equal(query.grad, torch.bmm(ones, key_t.detach().transpose(1, 2)))
    assert torch.equal(key_t.grad, torch.bmm(query.detach().transpose(1, 2), ones))


@pytest.mark.parametrize("dtype", HALF)
def test_score_matmul_keeps_half_cuda_operands(bmm, dtype):
    query, key_t = (t.as_subclass(CudaLike) for t in _operands(dtype, seed=3))
    scores = _Fp32ScoreMatmul.apply(query, key_t)
    assert bmm.calls == [(dtype, dtype, torch.float32)]
    assert scores.dtype == torch.float32


class _CudaMm:
    """Records each call's name, operand dtypes and ``out_dtype``."""

    def __init__(self, mm, addmm):
        self._mm, self._addmm = mm, addmm
        self.calls: list[tuple[str, torch.dtype, torch.dtype, torch.dtype | None]] = []

    def mm(self, a, b, *, out_dtype=None):
        self.calls.append(("mm", a.dtype, b.dtype, out_dtype))
        return self._mm(*(t.as_subclass(torch.Tensor).to(out_dtype) for t in (a, b)))

    def addmm(self, acc, a, b, *, out_dtype=None, out=None):
        self.calls.append(("addmm", a.dtype, b.dtype, out_dtype))
        plain = [t.as_subclass(torch.Tensor).to(out_dtype) for t in (a, b)]
        return self._addmm(acc.as_subclass(torch.Tensor), *plain, out=out.as_subclass(torch.Tensor))


@pytest.fixture
def mm(monkeypatch):
    recorder = _CudaMm(torch.mm, torch.addmm)
    monkeypatch.setattr(torch, "mm", recorder.mm)
    monkeypatch.setattr(torch, "addmm", recorder.addmm)
    return recorder


def _mm_operands(seed=4):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(17, 40, generator=generator), torch.randn(40, 9, generator=generator)


@pytest.mark.parametrize("dtype", [torch.float32, *HALF])
def test_mm_cpu_inputs_take_an_fp32_matmul_and_accumulate_into_out(dtype):
    a, b = (t.to(dtype) for t in _mm_operands())
    expected = a.float() @ b.float()
    got = mm_fp32(a, b)
    assert got.dtype == torch.float32
    assert torch.equal(got, expected)
    out = torch.ones(17, 9)
    assert mm_fp32(a, b, out=out) is out
    assert torch.equal(out, torch.ones(17, 9) + expected)


@pytest.mark.parametrize("dtype", HALF)
def test_mm_half_cuda_b_casts_a_to_its_dtype_with_an_fp32_result(mm, dtype):
    a, b = _mm_operands()
    a = a.as_subclass(CudaLike)
    b = b.to(dtype).as_subclass(CudaLike)
    got = mm_fp32(a, b)
    out = torch.zeros(17, 9).as_subclass(CudaLike)
    mm_fp32(a, b, out=out)
    assert mm.calls == [("mm", dtype, dtype, torch.float32), ("addmm", dtype, dtype, torch.float32)], (
        "an fp32 a must be cast to b's half dtype, and both GEMMs must return fp32"
    )
    assert got.dtype == out.dtype == torch.float32
    assert torch.equal(out.as_subclass(torch.Tensor), got.as_subclass(torch.Tensor))


def test_to_autocast_dtype_casts_what_autocast_casts():
    single, double = torch.ones(2, 2), torch.ones(2, 2, dtype=torch.float64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast, kept = to_autocast_dtype(single, double)
        autocast_out = F.linear(double, double)
    assert cast.dtype == torch.bfloat16
    assert kept is double and autocast_out.dtype == torch.float64, "autocast leaves fp64 alone"
    off_single, off_double = to_autocast_dtype(single, double)
    assert off_single is single and off_double is double


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
