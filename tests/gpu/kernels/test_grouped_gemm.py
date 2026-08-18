#!/usr/bin/env python
"""GPU tests for the grouped-GEMM dispatch (bf16 correctness + precision dispatch).

Validates:
- bf16 ``grouped_gemm`` matches a per-expert reference loop, for EP-like (few experts)
  and non-EP-like (many experts) group counts, including a variable-token routing.
- low-precision ``grouped_gemm`` (mxfp8 / nvfp4) uses the SIMULATED fake-quant path by default
  (the native DeepGEMM kernel is opt-in and net-slower than bf16, so never auto-selected), producing
  the expected block-scaled error vs bf16 with finite grads flowing to the bf16 masters.

Run: python tests/gpu/kernels/test_grouped_gemm.py
"""

import sys

import torch

from src.kernels.grouped_gemm import GroupedGemmPrecision, grouped_gemm
from src.kernels.lowp.deepgemm import deepgemm_available

DEV = "cuda"


def _reference(x, w, offs):
    outs, start = [], 0
    for e in range(w.shape[0]):
        end = int(offs[e])
        outs.append(x[start:end] @ w[e])
        start = end
    return torch.cat(outs, 0)


def test_bf16_grouped_matches_reference():
    torch.manual_seed(0)
    # EP-like (16 experts) and non-EP-like (128 experts), balanced + variable routing.
    for num_experts, per_expert in ((16, 512), (128, 128)):
        K, N = 2880, 2880
        x = torch.randn(num_experts * per_expert, K, device=DEV, dtype=torch.bfloat16) * 0.5
        w = torch.randn(num_experts, K, N, device=DEV, dtype=torch.bfloat16) * 0.02
        offs = torch.arange(per_expert, num_experts * per_expert + 1, per_expert, device=DEV, dtype=torch.int32)
        out = grouped_gemm(x, w, offs=offs)
        ref = _reference(x, w, offs)
        diff = (out.float() - ref.float()).abs().max().item()
        assert diff < 1e-1, f"E={num_experts}: grouped vs reference maxdiff {diff:.3e}"
        print(f"  bf16 grouped E={num_experts} M={per_expert}: maxdiff {diff:.2e} PASS")


def test_bf16_grouped_backward():
    torch.manual_seed(0)
    E, M, K, N = 8, 256, 512, 512
    x = (torch.randn(E * M, K, device=DEV, dtype=torch.bfloat16) * 0.5).requires_grad_(True)
    w = (torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
    offs = torch.arange(M, E * M + 1, M, device=DEV, dtype=torch.int32)
    out = grouped_gemm(x, w, offs=offs)
    (out * torch.randn_like(out)).sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
    print("  bf16 grouped backward: finite grads PASS")


def test_simulated_lowprecision_matches_format_error():
    """SIMULATED mxfp8/nvfp4 produce the expected block-scaled error vs bf16, with finite grads
    flowing to the bf16 masters (the mixed-precision contract: bf16 master, low-precision compute).

    Banded like test_deepgemm.py: the LOWER bound is what makes the assertion mean anything — a
    regression routing MXFP8/NVFP4 to the plain bf16 path scores rel ~ 0 and would sail under any
    upper tolerance. Measured rel at this shape is ~0.038 (mxfp8) / ~0.134 (nvfp4), stable to the
    4th decimal across seeds, so the bounds carry a wide margin on both sides.
    """
    torch.manual_seed(0)
    E, M, K, N = 8, 256, 2880, 2880
    bands = {GroupedGemmPrecision.MXFP8: (0.01, 0.08), GroupedGemmPrecision.NVFP4: (0.04, 0.25)}
    for prec, (lo, hi) in bands.items():
        x = (torch.randn(E * M, K, device=DEV, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        w = (torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
        offs = torch.arange(M, E * M + 1, M, device=DEV, dtype=torch.int32)
        out = grouped_gemm(x, w, offs=offs, precision=prec)  # fine-grained shape -> simulated fake-quant
        (out * torch.randn_like(out)).sum().backward()
        ref = _reference(x.detach(), w.detach(), offs).float()
        rel = ((out.detach().float() - ref).norm() / ref.norm()).item()
        assert lo < rel < hi, (
            f"{prec.value}: rel {rel:.4f} outside band {lo}-{hi} "
            f"(rel≈0 ⇒ silent bf16 fallback; rel>{hi} ⇒ broken quantization)"
        )
        assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
        print(f"  simulated {prec.value}: rel {rel:.3f} (band {lo}-{hi}) grads finite PASS")


def test_dispatch_contract():
    torch.manual_seed(0)
    E, M, K, N = 4, 64, 512, 512
    x = torch.randn(E * M, K, device=DEV, dtype=torch.bfloat16) * 0.5
    w = torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * 0.02
    offs = torch.arange(M, E * M + 1, M, device=DEV, dtype=torch.int32)

    # bf16 always works; low precision (mxfp8 / nvfp4) at this fine-grained shape routes to the SIMULATED
    # fake-quant path (the DeepGEMM gate needs wide experts + large tokens/expert, so it stays off here).
    assert grouped_gemm(x, w, offs=offs, precision=GroupedGemmPrecision.BF16).shape == (E * M, N)
    for prec in (GroupedGemmPrecision.MXFP8, GroupedGemmPrecision.NVFP4):
        out = grouped_gemm(x, w, offs=offs, precision=prec)
        assert out.shape == (E * M, N), f"{prec.value} wrong shape"
    print(f"  dispatch contract PASS (lowp=simulated at this shape; deepgemm_available={deepgemm_available()})")


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")  # the sentinel the launcher skips on; a bare exit 0 reads as a PASS
        return 0
    print(f"Grouped-GEMM tests on {torch.cuda.get_device_name()}")
    failures = []
    for test in (
        test_bf16_grouped_matches_reference,
        test_bf16_grouped_backward,
        test_simulated_lowprecision_matches_format_error,
        test_dispatch_contract,
    ):
        try:
            test()
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"  FAIL {test.__name__}: {exc}")
    if failures:
        print(f"\n{len(failures)} test(s) FAILED")
        return 1
    print("\nAll grouped-GEMM tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
