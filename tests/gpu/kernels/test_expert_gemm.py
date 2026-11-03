#!/usr/bin/env python
"""Grouped-GEMM dispatch: bf16 matches a per-expert loop, every format takes the simulated fake-quant path by default,
and ``HALO_DEEPGEMM_NATIVE=1`` takes mxfp8 / mxfp4 native at every shape."""

import os

import torch

from halo.kernels.expert_gemm import expert_gemm
from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.deepgemm import deepgemm_available
from tests.common.harness import gpu_test_main, record_check
from tests.common.numerics import fro_rel_err
from tests.common.rank_log import log

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
        out = expert_gemm(x, w, offs=offs)
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
    out = expert_gemm(x, w, offs=offs)
    (out * torch.randn_like(out)).sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
    print("  bf16 grouped backward: finite grads PASS")


def test_simulated_lowprecision_matches_format_error():
    """Banded: routing low precision to plain bf16 scores rel ~ 0 and passes any upper tolerance. Measured ~0.038
    (mxfp8) / ~0.134 (nvfp4), stable to the 4th decimal across seeds."""
    torch.manual_seed(0)
    E, M, K, N = 8, 256, 2880, 2880
    bands = {"mxfp8": (0.01, 0.08), "nvfp4": (0.04, 0.25)}
    for prec, (lo, hi) in bands.items():
        x = (torch.randn(E * M, K, device=DEV, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        w = (torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
        offs = torch.arange(M, E * M + 1, M, device=DEV, dtype=torch.int32)
        out = expert_gemm(x, w, offs=offs, quant_format=prec)  # The flag is off: simulated fake-quant.
        (out * torch.randn_like(out)).sum().backward()
        ref = _reference(x.detach(), w.detach(), offs).float()
        rel = fro_rel_err(out.detach(), ref)
        assert lo < rel < hi, (
            f"{prec}: rel {rel:.4f} outside band {lo}-{hi} "
            f"(rel≈0 ⇒ silent bf16 fallback; rel>{hi} ⇒ broken quantization)"
        )
        assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
        print(f"  simulated {prec}: rel {rel:.3f} (band {lo}-{hi}) grads finite PASS")


def test_dispatch_contract():
    torch.manual_seed(0)
    E, M, K, N = 4, 64, 512, 512
    x = torch.randn(E * M, K, device=DEV, dtype=torch.bfloat16) * 0.5
    w = torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * 0.02
    offs = torch.arange(M, E * M + 1, M, device=DEV, dtype=torch.int32)

    assert expert_gemm(x, w, offs=offs).shape == (E * M, N)
    real_forward = deepgemm._deepgemm_forward
    native_calls = []

    def counted_forward(*args):
        native_calls.append(args[3])
        return real_forward(*args)

    deepgemm._deepgemm_forward = counted_forward
    try:
        for prec in ("mxfp8", "mxfp4", "nvfp4"):
            out = expert_gemm(x, w, offs=offs, quant_format=prec)
            assert out.shape == (E * M, N), f"{prec} wrong shape"
        assert not native_calls, f"the default ran native: {native_calls}"

        os.environ["HALO_DEEPGEMM_NATIVE"] = "1"
        for prec in ("mxfp8", "mxfp4", "nvfp4"):
            out = expert_gemm(x, w, offs=offs, quant_format=prec)
            assert out.shape == (E * M, N), f"opted-in {prec} wrong shape"
    finally:
        os.environ.pop("HALO_DEEPGEMM_NATIVE", None)
        deepgemm._deepgemm_forward = real_forward
    expected = ["mxfp8", "mxfp4"] if deepgemm_available() else []
    assert native_calls == expected, f"opted in, native ran {native_calls}, expected {expected}"
    print(f"  dispatch contract PASS (deepgemm_available={deepgemm_available()}; opted-in native {native_calls})")


@gpu_test_main(exact_world_size=1, prefix="grouped_gemm", partial_state=False)
def run(ctx) -> dict:
    log(f"Grouped-GEMM tests on {torch.cuda.get_device_name()}")
    checks: dict[str, bool] = {}
    record_check(checks, "bf16_grouped_matches_reference", test_bf16_grouped_matches_reference)
    record_check(checks, "bf16_grouped_backward", test_bf16_grouped_backward)
    record_check(
        checks, "simulated_lowprecision_matches_format_error", test_simulated_lowprecision_matches_format_error
    )
    record_check(checks, "dispatch_contract", test_dispatch_contract)
    return {"checks": checks}


if __name__ == "__main__":
    run()
