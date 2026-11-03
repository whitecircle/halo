#!/usr/bin/env python
"""The opt-in native DeepGEMM fp8 / fp4 path: correctness, since it is net-slower than bf16 at training shapes.

``deepgemm_available`` swallows every import error, so a broken ``deep_gemm`` must be told apart from an absent one, or
both kernel checks vanish while the node exits 0. fp4 also exercises the K-not-÷256 zero-pad (K=2880 → 3072).
"""

import importlib
import os
import sys

import torch
import torch.nn.functional as F

from halo.kernels.lowp.deepgemm import deepgemm_available, deepgemm_grouped_gemm, use_deepgemm
from tests.common.numerics import fro_rel_err
from tests.common.tolerances import TOL

DEV = "cuda"

# Lower bound > 0, so a silent bf16 fallback (relerr ≈ 0) fails.
_BANDS = {"mxfp8": (0.005, 0.10), "nvfp4": (0.03, 0.30)}


def _bf16_ref(x, w, offs):
    out = torch.empty(x.shape[0], w.shape[2], device=DEV, dtype=torch.bfloat16)
    s = 0
    for e in range(w.shape[0]):
        e_end = int(offs[e])
        if e_end > s:
            out[s:e_end] = x[s:e_end] @ w[e]
        s = e_end
    return out


def test_deepgemm_forward_backward():
    torch.manual_seed(0)
    for precision in ("mxfp8", "nvfp4"):
        lo, hi = _BANDS[precision]
        for name, E, K, N, counts in [
            ("even", 8, 2880, 2880, [256] * 8),
            ("uneven+empty", 8, 2880, 2880, [100, 245, 0, 156, 203, 0, 134, 90]),
        ]:
            T = sum(counts)
            x = (torch.randn(T, K, device=DEV, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
            w = (torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * (K**-0.5)).requires_grad_(True)
            offs = torch.cumsum(torch.tensor(counts, device=DEV, dtype=torch.int32), 0).to(torch.int32)
            out = deepgemm_grouped_gemm(x, w, offs=offs, quant_format=precision)
            ref = _bf16_ref(x.detach(), w.detach(), offs)
            rel = fro_rel_err(out.detach(), ref)
            assert lo < rel < hi, (
                f"{precision} {name}: fwd rel-vs-bf16 {rel:.4f} outside band {lo}-{hi} "
                f"(rel≈0 ⇒ silent bf16 fallback; rel>{hi} ⇒ broken)"
            )
            g = torch.randn_like(out)
            out.backward(g)
            gx_ref = F.grouped_mm(g, w.detach().transpose(-2, -1), offs=offs)
            gw_ref = F.grouped_mm(x.detach().transpose(-2, -1), g, offs=offs)
            assert torch.allclose(x.grad, gx_ref, atol=TOL.kernel_atol, rtol=TOL.kernel_rtol), (
                f"{precision} {name}: grad_x not bf16-exact"
            )
            assert torch.allclose(w.grad, gw_ref, atol=TOL.kernel_atol, rtol=TOL.kernel_rtol), (
                f"{precision} {name}: grad_w not bf16-exact"
            )
            assert x.grad.isfinite().all() and w.grad.isfinite().all()
            print(
                f"  {precision} {name:14s} T={T:5d} fwd-rel {rel:.4f} (band {lo}-{hi}), "
                f"backward bf16-exact, finite — OK"
            )


def test_weight_cache_across_steps():
    torch.manual_seed(0)
    E, K, N = 8, 2048, 2048
    w = (torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * (K**-0.5)).requires_grad_(True)
    opt = torch.optim.SGD([w], lr=1e-2)
    offs = torch.cumsum(torch.full((E,), 256, device=DEV, dtype=torch.int32), 0).to(torch.int32)
    losses = []
    for _ in range(3):
        x = torch.randn(8 * 256, K, device=DEV, dtype=torch.bfloat16) * 0.3
        out = deepgemm_grouped_gemm(x, w, offs=offs, quant_format="mxfp8")
        loss = out.float().pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), losses
    print(f"  weight cache across 3 opt steps: losses {[f'{l:.4f}' for l in losses]} finite — OK")


def test_gate():
    # Off by default at every shape: native is never auto-selected.
    assert not use_deepgemm(256, 2880), "default must not route to DeepGEMM"
    assert not use_deepgemm(8192, 8192), "default must not route to DeepGEMM even at a wide shape"
    # Opt-in: only for qualifying (wide / large-M) shapes.
    prev = os.environ.get("HALO_DEEPGEMM_NATIVE")
    os.environ["HALO_DEEPGEMM_NATIVE"] = "1"
    try:
        assert not use_deepgemm(256, 2880), "opted-in but below the shape floor → still off"
        # The gate also requires deep_gemm importable, an opt-in install, so it stays off without it even at a
        # qualifying shape.
        if deepgemm_available():
            assert use_deepgemm(2048, 8192), "opted-in at a qualifying wide/large-M shape → on"
        else:
            assert not use_deepgemm(2048, 8192), "deep_gemm not importable → off even when opted-in"
    finally:
        if prev is None:
            os.environ.pop("HALO_DEEPGEMM_NATIVE", None)
        else:
            os.environ["HALO_DEEPGEMM_NATIVE"] = prev
    print("  use_deepgemm gate: OFF by default; opt-in only above the wide/large-M floor — OK")


def test_availability_gate_is_honest():
    """The only acceptable reason for False is the package being absent: a missing symbol or CUDA ABI mismatch reads
    the same, and would silently delete both kernel tests from the suite."""
    try:
        importlib.import_module("deep_gemm")
        error = None
    except BaseException as exc:
        error = exc

    assert deepgemm_available() == (error is None), (
        f"deepgemm_available() says {deepgemm_available()} but `import deep_gemm` "
        f"{'raised ' + type(error).__name__ if error else 'succeeded'} — the gate misreports the runtime"
    )

    if error is not None:
        # Absence is a supported state (agent-docs/optimization/low-precision-moe-kernels.md); anything else must be
        # loud.
        absent = isinstance(error, ModuleNotFoundError) and error.name == "deep_gemm"
        assert absent, (
            f"deep_gemm is installed but does not import: {type(error).__name__}: {error}. "
            "That is a regression, not the opt-in package being absent — the fp8/fp4 kernel tests "
            "below would otherwise be skipped silently."
        )
        print("  deep_gemm not installed (opt-in); gate reports unavailable and the fp8/fp4 tests are skipped — OK")
    else:
        print("  deep_gemm imports and the gate reports available — OK")
    return error is None


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        sys.exit(0)
    print("test_gate")
    test_gate()
    print("test_availability_gate_is_honest")
    installed = test_availability_gate_is_honest()
    if not installed:
        # The normal path, since neither image ships deep_gemm. A SKIP, since the fp8/fp4 kernel tests below never ran.
        print("SKIP: deep_gemm absent (opt-in); the gate is verified, the fp8/fp4 kernel tests are not")
        sys.exit(0)
    print("test_deepgemm_forward_backward")
    test_deepgemm_forward_backward()
    print("test_weight_cache_across_steps")
    test_weight_cache_across_steps()
    print("ALL PASS")
