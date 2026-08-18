#!/usr/bin/env python
"""GPU test for the native DeepGEMM fp8 / fp4 grouped MoE path (src/kernels/lowp/deepgemm.py).

This is the opt-in native low-precision capability: it runs on Blackwell's fp8/fp4 tensor cores and
converges, but is net-slower than bf16 at every training shape (so it is never auto-selected — see
:func:`~src.kernels.lowp.deepgemm.use_deepgemm`). The test validates *correctness*, not speed. ``deep_gemm``
is an opt-in install, so the fp8/fp4 kernel checks only run where it is present — but "not present" and
"present and broken" are separated first (``test_availability_gate_is_honest``), because
``deepgemm_available`` swallows every import error and would otherwise let a broken install delete both
kernel checks from the suite while the node still exits 0.

- the fp8 and fp4 forwards match a bf16 reference within — and crucially *not below* — the inherent block
  error (a relerr near 0 would mean the kernel silently fell back to bf16; the band lower-bound catches
  that), for even and uneven (incl. empty-expert) routing. fp4 also exercises the K-not-÷256 zero-pad
  (K=2880 → 3072) that lets gpt-oss-width experts run on the fp4 tensor cores,
- the backward is bf16-exact (Wgrad-in-HP) and finite,
- the per-step weight cache keeps training finite across optimizer steps,
- ``use_deepgemm`` is OFF by default and only opts in under ``HALO_DEEPGEMM_NATIVE=1`` at a qualifying shape.

Run: python tests/gpu/kernels/test_deepgemm.py
"""

import importlib
import os
import sys

import torch
import torch.nn.functional as F

from src.kernels.lowp.deepgemm import deepgemm_available, deepgemm_grouped_gemm, use_deepgemm

DEV = "cuda"

# Per-format relerr band vs bf16. Lower bound > 0 so a silent bf16 fallback (relerr ≈ 0) FAILS the test —
# i.e. we assert the kernel really ran in low precision on the tensor cores.
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
            out = deepgemm_grouped_gemm(x, w, offs=offs, precision=precision)
            ref = _bf16_ref(x.detach(), w.detach(), offs)
            rel = ((out.detach().float() - ref.float()).norm() / ref.float().norm()).item()
            assert lo < rel < hi, (
                f"{precision} {name}: fwd rel-vs-bf16 {rel:.4f} outside band {lo}-{hi} "
                f"(rel≈0 ⇒ silent bf16 fallback; rel>{hi} ⇒ broken)"
            )
            g = torch.randn_like(out)
            out.backward(g)
            gx_ref = F.grouped_mm(g, w.detach().transpose(-2, -1), offs=offs)
            gw_ref = F.grouped_mm(x.detach().transpose(-2, -1), g, offs=offs)
            assert torch.allclose(x.grad, gx_ref, atol=1e-2, rtol=1e-2), f"{precision} {name}: grad_x not bf16-exact"
            assert torch.allclose(w.grad, gw_ref, atol=1e-2, rtol=1e-2), f"{precision} {name}: grad_w not bf16-exact"
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
        out = deepgemm_grouped_gemm(x, w, offs=offs, precision="mxfp8")
        loss = out.float().pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), losses
    print(f"  weight cache across 3 opt steps: losses {[f'{l:.4f}' for l in losses]} finite — OK")


def test_gate():
    # OFF by default (not opted in), at every shape — native is never auto-selected.
    assert not use_deepgemm(256, 2880), "default must not route to DeepGEMM"
    assert not use_deepgemm(8192, 8192), "default must not route to DeepGEMM even at a wide shape"
    # Opt-in: enabled only for qualifying (wide / large-M) shapes; still off below the floor.
    prev = os.environ.get("HALO_DEEPGEMM_NATIVE")
    os.environ["HALO_DEEPGEMM_NATIVE"] = "1"
    try:
        assert not use_deepgemm(256, 2880), "opted-in but below the shape floor → still off"
        # The gate also requires deep_gemm to be importable. On an image without it (the default —
        # deep_gemm is an opt-in install), use_deepgemm stays off even at a qualifying shape.
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
    """``deepgemm_available()`` must agree with what ``import deep_gemm`` actually does, and the only
    acceptable reason for it to be False is that the package is genuinely NOT INSTALLED.

    It swallows every ``Exception``, so an installed-but-broken ``deep_gemm`` (a missing symbol, a
    broken transitive dependency, a CUDA ABI mismatch) reads exactly like the opt-in package being
    absent. Without this check that regression deletes both kernel tests below from the suite while
    the node still exits 0 — the failure mode is a silently shrinking test suite, not a red test.
    """
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
        # deep_gemm is an opt-in install (see agent-docs/optimization/low-precision-moe-kernels.md), so its
        # ABSENCE is a supported state. Anything else is a broken install, and must be loud.
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
        # deep_gemm ships in neither image, so this is the normal path. Report it as a SKIP, not a
        # PASS: the gate above is verified, but the fp8/fp4 kernel tests below never ran.
        print("SKIP: deep_gemm absent (opt-in); the gate is verified, the fp8/fp4 kernel tests are not")
        sys.exit(0)
    print("test_deepgemm_forward_backward")
    test_deepgemm_forward_backward()
    print("test_weight_cache_across_steps")
    test_weight_cache_across_steps()
    print("ALL PASS")
