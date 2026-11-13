#!/usr/bin/env python
"""The native DeepGEMM fp8 / mxfp4 path: correctness of its forward, its bf16 and fp8 backward and its per-step weight
cache.

``deepgemm_available`` swallows every import error, so a broken ``deep_gemm`` must be told apart from an absent one, or
both kernel checks vanish while the node exits 0. mxfp4 also exercises the K-not-÷256 zero-pad (K=2880 → 3072).
"""

import importlib
import itertools
import sys

import torch
import torch.nn.functional as F

from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.deepgemm import DEEPGEMM_FORMATS, deepgemm_available, deepgemm_grouped_gemm
from tests.common.numerics import fro_rel_err
from tests.common.tolerances import TOL

DEV = "cuda"

# Lower bound > 0, so a silent bf16 fallback (relerr ≈ 0) fails.
_BANDS = {"mxfp8": (0.005, 0.10), "mxfp4": (0.03, 0.30)}
_FP8_GRAD_BAND = (0.005, 0.10)
_SHAPES = [
    ("even", 8, 2880, 2880, [256] * 8),
    ("uneven+empty", 8, 2880, 2880, [100, 245, 0, 156, 203, 0, 134, 90]),
    # The narrow and few-row shapes a gate on width or rows per expert would divert.
    ("one row", 4, 2048, 1024, [1] * 4),
    ("narrow gate_up", 4, 256, 256, [64] * 4),
    ("narrow down", 4, 128, 256, [64] * 4),
    # Past the wgrad kernel's 128 experts per call.
    ("130 experts", 130, 256, 512, [3, 0, 130] * 43 + [7]),
]


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
    for precision in DEEPGEMM_FORMATS:
        lo, hi = _BANDS[precision]
        for name, E, K, N, counts in _SHAPES:
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


def test_deepgemm_fp8_backward():
    """Both gradients within fp8's band of the bf16 ones, for either forward weight format, as the MXFP4 forward's
    dgrad runs on an FP8 weight too; the lower bound fails a silent bf16 backward."""
    torch.manual_seed(0)
    lo, hi = _FP8_GRAD_BAND
    # An fp32 master takes its gradient in fp32, which the kernel accumulates into directly.
    cases = [(*s, torch.bfloat16) for s in _SHAPES] + [(*_SHAPES[1], torch.float32)]
    for precision, (name, E, K, N, counts, w_dtype) in itertools.product(DEEPGEMM_FORMATS, cases):
        T = sum(counts)
        x = (torch.randn(T, K, device=DEV, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        w = (torch.randn(E, K, N, device=DEV, dtype=w_dtype) * (K**-0.5)).requires_grad_(True)
        offs = torch.cumsum(torch.tensor(counts, device=DEV, dtype=torch.int32), 0).to(torch.int32)
        g = torch.randn(T, N, device=DEV, dtype=torch.bfloat16)
        deepgemm_grouped_gemm(x, w, offs=offs, quant_format=precision, fp8_backward=True).backward(g)
        gx_ref = F.grouped_mm(g, w.detach().to(torch.bfloat16).transpose(-2, -1), offs=offs)
        gw_ref = F.grouped_mm(x.detach().transpose(-2, -1), g, offs=offs)
        assert w.grad.shape == w.shape and w.grad.dtype == w.dtype
        rels = {"grad_x": fro_rel_err(x.grad, gx_ref), "grad_w": fro_rel_err(w.grad, gw_ref)}
        for grad, rel in rels.items():
            assert lo < rel < hi, f"{precision} {name}: {grad} rel-vs-bf16 {rel:.4f} outside band {lo}-{hi}"
        empty = [e for e, c in enumerate(counts) if c == 0]
        assert not w.grad[empty].any(), f"{precision} {name}: an expert routed no token got a weight gradient"
        print(
            f"  {precision} fp8 backward {name:14s} {str(w_dtype)[6:]:8s} T={T:5d} grad_x {rels['grad_x']:.4f} grad_w {rels['grad_w']:.4f} — OK"
        )


def _same_cast(a: tuple, b: tuple) -> bool:
    """As floats, the one read of fp8 elements the production path itself runs on CUDA."""
    return all(torch.equal(x.float(), y.float()) for x, y in zip(a, b, strict=True))


def test_weight_cache_across_steps():
    """Every step's forward runs on a cast of the weight as it is then: the entry the forward leaves equals a fresh cast
    and moves with the optimizer's write. The summed loss sizes that write past bf16's rounding, as a mean over these
    rows moves no block's largest element and so no scale."""
    E, K, N = 8, 2048, 2048
    offs = torch.cumsum(torch.full((E,), 256, device=DEV, dtype=torch.int32), 0).to(torch.int32)
    for precision in DEEPGEMM_FORMATS:
        torch.manual_seed(0)
        w = (torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * (K**-0.5)).requires_grad_(True)
        opt = torch.optim.SGD([w], lr=1e-3)
        previous = None
        for step in range(3):
            x = torch.randn(8 * 256, K, device=DEV, dtype=torch.bfloat16) * 0.3
            out = deepgemm_grouped_gemm(x, w, offs=offs, quant_format=precision)
            served = deepgemm._weight_caches[precision].get(w, (w._version, precision))
            assert served is not None, f"{precision} step {step}: the forward left no entry for this version"
            with torch.no_grad():
                fresh = deepgemm._quant_weight(w, precision, K, cacheable=False)
            assert _same_cast(served, fresh), f"{precision} step {step}: the cached cast is not of the current weight"
            if previous is not None:
                assert not torch.equal(previous, served[0].float()), f"{precision} step {step}: the cast did not move"
            previous = served[0].float()
            loss = out.float().pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            assert torch.isfinite(loss), f"{precision} step {step}: loss {loss.item()}"
        print(f"  {precision} weight cache across 3 opt steps tracks the weight — OK")


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
            "That is a regression, not the package being absent — the fp8/mxfp4 kernel tests "
            "below would otherwise be skipped silently."
        )
        print("  deep_gemm not installed; the gate reports unavailable and the fp8/mxfp4 tests are skipped — OK")
    else:
        print("  deep_gemm imports and the gate reports available — OK")
    return error is None


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        sys.exit(0)
    print("test_availability_gate_is_honest")
    installed = test_availability_gate_is_honest()
    if not installed:
        # The Hopper image ships no deep_gemm. A SKIP, since the fp8/mxfp4 kernel tests below never ran.
        print("SKIP: deep_gemm absent; the gate is verified, the fp8/mxfp4 kernel tests are not")
        sys.exit(0)
    print("test_deepgemm_forward_backward")
    test_deepgemm_forward_backward()
    print("test_deepgemm_fp8_backward")
    test_deepgemm_fp8_backward()
    print("test_weight_cache_across_steps")
    test_weight_cache_across_steps()
    print("ALL PASS")
