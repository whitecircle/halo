#!/usr/bin/env python
"""The native DeepGEMM fp8 / mxfp4 path: correctness of its forward, its bf16 and fp8 backward and its per-step weight
cache, and the warm-up of the weight-gradient kernel DeepGEMM compiles for long groups. mxfp4 also exercises the
K-not-÷256 zero-pad (K=2880 → 3072)."""

import contextlib
import itertools
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.deepgemm import deepgemm_available, deepgemm_grouped_gemm
from halo.kernels.lowp.spec import LOWP_RECIPES, LowpSpec
from tests.common.numerics import fro_rel_err
from tests.common.tolerances import TOL

DEV = "cuda"

# Lower bound > 0, so a silent bf16 fallback (relerr ≈ 0) fails.
_BANDS = {"fp8": (0.005, 0.10), "mxfp4": (0.03, 0.30)}
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
    # A mean of 2048 rows per expert and more: the weight gradient's one-store-stage kernel, with an empty expert.
    ("long groups", 4, 256, 256, [4096, 0, 4096, 2048]),
]
# A mean-reduced loss's scale: every gradient tile's largest element far under the reference cast's 1e-4 floor.
_SMALL_GRADIENT = 1e-9


@contextlib.contextmanager
def _wgrad_output_poisoned():
    """NaN in the weight-gradient kernel's unfilled output before each launch, so a block it skips reads as NaN."""
    kernels, _ = deepgemm._deep_gemm()
    real = kernels.k_grouped_fp8_gemm_tn_contiguous

    def poisoned(a, b, d, *args, **kwargs):
        d.fill_(float("nan"))
        return real(a, b, d, *args, **kwargs)

    kernels.k_grouped_fp8_gemm_tn_contiguous = poisoned
    try:
        yield
    finally:
        kernels.k_grouped_fp8_gemm_tn_contiguous = real


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
    for precision, (lo, hi) in _BANDS.items():
        for name, E, K, N, counts in _SHAPES:
            T = sum(counts)
            x = (torch.randn(T, K, device=DEV, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
            w = (torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * (K**-0.5)).requires_grad_(True)
            offs = torch.cumsum(torch.tensor(counts, device=DEV, dtype=torch.int32), 0).to(torch.int32)
            out = deepgemm_grouped_gemm(x, w, offs=offs, lowp=LowpSpec(precision=precision, backend="native"))
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
    dgrad runs on an FP8 weight too, at a unit-scale gradient and at a mean-reduced loss's; the lower bound fails a
    silent bf16 backward."""
    torch.manual_seed(0)
    lo, hi = _FP8_GRAD_BAND
    cases = [(*s, torch.bfloat16, 1.0) for s in _SHAPES]
    # An fp32 master takes its gradient in fp32, which the kernel accumulates into directly.
    cases += [(*_SHAPES[1], torch.float32, 1.0), (*_SHAPES[1], torch.bfloat16, _SMALL_GRADIENT)]
    for precision, (name, E, K, N, counts, w_dtype, grad_scale) in itertools.product(_BANDS, cases):
        T = sum(counts)
        x = (torch.randn(T, K, device=DEV, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        w = (torch.randn(E, K, N, device=DEV, dtype=w_dtype) * (K**-0.5)).requires_grad_(True)
        offs = torch.cumsum(torch.tensor(counts, device=DEV, dtype=torch.int32), 0).to(torch.int32)
        g = torch.randn(T, N, device=DEV, dtype=torch.bfloat16) * grad_scale
        out = deepgemm_grouped_gemm(
            x, w, offs=offs, lowp=LowpSpec(precision=precision, backend="native", backward_precision="fp8")
        )
        with _wgrad_output_poisoned():
            out.backward(g)
        gx_ref = F.grouped_mm(g, w.detach().to(torch.bfloat16).transpose(-2, -1), offs=offs)
        gw_ref = F.grouped_mm(x.detach().transpose(-2, -1), g, offs=offs)
        assert w.grad.shape == w.shape and w.grad.dtype == w.dtype
        rels = {"grad_x": fro_rel_err(x.grad, gx_ref), "grad_w": fro_rel_err(w.grad, gw_ref)}
        for grad, rel in rels.items():
            assert lo < rel < hi, f"{precision} {name}: {grad} rel-vs-bf16 {rel:.4f} outside band {lo}-{hi}"
        empty = [e for e, c in enumerate(counts) if c == 0]
        assert not w.grad[empty].any(), f"{precision} {name}: an expert routed no token got a weight gradient"
        print(
            f"  {precision} fp8 backward {name:14s} {str(w_dtype)[6:]:8s} T={T:5d} grad x{grad_scale:g} "
            f"grad_x {rels['grad_x']:.4f} grad_w {rels['grad_w']:.4f} — OK"
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
    for precision in _BANDS:
        torch.manual_seed(0)
        w = (torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * (K**-0.5)).requires_grad_(True)
        opt = torch.optim.SGD([w], lr=1e-3)
        previous = None
        for step in range(3):
            x = torch.randn(8 * 256, K, device=DEV, dtype=torch.bfloat16) * 0.3
            out = deepgemm_grouped_gemm(x, w, offs=offs, lowp=LowpSpec(precision=precision, backend="native"))
            fmt = LOWP_RECIPES[precision].quant_format
            served = deepgemm._weight_caches[fmt].get(w, w._version)
            assert served is not None, f"{precision} step {step}: the forward left no entry for this version"
            with torch.no_grad():
                fresh = deepgemm._quant_weight(w, fmt, cacheable=False)
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


# A child of its own, as DeepGEMM reads its JIT cache directory once per process. Each line: rows per expert, then the
# kernels that forward and backward compiled.
_WGRAD_COMPILE_PROBE = """
import os, sys, torch
from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.spec import LowpSpec

LOWP = LowpSpec(precision=sys.argv[3], backend="native", backward_precision="fp8")

def compiled():
    return sum(len(dirs) for _, dirs, _ in os.walk(os.environ["DG_JIT_CACHE_DIR"]))

E, K, N = 4, 256, 512
w_dtype = getattr(torch, sys.argv[2])
w = (torch.randn(E, K, N, device="cuda", dtype=w_dtype) * K**-0.5).requires_grad_(True)

def step(rows):
    x = torch.randn(E * rows, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    offs = torch.arange(1, E + 1, device="cuda", dtype=torch.int32) * rows
    before = compiled()
    out = deepgemm.deepgemm_grouped_gemm(x, w, offs=offs, lowp=LOWP)
    out.backward(torch.randn_like(out))
    torch.cuda.synchronize()
    return compiled() - before

step(1)
if sys.argv[1] == "warm":
    deepgemm.warm_fp8_wgrad(E, K, N, w_dtype, torch.device("cuda"))
for rows in (3, 129, 1024, 2048, 4096, 8192):
    print(rows, step(rows))
"""


def _wgrad_compiles(mode: str, w_dtype: str, precision: str) -> dict[int, int]:
    with tempfile.TemporaryDirectory() as cache:
        out = subprocess.run(
            [sys.executable, "-c", _WGRAD_COMPILE_PROBE, mode, w_dtype, precision],
            env={**os.environ, "DG_JIT_CACHE_DIR": cache},
            capture_output=True,
            text=True,
            check=False,
        )
    assert out.returncode == 0, f"the {mode} {w_dtype} probe failed:\n{out.stderr[-2000:]}"
    return {int(rows): int(count) for rows, count in (line.split() for line in out.stdout.splitlines())}


def test_the_wgrad_warm_up_compiles_the_long_group_kernel():
    """On an empty JIT cache, a forward and backward past one step and ``warm_fp8_wgrad`` compile nothing at any rows
    per expert. Without the warm-up the first long receive compiles a kernel mid-step, which stalls every EP peer: the control, and what tells
    a DeepGEMM pin whose heuristic no longer needs the warm-up."""
    for precision, w_dtype in (("fp8", "bfloat16"), ("fp8", "float32"), ("mxfp4", "bfloat16")):
        cold = _wgrad_compiles("cold", w_dtype, precision)
        assert sum(cold.values()) > 0, f"{precision} {w_dtype}: no long-group kernel to warm, {cold}"
        warm = _wgrad_compiles("warm", w_dtype, precision)
        assert not any(warm.values()), f"{precision} {w_dtype}: a step compiled a kernel after the warm-up, {warm}"
        print(f"  {precision:5s} {w_dtype:8s} cold compiles {cold}, none after the warm-up — OK")


def test_a_non_finite_mxfp4_weight_raises_at_the_one_flag_read():
    """The cast decodes a NaN weight as zeros and flags it on the device; the read raises, cached weight or not."""
    lowp = LowpSpec(precision="mxfp4", backend="native")
    x = torch.randn(256, 256, device=DEV, dtype=torch.bfloat16)
    offs = torch.tensor([128, 256], device=DEV, dtype=torch.int32)
    for cacheable in (True, False):
        w = torch.randn(2, 256, 256, device=DEV, dtype=torch.bfloat16) / 16
        deepgemm_grouped_gemm(x, w, offs=offs, lowp=lowp, weight_cacheable=cacheable)
        deepgemm.raise_on_non_finite_weight()
        w[1, 3, 5] = float("nan")
        out = deepgemm_grouped_gemm(x, w, offs=offs, lowp=lowp, weight_cacheable=cacheable)
        assert out.isfinite().all(), "premise: the output hides the NaN weight"
        try:
            deepgemm.raise_on_non_finite_weight()
        except ValueError as exc:
            assert "Non-finite expert weight" in str(exc)
        else:
            raise AssertionError(f"a NaN weight (cacheable={cacheable}) passed the flag read")
        deepgemm.raise_on_non_finite_weight()  # the raise cleared the flag
    print("  mxfp4 NaN weight: finite output, raised at the flag read, cached and uncached — OK")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        sys.exit(0)
    if not deepgemm_available():  # the Hopper image ships no deep_gemm; a broken install raises here
        print("SKIP: deep_gemm absent")
        sys.exit(0)
    print("test_deepgemm_forward_backward")
    test_deepgemm_forward_backward()
    print("test_deepgemm_fp8_backward")
    test_deepgemm_fp8_backward()
    print("test_weight_cache_across_steps")
    test_weight_cache_across_steps()
    print("test_the_wgrad_warm_up_compiles_the_long_group_kernel")
    test_the_wgrad_warm_up_compiles_the_long_group_kernel()
    print("test_a_non_finite_mxfp4_weight_raises_at_the_one_flag_read")
    test_a_non_finite_mxfp4_weight_raises_at_the_one_flag_read()
    print("ALL PASS")
