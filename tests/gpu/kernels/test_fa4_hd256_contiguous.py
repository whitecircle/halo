#!/usr/bin/env python
"""FA4's dedicated head_dim-256 kernel, behind ``install_fa4_contiguous_dedicated_head_inputs``, is as accurate
as FA2 on the strided q/k/v the head_dim-256 families hand it.

flash-attn-4 4.0.0b16 builds the hd256 kernel's forward layouts with packed strides, so a head-major q/k (Qwen3.5's
partial-rotary ``torch.cat``, GLM-4.7-Flash's MLA) or a v sliced off a fused K/V tensor (MLA, the Ulysses
all-to-all) is read scrambled. Against an fp64 reference on those exact layouts, dense and varlen, at the Qwen3.5
(16:2) and GLM-4.7-Flash MLA (20:20) head counts: the shimmed entry points' output and dq/dk/dv must stay within
1.5x of FA2's error, and the unwrapped kernel must still misread the same inputs (when it stops, upstream fixed the
layouts and the shim can go). The output gradient reaches the kernel contiguous, as from every FA4 caller; the shim
leaves it alone, so a second backward with the same gradient head-major must stay within FA2's error too, or the
backward misreads a strided dout and the shim has to copy it as well. The hd256 backward must also refuse
``deterministic=True``, which is why the FA4 veto stays on for deterministic runs. Fails without the shim (its arm
then is the unwrapped kernel), or if a wrapper drops the copy for any of q, k or v.

    torchrun --nproc_per_node=1 tests/gpu/kernels/test_fa4_hd256_contiguous.py
"""

import torch

from halo.models.attention.backend_setup import install_fa4_contiguous_dedicated_head_inputs
from halo.models.attention.selection import FA4_DEDICATED_HEAD_DIM
from tests.common.harness import gpu_test_main
from tests.common.numerics import max_abs_rel_err
from tests.common.rank_log import log

D = FA4_DEDICATED_HEAD_DIM
# (query heads, key/value heads): Qwen3.5/3.6 and Qwen3-Next GQA, GLM-4.7-Flash MLA.
HEAD_COUNTS = ((16, 2), (20, 20))
DOCUMENT_LENGTHS = (700, 900, 448)
S = sum(DOCUMENT_LENGTHS)
# The shimmed kernel's error may exceed FA2's (two bf16 kernels, different accumulation order) by this factor.
FA2_ERROR_RATIO = 1.5
# The unwrapped kernel on head-major inputs reads another head's rows: its output error is O(1), FA2's O(1e-3).
MISREAD_RATIO = 10.0


def _strided_inputs(heads: int, kv_heads: int, *, varlen: bool, seed: int):
    """q and k head-major (``[.., H, S, D]`` transposed to ``[.., S, H, D]``), v the second half of a fused
    ``[.., S, KV, 2D]`` tensor: last dim contiguous, every other stride not the packed one."""
    torch.manual_seed(seed)
    lead = () if varlen else (1,)
    q = torch.randn(*lead, heads, S, D, device="cuda", dtype=torch.bfloat16).transpose(-3, -2)
    k = torch.randn(*lead, kv_heads, S, D, device="cuda", dtype=torch.bfloat16).transpose(-3, -2)
    v = torch.randn(*lead, S, kv_heads, 2 * D, device="cuda", dtype=torch.bfloat16)[..., D:]
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    assert not (q.is_contiguous() or k.is_contiguous() or v.is_contiguous())
    return q, k, v


def _varlen_kwargs():
    cu = torch.tensor([0, *torch.tensor(DOCUMENT_LENGTHS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
    return {
        "cu_seqlens_q": cu,
        "cu_seqlens_k": cu,
        "max_seqlen_q": max(DOCUMENT_LENGTHS),
        "max_seqlen_k": max(DOCUMENT_LENGTHS),
    }


def _reference(q, k, v, *, varlen: bool):
    """fp64 causal attention, per document under ``varlen``, in the input's own layout."""
    heads, kv_heads = q.shape[-2], k.shape[-2]
    q, k, v = (t.double().reshape(S, t.shape[-2], D) for t in (q, k, v))
    k, v = (t.repeat_interleave(heads // kv_heads, dim=1) for t in (k, v))
    scores = torch.einsum("qhd,khd->hqk", q, k) / D**0.5
    position = torch.arange(S, device="cuda")
    allowed = position[:, None] >= position[None, :]
    if varlen:
        document = torch.repeat_interleave(
            torch.arange(len(DOCUMENT_LENGTHS), device="cuda"), torch.tensor(DOCUMENT_LENGTHS, device="cuda")
        )
        allowed &= document[:, None] == document[None, :]
    out = torch.einsum("hqk,khd->qhd", scores.masked_fill(~allowed, float("-inf")).softmax(-1), v)
    return out if varlen else out.unsqueeze(0)


def _attend(fn, q, k, v, *, varlen: bool, **kwargs):
    out = fn(q, k, v, causal=True, **(_varlen_kwargs() if varlen else {}), **kwargs)
    return out[0] if isinstance(out, tuple) else out


def _errors(
    fn, heads, kv_heads, *, varlen: bool, backward: bool = True, head_major_dout: bool = False
) -> dict[str, float]:
    """Output (and dq/dk/dv) error of ``fn`` against the fp64 reference on the same strided leaves, the output
    gradient handed to ``fn``'s backward contiguous or, with ``head_major_dout``, head-major."""
    q, k, v = (t.requires_grad_(True) for t in _strided_inputs(heads, kv_heads, varlen=varlen, seed=0))
    out = _attend(fn, q, k, v, varlen=varlen)
    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    ref = _reference(q64, k64, v64, varlen=varlen)
    errors = {"o": max_abs_rel_err(out, ref)}
    if backward:
        grad = torch.randn(ref.shape, device="cuda", dtype=ref.dtype)
        dout = grad.to(out.dtype)
        if head_major_dout:
            dout = dout.transpose(-3, -2).contiguous().transpose(-3, -2)
            assert dout.stride(-1) == 1 and not dout.is_contiguous()
        out.backward(dout)
        ref.backward(grad)
        for name, got, want in (("dq", q.grad, q64.grad), ("dk", k.grad, k64.grad), ("dv", v.grad, v64.grad)):
            errors[name] = max_abs_rel_err(got, want)
    return errors


def _deterministic_backward_refused(fn, heads, kv_heads) -> bool:
    q, k, v = (t.requires_grad_(True) for t in _strided_inputs(heads, kv_heads, varlen=False, seed=1))
    try:
        _attend(fn, q, k, v, varlen=False, deterministic=True).float().sum().backward()
    except AssertionError as refusal:
        return "deterministic" in str(refusal)
    return False


@gpu_test_main(exact_world_size=1, partial_state=False, prefix="fa4_hd256_contiguous")
def run(ctx):
    import flash_attn  # noqa: PLC0415 - optional dependency (FA2, the reference kernel)
    import flash_attn.cute as cute  # noqa: PLC0415 - optional Blackwell-only dependency

    checks = {"installs": install_fa4_contiguous_dedicated_head_inputs()}
    entry_points = (cute.flash_attn_func, cute.flash_attn_varlen_func)
    checks["install_is_idempotent"] = install_fa4_contiguous_dedicated_head_inputs() and entry_points == (
        cute.flash_attn_func,
        cute.flash_attn_varlen_func,
    )
    metrics = {}
    for heads, kv_heads in HEAD_COUNTS:
        for varlen in (False, True):
            label = f"{heads}x{kv_heads}_{'varlen' if varlen else 'dense'}"
            shimmed = cute.flash_attn_varlen_func if varlen else cute.flash_attn_func
            fa2 = flash_attn.flash_attn_varlen_func if varlen else flash_attn.flash_attn_func
            fa4_errors = _errors(shimmed, heads, kv_heads, varlen=varlen)
            fa2_errors = _errors(fa2, heads, kv_heads, varlen=varlen)
            head_major_dout = _errors(shimmed, heads, kv_heads, varlen=varlen, head_major_dout=True)
            misread = _errors(shimmed.__wrapped__, heads, kv_heads, varlen=varlen, backward=False)["o"]
            log(
                f"  [{label}] fa4_shim {fa4_errors}  head-major dout {head_major_dout}  fa2 {fa2_errors}  "
                f"unwrapped o {misread:.2e}"
            )
            for name, error in fa4_errors.items():
                checks[f"{label}_{name}_within_fa2_error"] = error <= FA2_ERROR_RATIO * fa2_errors[name]
                metrics[f"{label}_{name}_fa4_shim"] = error
                metrics[f"{label}_{name}_fa2"] = fa2_errors[name]
            for name in ("dq", "dk", "dv"):
                checks[f"{label}_{name}_head_major_dout_within_fa2_error"] = (
                    head_major_dout[name] <= FA2_ERROR_RATIO * fa2_errors[name]
                )
                metrics[f"{label}_{name}_fa4_shim_head_major_dout"] = head_major_dout[name]
            checks[f"{label}_unwrapped_kernel_still_misreads_strided_inputs"] = not (
                misread < MISREAD_RATIO * fa2_errors["o"]
            )
            metrics[f"{label}_o_unwrapped"] = misread
        checks[f"{heads}x{kv_heads}_deterministic_backward_refused"] = _deterministic_backward_refused(
            cute.flash_attn_func, heads, kv_heads
        )
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
