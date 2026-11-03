#!/usr/bin/env python
"""FA4's head_dim-256 path is as accurate as FA2 on those families' strided q/k/v, against an fp64 reference.

Head-major q/k, a v sliced off a fused K/V, or a Ulysses receive buffer reach the kernel with every stride but the last
unpacked; flash-attn-4 4.0.0b16 read those scrambled. ``deterministic=True`` is measured, not gated: a deterministic
run keeps head_dim-256 attention off FA4.
"""

import torch

from halo.models.attention.selection import FA4_DEDICATED_HEAD_DIM
from tests.common.harness import gpu_test_main
from tests.common.numerics import max_abs_rel_err
from tests.common.rank_log import log

D = FA4_DEDICATED_HEAD_DIM
# (query heads, key/value heads): Qwen3.5/3.6 and Qwen3-Next GQA, GLM-4.7-Flash MLA.
HEAD_COUNTS = ((16, 2), (20, 20))
DOCUMENT_LENGTHS = (700, 900, 448)
S = sum(DOCUMENT_LENGTHS)
# Two bf16 kernels, different accumulation order; a misread is O(1).
FA2_ERROR_RATIO = 1.5
# "head_major": q/k head-major and v off a fused K/V; "one_buffer": the Ulysses all-to-all's received buffer.
LAYOUTS = ("head_major", "one_buffer")


def _strided_inputs(heads: int, kv_heads: int, *, varlen: bool, seed: int, layout: str = "head_major"):
    """q, k and v ``[.., S, H, D]`` with the last dim contiguous and every other stride not the packed one."""
    torch.manual_seed(seed)
    lead = () if varlen else (1,)
    if layout == "one_buffer":
        buffer = torch.randn(*lead, S, (heads + 2 * kv_heads) * D, device="cuda", dtype=torch.bfloat16)
        q, k, v = (
            part.unflatten(-1, (-1, D)) for part in buffer.split([heads * D, kv_heads * D, kv_heads * D], dim=-1)
        )
    else:
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


def _errors(fn, heads, kv_heads, *, varlen: bool, layout: str, head_major_dout: bool = False) -> dict[str, float]:
    """Against the fp64 reference on the same leaves; ``head_major_dout`` hands the output gradient head-major."""
    q, k, v = (t.requires_grad_(True) for t in _strided_inputs(heads, kv_heads, varlen=varlen, seed=0, layout=layout))
    out = _attend(fn, q, k, v, varlen=varlen)
    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    ref = _reference(q64, k64, v64, varlen=varlen)
    errors = {"o": max_abs_rel_err(out, ref)}
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


def _deterministic_backward_repeats(fn, heads, kv_heads) -> float:
    """1.0 when two ``deterministic=True`` backwards are bitwise-equal, 0.0 when they differ, -1.0 when the kernel
    refuses the flag."""
    q, k, v = (t.requires_grad_(True) for t in _strided_inputs(heads, kv_heads, varlen=False, seed=1))
    dout = torch.randn(q.shape, device="cuda", dtype=q.dtype)
    grads = []
    for _ in range(2):
        try:
            _attend(fn, q, k, v, varlen=False, deterministic=True).backward(dout)
        except AssertionError as refusal:
            if "deterministic" not in str(refusal):
                raise
            return -1.0
        grads.append([t.grad.clone() for t in (q, k, v)])
        q.grad = k.grad = v.grad = None
    return float(all(torch.equal(a, b) for a, b in zip(*grads, strict=True)))


@gpu_test_main(exact_world_size=1, partial_state=False, prefix="fa4_hd256_strided_inputs")
def run(ctx):
    import flash_attn  # noqa: PLC0415 - optional dependency (FA2, the reference kernel)
    import flash_attn.cute as cute  # noqa: PLC0415 - optional Blackwell-only dependency

    checks, metrics = {}, {}
    for heads, kv_heads in HEAD_COUNTS:
        for layout in LAYOUTS:
            for varlen in (False, True):
                label = f"{heads}x{kv_heads}_{layout}_{'varlen' if varlen else 'dense'}"
                fa4 = cute.flash_attn_varlen_func if varlen else cute.flash_attn_func
                fa2 = flash_attn.flash_attn_varlen_func if varlen else flash_attn.flash_attn_func
                fa4_errors = _errors(fa4, heads, kv_heads, varlen=varlen, layout=layout)
                fa2_errors = _errors(fa2, heads, kv_heads, varlen=varlen, layout=layout)
                head_major_dout = _errors(fa4, heads, kv_heads, varlen=varlen, layout=layout, head_major_dout=True)
                log(f"  [{label}] fa4 {fa4_errors}  head-major dout {head_major_dout}  fa2 {fa2_errors}")
                for name, error in fa4_errors.items():
                    checks[f"{label}_{name}_within_fa2_error"] = error <= FA2_ERROR_RATIO * fa2_errors[name]
                    metrics[f"{label}_{name}_fa4"] = error
                    metrics[f"{label}_{name}_fa2"] = fa2_errors[name]
                for name in ("dq", "dk", "dv"):
                    checks[f"{label}_{name}_head_major_dout_within_fa2_error"] = (
                        head_major_dout[name] <= FA2_ERROR_RATIO * fa2_errors[name]
                    )
                    metrics[f"{label}_{name}_fa4_head_major_dout"] = head_major_dout[name]
        repeats = _deterministic_backward_repeats(cute.flash_attn_func, heads, kv_heads)
        log(f"  [{heads}x{kv_heads}] deterministic backward bitwise across two calls: {repeats}")
        metrics[f"{heads}x{kv_heads}_deterministic_backward_bitwise"] = repeats
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
