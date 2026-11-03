#!/usr/bin/env python
"""FA4 at head_dim 128 and 16 query heads per KV head, against an fp64 reference, forward and backward: causal, varlen,
non-causal, and the rectangular log-sum-exp call a per-block attention merges over.

Every other FA4 row runs at most 8:1 here. ``--heads``/``--kv-heads`` pick the geometry; the manifest holds one row per
release ratio. Rotating the query heads must move the error by far more than the kernel's, or the row compares nothing.
"""

import argparse
from dataclasses import dataclass

import torch

from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.numerics import max_abs_rel_err
from tests.common.rank_log import log

D = 128
QUERY_LENGTHS = (700, 900, 448)
KEY_LENGTHS = (1300, 1700, 900)
# Two bf16 kernels, different accumulation order; a misread is O(1).
FA2_ERROR_RATIO = 1.5
# FA2 returns no log-sum-exp gradient to compare with: bf16 round-off of a head-summed score, far below a misread.
LSE_REL_TOL = 3e-2
MISREAD_REL_FLOOR = 0.3

parser = argparse.ArgumentParser()
parser.add_argument("--heads", type=int, required=True)
parser.add_argument("--kv-heads", type=int, required=True)
ARGS = parser.parse_args()
skip_unless_blackwell("the 16:1 GQA row")


@dataclass(frozen=True)
class Form:
    name: str
    varlen: bool
    causal: bool
    rectangular: bool = False
    # FA2 has no gradient through the log-sum-exp, so these are held to ``LSE_REL_TOL`` alone.
    lse: bool = False


FORMS = (
    Form("dense_causal", varlen=False, causal=True),
    Form("varlen_causal", varlen=True, causal=True),
    Form("dense_noncausal", varlen=False, causal=False),
    Form("varlen_causal_lse", varlen=True, causal=True, lse=True),
    Form("varlen_rectangular_lse", varlen=True, causal=False, rectangular=True, lse=True),
)


def _inputs(form: Form, *, seed: int):
    torch.manual_seed(seed)
    queries, keys = sum(QUERY_LENGTHS), sum(KEY_LENGTHS if form.rectangular else QUERY_LENGTHS)
    lead = () if form.varlen else (1,)
    q = torch.randn(*lead, queries, ARGS.heads, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(*lead, keys, ARGS.kv_heads, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(*lead, keys, ARGS.kv_heads, D, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def _segments(lengths: tuple[int, ...]) -> torch.Tensor:
    return torch.repeat_interleave(torch.arange(len(lengths), device="cuda"), torch.tensor(lengths, device="cuda"))


def _allowed(form: Form) -> torch.Tensor:
    """``[query, key]``: the keys each query attends in this form."""
    query_lengths = QUERY_LENGTHS
    key_lengths = KEY_LENGTHS if form.rectangular else QUERY_LENGTHS
    if form.varlen:
        allowed = _segments(query_lengths)[:, None] == _segments(key_lengths)[None, :]
    else:
        allowed = torch.ones(sum(query_lengths), sum(key_lengths), dtype=torch.bool, device="cuda")
    if form.causal:
        positions = torch.arange(sum(query_lengths), device="cuda")
        allowed = allowed & (positions[:, None] >= positions[None, :])
    return allowed


def _reference(q, k, v, form: Form) -> tuple[torch.Tensor, torch.Tensor]:
    """Output ``[..., query, head, D]`` and log-sum-exp ``[head, query]``, in fp64."""
    q, k, v = (t.reshape(-1, t.shape[-2], D) for t in (q, k, v))
    k, v = (t.repeat_interleave(ARGS.heads // ARGS.kv_heads, dim=1) for t in (k, v))
    scores = torch.einsum("qhd,khd->hqk", q, k) / D**0.5
    scores = scores.masked_fill(~_allowed(form), float("-inf"))
    out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
    return (out if form.varlen else out.unsqueeze(0)), scores.logsumexp(-1)


def _call(fn, q, k, v, form: Form, **kwargs):
    if form.varlen:
        key_lengths = KEY_LENGTHS if form.rectangular else QUERY_LENGTHS
        cu_q = torch.tensor([0, *torch.tensor(QUERY_LENGTHS).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, *torch.tensor(key_lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32)
        kwargs |= {
            "cu_seqlens_q": cu_q,
            "cu_seqlens_k": cu_k,
            "max_seqlen_q": max(QUERY_LENGTHS),
            "max_seqlen_k": max(key_lengths),
        }
    return fn(q, k, v, causal=form.causal, **kwargs)


def _errors(fn, form: Form) -> dict[str, float]:
    q, k, v = (t.requires_grad_(True) for t in _inputs(form, seed=0))
    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    ref, ref_lse = _reference(q64, k64, v64, form)
    grad = torch.randn(ref.shape, device="cuda", dtype=ref.dtype)
    if form.lse:
        out, lse = _call(fn, q, k, v, form, return_lse=True)
        grad_lse = torch.randn(ref_lse.shape, device="cuda", dtype=ref_lse.dtype)
        errors = {"o": max_abs_rel_err(out, ref), "lse": max_abs_rel_err(lse, ref_lse)}
        torch.autograd.backward([out, lse], [grad.to(out.dtype), grad_lse.to(lse.dtype)])
        torch.autograd.backward([ref, ref_lse], [grad, grad_lse])
    else:
        out = _call(fn, q, k, v, form)
        out = out[0] if isinstance(out, tuple) else out
        errors = {"o": max_abs_rel_err(out, ref)}
        out.backward(grad.to(out.dtype))
        ref.backward(grad)
    for name, got, want in (("dq", q.grad, q64.grad), ("dk", k.grad, k64.grad), ("dv", v.grad, v64.grad)):
        errors[name] = max_abs_rel_err(got, want)
    return errors


def _misread_error(fn, form: Form) -> float:
    """The kernel handed each query head's queries to its neighbour, read against the true reference."""
    q, k, v = _inputs(form, seed=0)
    ref, _ = _reference(q.double(), k.double(), v.double(), form)
    with torch.no_grad():
        out = _call(fn, q.roll(1, dims=-2), k, v, form)
    return max_abs_rel_err(out[0] if isinstance(out, tuple) else out, ref)


@gpu_test_main(exact_world_size=1, partial_state=False, prefix=f"fa4_gqa_{ARGS.heads}x{ARGS.kv_heads}")
def run(ctx):
    import flash_attn  # noqa: PLC0415 - optional dependency (FA2, the reference kernel)
    import flash_attn.cute as cute  # noqa: PLC0415 - optional Blackwell-only dependency

    checks, metrics = {"sixteen_query_heads_per_kv_head": ARGS.heads == 16 * ARGS.kv_heads}, {}
    for form in FORMS:
        fa4 = cute.flash_attn_varlen_func if form.varlen else cute.flash_attn_func
        fa2 = flash_attn.flash_attn_varlen_func if form.varlen else flash_attn.flash_attn_func
        fa4_errors = _errors(fa4, form)
        fa2_errors = {} if form.lse else _errors(fa2, form)
        log(f"  [{form.name}] fa4 {fa4_errors}  fa2 {fa2_errors}")
        for name, error in fa4_errors.items():
            metrics[f"{form.name}_{name}_fa4"] = error
            if form.lse:
                checks[f"{form.name}_{name}_within_bf16_round_off"] = error <= LSE_REL_TOL
            else:
                metrics[f"{form.name}_{name}_fa2"] = fa2_errors[name]
                checks[f"{form.name}_{name}_within_fa2_error"] = error <= FA2_ERROR_RATIO * fa2_errors[name]
        misread = _misread_error(fa4, form)
        metrics[f"{form.name}_misread_heads"] = misread
        checks[f"{form.name}_misread_heads_shows"] = misread > MISREAD_REL_FLOOR
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
