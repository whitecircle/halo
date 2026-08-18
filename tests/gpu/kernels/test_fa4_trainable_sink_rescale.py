#!/usr/bin/env python
"""The FA4 trainable-sink rescale is exact on both kernel entry points.

The fused FA4 backward returns no sink gradient, so ``install_fa4_trainable_sink_rescale`` routes
grad-requiring sinks through ``out * sigmoid(lse - sink)`` on the sink-less kernel — for the varlen
kernel (packing / padding-free / padded batches) and the dense one (unpadded batches). Checks: the
install is idempotent; a frozen sink is a bit-identical passthrough; the rescaled forward and the
returned lse match the fused kernel; all four gradients (dq, dk, dv, d_sink) match an eager sink
reference. Fails if the kernel's lse contract, its ``dlse`` backward path, or the wrapper's dispatch
(frozen passthrough vs trainable rescale) breaks on either entry point.
"""

import torch

from src.models.patches.gpt_oss_sinks import install_fa4_trainable_sink_rescale
from tests.common.harness import gpu_test_main
from tests.common.utils import log

H, HKV, D, S, B = 8, 2, 64, 256, 2
FWD_REL_TOL = 2e-2  # bf16 output: two roundings (kernel out, gate multiply) vs the fused kernel's one
GRAD_REL_TOL = 6e-2  # bf16 dq/dk/dv/d_sink vs an fp32 eager reference


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-9)).item()


def _inputs(seed, *, dense: bool):
    torch.manual_seed(seed)
    shape = (B, S) if dense else (B * S,)
    q = torch.randn(*shape, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(*shape, HKV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(*shape, HKV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    return q, k, v, sink


def _call_kwargs(*, dense: bool):
    if dense:
        return {"causal": True}
    cu = torch.arange(0, (B + 1) * S, S, device="cuda", dtype=torch.int32)
    return {"cu_seqlens_q": cu, "cu_seqlens_k": cu, "max_seqlen_q": S, "max_seqlen_k": S, "causal": True}


def _eager_sink_reference(q, k, v, sink):
    """Per-sequence causal softmax over [scores | sink], sink column dropped from the output."""
    q, k, v = (t.reshape(B, S, -1, D) for t in (q, k, v))
    kr, vr = (t.repeat_interleave(H // HKV, dim=2) for t in (k, v))
    qq, kk, vv = (t.transpose(1, 2) for t in (q, kr, vr))  # [B, H, S, D]
    scores = (qq @ kk.transpose(-1, -2)) / D**0.5
    scores = scores.masked_fill(torch.triu(torch.ones(S, S, device="cuda", dtype=torch.bool), 1), float("-inf"))
    p = torch.softmax(torch.cat([scores, sink.view(1, H, 1, 1).expand(B, H, S, 1)], -1), -1)[..., :-1]
    return (p @ vv).transpose(1, 2)  # [B, S, H, D]


def _entry_point(cute, *, dense: bool):
    return cute.flash_attn_func if dense else cute.flash_attn_varlen_func


def _check_entry_point(cute, *, dense: bool) -> dict:
    label = "dense" if dense else "varlen"
    fn = _entry_point(cute, dense=dense)
    fused = fn.__wrapped__
    kwargs = _call_kwargs(dense=dense)
    checks = {}

    q, k, v, sink = _inputs(0, dense=dense)
    with torch.no_grad():
        wrapped = fn(q, k, v, **kwargs, learnable_sink=sink.detach())
        direct = fused(q, k, v, **kwargs, learnable_sink=sink.detach())
    checks[f"{label}_frozen_sink_bit_identical_passthrough"] = torch.equal(wrapped[0], direct[0])
    # The trainable branch must return what the pinned kernel returns at the default ``return_lse``
    # (this build always hands back ``(out, lse)``); a kernel bump that changes the arity fails here.
    checks[f"{label}_trainable_return_arity_matches_kernel"] = isinstance(
        fn(q, k, v, **kwargs, learnable_sink=sink), tuple
    ) is isinstance(direct, tuple)

    q, k, v, sink = _inputs(1, dense=dense)
    with torch.no_grad():
        fused_out, fused_lse = fused(q, k, v, **kwargs, learnable_sink=sink, return_lse=True)
    out, lse = fn(q, k, v, **kwargs, learnable_sink=sink, return_lse=True)
    fwd_rel, lse_rel = _rel(out, fused_out), _rel(lse, fused_lse)
    log(f"  [{label}] fwd rescale-vs-fused rel {fwd_rel:.2e}; lse rel {lse_rel:.2e}")
    checks[f"{label}_forward_matches_fused_kernel"] = fwd_rel < FWD_REL_TOL
    checks[f"{label}_returned_lse_is_the_sinked_lse"] = lse_rel < 1e-3

    q, k, v, sink = _inputs(2, dense=dense)
    fn(q, k, v, **kwargs, learnable_sink=sink)[0].float().square().sum().backward()
    qe, ke, ve, se = (t.detach().float().clone().requires_grad_(True) for t in (q, k, v, sink))
    _eager_sink_reference(qe, ke, ve, se).reshape(q.shape).square().sum().backward()
    for name, got, ref in (
        ("dq", q.grad, qe.grad),
        ("dk", k.grad, ke.grad),
        ("dv", v.grad, ve.grad),
        ("d_sink", sink.grad, se.grad),
    ):
        rel = _rel(got, ref)
        log(f"  [{label}] {name:6} rel vs eager {rel:.2e}")
        checks[f"{label}_{name}_matches_eager_reference"] = rel < GRAD_REL_TOL
    checks[f"{label}_sink_gradient_is_nonzero"] = bool(sink.grad.abs().sum() > 0)
    return checks


@gpu_test_main(exact_world_size=1, prefix="fa4_sink_rescale")
def run(ctx):
    import flash_attn.cute as cute

    checks = {"installs": install_fa4_trainable_sink_rescale()}
    first = (cute.flash_attn_varlen_func, cute.flash_attn_func)
    checks["install_is_idempotent"] = (
        install_fa4_trainable_sink_rescale() and (cute.flash_attn_varlen_func, cute.flash_attn_func) == first
    )
    checks.update(_check_entry_point(cute, dense=False))
    checks.update(_check_entry_point(cute, dense=True))
    return {"checks": checks}


if __name__ == "__main__":
    run()
