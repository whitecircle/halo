#!/usr/bin/env python
"""FSDP2 pins the unsharded param's version counter — so the low-precision weight-quant cache must
be bypassed for FSDP2-managed expert weights (``grouped_gemm(..., weight_cacheable=False)``).

``cached_fake_quant`` keys its once-per-step round-trip on ``weight._version``. FSDP2 refreshes the
unsharded param's data through ``_unsafe_preserve_version_counter``, so that key never advances and
a cache keyed on it serves the step-0 quantization for the whole run. Two halves pin this:

1. Torch premise (Linear-as-expert-weight): across 3 AdamW steps on a ``fully_shard``-ed module
   (``reshard_after_forward=False``), a forward pre-hook sees the unsharded param's ``_version``
   FROZEN while its data moves; ``cached_fake_quant`` keeps serving the step-0 quantization while
   ``fake_quant`` tracks the live weight. If FSDP2 ever stops pinning the counter, this half fails
   and the ``weight_cacheable=False`` special-casing can be reconsidered.
2. ``grouped_gemm`` contract ([E,K,N] FSDP2-managed expert weight, MXFP8): with
   ``weight_cacheable=False`` the output tracks the master (differs after an optimizer step); with
   ``weight_cacheable=True`` on the same frozen-version weight the output provably stays the stale
   step-0 result while the master moves underneath — the control that fails without the fix.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/kernels/test_lowp_fsdp2_weight_cache.py
"""

import os

# Eager quantization — numerically identical to the compiled path, skips the per-rank compile warm-up.
os.environ.setdefault("HALO_LOWP_COMPILE", "0")

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from src.kernels.grouped_gemm import GroupedGemmPrecision, grouped_gemm
from src.kernels.lowp.quantization import cached_fake_quant, fake_quant
from tests.common.harness import gpu_test_main

FMT = "mxfp8"
STEPS = 3
LR = 0.05  # large vs the ~0.05-scale weights so 2 Adam steps move well past mxfp8's resolution


def _linear_version_pin(ctx) -> dict:
    """Half 1: unsharded ``_version`` frozen while the master moves; cached quant goes stale."""
    torch.manual_seed(7)
    lin = nn.Linear(64, 64, bias=False, device=ctx.device, dtype=torch.bfloat16)
    fully_shard(lin, reshard_after_forward=False)

    captures = []

    def capture(module, args):
        w = module.weight
        with torch.no_grad():
            captures.append(
                (
                    not isinstance(w, DTensor),
                    w._version,
                    w.detach().clone(),
                    fake_quant(w, FMT, -1),
                    cached_fake_quant(w, FMT, -1),
                )
            )

    lin.register_forward_pre_hook(capture)
    opt = torch.optim.AdamW(lin.parameters(), lr=LR)
    torch.manual_seed(101 + ctx.rank)
    x = torch.randn(32, 64, device=ctx.device, dtype=torch.bfloat16)
    for _ in range(STEPS):
        lin(x).float().pow(2).mean().backward()
        opt.step()
        opt.zero_grad()

    unsharded_flags = [c[0] for c in captures]
    versions = [c[1] for c in captures]
    w_first, w_last = captures[0][2], captures[-1][2]
    fq_first, fq_last = captures[0][3], captures[-1][3]
    cq_first, cq_last = captures[0][4], captures[-1][4]
    return {
        "linear_hook_saw_unsharded_plain_param": all(unsharded_flags) and len(captures) == STEPS,
        "linear_version_frozen": len(set(versions)) == 1,
        "linear_master_moved": not torch.equal(w_first, w_last),
        "linear_fake_quant_tracks_master": not torch.equal(fq_first, fq_last),
        "linear_cached_serves_step0_forever": torch.equal(cq_first, cq_last) and torch.equal(cq_last, fq_first),
        "linear_cached_diverged_from_live_weight": not torch.equal(cq_last, fq_last),
    }


class _Experts(nn.Module):
    """Tiny [E, K, N] expert bank whose forward is exactly the grouped_gemm under test."""

    def __init__(self, device):
        super().__init__()
        torch.manual_seed(7)
        self.weight = nn.Parameter(torch.randn(4, 64, 32, device=device, dtype=torch.bfloat16) * 0.05)

    def forward(self, x, offs, cacheable):
        return grouped_gemm(
            x, self.weight, offs=offs, precision=GroupedGemmPrecision.MXFP8, weight_cacheable=cacheable
        )


def _local(w: torch.Tensor) -> torch.Tensor:
    return w.to_local() if isinstance(w, DTensor) else w


def _train_grouped(ctx, cacheable: bool) -> tuple[list, bool]:
    """3 optimizer steps on an FSDP2-managed expert weight with a FIXED input; returns the per-step
    forward outputs and whether the sharded master actually moved."""
    mod = _Experts(ctx.device)
    fully_shard(mod, reshard_after_forward=False)
    opt = torch.optim.AdamW(mod.parameters(), lr=LR)
    torch.manual_seed(211 + ctx.rank)
    x = torch.randn(32, 64, device=ctx.device, dtype=torch.bfloat16) * 0.3
    offs = torch.arange(8, 33, 8, device=ctx.device, dtype=torch.int32)
    shard_before = _local(mod.weight.detach()).clone()

    outs = []
    for _ in range(STEPS):
        out = mod(x, offs, cacheable)
        outs.append(out.detach().clone())
        out.float().pow(2).mean().backward()
        opt.step()
        opt.zero_grad()

    master_moved = not torch.equal(shard_before, _local(mod.weight.detach()))
    return outs, master_moved


def _grouped_gemm_contract(ctx) -> dict:
    """Half 2: weight_cacheable=False tracks the master; True serves the stale step-0 entry."""
    outs_fresh, moved_fresh = _train_grouped(ctx, cacheable=False)
    outs_cached, moved_cached = _train_grouped(ctx, cacheable=True)
    return {
        # The fix: an uncached weight quant follows the master, so the output changes after a step.
        "gg_master_moved_uncached": moved_fresh,
        "gg_uncached_output_tracks_master": not torch.equal(outs_fresh[0], outs_fresh[-1]),
        # The control that fails without the fix: same frozen-version weight, cache enabled — the
        # output is bitwise the step-0 result on every step even though the master moved.
        "gg_master_moved_cached": moved_cached,
        "gg_cached_output_stale": all(torch.equal(outs_cached[0], o) for o in outs_cached[1:]),
    }


@gpu_test_main(exact_world_size=2, prefix="lowp_fsdp2_cache", partial_state=False)
def run(ctx):
    checks = {}
    checks.update(_linear_version_pin(ctx))
    checks.update(_grouped_gemm_contract(ctx))
    return {"checks": checks}


if __name__ == "__main__":
    run()
