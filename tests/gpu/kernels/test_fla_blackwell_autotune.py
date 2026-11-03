#!/usr/bin/env python
"""FLA's GatedDeltaNet op, after the post-load restriction, is exact and repeatable on packed rows on Blackwell.

On sm_10x under Triton 3.6 ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` varies run to run with ``num_warps=4``,
which the stock autotuner picks at Qwen3.5/3.6 shapes. Once the forced control stops missing, the restriction can go.
"""

import os
import sys

import torch
import torch.nn.functional as F
from fla.ops.common import chunk_delta_h
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule

from halo.kernels.fla_launch import triton_autotuner
from halo.model_loading.finalization import finalize_loaded_model
from tests.common.harness import gpu_test_main, record_check
from tests.common.numerics import cos_sim, max_abs_rel_err
from tests.common.rank_log import log

SEED = 1234
# Qwen3.5/3.6-35B-A3B's GatedDeltaNet after the q/k repeat.
HEADS, HEAD_DIM = 32, 128
# One packed row of four documents, the layout the padding-free collators emit.
CU_SEQLENS = (0, 896, 2048, 3584, 4096)
REPEATS = 20
CONTROL_REPEATS = 30
# The restricted configs measured 0.76% max relative error against fp32; the num_warps=4 race up to 28%.
TOL_O_MAXREL = 2e-2
TOL_GRAD_COS = 0.9999
CONTROL_CONFIG = {"BV": 32, "num_warps": 4, "num_stages": 2}
OP_INPUTS = ("q", "k", "v", "beta", "g")
GRADS = tuple(f"d{name}" for name in OP_INPUTS)
BLACKWELL_MAJOR = 10


def _inputs(device) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    total = CU_SEQLENS[-1]
    rand = lambda *shape: torch.randn(*shape, generator=gen)  # noqa: E731
    a_log = torch.empty(HEADS).uniform_(0.01, 16, generator=gen).log()
    leaves = {
        "q": rand(1, total, HEADS, HEAD_DIM).to(torch.bfloat16),
        "k": rand(1, total, HEADS, HEAD_DIM).to(torch.bfloat16),
        "v": rand(1, total, HEADS, HEAD_DIM).to(torch.bfloat16),
        "beta": rand(1, total, HEADS).sigmoid().to(torch.bfloat16),
        "g": -a_log.exp() * F.softplus(rand(1, total, HEADS) + 1.0),
        "do": rand(1, total, HEADS, HEAD_DIM),
    }
    return {name: tensor.to(device) for name, tensor in leaves.items()}


def _fresh(leaves, dtype=None) -> dict[str, torch.Tensor]:
    """Independent grad-carrying leaves, so no call reads or accumulates into another's tensors."""
    return {name: leaves[name].detach().to(dtype or leaves[name].dtype).clone().requires_grad_() for name in OP_INPUTS}


def _run(leaves) -> dict[str, torch.Tensor]:
    inputs = _fresh(leaves)
    cu_seqlens = torch.tensor(CU_SEQLENS, device=leaves["q"].device, dtype=torch.long)
    o, _ = chunk_gated_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    (o.float() * leaves["do"]).sum().backward()
    return {"o": o.detach(), **{f"d{name}": inputs[name].grad for name in OP_INPUTS}}


def _reference(leaves) -> dict[str, torch.Tensor]:
    """FLA's naive recurrence in fp32, one document at a time from a zero state."""
    inputs = _fresh(leaves, torch.float32)
    q, k = (F.normalize(inputs[name], dim=-1, eps=1e-6) for name in ("q", "k"))
    outputs = [
        naive_recurrent_gated_delta_rule(
            q[:, s:e], k[:, s:e], inputs["v"][:, s:e], inputs["beta"][:, s:e], inputs["g"][:, s:e]
        )[0]
        for s, e in zip(CU_SEQLENS[:-1], CU_SEQLENS[1:], strict=True)
    ]
    o = torch.cat(outputs, dim=1)
    (o * leaves["do"]).sum().backward()
    return {"o": o.detach(), **{f"d{name}": inputs[name].grad for name in OP_INPUTS}}


def _repeat_and_score(leaves, reference, repeats: int) -> tuple[bool, float, float]:
    """(every repeat bitwise equal to the first, worst o max-rel error, worst gradient cosine)."""
    first = _run(leaves)
    stable, worst_rel, worst_cos = True, 0.0, 1.0
    for repeat in range(repeats):
        result = first if repeat == 0 else _run(leaves)
        stable &= all(torch.equal(result[name], first[name]) for name in first)
        worst_rel = max(worst_rel, max_abs_rel_err(result["o"], reference["o"]))
        worst_cos = min(worst_cos, *(cos_sim(result[name], reference[name], label=name) for name in GRADS))
    return stable, worst_rel, worst_cos


def _fwd_h_autotuner():
    return triton_autotuner(chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64)


def _check_restricted(leaves, reference) -> None:
    finalize_loaded_model(torch.nn.Module())
    warps = {config.num_warps for config in _fwd_h_autotuner().configs}
    assert warps == {2}, f"the state kernel still autotunes num_warps {sorted(warps)} on Blackwell"
    stable, worst_rel, worst_cos = _repeat_and_score(leaves, reference, REPEATS)
    log(f"  restricted: stable={stable} o max-rel={worst_rel:.2e} worst grad cos={worst_cos:.6f}")
    assert stable, f"the restricted op is not repeatable over {REPEATS} calls"
    assert worst_rel < TOL_O_MAXREL, f"restricted o max-rel {worst_rel:.2e} exceeds {TOL_O_MAXREL:.0e}"
    assert worst_cos > TOL_GRAD_COS, f"restricted worst gradient cosine {worst_cos:.6f} under {TOL_GRAD_COS}"


def _check_control(leaves, reference) -> None:
    autotuner = _fwd_h_autotuner()
    restricted = autotuner.configs
    control = type(restricted[0])(
        {"BV": CONTROL_CONFIG["BV"]}, num_warps=CONTROL_CONFIG["num_warps"], num_stages=CONTROL_CONFIG["num_stages"]
    )
    autotuner.configs, autotuner.cache = [control], {}
    try:
        stable, worst_rel, worst_cos = _repeat_and_score(leaves, reference, CONTROL_REPEATS)
    finally:
        autotuner.configs, autotuner.cache = restricted, {}
    log(f"  control {CONTROL_CONFIG}: stable={stable} o max-rel={worst_rel:.2e} worst grad cos={worst_cos:.6f}")
    assert not (stable and worst_rel < TOL_O_MAXREL), (
        f"the num_warps=4 control was repeatable and within {TOL_O_MAXREL:.0e} over {CONTROL_REPEATS} calls: the "
        f"race no longer reproduces here, so the Blackwell restriction in halo/kernels/fla_launch.py may be obsolete"
    )


@gpu_test_main(exact_world_size=1, partial_state=False, prefix="test_fla_blackwell_autotune")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    # An empty cache, as on a node's first run. Triton reads it at first launch.
    os.environ["TRITON_CACHE_DIR"] = os.path.join(ctx.cache_dir, "triton")
    torch.cuda.set_device(ctx.device)
    leaves = _inputs(ctx.device)
    reference = _reference(leaves)
    log("\n--- after finalize_loaded_model's restriction ---")
    record_check(checks, "restricted_op_exact_and_repeatable", lambda: _check_restricted(leaves, reference))
    log("\n--- num_warps=4 control ---")
    record_check(checks, "num_warps_4_control_misses", lambda: _check_control(leaves, reference))
    return {"checks": checks}


if __name__ == "__main__":
    if torch.cuda.get_device_capability()[0] != BLACKWELL_MAJOR:
        log("SKIP: the restriction applies to Blackwell (sm_10x) only")
        sys.exit(0)
    run()
