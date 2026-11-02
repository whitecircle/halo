#!/usr/bin/env python
"""Liger's rotary, as the toolkit binds it, is exact on every call at a production shape.

Liger auto-selects its CuTe DSL RoPE ahead of Triton once a grad-carrying call reaches 131072 query rows (``bsz *
heads * seq``) at ``head_dim >= 128``, and under ``CUTE_DSL_ENABLE_TVM_FFI`` that kernel is wrong from the second
launch of a shape at the latest. The toolkit pins the op to Triton (``pin_liger_impls``). This binds Qwen3's rotary
through ``apply_liger_kernel`` and checks several same-shape calls, forward and backward, against the family's own
rotation in fp32, on fresh ``[batch, seq, heads, head_dim]`` projections viewed as ``[batch, heads, seq, head_dim]``
the way the attention builds them.

The control runs the CuTe DSL implementation directly at the same shape and must miss on a later call:
that keeps the shape inside the defect, so the pinned check is not passing below the gate. Once the
control stops missing, Liger has fixed the kernel and the pin can go.

    torchrun --nproc_per_node=1 tests/gpu/kernels/test_liger_rope_repeated_calls.py
"""

import os

import torch
from liger_kernel.backends.dispatch import ImplNotAvailableError, dispatch, resolve_impl
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.qwen3 import modeling_qwen3

from halo.model_loading.liger.dispatch import apply_liger_kernel
from tests.common.harness import gpu_test_main, record_check
from tests.common.numerics import max_abs_rel_err
from tests.common.rank_log import log

SEED = 42
# Qwen3-8B's attention at 4096 tokens: 32 query heads x 4096 positions is exactly Liger's CuTe DSL gate.
BATCH, SEQ, Q_HEADS, KV_HEADS, HEAD_DIM = 1, 4096, 32, 8, 128
ROPE_THETA = 1_000_000.0
CALLS = 3
# bf16 storage rounding against an fp32 rotation (the bound `test_liger_family_kernels.py` uses); a stale
# buffer misses by the size of the activations.
TOL_BF16 = 3e-2
CUTEDSL_IMPL = "nvidia-cutedsl"


def _rotary_tables(device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM))
    freqs = torch.arange(SEQ, device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    angles = torch.cat((freqs, freqs), dim=-1)[None]
    return angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)


def _residuals_per_call(rope, reference, device) -> list[dict[str, float]]:
    """``max_abs_rel_err`` of q, k, dq and dk against ``reference`` for each of ``CALLS`` fresh inputs.

    Liger's RoPE rotates in place, forward and backward, so the rotary gets a fresh projection-shaped copy of
    each input and a clone of each incoming gradient, while the reference reads the untouched sources.
    """
    cos, sin = _rotary_tables(device)
    results = []
    for _ in range(CALLS):
        q_src = torch.randn(BATCH, SEQ, Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
        k_src = torch.randn(BATCH, SEQ, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
        q_ref = q_src.detach().float().requires_grad_(True)
        k_ref = k_src.detach().float().requires_grad_(True)
        expected_q, expected_k = reference(q_ref.transpose(1, 2), k_ref.transpose(1, 2), cos.float(), sin.float())
        dq_out, dk_out = torch.randn_like(expected_q), torch.randn_like(expected_k)
        expected_dq, expected_dk = torch.autograd.grad((expected_q, expected_k), (q_ref, k_ref), (dq_out, dk_out))

        q, k = (q_src * 1).transpose(1, 2), (k_src * 1).transpose(1, 2)
        rotated_q, rotated_k = rope(q, k, cos, sin)
        dq, dk = torch.autograd.grad(
            (rotated_q, rotated_k), (q_src, k_src), (dq_out.to(torch.bfloat16), dk_out.to(torch.bfloat16))
        )
        results.append(
            {
                "q": max_abs_rel_err(rotated_q, expected_q),
                "k": max_abs_rel_err(rotated_k, expected_k),
                "dq": max_abs_rel_err(dq, expected_dq),
                "dk": max_abs_rel_err(dk, expected_dk),
            }
        )
    return results


def _log_calls(label: str, results: list[dict[str, float]]) -> None:
    for call, residuals in enumerate(results):
        log(f"  {label} call {call}: " + " ".join(f"{name}={value:.1e}" for name, value in residuals.items()))


def _check_bound_rotary(stock_rotary, device) -> None:
    apply_liger_kernel(CONFIG_MAPPING["qwen3"](), {"rope": True})
    bound = modeling_qwen3.apply_rotary_pos_emb
    assert bound is not stock_rotary, "apply_liger_kernel did not swap Qwen3's rotary"
    selected = resolve_impl("rope", device=device).impl_name
    assert selected == "nvidia-triton", f"Liger resolves rope to {selected}, not the pinned nvidia-triton"

    results = _residuals_per_call(bound, stock_rotary, device)
    _log_calls("bound", results)
    failing = [(call, name) for call, r in enumerate(results) for name, value in r.items() if not value < TOL_BF16]
    assert not failing, f"bound rotary exceeds {TOL_BF16:.0e} at (call, tensor) {failing}"


def _check_cutedsl_control(stock_rotary, device) -> None:
    if os.environ.get("CUTE_DSL_ENABLE_TVM_FFI") != "1":
        log("  control skipped: CUTE_DSL_ENABLE_TVM_FFI is not 1, so the defect cannot arise")
        return
    try:
        resolve_impl("rope", requested=CUTEDSL_IMPL, device=device)
    except ImplNotAvailableError as e:
        log(f"  control skipped: {e}")
        return

    def cutedsl_rotary(q, k, cos, sin):
        return dispatch("rope", q, k, cos, sin, None, 1, impl=CUTEDSL_IMPL)

    results = _residuals_per_call(cutedsl_rotary, stock_rotary, device)
    _log_calls("cutedsl", results)
    worst_later = max(value for r in results[1:] for value in r.values())
    assert not worst_later < TOL_BF16, (
        f"the CuTe DSL RoPE matched on every later call (worst {worst_later:.1e}): Liger fixed the stale "
        f"buffer, so `_PINNED_LIGER_IMPLS` in halo/kernels/jit_cache.py can drop rope"
    )


@gpu_test_main(min_world_size=1, prefix="test_liger_rope_repeated_calls")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    torch.cuda.set_device(ctx.device)
    torch.manual_seed(SEED)
    # Before any applier swaps it: the reference must be the family's own rotation.
    stock_rotary = modeling_qwen3.apply_rotary_pos_emb

    log("\n--- CuTe DSL control ---")
    record_check(checks, "cutedsl_control_misses", lambda: _check_cutedsl_control(stock_rotary, ctx.device))
    log("\n--- rotary bound by apply_liger_kernel ---")
    record_check(checks, "bound_rotary_exact", lambda: _check_bound_rotary(stock_rotary, ctx.device))
    return {"checks": checks}


if __name__ == "__main__":
    run()
