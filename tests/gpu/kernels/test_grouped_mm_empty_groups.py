#!/usr/bin/env python
"""Grouped GEMM with empty expert groups, repeated ``offs`` entries that top-k routing produces routinely.

A dropped zero-width group slides later experts' rows down a slot with no crash or NaN. ``F.grouped_mm`` rejects int64
offsets, so dropping the wrapper's cast of ``torch.cumsum``'s output takes every MoE forward down.
"""

import torch
import torch.nn.functional as F

from halo.kernels.expert_gemm import expert_gemm
from halo.kernels.grouped_mm import grouped_mm
from tests.common.harness import gpu_test_main, record_check
from tests.common.numerics import fro_rel_err
from tests.common.rank_log import log

DEV = "cuda"
K, N = 128, 96
# One interior and one trailing empty expert; a dropped zero-width group shifts the tail past the interior one.
SKEWED_COUNTS = (7, 33, 0, 1, 64, 0)


def _offsets(counts, dtype=torch.int32):
    """Cumulative end offsets, the layout ``F.grouped_mm`` and the EP router both use."""
    return torch.cumsum(torch.tensor(counts, device=DEV), 0).to(dtype)


def _reference(x, w, offs):
    outs, start = [], 0
    for expert in range(w.shape[0]):
        end = int(offs[expert])
        outs.append(x[start:end] @ w[expert])
        start = end
    return torch.cat(outs, 0)


def _operands(counts, requires_grad=False):
    torch.manual_seed(0)
    total = sum(counts)
    x = torch.randn(total, K, device=DEV, dtype=torch.bfloat16) * 0.5
    w = torch.randn(len(counts), K, N, device=DEV, dtype=torch.bfloat16) * 0.05
    return x.requires_grad_(requires_grad), w.requires_grad_(requires_grad)


def test_empty_groups_forward_matches_the_per_expert_loop():
    x, w = _operands(SKEWED_COUNTS)
    offs = _offsets(SKEWED_COUNTS)
    out = grouped_mm(x, w, offs=offs)
    reference = _reference(x, w, offs)

    assert out.shape == (sum(SKEWED_COUNTS), N), f"wrong row count with empty groups: {tuple(out.shape)}"
    diff = (out.float() - reference.float()).abs().max().item()
    # Exact: shifted rows differ by O(1), and a tolerance could hide them.
    assert diff == 0.0, f"empty-group forward diverged from the loop: maxdiff {diff:.3e}"
    print(f"  forward counts={SKEWED_COUNTS} offs={offs.tolist()} maxdiff {diff:.1e} PASS")


def test_an_empty_expert_gets_an_allocated_all_zero_weight_gradient():
    """A missing or garbage row corrupts the optimizer state of an expert that sat out a step, and FSDP2's
    reduce-scatter would spread it to every replica."""
    x, w = _operands(SKEWED_COUNTS, requires_grad=True)
    offs = _offsets(SKEWED_COUNTS)

    out = grouped_mm(x, w, offs=offs)
    (out * torch.randn_like(out)).sum().backward()

    assert w.grad.shape == w.shape, f"weight grad lost its expert axis: {tuple(w.grad.shape)}"
    assert torch.isfinite(w.grad).all() and torch.isfinite(x.grad).all()
    for expert, count in enumerate(SKEWED_COUNTS):
        absmax = w.grad[expert].abs().max().item()
        if count == 0:
            assert absmax == 0.0, f"empty expert {expert} received a nonzero weight grad ({absmax:.3e})"
        else:
            # Anti-vacuity: an all-zero grad tensor would satisfy the clause above for every expert.
            assert absmax > 0.0, f"live expert {expert} ({count} tokens) received a zero weight grad"
    print("  backward: empty experts zero-grad, live experts nonzero PASS")


def test_empty_group_gradients_match_the_per_expert_loop():
    x, w = _operands(SKEWED_COUNTS, requires_grad=True)
    offs = _offsets(SKEWED_COUNTS)
    seed_grad = torch.randn(sum(SKEWED_COUNTS), N, device=DEV, dtype=torch.bfloat16)

    grouped_mm(x, w, offs=offs).backward(seed_grad)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_w = w.detach().clone().requires_grad_(True)
    _reference(ref_x, ref_w, offs).backward(seed_grad)

    dx = (x.grad.float() - ref_x.grad.float()).abs().max().item()
    dw = (w.grad.float() - ref_w.grad.float()).abs().max().item()
    assert dx == 0.0 and dw == 0.0, f"empty-group grads diverged from the loop: dx {dx:.3e}, dw {dw:.3e}"
    print(f"  backward vs loop: dx {dx:.1e} dw {dw:.1e} PASS")


def test_a_single_expert_holding_every_token_is_still_correct():
    counts = (0, 0, 0, 0, 105, 0)
    x, w = _operands(counts)
    offs = _offsets(counts)
    diff = (grouped_mm(x, w, offs=offs).float() - _reference(x, w, offs).float()).abs().max().item()
    assert diff == 0.0, f"all-but-one-empty routing diverged: {diff:.3e}"
    print("  single live expert PASS")


def test_int64_offsets_are_normalized_by_the_wrapper():
    """The raw-kernel half is the anti-vacuity: a wrapper that stopped casting passes if a future kernel accepts
    int64."""
    x, w = _operands(SKEWED_COUNTS)
    offs64 = _offsets(SKEWED_COUNTS, dtype=torch.int64)
    assert offs64.dtype == torch.int64

    raw_rejected = False
    try:
        F.grouped_mm(x, w, offs=offs64)
    except RuntimeError as err:
        raw_rejected = "int32" in str(err).lower()
    assert raw_rejected, "F.grouped_mm accepted int64 offsets; the wrapper's cast is no longer load-bearing"

    diff = (grouped_mm(x, w, offs=offs64).float() - _reference(x, w, _offsets(SKEWED_COUNTS)).float()).abs().max()
    assert diff.item() == 0.0, "int64 offsets took a different path than int32"
    print("  int64 offsets normalized (raw kernel refuses them) PASS")


def test_the_precision_dispatch_survives_empty_groups():
    """The low-precision wrapper quantizes both operands; an empty group must not become a NaN block scale."""
    offs = _offsets(SKEWED_COUNTS)
    x_ref, w_ref = _operands(SKEWED_COUNTS)
    reference = _reference(x_ref, w_ref, offs).float()

    for quant_format in (None, "mxfp8", "nvfp4"):
        label = quant_format or "bf16"
        x, w = _operands(SKEWED_COUNTS, requires_grad=True)
        out = expert_gemm(x, w, offs=offs, quant_format=quant_format)
        (out * torch.randn_like(out)).sum().backward()

        assert out.shape == (sum(SKEWED_COUNTS), N)
        assert torch.isfinite(out).all(), f"{label}: empty groups produced non-finite output"
        assert torch.isfinite(w.grad).all(), f"{label}: empty groups produced non-finite grads"
        rel = fro_rel_err(out.detach(), reference)
        assert rel < 0.3, f"{label}: rel {rel:.3f} — empty groups broke the quantized path"
        for expert, count in enumerate(SKEWED_COUNTS):
            if count == 0:
                assert w.grad[expert].abs().max().item() == 0.0, f"{label}: expert {expert} not zero"
        print(f"  {label} with empty groups: rel {rel:.3f} PASS")


def test_zero_total_tokens_with_a_broadcast_grad_backpropagates():
    """``sum()``'s broadcast grad carries [0, 0] strides a 0-element ``contiguous()`` keeps, and the kernel validates
    strides against sizes even at M == 0, raising ``Invalid strides/sizes`` if unnormalized."""
    counts = [0] * len(SKEWED_COUNTS)
    x, w = _operands(counts, requires_grad=True)
    offs = _offsets(counts)

    grouped_mm(x, w, offs=offs).sum().backward()

    assert x.grad.shape == x.shape and w.grad.shape == w.shape
    assert torch.isfinite(w.grad).all()
    assert w.grad.abs().max().item() == 0.0, "no tokens were routed; every weight grad must be zero"
    print("  zero-total-token broadcast-grad backward PASS")


@gpu_test_main(exact_world_size=1, prefix="grouped_mm_empty_groups", partial_state=False)
def run(ctx) -> dict:
    log(f"Grouped-GEMM empty-group tests on {torch.cuda.get_device_name()}")
    checks: dict[str, bool] = {}
    record_check(checks, "forward_matches_the_per_expert_loop", test_empty_groups_forward_matches_the_per_expert_loop)
    record_check(
        checks,
        "empty_expert_gets_an_allocated_all_zero_weight_gradient",
        test_an_empty_expert_gets_an_allocated_all_zero_weight_gradient,
    )
    record_check(checks, "gradients_match_the_per_expert_loop", test_empty_group_gradients_match_the_per_expert_loop)
    record_check(
        checks, "single_expert_holding_every_token", test_a_single_expert_holding_every_token_is_still_correct
    )
    record_check(checks, "int64_offsets_normalized_by_the_wrapper", test_int64_offsets_are_normalized_by_the_wrapper)
    record_check(checks, "precision_dispatch_survives_empty_groups", test_the_precision_dispatch_survives_empty_groups)
    record_check(
        checks,
        "zero_total_tokens_with_a_broadcast_grad",
        test_zero_total_tokens_with_a_broadcast_grad_backpropagates,
    )
    return {"checks": checks}


if __name__ == "__main__":
    run()
