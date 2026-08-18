#!/usr/bin/env python
"""Grouped GEMM with EMPTY expert groups — the routing shape every real MoE step produces.

``offs`` is a cumulative end-offset vector, so an expert that received no tokens appears as a
REPEATED offset (``offs[e] == offs[e-1]``) rather than as a missing entry. Top-k routing over a real
batch leaves experts empty routinely — a fresh router, a short sequence, a rank whose DeepEP dispatch
handed it nothing for some experts — and under EP the per-rank token counts are skewed by
construction. Every existing grouped-GEMM test builds ``offs`` with ``torch.arange`` (uniform) or a
hand-written list whose minimum is 1, so the zero-width group is unexercised at every tier.

What would break silently: an empty group must contribute an all-ZERO weight gradient that is still
ALLOCATED (the optimizer and the FSDP reduce-scatter both expect the full ``[E, K, N]`` grad), and
must not shift the tokens of the experts after it. A kernel or offset regression that dropped the
zero-width group would slide every later expert's rows down one slot — wrong tokens through the right
weights, no crash, no NaN.

Also pinned here: the wrapper's int32 offset normalization. ``torch.cumsum`` returns int64 and the
production router builds ``offs`` with it; ``F.grouped_mm`` rejects int64 outright
(``RuntimeError: Offsets have to be int32``), so dropping the cast at ``grouped_mm.py:37-38`` takes
every MoE forward down.

Run: python tests/gpu/kernels/test_grouped_mm_empty_groups.py
"""

import sys

import torch
import torch.nn.functional as F

from src.kernels.grouped_gemm import GroupedGemmPrecision, grouped_gemm
from src.kernels.grouped_mm_autograd import grouped_mm

DEV = "cuda"
K, N = 128, 96
# Two empty experts (one interior, one trailing) beside wildly uneven live ones — the interior empty
# is what a "drop the zero-width group" regression shifts the tail past.
SKEWED_COUNTS = (7, 33, 0, 1, 64, 0)


def _offsets(counts, dtype=torch.int32):
    """Cumulative END offsets, the layout ``F.grouped_mm`` and the EP router both use."""
    return torch.cumsum(torch.tensor(counts, device=DEV), 0).to(dtype)


def _reference(x, w, offs):
    """Per-expert loop. A zero-width slice yields an empty product, contributing no rows."""
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
    """Skewed routing with interior and trailing empty experts must match the loop exactly."""
    x, w = _operands(SKEWED_COUNTS)
    offs = _offsets(SKEWED_COUNTS)
    out = grouped_mm(x, w, offs=offs)
    reference = _reference(x, w, offs)

    assert out.shape == (sum(SKEWED_COUNTS), N), f"wrong row count with empty groups: {tuple(out.shape)}"
    diff = (out.float() - reference.float()).abs().max().item()
    # Same kernel, same accumulation order per group: exact, not merely close. A tolerance here would
    # hide the row-shift this test exists to catch (shifted rows differ by O(1), not O(eps)).
    assert diff == 0.0, f"empty-group forward diverged from the loop: maxdiff {diff:.3e}"
    print(f"  forward counts={SKEWED_COUNTS} offs={offs.tolist()} maxdiff {diff:.1e} PASS")


def test_an_empty_expert_gets_an_allocated_all_zero_weight_gradient():
    """The zero-width group contributes nothing, but its slot must still exist and be zero.

    A missing or garbage row here corrupts the optimizer state for an expert that simply sat out a
    step — and under FSDP2 the reduce-scatter would broadcast the garbage to every DP replica.
    """
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
    """Both grads, against a loop reference built from the same tensors."""
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
    """The degenerate end of the skew — every other expert empty, one expert holding the batch."""
    counts = (0, 0, 0, 0, 105, 0)
    x, w = _operands(counts)
    offs = _offsets(counts)
    diff = (grouped_mm(x, w, offs=offs).float() - _reference(x, w, offs).float()).abs().max().item()
    assert diff == 0.0, f"all-but-one-empty routing diverged: {diff:.3e}"
    print("  single live expert PASS")


def test_int64_offsets_are_normalized_by_the_wrapper():
    """``torch.cumsum`` returns int64 and the kernel rejects it — the wrapper's cast is load-bearing.

    The raw-kernel half is the anti-vacuity: without it, a wrapper that silently stopped casting
    would still pass as long as some future kernel accepted int64.
    """
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
    """The low-precision wrapper quantizes both operands before the same kernel; an empty group must
    not turn into a NaN block scale, and the empty expert's grad must stay zero."""
    offs = _offsets(SKEWED_COUNTS)
    x_ref, w_ref = _operands(SKEWED_COUNTS)
    reference = _reference(x_ref, w_ref, offs).float()

    for precision in (GroupedGemmPrecision.BF16, GroupedGemmPrecision.MXFP8, GroupedGemmPrecision.NVFP4):
        x, w = _operands(SKEWED_COUNTS, requires_grad=True)
        out = grouped_gemm(x, w, offs=offs, precision=precision)
        (out * torch.randn_like(out)).sum().backward()

        assert out.shape == (sum(SKEWED_COUNTS), N)
        assert torch.isfinite(out).all(), f"{precision.value}: empty groups produced non-finite output"
        assert torch.isfinite(w.grad).all(), f"{precision.value}: empty groups produced non-finite grads"
        rel = ((out.detach().float() - reference).norm() / reference.norm()).item()
        assert rel < 0.3, f"{precision.value}: rel {rel:.3f} — empty groups broke the quantized path"
        for expert, count in enumerate(SKEWED_COUNTS):
            if count == 0:
                assert w.grad[expert].abs().max().item() == 0.0, f"{precision.value}: expert {expert} not zero"
        print(f"  {precision.value} with empty groups: rel {rel:.3f} PASS")


def test_zero_total_tokens_with_a_broadcast_grad_backpropagates():
    """A rank whose dispatch routes ZERO tokens still runs backward. ``sum()``'s broadcast grad
    carries [0, 0] strides that a 0-element ``contiguous()`` keeps, and the kernel validates
    strides against sizes even at M == 0, where an unnormalized stride raises ``Invalid
    strides/sizes``."""
    counts = [0] * len(SKEWED_COUNTS)
    x, w = _operands(counts, requires_grad=True)
    offs = _offsets(counts)

    grouped_mm(x, w, offs=offs).sum().backward()

    assert x.grad.shape == x.shape and w.grad.shape == w.shape
    assert torch.isfinite(w.grad).all()
    assert w.grad.abs().max().item() == 0.0, "no tokens were routed; every weight grad must be zero"
    print("  zero-total-token broadcast-grad backward PASS")


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")  # the sentinel the launcher skips on; a bare exit 0 reads as a PASS
        return 0
    print(f"Grouped-GEMM empty-group tests on {torch.cuda.get_device_name()}")
    failures = []
    for test in (
        test_empty_groups_forward_matches_the_per_expert_loop,
        test_an_empty_expert_gets_an_allocated_all_zero_weight_gradient,
        test_empty_group_gradients_match_the_per_expert_loop,
        test_a_single_expert_holding_every_token_is_still_correct,
        test_int64_offsets_are_normalized_by_the_wrapper,
        test_the_precision_dispatch_survives_empty_groups,
        test_zero_total_tokens_with_a_broadcast_grad_backpropagates,
    ):
        try:
            test()
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"  FAIL {test.__name__}: {exc}")
    if failures:
        print(f"\n{len(failures)} test(s) FAILED")
        return 1
    print("\nAll empty-group grouped-GEMM tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
