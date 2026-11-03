#!/usr/bin/env python
"""``MoEExpertBiasGather``'s atomic-free expert-bias gradient, a GEMM in place of the ``index_add_`` that serializes
under eids duplication (~12% of the step)."""

import torch

from halo.kernels.moe_permute import MoEExpertBiasGather
from tests.common.harness import gpu_test_main, record_check
from tests.common.numerics import max_abs_rel_err

E, DIM, N = 32, 256, 512


def _case(dtype, device="cuda", empty_expert=None):
    """``eids`` need not be sorted (the GEMM backward is order-independent); sorted to mirror the real dispatch."""
    g = torch.Generator(device=device).manual_seed(0)
    eids = torch.randint(0, E, (N,), generator=g, device=device)
    if empty_expert is not None:
        eids = eids[eids != empty_expert]  # Guarantees one expert receives no tokens.
    eids = eids.sort().values
    bias = torch.randn(E, DIM, generator=g, device=device, dtype=dtype)
    grad = torch.randn(eids.shape[0], DIM, generator=g, device=device, dtype=dtype)
    return bias, eids, grad


def _fp64_bias_grad(eids, grad):
    """The fp64 sum of each expert's rows, and the fp32 rounding bound of a sum of that many."""
    want = torch.zeros(E, DIM, device=grad.device, dtype=torch.float64).index_add_(0, eids, grad.double())
    magnitude = torch.zeros_like(want).index_add_(0, eids, grad.double().abs())
    return want, eids.numel() * 2**-24 * magnitude


def _reference_bias_grad(bias, eids, grad):
    b = bias.clone().requires_grad_(True)
    (b.index_select(0, eids) * grad).sum().backward()
    return b.grad


def _fix_bias_grad(bias, eids, grad):
    b = bias.clone().requires_grad_(True)
    (MoEExpertBiasGather.apply(b, eids) * grad).sum().backward()
    return b.grad


def test_forward_is_plain_gather():
    bias, eids, _ = _case(torch.bfloat16)
    out = MoEExpertBiasGather.apply(bias, eids)
    assert torch.equal(out, bias.index_select(0, eids))  # Byte-identical forward.


def test_bias_grad_matches_reference_fp32():
    bias, eids, grad = _case(torch.float32)
    (want, rounding), fix = _fp64_bias_grad(eids, grad), _fix_bias_grad(bias, eids, grad)
    assert bool(((fix.double() - want).abs() <= rounding).all()), (
        f"fp32 bias-grad error {max_abs_rel_err(fix, want):.2e}"
    )
    assert torch.equal(fix, _fix_bias_grad(bias, eids, grad)), "two runs of the bias gradient differ"


def test_bias_grad_matches_reference_bf16():
    bias, eids, grad = _case(torch.bfloat16)
    ref, fix = _reference_bias_grad(bias, eids, grad), _fix_bias_grad(bias, eids, grad)
    rel = max_abs_rel_err(fix, ref)
    assert rel < 5e-2, f"bf16 bias-grad rel error {rel:.2e} exceeds accumulation tolerance"


def test_empty_expert_gets_zero_grad():
    empty = 7
    bias, eids, grad = _case(torch.float32, empty_expert=empty)
    assert (eids == empty).sum() == 0
    fix = _fix_bias_grad(bias, eids, grad)
    assert fix[empty].abs().max().item() == 0.0


@gpu_test_main(prefix="gptoss_expert_bias_grad", partial_state=False)
def run(ctx):
    checks: dict[str, bool] = {}
    record_check(checks, "forward_is_plain_gather", test_forward_is_plain_gather)
    record_check(checks, "bias_grad_matches_reference_fp32", test_bias_grad_matches_reference_fp32)
    record_check(checks, "bias_grad_matches_reference_bf16", test_bias_grad_matches_reference_bf16)
    record_check(checks, "empty_expert_gets_zero_grad", test_empty_expert_gets_zero_grad)
    return {"checks": checks}


if __name__ == "__main__":
    run()
