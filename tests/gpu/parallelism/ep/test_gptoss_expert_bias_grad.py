#!/usr/bin/env python
"""Correctness of the atomic-free expert-bias gradient (``MoEExpertBiasGather``).

The gpt-oss expert bias is broadcast to expert-sorted tokens (``bias[eids]``). The default gather
backward is an ``index_add_`` — a bf16 atomic scatter that serialises under eids duplication (~12%
of the step). ``MoEExpertBiasGather`` keeps the gather forward but computes the bias gradient as a
GEMM (``onehotᵀ @ grad``). This test pins it to the reference:

  - forward is byte-identical to ``bias.index_select(0, eids)``;
  - the bias gradient equals the autograd ``index_select``/``index_add_`` reference — exactly in
    fp32 (tight tol), within bf16 accumulation noise in bf16;
  - an expert with no routed tokens gets exactly zero gradient.

These assertions FAIL if the backward is wired wrong (wrong reduction, wrong dtype, sort assumed),
so the test is not vacuous. Single GPU; no DeepEP.

    python tests/gpu/parallelism/ep/test_gptoss_expert_bias_grad.py
"""

import sys

import pytest
import torch

from src.distributed.expert_parallel.autograd import MoEExpertBiasGather

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, DIM, N = 32, 256, 512


def _case(dtype, device="cuda", empty_expert=None):
    """Random (bias, eids) like the gpt-oss grouped path. ``eids`` need not be sorted — the
    GEMM backward is order-independent — but we sort to mirror the real dispatch layout."""
    g = torch.Generator(device=device).manual_seed(0)
    eids = torch.randint(0, E, (N,), generator=g, device=device)
    if empty_expert is not None:
        eids = eids[eids != empty_expert]  # guarantee one expert receives no tokens
    eids = eids.sort().values
    bias = torch.randn(E, DIM, generator=g, device=device, dtype=dtype)
    grad = torch.randn(eids.shape[0], DIM, generator=g, device=device, dtype=dtype)
    return bias, eids, grad


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
    assert torch.equal(out, bias.index_select(0, eids))  # byte-identical forward


def test_bias_grad_matches_reference_fp32():
    bias, eids, grad = _case(torch.float32)
    ref, fix = _reference_bias_grad(bias, eids, grad), _fix_bias_grad(bias, eids, grad)
    rel = ((fix - ref).abs().max() / ref.abs().max()).item()
    assert rel < 1e-3, f"fp32 bias-grad rel error {rel:.2e} too large"


def test_bias_grad_matches_reference_bf16():
    bias, eids, grad = _case(torch.bfloat16)
    ref, fix = _reference_bias_grad(bias, eids, grad), _fix_bias_grad(bias, eids, grad)
    rel = ((fix.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    assert rel < 5e-2, f"bf16 bias-grad rel error {rel:.2e} exceeds accumulation tolerance"


def test_empty_expert_gets_zero_grad():
    empty = 7
    bias, eids, grad = _case(torch.float32, empty_expert=empty)
    assert (eids == empty).sum() == 0
    fix = _fix_bias_grad(bias, eids, grad)
    assert fix[empty].abs().max().item() == 0.0


def _main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")  # the sentinel the launcher skips on; a bare exit 0 reads as a PASS
        return 0
    failed = 0
    for fn in [
        test_forward_is_plain_gather,
        test_bias_grad_matches_reference_fp32,
        test_bias_grad_matches_reference_bf16,
        test_empty_expert_gets_zero_grad,
    ]:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{'ALL PASSED' if not failed else f'{failed} FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
