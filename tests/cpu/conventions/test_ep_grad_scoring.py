#!/usr/bin/env python
"""``score_ep_grad_pairs`` fails each way an EP gradient can disagree with its reference.

Seven GPU suites (the EP-vs-FSDP family tests, Inkling, Bailing V3) gate their gradient checks on
this helper, and they run only on the GPU tiers. Each failure mode it exists for is pinned here: a
missing or misshapen EP gradient, a doubled or halved cross-rank divide, and a reoriented gradient.

Run: python tests/cpu/conventions/test_ep_grad_scoring.py
"""

import pytest
import torch

from tests.common.ep_reference import MISSING_GRAD_SCORE, score_ep_grad_pairs
from tests.common.tolerances import TOL

REFERENCE = torch.randn(4, 16, 8, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)


def _score(got):
    checks, metrics = {}, {}
    score_ep_grad_pairs({"l0_gate_up_grad": (got, REFERENCE)}, checks, metrics, cos_min=TOL.ep_grad_cosine_min)
    return checks["l0_gate_up_grad_matches"], metrics


def test_a_matching_gradient_passes():
    matches, metrics = _score(REFERENCE.clone())
    assert matches
    assert metrics["l0_gate_up_grad_cos"] == pytest.approx(1.0) and metrics["l0_gate_up_grad_norm_ratio"] == 1.0


@pytest.mark.parametrize("got", [None, REFERENCE[:2]], ids=["missing", "misshapen"])
def test_a_missing_or_misshapen_gradient_fails_by_name(got):
    matches, metrics = _score(got)
    assert not matches
    assert metrics["l0_gate_up_grad_cos"] == metrics["l0_gate_up_grad_norm_ratio"] == MISSING_GRAD_SCORE


@pytest.mark.parametrize("scale", [2.0, 0.5], ids=["doubled", "halved"])
def test_a_rescaled_gradient_fails_on_the_norm_ratio(scale):
    matches, metrics = _score(REFERENCE * scale)
    assert not matches
    assert metrics["l0_gate_up_grad_cos"] == pytest.approx(1.0)


def test_a_reoriented_gradient_fails_on_the_cosine():
    matches, metrics = _score(REFERENCE.roll(1, dims=0))
    assert not matches
    assert metrics["l0_gate_up_grad_norm_ratio"] == pytest.approx(1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
