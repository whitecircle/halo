"""Bounded tail for TRL's k3 KL estimator (``clamp_ref_logps``).

``per_token_kl = exp(ref - logp) - (ref - logp) - 1`` is unbounded where the policy suppresses a token
the reference likes. Unclamped, a single token drove the batch KL to 19.3 and grad_norm to 39.5 (step 64
of the codeforces run), hijacking the whole optimizer step. Clamping the log-ratio bounds both.
"""

import math
import sys

import pytest
import torch

from src.trainers.grpo.objective.logratio import KL_LOGRATIO_CLAMP, clamp_ref_logps


def _k3(ref: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    """TRL's k3 estimator, verbatim (grpo_trainer.py)."""
    d = ref - logp
    return torch.exp(d) - d - 1


def test_typical_tokens_are_untouched():
    """A token at the observed median log-ratio (~0.4 nats) must pass through unchanged."""
    logp = torch.tensor([[-1.0, -2.0]])
    ref = logp + 0.4
    clamped, frac = clamp_ref_logps(ref, logp)
    torch.testing.assert_close(clamped, ref)
    assert frac.item() == 0.0


def test_exploding_tail_is_bounded():
    """The real step-64 outlier: delta ~11.7 nats -> per-token kl ~1.2e5 unclamped."""
    logp = torch.tensor([[-12.0]])
    ref = torch.tensor([[-0.3]])  # delta = 11.7
    assert _k3(ref, logp).item() > 1e4, "precondition: unclamped k3 explodes"

    clamped, frac = clamp_ref_logps(ref, logp)
    bounded = _k3(clamped, logp).item()
    assert bounded == pytest.approx(math.exp(KL_LOGRATIO_CLAMP) - KL_LOGRATIO_CLAMP - 1, rel=1e-4)
    assert bounded < 200, bounded
    assert frac.item() == 1.0


def test_only_the_tail_is_clamped_not_the_batch():
    """One bad token must not truncate its well-behaved neighbours."""
    logp = torch.tensor([[-1.0, -1.0, -20.0]])
    ref = torch.tensor([[-0.6, -1.4, -0.5]])  # deltas: +0.4, -0.4, +19.5
    clamped, frac = clamp_ref_logps(ref, logp)
    torch.testing.assert_close(clamped[0, :2], ref[0, :2])  # neighbours untouched
    assert clamped[0, 2].item() == pytest.approx(-20.0 + KL_LOGRATIO_CLAMP)
    assert frac.item() == pytest.approx(1 / 3)


def test_policy_above_reference_is_never_clamped():
    """The estimator's other side (logp > ref) is already bounded; do not touch it."""
    logp = torch.tensor([[-0.5, -0.1]])
    ref = torch.tensor([[-6.0, -9.0]])  # deltas strongly negative
    clamped, frac = clamp_ref_logps(ref, logp)
    torch.testing.assert_close(clamped, ref)
    assert frac.item() == 0.0


def test_gradient_wrt_policy_logp_is_bounded():
    """The KL gradient scales as exp(ref - logp); clamping must bound it (that is what clipped the step)."""
    logp = torch.tensor([[-12.0]], requires_grad=True)
    ref = torch.tensor([[-0.3]])
    clamped, _ = clamp_ref_logps(ref, logp.detach())
    _k3(clamped, logp).sum().backward()
    assert abs(logp.grad.item()) < math.exp(KL_LOGRATIO_CLAMP) + 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
