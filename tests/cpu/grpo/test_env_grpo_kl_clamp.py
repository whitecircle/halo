"""Bounded tail for TRL's k3 KL estimator (``clamp_ref_logps``). Unclamped, ``exp(ref - logp) - (ref - logp) - 1`` on
one token the policy suppresses drives the batch KL and grad_norm up by an order of magnitude."""

import math

import pytest
import torch

from halo.trainers.grpo.objective.logratio import KL_LOGRATIO_CLAMP, clamp_ref_logps


def _k3(ref: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    """TRL's k3 estimator, verbatim (grpo_trainer.py)."""
    d = ref - logp
    return torch.exp(d) - d - 1


def test_typical_tokens_are_untouched():
    logp = torch.tensor([[-1.0, -2.0]])
    ref = logp + 0.4
    clamped, over = clamp_ref_logps(ref, logp)
    torch.testing.assert_close(clamped, ref)
    assert not over.any()


def test_exploding_tail_is_bounded():
    """Per-token KL is ~1.2e5 unclamped at this delta."""
    logp = torch.tensor([[-12.0]])
    ref = torch.tensor([[-0.3]])  # delta = 11.7
    assert _k3(ref, logp).item() > 1e4, "precondition: unclamped k3 explodes"

    clamped, over = clamp_ref_logps(ref, logp)
    bounded = _k3(clamped, logp).item()
    assert bounded == pytest.approx(math.exp(KL_LOGRATIO_CLAMP) - KL_LOGRATIO_CLAMP - 1, rel=1e-4)
    assert bounded < 200, bounded
    assert over.tolist() == [[True]]


def test_only_the_tail_is_clamped_not_the_batch():
    logp = torch.tensor([[-1.0, -1.0, -20.0]])
    ref = torch.tensor([[-0.6, -1.4, -0.5]])  # deltas: +0.4, -0.4, +19.5
    clamped, over = clamp_ref_logps(ref, logp)
    torch.testing.assert_close(clamped[0, :2], ref[0, :2])
    assert clamped[0, 2].item() == pytest.approx(-20.0 + KL_LOGRATIO_CLAMP)
    assert over.tolist() == [[False, False, True]], "the clamp mask names the one clamped position"


def test_policy_above_reference_is_never_clamped():
    """The estimator's other side is already bounded."""
    logp = torch.tensor([[-0.5, -0.1]])
    ref = torch.tensor([[-6.0, -9.0]])
    clamped, over = clamp_ref_logps(ref, logp)
    torch.testing.assert_close(clamped, ref)
    assert not over.any()


def test_gradient_wrt_policy_logp_is_bounded():
    """The KL gradient scales as exp(ref - logp)."""
    logp = torch.tensor([[-12.0]], requires_grad=True)
    ref = torch.tensor([[-0.3]])
    clamped, _ = clamp_ref_logps(ref, logp.detach())
    _k3(clamped, logp).sum().backward()
    assert abs(logp.grad.item()) < math.exp(KL_LOGRATIO_CLAMP) + 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
