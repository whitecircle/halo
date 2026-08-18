#!/usr/bin/env python
"""
Tests for SDPG-style on-policy self-distillation losses.

Run: python tests/cpu/trainers/test_distillation_shared_losses.py
"""

import pytest
import torch

from src.trainers.distillation.losses import (
    beta_warmup_decay,
    forward_kl_opd_loss,
    get_self_distillation_loss_fn,
    masked_token_mean,
    reverse_kl_opd_loss,
    unnormalized_kl_loss,
)

BS, SEQ, VOCAB = 2, 4, 8
torch.manual_seed(0)
STUDENT = torch.randn(BS, SEQ, VOCAB)
TEACHER = torch.randn(BS, SEQ, VOCAB)


def test_reverse_kl_shape():
    loss = reverse_kl_opd_loss(STUDENT, TEACHER)
    assert loss.shape == (BS, SEQ, VOCAB), loss.shape


def test_reverse_kl_zero_at_identity():
    loss = reverse_kl_opd_loss(STUDENT, STUDENT).sum(-1)
    assert loss.abs().max().item() < 1e-5, loss.abs().max().item()


def test_reverse_kl_nonnegative_when_summed():
    loss = reverse_kl_opd_loss(STUDENT, TEACHER).sum(-1)
    assert loss.min().item() > -1e-6, loss.min().item()


def test_reverse_kl_teacher_detached():
    s = STUDENT.clone().requires_grad_(True)
    t = TEACHER.clone().requires_grad_(True)
    reverse_kl_opd_loss(s, t).sum().backward()
    assert s.grad is not None and s.grad.abs().sum() > 0, "student must receive gradient"
    assert t.grad is None or t.grad.abs().sum() == 0, "teacher branch must be stop-gradient"


def test_reverse_kl_matches_manual():
    p = torch.softmax(STUDENT, dim=-1)
    logp = torch.log_softmax(STUDENT, dim=-1)
    logq = torch.log_softmax(TEACHER, dim=-1)
    manual = (p * (logp - logq)).sum(-1)
    got = reverse_kl_opd_loss(STUDENT, TEACHER).sum(-1)
    assert torch.allclose(got, manual, atol=1e-6), (got - manual).abs().max().item()


def test_forward_kl_zero_at_identity():
    loss = forward_kl_opd_loss(STUDENT, STUDENT).sum(-1)
    assert loss.abs().max().item() < 1e-5, loss.abs().max().item()


def test_forward_kl_differs_from_reverse():
    fwd = forward_kl_opd_loss(STUDENT, TEACHER).sum().item()
    rev = reverse_kl_opd_loss(STUDENT, TEACHER).sum().item()
    assert abs(fwd - rev) > 1e-6, "forward and reverse KL should differ"


def test_ukl_zero_at_identity():
    loss = unnormalized_kl_loss(STUDENT, STUDENT).sum(-1)
    assert loss.abs().max().item() < 1e-5, loss.abs().max().item()


def test_ukl_nonnegative_elementwise():
    # The k3 estimator is non-negative element-wise, not merely once summed.
    loss = unnormalized_kl_loss(STUDENT, TEACHER)
    assert loss.min().item() > -1e-6, loss.min().item()


def test_ukl_sum_equals_kl():
    # Over a normalized vocab the mass-correction term sums to zero.
    ukl = unnormalized_kl_loss(STUDENT, TEACHER).sum(-1)
    kl = reverse_kl_opd_loss(STUDENT, TEACHER).sum(-1)
    assert torch.allclose(ukl, kl, atol=1e-5), (ukl - kl).abs().max().item()


def test_ukl_teacher_detached():
    s = STUDENT.clone().requires_grad_(True)
    t = TEACHER.clone().requires_grad_(True)
    unnormalized_kl_loss(s, t).sum().backward()
    assert t.grad is None or t.grad.abs().sum() == 0, "teacher branch must be stop-gradient"


def test_factory_resolves():
    assert get_self_distillation_loss_fn("reverse_kl") is reverse_kl_opd_loss
    assert get_self_distillation_loss_fn("forward_kl") is forward_kl_opd_loss
    assert get_self_distillation_loss_fn("unnormalized_kl") is unnormalized_kl_loss


def test_factory_rejects_unknown():
    try:
        get_self_distillation_loss_fn("nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown loss")


def test_masked_token_mean_ignores_masked_tokens():
    loss = torch.ones(BS, SEQ, VOCAB)
    mask = torch.zeros(BS, SEQ)
    mask[:, :2] = 1.0  # only first two tokens count
    out = masked_token_mean(loss, mask)
    # VOCAB per token ⇒ both the token-mean and the batch mean are VOCAB.
    assert abs(out.item() - VOCAB) < 1e-6, out.item()


def test_masked_token_mean_confidence_weighting():
    loss = torch.ones(BS, SEQ)  # already vocab-reduced [B, S]
    mask = torch.ones(BS, SEQ)
    weights = torch.tensor([0.0, 2.0])
    out = masked_token_mean(loss, mask, sample_weights=weights)
    # token-means of 1.0 weighted to [0, 2] ⇒ batch mean 1.0.
    assert abs(out.item() - 1.0) < 1e-6, out.item()


def test_masked_token_mean_2d_and_3d_agree():
    loss3d = reverse_kl_opd_loss(STUDENT, TEACHER)
    loss2d = loss3d.sum(-1)
    mask = torch.ones(BS, SEQ)
    a = masked_token_mean(loss3d, mask)
    b = masked_token_mean(loss2d, mask)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().item()


def test_beta_warmup_ramps_up():
    assert beta_warmup_decay(0, 100, 1.0, 10, 10) == 0.0
    assert abs(beta_warmup_decay(5, 100, 1.0, 10, 10) - 0.5) < 1e-9
    assert abs(beta_warmup_decay(10, 100, 1.0, 10, 10) - 1.0) < 1e-9


def test_beta_hold_at_base():
    assert abs(beta_warmup_decay(50, 100, 1.0, 10, 10) - 1.0) < 1e-9


def test_beta_decays_to_zero():
    assert abs(beta_warmup_decay(95, 100, 1.0, 10, 10) - 0.5) < 1e-9
    assert abs(beta_warmup_decay(100, 100, 1.0, 10, 10) - 0.0) < 1e-9


def test_beta_constant_when_no_windows():
    assert abs(beta_warmup_decay(7, 100, 0.3, 0, 0) - 0.3) < 1e-9


def test_beta_disabled_when_base_zero():
    assert beta_warmup_decay(50, 100, 0.0, 10, 10) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
