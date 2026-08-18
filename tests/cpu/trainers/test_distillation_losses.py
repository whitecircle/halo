#!/usr/bin/env python
"""Tests for the distillation loss functions and masking utilities.

Pure tensor math (``src/training/distillation.py``), so the assertions are
on closed-form values, symmetry, temperature scaling, gradient finiteness and the
reduction semantics — not just "runs". Run: ``pytest tests/cpu/trainers/test_distillation_losses.py``.
"""

import inspect
import math

import pytest
import torch
from torch.nn.functional import softmax

from src.trainers.distillation.losses import masked_token_mean
from src.trainers.distillation.teacher_losses import (
    _DISTILLATION_LOSSES,
    alpha_beta_divergence_loss,
    apply_hard_labels_mask,
    call_distillation_loss,
    cosine_similarity_loss,
    earth_mover_distance,
    get_distillation_loss_fn,
    hard_labels_coefficient,
    jensen_shannon_divergence,
    kl_divergence_loss,
    mse_loss_fn,
    slim_loss,
    soft_target_cross_entropy_loss,
)

BS, SEQ, VOCAB = 2, 4, 8
torch.manual_seed(42)
LOGITS_A = torch.randn(BS, SEQ, VOCAB)
LOGITS_B = torch.randn(BS, SEQ, VOCAB)

# Temperatures the gradient-magnitude sweep spans (an 8x range: without the T**2 factor a softened
# divergence's gradient falls 64x across it).
_TEMPERATURES = (1.0, 2.0, 4.0, 8.0)
# Small logits — the near-uniform-softmax regime Hinton's T**2 argument is derived in, and the
# converged regime distillation actually ends in.
_SMALL_LOGIT_SIGMA = 0.02
# Largest gradient-norm spread across _TEMPERATURES that still counts as invariant.
_INVARIANCE_SPREAD = 1.01
# SLIM reaches neither convention's invariance (see its test); this separates its unscaled
# behaviour (~1.5x) from what inheriting the T**2 factor would do (~42x).
_SLIM_SPREAD_CEILING = 2.0


def test_kl_identical_is_zero():
    loss = kl_divergence_loss(LOGITS_A, LOGITS_A, temperature=1.0)
    assert loss.shape == (BS, SEQ, VOCAB)
    assert loss.sum(-1).abs().max().item() < 1e-5


def test_kl_known_closed_form():
    # KL(teacher||student) summed over vocab, single-token distributions.
    student = torch.tensor([[[2.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 0.0]]])  # uniform after softmax
    p = softmax(teacher[0, 0], dim=-1)
    q = softmax(student[0, 0], dim=-1)
    expected = (p * (p / q).log()).sum().item()
    loss = kl_divergence_loss(student, teacher, temperature=1.0).sum(-1)
    assert abs(loss.item() - expected) < 1e-5


def test_kl_is_nonnegative():
    loss = kl_divergence_loss(LOGITS_A, LOGITS_B, temperature=1.0).sum(-1)
    assert (loss >= -1e-6).all()


def test_kl_asymmetric():
    kl_ab = kl_divergence_loss(LOGITS_A, LOGITS_B, temperature=1.0).sum(-1)
    kl_ba = kl_divergence_loss(LOGITS_B, LOGITS_A, temperature=1.0).sum(-1)
    assert not torch.allclose(kl_ab, kl_ba, atol=1e-4)


def test_kl_temperature_squared_scaling():
    """The T**2 prefactor keeps the distillation term commensurate with the SFT term as T grows.

    Compared against the same KL computed WITHOUT the factor: a bare "temperature changed the value"
    bound passes more easily with the factor deleted, since softening alone moves it several-fold.
    """
    t = 2.0
    unscaled = torch.nn.functional.kl_div(
        torch.log_softmax(LOGITS_A / t, dim=-1, dtype=torch.float32),
        torch.softmax(LOGITS_B / t, dim=-1, dtype=torch.float32),
        reduction="none",
        log_target=False,
    ).sum()
    scaled = kl_divergence_loss(LOGITS_A, LOGITS_B, temperature=t).sum()
    assert torch.allclose(scaled, unscaled * t**2, rtol=1e-5), (
        f"expected the T**2={t**2} prefactor: got {scaled.item()} vs {unscaled.item() * t**2}"
    )


def test_kl_gradient_flows_to_student():
    student = LOGITS_A.clone().requires_grad_(True)
    kl_divergence_loss(student, LOGITS_B, temperature=1.5).sum().backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0


def test_mse_identical_is_zero():
    assert mse_loss_fn(LOGITS_A, LOGITS_A).sum().item() == 0.0


def test_mse_known_value_and_shape():
    a = torch.tensor([[[1.0, 0.0]]])
    b = torch.tensor([[[0.0, 1.0]]])
    loss = mse_loss_fn(a, b)
    assert loss.shape == a.shape
    assert torch.allclose(loss, torch.ones_like(loss))  # (1-0)^2 and (0-1)^2


def test_soft_ce_identical_equals_entropy():
    # student==teacher ⇒ -sum(p*log p) is exactly the entropy of the softened distribution.
    loss = soft_target_cross_entropy_loss(LOGITS_A, LOGITS_A, temperature=1.0)
    p = softmax(LOGITS_A, dim=-1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
    assert torch.allclose(loss.sum(-1), entropy, atol=1e-5)


def test_soft_ce_ge_entropy_lower_bound():
    # Gibbs: H(p,q) >= H(p), equality iff q==p.
    ce = soft_target_cross_entropy_loss(LOGITS_A, LOGITS_B, temperature=1.0).sum(-1)
    p = softmax(LOGITS_B, dim=-1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
    assert (ce - entropy >= -1e-5).all()


def test_cosine_identical_is_zero():
    assert cosine_similarity_loss(LOGITS_A, LOGITS_A).abs().max().item() < 1e-5


def test_cosine_orthogonal_is_one():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0]])
    assert abs(cosine_similarity_loss(a, b).item() - 1.0) < 1e-5


def test_cosine_opposite_is_two():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    assert abs(cosine_similarity_loss(a, -a).item() - 2.0) < 1e-5


def test_cosine_scale_invariant():
    base = cosine_similarity_loss(LOGITS_A, LOGITS_B)
    scaled = cosine_similarity_loss(LOGITS_A * 5.0, LOGITS_B)
    assert torch.allclose(base, scaled, atol=1e-5)


def test_jsd_symmetric():
    jsd_ab = jensen_shannon_divergence(LOGITS_A, LOGITS_B, temperature=1.0).sum(-1)
    jsd_ba = jensen_shannon_divergence(LOGITS_B, LOGITS_A, temperature=1.0).sum(-1)
    assert torch.allclose(jsd_ab, jsd_ba, atol=1e-5)


def test_jsd_identical_is_zero():
    jsd = jensen_shannon_divergence(LOGITS_A, LOGITS_A, temperature=1.0)
    assert jsd.sum(-1).abs().max().item() < 1e-5


def test_jsd_bounded_by_ln2():
    jsd = jensen_shannon_divergence(LOGITS_A, LOGITS_B, temperature=1.0).sum(-1)
    assert (jsd >= -1e-6).all()
    assert jsd.max().item() <= math.log(2) + 1e-5


def test_jsd_sharp_opposite_bounded_by_ln2():
    # Saturates at ln 2; the reversed-argument form KL(M||P)+KL(M||Q) is unbounded and fails here.
    s = torch.tensor([[[50.0, -50.0]]])  # ~[1, 0]
    t = torch.tensor([[[-50.0, 50.0]]])  # ~[0, 1]
    jsd = jensen_shannon_divergence(s, t, temperature=1.0).sum(-1)
    assert jsd.max().item() <= math.log(2) + 1e-4
    assert abs(jsd.item() - math.log(2)) < 1e-3


def test_emd_per_token_shape_and_identical_is_zero():
    # Must stay element-wise [batch, seq, vocab] — masked_token_mean sums over the vocab axis.
    loss = earth_mover_distance(LOGITS_A, LOGITS_A)
    assert loss.shape == (BS, SEQ, VOCAB)
    assert loss.sum(-1).abs().max().item() < 1e-5


def test_emd_known_wasserstein_value():
    # 1-Wasserstein over an ordered 2-symbol support: one step of mass ⇒ 1, not the L1/TV value 2.
    s = torch.tensor([[100.0, -100.0]])  # ~[1, 0]
    t = torch.tensor([[-100.0, 100.0]])  # ~[0, 1]
    d = earth_mover_distance(s, t).sum(-1)
    assert abs(d.item() - 1.0) < 1e-3  # |CDF_s - CDF_t| = |1-0| + |1-1|


def test_emd_symmetric():
    assert torch.allclose(
        earth_mover_distance(LOGITS_A, LOGITS_B), earth_mover_distance(LOGITS_B, LOGITS_A), atol=1e-5
    )


def test_emd_gradient_flows_to_student():
    student = LOGITS_A.clone().requires_grad_(True)
    earth_mover_distance(student, LOGITS_B).sum().backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0


def test_ab_zero_when_identical():
    loss = alpha_beta_divergence_loss(LOGITS_A, LOGITS_A, alpha=1.0, beta=2.0)
    assert loss.shape == LOGITS_A.shape
    assert torch.allclose(loss.sum(-1), torch.zeros(LOGITS_A.shape[:-1]), atol=1e-6)


def test_ab_finite_for_distinct_inputs():
    for alpha, beta in [(1.0, 2.0), (2.0, 2.0), (0.5, 1.5)]:
        loss = alpha_beta_divergence_loss(LOGITS_A, LOGITS_B, alpha=alpha, beta=beta)
        assert torch.isfinite(loss).all()


def test_ab_gradient_flows_to_student():
    student = LOGITS_A.clone().requires_grad_(True)
    alpha_beta_divergence_loss(student, LOGITS_B, alpha=1.0, beta=2.0).sum().backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0


@pytest.mark.parametrize("alpha,beta", [(0.0, 2.0), (1.0, 0.0), (1.0, -1.0)])
def test_ab_rejects_degenerate_params(alpha, beta):
    with pytest.raises(ValueError):
        alpha_beta_divergence_loss(LOGITS_A, LOGITS_B, alpha=alpha, beta=beta)


def test_slim_shape_and_finite():
    hard_labels = torch.randint(0, VOCAB, (BS, SEQ))
    loss = slim_loss(LOGITS_A, LOGITS_B, temperature=1.0, hard_labels=hard_labels)
    assert loss.shape == (BS, SEQ, VOCAB)
    assert torch.isfinite(loss).all()


def test_slim_handles_ignored_minus_100_labels():
    # SFT collators emit -100 at masked positions; one_hot must not raise on them.
    hard_labels = torch.randint(0, VOCAB, (BS, SEQ))
    hard_labels[0, 0] = -100
    loss = slim_loss(LOGITS_A, LOGITS_B, temperature=1.0, hard_labels=hard_labels)
    assert loss.shape == (BS, SEQ, VOCAB)
    assert torch.isfinite(loss).all()


def test_slim_zero_when_identical():
    # Not zero even when identical: the label term keeps a coef-weighted entropy, so only
    # finiteness and non-negativity hold.
    hard_labels = torch.randint(0, VOCAB, (BS, SEQ))
    loss = slim_loss(LOGITS_A, LOGITS_A, temperature=1.0, hard_labels=hard_labels)
    assert torch.isfinite(loss).all()
    assert (loss >= -1e-6).all()


def test_attention_mask_all_ones_sums_vocab():
    loss = torch.ones(2, 4, 8)
    result = masked_token_mean(loss, torch.ones(2, 4))
    # 8 = vocab sum per token, then means over tokens and batch.
    assert abs(result.item() - 8.0) < 1e-5


def test_attention_mask_ignores_padding_tokens():
    loss = torch.ones(1, 4, 8)
    full = masked_token_mean(loss, torch.ones(1, 4))
    padded = masked_token_mean(loss, torch.tensor([[1, 1, 0, 0]]).float())
    assert abs(full.item() - padded.item()) < 1e-5


def test_attention_mask_all_padding_no_div_by_zero():
    loss = torch.ones(1, 4, 8)
    result = masked_token_mean(loss, torch.zeros(1, 4))
    assert torch.isfinite(result)
    assert abs(result.item()) < 1e-3  # clamp(min=1e-9) keeps the empty-mask divide finite


def test_hard_labels_mask_ignores_minus_100():
    """Per-token values differ so the KEPT set is identifiable: a uniform fixture returns the same
    number for any mask of the same size, including the inverted one that distills on ignored
    positions only."""
    loss = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1).expand(1, 4, 8).contiguous()
    labels = torch.tensor([[1, -100, 3, -100]])
    # Kept positions 0,2 → mean 8; the inverted mask would keep 1,3 → mean 16.
    result = apply_hard_labels_mask(loss, labels)
    assert abs(result.item() - 8.0) < 1e-5


def test_hard_labels_coefficient_value_and_shape():
    student = torch.tensor([[[2.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 2.0]]])
    labels = torch.tensor([[0]])
    coef = hard_labels_coefficient(student, teacher, labels)
    assert coef.shape == (1, 1, 1)
    sp = softmax(student[0, 0], dim=-1)[0].item()
    tp = softmax(teacher[0, 0], dim=-1)[0].item()
    assert abs(coef.item() - (1 - sp) * tp) < 1e-5


def test_hard_labels_coefficient_handles_ignored_minus_100():
    # -100 must not raise in .gather; the value is dropped later by apply_hard_labels_mask.
    labels = torch.randint(0, VOCAB, (BS, SEQ))
    labels[0, 0] = -100
    coef = hard_labels_coefficient(LOGITS_A, LOGITS_B, labels)
    assert coef.shape == (BS, SEQ, 1)
    assert torch.isfinite(coef).all()


def _gradient_norms_across_temperature(loss_fn, seed=0):
    """Student-gradient norm of ``loss_fn.sum()`` at each of :data:`_TEMPERATURES`."""
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(BS, SEQ, VOCAB * 2, generator=generator) * _SMALL_LOGIT_SIGMA
    base_student = teacher + torch.randn(teacher.shape, generator=generator) * _SMALL_LOGIT_SIGMA * 0.5
    hard_labels = torch.randint(0, teacher.size(-1), teacher.shape[:2], generator=generator)

    norms = []
    for temperature in _TEMPERATURES:
        student = base_student.clone().requires_grad_(True)
        call_distillation_loss(loss_fn, student, teacher, temperature, hard_labels).sum().backward()
        norms.append(student.grad.norm().item())
    return norms


def test_every_soft_loss_holds_its_gradient_magnitude_across_temperature():
    """The point of the ``T**2`` factor, asserted on every registered loss at once.

    Softening divides the logits by ``T``, shrinking a divergence and its gradient as ``1/T**2``;
    multiplying back by ``T**2`` holds the distillation term's pull on the student — and therefore
    its weight against the hard-label term — fixed as ``distill_temperature`` moves. Losses that
    take no temperature are trivially invariant and are swept here too, so the registry is covered
    exhaustively and a new loss cannot join it without meeting the convention.

    ``slim`` is the one exception and has its own test; leaving it silently out of the sweep is what
    let the convention drift apart in the first place.
    """
    swept = []
    for name, loss_fn in _DISTILLATION_LOSSES.items():
        if name == "slim":
            continue
        norms = _gradient_norms_across_temperature(loss_fn)
        spread = max(norms) / min(norms)
        if "temperature" not in inspect.signature(loss_fn).parameters:
            # Not an invariance result: these never see T, so the sweep pins only that they stay out
            # of the convention. Exact equality, since nothing in them can differ.
            assert len(set(norms)) == 1, f"{name} declares no temperature yet its gradient moved: {norms}"
        else:
            assert spread < _INVARIANCE_SPREAD, (
                f"{name}: gradient magnitude moves {spread:.1f}x across T={_TEMPERATURES} "
                f"({norms}) — the T**2 rescale is missing or doubled"
            )
        swept.append(name)
    assert len(swept) == len(_DISTILLATION_LOSSES) - 1, "a loss silently dropped out of the sweep"


def test_slim_is_the_documented_exception_and_keeps_the_unscaled_core():
    """SLIM's coefficient depends on the STUDENT, so it multiplies the cross-entropy's
    teacher-entropy offset — which does not shrink as ``1/T**2`` — as well as the divergence.
    Rescaling that product by ``T**2`` grows its gradient ~``T**2`` (≈42x over this sweep) instead of
    holding it fixed, so ``slim_loss`` deliberately calls the UNSCALED core.

    Pinned from both sides: inheriting the factor blows the ceiling, and dropping the exception's
    justification (making slim's core scaled) is exactly what this catches.
    """
    norms = _gradient_norms_across_temperature(slim_loss)
    spread = max(norms) / min(norms)
    assert 1.0 < spread < _SLIM_SPREAD_CEILING, (
        f"slim's gradient moves {spread:.1f}x across T={_TEMPERATURES} ({norms}); "
        f"above {_SLIM_SPREAD_CEILING} it is inheriting the T**2 rescale"
    )


def test_soft_cross_entropy_and_jsd_carry_the_temperature_squared_prefactor():
    """The two losses that used to omit it, against their own unscaled definitions."""
    t = 3.0
    unscaled_ce = -(torch.softmax(LOGITS_B / t, dim=-1) * torch.log_softmax(LOGITS_A / t, dim=-1)).sum()
    assert torch.allclose(
        soft_target_cross_entropy_loss(LOGITS_A, LOGITS_B, temperature=t).sum(), unscaled_ce * t**2, rtol=1e-5
    )

    student_probs = torch.softmax(LOGITS_A / t, dim=-1)
    teacher_probs = torch.softmax(LOGITS_B / t, dim=-1)
    m = 0.5 * (student_probs + teacher_probs)
    unscaled_jsd = 0.5 * (
        (student_probs * (student_probs.log() - m.log())).sum()
        + (teacher_probs * (teacher_probs.log() - m.log())).sum()
    )
    assert torch.allclose(
        jensen_shannon_divergence(LOGITS_A, LOGITS_B, temperature=t).sum(), unscaled_jsd * t**2, rtol=1e-4
    )


def test_factory_returns_correct_function():
    mapping = {
        "kl_divergence": kl_divergence_loss,
        "mse": mse_loss_fn,
        "soft_cross_entropy": soft_target_cross_entropy_loss,
        "cosine_similarity": cosine_similarity_loss,
        "jensen_shannon": jensen_shannon_divergence,
        "earth_mover_distance": earth_mover_distance,
        "alpha_beta_divergence": alpha_beta_divergence_loss,
        "slim": slim_loss,
    }
    for name, fn in mapping.items():
        assert get_distillation_loss_fn(name) is fn


def test_factory_invalid_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        get_distillation_loss_fn("nonexistent_loss")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
