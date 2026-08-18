#!/usr/bin/env python
"""Distillation losses must evaluate in fp32 even when the logits arrive bf16.

Models load bf16 by default, and every one of these objectives subtracts nearly equal quantities
(``log p - log q``, ``1 - cos``). In bf16 that cancellation dominates the result exactly in the
converged regime distillation ends in — far enough to flip the sign of a provably non-negative
divergence. These tests pin the fp32 evaluation, not a tolerance: they assert the bf16-input result
matches the fp32-input result and that non-negativity survives.
"""

import pytest
import torch

from src.trainers.distillation.losses import (
    forward_kl_opd_loss,
    masked_token_mean,
    reverse_kl_opd_loss,
    unnormalized_kl_loss,
)
from src.trainers.distillation.teacher_losses import (
    alpha_beta_divergence_loss,
    call_distillation_loss,
    cosine_similarity_loss,
    earth_mover_distance,
    get_distillation_loss_fn,
    jensen_shannon_divergence,
    kl_divergence_loss,
    soft_target_cross_entropy_loss,
)

# Vocab wide enough that a bf16 log-sum-exp actually loses the small per-element differences.
VOCAB = 4096
# Teacher/student separated by this much logit noise = a near-converged student, where the bf16
# per-element KL terms (O(1e-6)) fall below bf16 resolution around log-probs of O(-8).
CONVERGED_SIGMA = 0.02


def _near_converged_pair(seed: int = 0):
    """(student, teacher) bf16 logits for a student that has nearly matched its teacher."""
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(2, 8, VOCAB, generator=generator) * 3.0
    student = teacher + torch.randn(teacher.shape, generator=generator) * CONVERGED_SIGMA
    return student.bfloat16(), teacher.bfloat16()


def test_kl_divergence_is_non_negative_on_bf16_logits():
    """KL >= 0 elementwise-summed; bf16 arithmetic drove per-token values negative."""
    student, teacher = _near_converged_pair()
    per_token = kl_divergence_loss(student, teacher, temperature=2.0).sum(-1)
    assert per_token.min().item() >= 0.0, (
        f"KL divergence went negative ({per_token.min().item():.3e}) — the loss is being computed "
        f"in bf16, so log p - log q is pure rounding noise near convergence"
    )


@pytest.mark.parametrize(
    "loss_name",
    ["kl_divergence", "soft_cross_entropy", "jensen_shannon", "earth_mover_distance", "alpha_beta_divergence"],
)
def test_teacher_losses_match_fp32_reference_on_bf16_logits(loss_name):
    """A bf16 input must give (near) the fp32 answer — i.e. the math ran in fp32, not bf16."""
    student, teacher = _near_converged_pair()
    loss_fn = get_distillation_loss_fn(loss_name)
    hard_labels = torch.zeros(student.shape[:2], dtype=torch.long)

    from_bf16 = call_distillation_loss(loss_fn, student, teacher, 2.0, hard_labels).sum()
    reference = call_distillation_loss(loss_fn, student.float(), teacher.float(), 2.0, hard_labels).sum()

    assert from_bf16.dtype == torch.float32, f"{loss_name} returned {from_bf16.dtype}, so it reduced in bf16"
    scale = max(abs(reference.item()), 1e-12)
    relative_error = abs(from_bf16.item() - reference.item()) / scale
    assert relative_error < 1e-3, (
        f"{loss_name} on bf16 logits differs from the fp32 result by {relative_error:.1%} — "
        f"the loss is evaluating in bf16"
    )


def test_soft_cross_entropy_and_jsd_track_kl_ordering():
    """The probability-space losses must stay ordered vs a strictly-worse student.

    bf16 noise at these magnitudes exceeded the gap between a converged and a diverged student,
    inverting the comparison the objective exists to make.
    """
    student, teacher = _near_converged_pair()
    generator = torch.Generator().manual_seed(7)
    worse = (teacher.float() + torch.randn(teacher.shape, generator=generator) * 0.5).bfloat16()

    for loss_fn in (kl_divergence_loss, soft_target_cross_entropy_loss, jensen_shannon_divergence):
        close = loss_fn(student, teacher, 2.0).sum()
        far = loss_fn(worse, teacher, 2.0).sum()
        assert close < far, f"{loss_fn.__name__} ranked a diverged student below a converged one"


def test_unnormalized_kl_is_non_negative_on_bf16_logits():
    """UKL's (Q - P) mass correction makes it non-negative elementwise — bf16 cancellation breaks that.

    Pins the sibling-consistency of ``losses``: ``reverse_kl``/``forward_kl`` upcast, and
    ``unnormalized_kl`` (the default self-distillation loss) must too.

    The floor is ``-1e-6``, not ``0``: ``p*log(p/q) + (q - p)`` is analytically non-negative but
    both terms are O(p*delta) while their sum is O(p*delta**2), so fp32 arithmetic leaves residual
    negatives around ``-5e-9``. bf16 arithmetic on the same inputs reaches ``-5e-4`` — five orders
    of magnitude away, so this threshold discriminates the two rather than tracking fp32 noise.
    """
    student, teacher = _near_converged_pair()
    per_element = unnormalized_kl_loss(student, teacher, temperature=1.0)
    assert per_element.dtype == torch.float32
    assert per_element.min().item() >= -1e-6, (
        f"unnormalized_kl went negative ({per_element.min().item():.3e}); its mass-correction term "
        f"subtracts two nearly equal bf16 probabilities"
    )


@pytest.mark.parametrize("loss_fn", [reverse_kl_opd_loss, forward_kl_opd_loss, unnormalized_kl_loss])
def test_shared_distillation_losses_match_fp32_reference(loss_fn):
    """All three OPD losses must agree with their fp32 reference — none may skip the upcast."""
    student, teacher = _near_converged_pair()
    from_bf16 = loss_fn(student, teacher, 1.0).sum()
    reference = loss_fn(student.float(), teacher.float(), 1.0).sum()
    scale = max(abs(reference.item()), 1e-12)
    assert abs(from_bf16.item() - reference.item()) / scale < 1e-3, (
        f"{loss_fn.__name__} on bf16 logits diverges from its fp32 reference"
    )


def test_cosine_similarity_loss_resolves_near_identical_logits():
    """``1 - cos`` near cos=1 is catastrophic cancellation: bf16 spacing at 1.0 is ~0.0078."""
    student, teacher = _near_converged_pair()
    loss = cosine_similarity_loss(student, teacher)
    reference = cosine_similarity_loss(student.float(), teacher.float())
    assert loss.dtype == torch.float32
    # The true value is ~1e-5; a bf16 evaluation quantizes it to 0 or to a multiple of 0.0078.
    assert reference.mean().item() > 0.0
    assert torch.allclose(loss, reference, atol=1e-6)


def test_earth_mover_and_alpha_beta_return_fp32():
    """The two CDF/power losses also run their softmax in fp32."""
    student, teacher = _near_converged_pair()
    assert earth_mover_distance(student, teacher).dtype == torch.float32
    assert alpha_beta_divergence_loss(student, teacher).dtype == torch.float32


@pytest.mark.parametrize("count", [1500, 3000, 12345])
def test_masked_mean_token_count_is_exact_for_bf16_losses(count):
    """The kept-token denominator (and the divide itself) must stay fp32, not the loss dtype.

    None of these counts is bf16-representable (1500 -> 1504, 3000 -> 3008, 12345 -> 12352), so a
    denominator or quotient rounded to bf16 rescales the mean by ~0.3%. A single exactly
    representable nonzero token makes the numerator exact, isolating the denominator's rounding.
    """
    loss = torch.zeros(1, count, dtype=torch.bfloat16)
    loss[0, 0] = 1024.0
    mask = torch.ones(1, count, dtype=torch.bfloat16)
    assert masked_token_mean(loss, mask).item() == pytest.approx(1024.0 / count, rel=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
