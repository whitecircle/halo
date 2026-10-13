#!/usr/bin/env python
"""Distillation divergences must evaluate in fp32 even when the logits arrive bf16.

Models load bf16 by default, and every one of these objectives subtracts nearly equal quantities
(``log p - log q``, ``1 - cos``). In bf16 that cancellation dominates the result exactly in the
converged regime distillation ends in — far enough to flip the sign of a provably non-negative
divergence. These tests pin the fp32 evaluation, not a tolerance: a bf16 input scores bit for bit
like its fp32 copy, and the non-negativity a bf16 evaluation breaks survives.
"""

import pytest
import torch

from src.args.mixins import DEFAULT_JSD_BETA
from src.trainers.distillation.losses import (
    DIVERGENCES,
    call_divergence,
    forward_kl_loss,
    get_divergence,
    masked_token_mean,
    reverse_kl_loss,
    unnormalized_kl_loss,
)

# Vocab wide enough that a bf16 log-sum-exp actually loses the small per-element differences.
VOCAB = 4096
# Teacher/student separated by this much logit noise = a near-converged student, where the bf16
# per-element KL terms (O(1e-6)) fall below bf16 resolution around log-probs of O(-8).
CONVERGED_SIGMA = 0.02
# A temperature whose divide rounds in bf16, so a bf16-side divide is visible.
INEXACT_TEMPERATURE = 0.7


def _near_converged_pair(seed: int = 0):
    """(student, teacher) bf16 logits for a student that has nearly matched its teacher."""
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(2, 8, VOCAB, generator=generator) * 3.0
    student = teacher + torch.randn(teacher.shape, generator=generator) * CONVERGED_SIGMA
    return student.bfloat16(), teacher.bfloat16()


@pytest.mark.parametrize("name", sorted(DIVERGENCES))
def test_every_divergence_scores_bf16_logits_exactly_as_their_fp32_copy(name):
    """Bit-exact, so the upcast precedes every arithmetic step, the temperature divide included: a
    bf16 step anywhere would differ (the fixture's own check shows a bf16 divide does)."""
    student, teacher = _near_converged_pair()
    assert not torch.equal((student / INEXACT_TEMPERATURE).float(), student.float() / INEXACT_TEMPERATURE)
    hard_labels = torch.randint(0, VOCAB, student.shape[:2], generator=torch.Generator().manual_seed(1))
    divergence = get_divergence(name, jsd_beta=DEFAULT_JSD_BETA)
    from_bf16 = call_divergence(divergence, student, teacher, INEXACT_TEMPERATURE, hard_labels)
    from_fp32 = call_divergence(divergence, student.float(), teacher.float(), INEXACT_TEMPERATURE, hard_labels)
    assert from_bf16.dtype == torch.float32
    assert torch.equal(from_bf16, from_fp32), f"{name} evaluates some step in bf16"


@pytest.mark.parametrize("loss_fn", [reverse_kl_loss, forward_kl_loss])
def test_the_kl_divergences_stay_non_negative_on_bf16_logits(loss_fn):
    """KL >= 0 vocab-summed; bf16 arithmetic drives per-token values negative."""
    student, teacher = _near_converged_pair()
    per_token = loss_fn(student, teacher, 2.0).sum(-1)
    assert per_token.min().item() >= 0.0, f"{loss_fn.__name__} went negative ({per_token.min().item():.3e})"


def test_unnormalized_kl_stays_non_negative_elementwise_on_bf16_logits():
    """UKL's (Q - P) mass correction makes it non-negative elementwise — bf16 cancellation breaks that.

    The floor is ``-1e-6``, not ``0``: ``p*log(p/q) + (q - p)`` is analytically non-negative but
    both terms are O(p*delta) while their sum is O(p*delta**2), so fp32 arithmetic leaves residual
    negatives around ``-5e-9``. bf16 arithmetic on the same inputs reaches ``-5e-4`` — five orders
    of magnitude away, so this threshold discriminates the two rather than tracking fp32 noise.
    """
    student, teacher = _near_converged_pair()
    assert unnormalized_kl_loss(student, teacher, 1.0).min().item() >= -1e-6


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
