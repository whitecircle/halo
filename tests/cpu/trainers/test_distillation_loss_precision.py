#!/usr/bin/env python
"""Distillation divergences evaluate in fp32 even when the logits arrive bf16.
They subtract nearly equal quantities; near convergence bf16 cancellation can flip a non-negative divergence's sign."""

import pytest
import torch

from halo.trainers.distillation.divergences import (
    DIVERGENCES,
    call_divergence,
    forward_kl_loss,
    masked_token_mean,
    require_divergence,
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
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(2, 8, VOCAB, generator=generator) * 3.0
    student = teacher + torch.randn(teacher.shape, generator=generator) * CONVERGED_SIGMA
    return student.bfloat16(), teacher.bfloat16()


@pytest.mark.parametrize("name", sorted(DIVERGENCES))
def test_every_divergence_scores_bf16_logits_exactly_as_their_fp32_copy(name):
    """Bit-exact, so the upcast precedes every step; the first assert shows a bf16 temperature divide would differ."""
    student, teacher = _near_converged_pair()
    assert not torch.equal((student / INEXACT_TEMPERATURE).float(), student.float() / INEXACT_TEMPERATURE)
    hard_labels = torch.randint(0, VOCAB, student.shape[:2], generator=torch.Generator().manual_seed(1))
    divergence = require_divergence(name, jsd_beta=0.5)
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
    """Floor ``-1e-6``: fp32 leaves residual negatives near ``-5e-9`` (terms O(p*delta), their sum O(p*delta**2)).
    bf16 on the same inputs reaches ``-5e-4``, so the threshold separates the two."""
    student, teacher = _near_converged_pair()
    assert unnormalized_kl_loss(student, teacher, 1.0).min().item() >= -1e-6


@pytest.mark.parametrize("count", [1500, 3000, 12345])
def test_masked_mean_token_count_is_exact_for_bf16_losses(count):
    """None of these counts is bf16-representable (1500 -> 1504, 3000 -> 3008, 12345 -> 12352).
    A denominator rounded to bf16 rescales the mean ~0.3%; one exact nonzero token isolates it from the numerator."""
    loss = torch.zeros(1, count, dtype=torch.bfloat16)
    loss[0, 0] = 1024.0
    mask = torch.ones(1, count, dtype=torch.bfloat16)
    assert masked_token_mean(loss, mask).item() == pytest.approx(1024.0 / count, rel=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
