#!/usr/bin/env python
"""The teacher top-k + tail-bin objective (``distill_topk``).

The student and teacher softened distributions are restricted to the teacher's top-k tokens plus one
bin holding the rest of the vocabulary, and a teacher-weighted loss scores those ``k + 1`` bins.
Pinned against independent float64 spellings:

- the tail bin carries exactly the probability mass outside the support, so each binned row is a
  distribution, and a student whose support mass rounds to 1 in fp32 still gets the tail's value and
  gradient;
- the support is the teacher's top-k;
- at ``k = V`` the binned loss matches the full-vocab loss, value and student gradient;
- below that, binning never raises a divergence (a lower bound on the full-vocab KL);
- the softening runs over the full vocabulary and the ``T**2`` rescale is applied once.

    python tests/cpu/trainers/test_distillation_teacher_topk.py
"""

import pytest
import torch
from torch.nn.functional import log_softmax

from src.configs.distillation_config import TOPK_DISTILL_LOSSES
from src.trainers.distillation.losses import call_divergence, get_divergence, teacher_topk_log_probs

BATCH, SEQ, VOCAB = 2, 5, 64
TOPK = 6
TEMPERATURES = (1.0, 2.0)
# fp32 loss against a float64 reference.
REFERENCE_RTOL = 1e-5
REFERENCE_ATOL = 1e-6


def _logits(seed=0):
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn(BATCH, SEQ, VOCAB, generator=generator) * 2.0
    teacher = student + torch.randn(BATCH, SEQ, VOCAB, generator=generator)
    return student, teacher


def _binned_reference(logits, support, temperature):
    """float64 log-probs on ``support`` plus the log of the summed probability off it."""
    logprobs = log_softmax(logits.double() / temperature, dim=-1)
    off_support = logprobs.scatter(-1, support, float("-inf")).logsumexp(-1, keepdim=True)
    return torch.cat([logprobs.gather(-1, support), off_support], dim=-1)


def _forward_kl_on_bins(student_bins, teacher_bins):
    return (teacher_bins.exp() * (teacher_bins - student_bins)).sum(-1)


def _soft_cross_entropy_on_bins(student_bins, teacher_bins):
    return -(teacher_bins.exp() * student_bins).sum(-1)


REFERENCES_ON_BINS = {"kl_divergence": _forward_kl_on_bins, "soft_cross_entropy": _soft_cross_entropy_on_bins}


def _topk_loss(name, topk, student, teacher, temperature):
    hard_labels = torch.zeros(BATCH, SEQ, dtype=torch.long)
    return call_divergence(get_divergence(name, topk=topk), student, teacher, temperature, hard_labels)


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_tail_bin_carries_exactly_the_mass_off_the_teacher_topk(temperature):
    student, teacher = _logits()
    student_bins, teacher_bins = teacher_topk_log_probs(student, teacher, TOPK, temperature)
    support = teacher.topk(TOPK, dim=-1).indices

    assert student_bins.shape == (BATCH, SEQ, TOPK + 1)
    for bins, logits in ((student_bins, student), (teacher_bins, teacher)):
        reference = _binned_reference(logits, support, temperature)
        torch.testing.assert_close(bins.double(), reference, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
        torch.testing.assert_close(bins.logsumexp(-1), torch.zeros(BATCH, SEQ), rtol=0.0, atol=REFERENCE_ATOL)


def test_the_support_is_the_teacher_topk_not_the_student_topk():
    """Reversing the student's preferences must not move the bins: they follow the teacher alone."""
    student, teacher = _logits(seed=1)
    _, teacher_bins = teacher_topk_log_probs(student, teacher, TOPK, 1.0)
    _, teacher_bins_reversed_student = teacher_topk_log_probs(-student, teacher, TOPK, 1.0)
    assert torch.equal(teacher_bins, teacher_bins_reversed_student)


@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("name", TOPK_DISTILL_LOSSES)
def test_full_vocabulary_support_reproduces_the_full_vocab_loss(name, temperature):
    """At ``k = V`` the tail bin is empty, so the binned loss matches the plain loss, value and gradient."""
    student, teacher = _logits(seed=2)

    binned_student = student.clone().requires_grad_(True)
    binned = _topk_loss(name, VOCAB, binned_student, teacher, temperature).sum(-1)
    binned.sum().backward()

    full_student = student.clone().requires_grad_(True)
    full = _topk_loss(name, None, full_student, teacher, temperature).sum(-1)
    full.sum().backward()

    torch.testing.assert_close(binned, full, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
    torch.testing.assert_close(binned_student.grad, full_student.grad, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_binned_forward_kl_matches_its_reference_and_carries_the_temperature_squared(temperature):
    student, teacher = _logits(seed=3)
    support = teacher.topk(TOPK, dim=-1).indices
    reference = _forward_kl_on_bins(
        _binned_reference(student, support, temperature), _binned_reference(teacher, support, temperature)
    )
    binned = _topk_loss("kl_divergence", TOPK, student, teacher, temperature).sum(-1)
    torch.testing.assert_close(binned.double(), reference * temperature**2, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)


def test_binning_lower_bounds_the_full_vocab_kl():
    """Merging tokens into one bin never increases a KL (data processing inequality), and here it strictly drops it."""
    student, teacher = _logits(seed=4)
    binned = _topk_loss("kl_divergence", TOPK, student, teacher, 1.0).sum(-1)
    full = _topk_loss("kl_divergence", None, student, teacher, 1.0).sum(-1)
    assert (binned <= full + REFERENCE_ATOL).all()
    assert (binned < full).any()


@pytest.mark.parametrize("name", TOPK_DISTILL_LOSSES)
def test_a_student_with_all_its_mass_on_the_support_keeps_its_tail_gradient(name):
    """A student whose top-k mass rounds to 1 in fp32 still owes the teacher's tail mass.

    ``log(1 - support mass)`` would turn its tail bin into a constant: the loss stays off by the
    missing tail term and the gradient that moves mass back to the tail vanishes.
    """
    teacher = torch.zeros(1, 1, VOCAB)
    teacher[..., 0] = 1.0
    student = torch.full((1, 1, VOCAB), -40.0)
    student[..., 0] = 0.0
    support = teacher.topk(TOPK, dim=-1).indices

    fp32_student = student.clone().requires_grad_(True)
    value = _topk_loss(name, TOPK, fp32_student, teacher, 1.0).sum()
    value.backward()

    reference_student = student.double().requires_grad_(True)
    reference = REFERENCES_ON_BINS[name](
        _binned_reference(reference_student, support, 1.0), _binned_reference(teacher, support, 1.0)
    ).sum()
    reference.backward()

    torch.testing.assert_close(value.double(), reference, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
    torch.testing.assert_close(
        fp32_student.grad.double(), reference_student.grad, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
