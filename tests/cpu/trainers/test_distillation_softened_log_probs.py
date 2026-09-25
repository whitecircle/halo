#!/usr/bin/env python
"""The temperature-softened distillation losses share one fp32 log-prob path.

``softened_log_probs`` upcasts the logits before dividing by ``T``, and ``temperature_rescaled``
applies Hinton's ``T**2``. Pinned against independent spellings of each loss:

- the OPD losses (reverse, forward, unnormalized KL) equal the upcast-then-divide formulas bit for
  bit, values and student gradients, at every temperature, for bf16 and fp32 logits;
- the teacher arm's ``kl_divergence`` is the OPD forward KL, and it, ``soft_cross_entropy`` and
  ``jensen_shannon`` agree with the ``kl_div`` / ``softmax`` spellings to fp32 rounding;
- at a temperature bf16 cannot divide exactly, a bf16 input scores exactly like its fp32 copy: the
  divide runs once, after the upcast, never on the bf16 logits.

    python tests/cpu/trainers/test_distillation_softened_log_probs.py
"""

import pytest
import torch
from torch.nn.functional import kl_div, log_softmax, softmax

from src.trainers.distillation.losses import forward_kl_opd_loss, reverse_kl_opd_loss, unnormalized_kl_loss
from src.trainers.distillation.teacher_losses import call_distillation_loss, get_distillation_loss_fn

BATCH, SEQ, VOCAB = 2, 5, 257
TEMPERATURES = (1.0, 2.0, 0.7, 1.3)
# A temperature whose divide rounds in bf16, so a bf16-side divide is visible.
INEXACT_TEMPERATURE = 0.7
# fp32 rounding between two spellings of one loss (q log q - q log p vs q (log q - log p), softmax vs
# exp(log_softmax)).
FP32_RTOL = 1e-5
FP32_ATOL = 1e-6
TEACHER_SOFTENED_LOSSES = ("kl_divergence", "soft_cross_entropy", "jensen_shannon")


def _log_probs(logits, temperature):
    return log_softmax(logits.float() / temperature, dim=-1)


def _reverse_kl(student, teacher, temperature):
    student_logprobs = _log_probs(student, temperature)
    teacher_logprobs = _log_probs(teacher.detach(), temperature)
    return student_logprobs.exp() * (student_logprobs - teacher_logprobs) * (temperature**2)


def _forward_kl(student, teacher, temperature):
    student_logprobs = _log_probs(student, temperature)
    teacher_logprobs = _log_probs(teacher.detach(), temperature)
    teacher_probs = teacher_logprobs.exp()
    return teacher_probs * (teacher_logprobs - student_logprobs) * (temperature**2)


def _unnormalized_kl(student, teacher, temperature):
    student_logprobs = _log_probs(student, temperature)
    teacher_logprobs = _log_probs(teacher.detach(), temperature)
    student_probs = student_logprobs.exp()
    teacher_probs = teacher_logprobs.exp()
    log_ratio = student_logprobs - teacher_logprobs
    return (student_probs * log_ratio + (teacher_probs - student_probs)) * (temperature**2)


def _kl_div_spelling(student, teacher, temperature):
    teacher_probs = softmax(teacher.float() / temperature, dim=-1)
    return kl_div(_log_probs(student, temperature), teacher_probs, reduction="none") * (temperature**2)


def _soft_cross_entropy_spelling(student, teacher, temperature):
    teacher_probs = softmax(teacher.float() / temperature, dim=-1)
    return -(teacher_probs * _log_probs(student, temperature)) * (temperature**2)


def _jensen_shannon_spelling(student, teacher, temperature):
    student_probs = softmax(student.float() / temperature, dim=-1)
    teacher_probs = softmax(teacher.float() / temperature, dim=-1)
    log_m = (0.5 * (student_probs + teacher_probs)).clamp_min(1e-12).log()
    both = kl_div(log_m, student_probs, reduction="none") + kl_div(log_m, teacher_probs, reduction="none")
    return 0.5 * both * (temperature**2)


SPELLINGS = {
    "kl_divergence": _kl_div_spelling,
    "soft_cross_entropy": _soft_cross_entropy_spelling,
    "jensen_shannon": _jensen_shannon_spelling,
}


def _logits(dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn(BATCH, SEQ, VOCAB, generator=generator) * 4.0
    teacher = student + torch.randn(BATCH, SEQ, VOCAB, generator=generator)
    return student.to(dtype), teacher.to(dtype)


def _teacher_loss(name):
    loss_fn = get_distillation_loss_fn(name)
    hard_labels = torch.zeros(BATCH, SEQ, dtype=torch.long)
    return lambda student, teacher, temperature: call_distillation_loss(
        loss_fn, student, teacher, temperature, hard_labels
    )


def _value_and_grad(loss_fn, student, teacher, temperature):
    student = student.clone().requires_grad_(True)
    value = loss_fn(student, teacher, temperature)
    value.sum().backward()
    return value.detach(), student.grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize(
    ("loss_fn", "reference_fn"),
    [
        (reverse_kl_opd_loss, _reverse_kl),
        (forward_kl_opd_loss, _forward_kl),
        (unnormalized_kl_loss, _unnormalized_kl),
    ],
)
def test_opd_losses_are_bit_identical_to_upcast_then_divide(loss_fn, reference_fn, temperature, dtype):
    student, teacher = _logits(dtype)
    value, grad = _value_and_grad(loss_fn, student, teacher, temperature)
    reference, reference_grad = _value_and_grad(reference_fn, student, teacher, temperature)
    assert value.dtype == torch.float32
    assert torch.equal(value, reference), f"{loss_fn.__name__} value at T={temperature}, {dtype}"
    assert torch.equal(grad, reference_grad), f"{loss_fn.__name__} student gradient at T={temperature}, {dtype}"


def test_teacher_kl_is_the_opd_forward_kl():
    assert get_distillation_loss_fn("kl_divergence") is forward_kl_opd_loss


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("name", TEACHER_SOFTENED_LOSSES)
def test_teacher_softened_losses_match_their_reference_spelling(name, temperature, dtype):
    student, teacher = _logits(dtype, seed=1)
    value, grad = _value_and_grad(_teacher_loss(name), student, teacher, temperature)
    reference, reference_grad = _value_and_grad(SPELLINGS[name], student, teacher, temperature)
    torch.testing.assert_close(value.sum(-1), reference.sum(-1), rtol=FP32_RTOL, atol=FP32_ATOL)
    torch.testing.assert_close(grad.float(), reference_grad.float(), rtol=FP32_RTOL, atol=FP32_ATOL)


@pytest.mark.parametrize("name", TEACHER_SOFTENED_LOSSES)
def test_teacher_softened_losses_divide_after_the_upcast(name):
    student, teacher = _logits(torch.bfloat16, seed=2)
    loss = _teacher_loss(name)
    assert torch.equal(
        loss(student, teacher, INEXACT_TEMPERATURE), loss(student.float(), teacher.float(), INEXACT_TEMPERATURE)
    )
    bf16_divide = log_softmax(student / INEXACT_TEMPERATURE, dim=-1, dtype=torch.float32)
    assert not torch.equal(bf16_divide, _log_probs(student, INEXACT_TEMPERATURE)), (
        "fixture does not separate a bf16 divide from an fp32 one"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
