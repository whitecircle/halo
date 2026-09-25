#!/usr/bin/env python
"""One forward-KL implementation serves both distillation arms, and sharing it changed no OPD number.

The OPD losses (self-distillation, SDPG) and the teacher arm's ``kl_divergence`` each carried their
own KL. They now share ``softened_log_probs`` and ``temperature_rescaled``. The OPD losses must be
bit-identical to the formulas they replaced, values and gradients, at every temperature and for bf16
and fp32 logits. The teacher KL moves from ``kl_div`` over a bf16-divided softmax to the OPD form:
it must agree with the old value to fp32 rounding wherever the bf16 divide was exact (``T`` a power
of two), and elsewhere read the fp32-divided logits the old form rounded a second time.

    python tests/cpu/trainers/test_distillation_kl_unification.py
"""

import pytest
import torch
from torch.nn.functional import kl_div, log_softmax, softmax

from src.trainers.distillation.losses import forward_kl_opd_loss, reverse_kl_opd_loss, unnormalized_kl_loss
from src.trainers.distillation.teacher_losses import call_distillation_loss, get_distillation_loss_fn

BATCH, SEQ, VOCAB = 2, 5, 257
TEMPERATURES = (1.0, 2.0, 0.7, 1.3)
# fp32 rounding between the kl_div spelling (q log q - q log p) and the log-space one (q (log q - log p)).
FP32_RTOL = 1e-5
FP32_ATOL = 1e-6


def _old_reverse_kl(student, teacher, temperature):
    student_logprobs = log_softmax(student.float() / temperature, dim=-1)
    teacher_logprobs = log_softmax(teacher.detach().float() / temperature, dim=-1)
    return student_logprobs.exp() * (student_logprobs - teacher_logprobs) * (temperature**2)


def _old_forward_kl(student, teacher, temperature):
    student_logprobs = log_softmax(student.float() / temperature, dim=-1)
    teacher_logprobs = log_softmax(teacher.detach().float() / temperature, dim=-1)
    teacher_probs = teacher_logprobs.exp()
    return teacher_probs * (teacher_logprobs - student_logprobs) * (temperature**2)


def _old_unnormalized_kl(student, teacher, temperature):
    student_logprobs = log_softmax(student.float() / temperature, dim=-1)
    teacher_logprobs = log_softmax(teacher.detach().float() / temperature, dim=-1)
    student_probs = student_logprobs.exp()
    teacher_probs = teacher_logprobs.exp()
    log_ratio = student_logprobs - teacher_logprobs
    return (student_probs * log_ratio + (teacher_probs - student_probs)) * (temperature**2)


def _old_teacher_kl(student, teacher, temperature):
    student_logprobs = log_softmax(student / temperature, dim=-1, dtype=torch.float32)
    teacher_probs = softmax(teacher / temperature, dim=-1, dtype=torch.float32)
    return kl_div(student_logprobs, teacher_probs, reduction="none", log_target=False) * (temperature**2)


def _logits(dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn(BATCH, SEQ, VOCAB, generator=generator) * 4.0
    teacher = student + torch.randn(BATCH, SEQ, VOCAB, generator=generator)
    return student.to(dtype), teacher.to(dtype)


def _value_and_grad(loss_fn, student, teacher, temperature):
    student = student.clone().requires_grad_(True)
    value = loss_fn(student, teacher, temperature)
    value.sum().backward()
    return value.detach(), student.grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize(
    ("loss_fn", "old_fn"),
    [
        (reverse_kl_opd_loss, _old_reverse_kl),
        (forward_kl_opd_loss, _old_forward_kl),
        (unnormalized_kl_loss, _old_unnormalized_kl),
    ],
)
def test_opd_losses_are_bit_identical_to_the_replaced_formulas(loss_fn, old_fn, temperature, dtype):
    student, teacher = _logits(dtype)
    value, grad = _value_and_grad(loss_fn, student, teacher, temperature)
    old_value, old_grad = _value_and_grad(old_fn, student, teacher, temperature)
    assert value.dtype == torch.float32
    assert torch.equal(value, old_value), f"{loss_fn.__name__} changed value at T={temperature}, {dtype}"
    assert torch.equal(grad, old_grad), f"{loss_fn.__name__} changed its student gradient at T={temperature}, {dtype}"


def test_teacher_kl_is_the_opd_forward_kl():
    assert get_distillation_loss_fn("kl_divergence") is forward_kl_opd_loss


@pytest.mark.parametrize(
    ("dtype", "temperature"),
    [(torch.float32, t) for t in TEMPERATURES] + [(torch.bfloat16, 1.0), (torch.bfloat16, 2.0)],
)
def test_teacher_kl_matches_the_replaced_kl_div_to_fp32_rounding(dtype, temperature):
    """Every fp32 input, and bf16 wherever ``x / T`` is exact in bf16: the same KL, values and gradient."""
    student, teacher = _logits(dtype, seed=1)
    loss_fn = get_distillation_loss_fn("kl_divergence")
    hard_labels = torch.zeros(BATCH, SEQ, dtype=torch.long)
    value, grad = _value_and_grad(
        lambda s, t, temp: call_distillation_loss(loss_fn, s, t, temp, hard_labels), student, teacher, temperature
    )
    old_value, old_grad = _value_and_grad(_old_teacher_kl, student, teacher, temperature)
    torch.testing.assert_close(value.sum(-1), old_value.sum(-1), rtol=FP32_RTOL, atol=FP32_ATOL)
    torch.testing.assert_close(grad.float(), old_grad.float(), rtol=FP32_RTOL, atol=FP32_ATOL)


def test_teacher_kl_divides_bf16_logits_after_the_upcast():
    """At a temperature bf16 cannot divide exactly, the old form rounded ``x / T`` to bf16 before the
    upcast; the shared one reads the bf16 logits exactly as their fp32 copies."""
    temperature = 0.7
    student, teacher = _logits(torch.bfloat16, seed=2)
    from_bf16 = forward_kl_opd_loss(student, teacher, temperature)
    from_fp32 = forward_kl_opd_loss(student.float(), teacher.float(), temperature)
    assert torch.equal(from_bf16, from_fp32)
    old_from_bf16 = _old_teacher_kl(student, teacher, temperature)
    old_from_fp32 = _old_teacher_kl(student.float(), teacher.float(), temperature)
    assert not torch.allclose(old_from_bf16, old_from_fp32, rtol=FP32_RTOL, atol=FP32_ATOL), (
        "fixture does not separate a bf16 divide from an fp32 one"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
