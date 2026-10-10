#!/usr/bin/env python
"""The temperature-softened distillation losses share one fp32 log-prob path.

``softened_log_probs`` upcasts the logits before dividing by ``T``, and ``temperature_rescaled``
applies Hinton's ``T**2``. Pinned against independent spellings of each loss:

- the OPD losses (reverse, forward, unnormalized KL) equal the upcast-then-divide formulas bit for
  bit, values and student gradients, at every temperature, for bf16 and fp32 logits;
- the teacher arm's ``kl_divergence``, ``soft_cross_entropy`` and ``jensen_shannon`` (at its default,
  symmetric β) agree with the ``kl_div`` / ``softmax`` spellings to fp32 rounding;
- a ``-inf`` logit (padded vocabulary, a top-k-truncated teacher) is a zero-probability entry that adds
  exactly 0, not NaN, to the value and the gradient;
- the student's backward keeps one fp32 ``[..., V]`` plane (``log_softmax``'s output): flooring the
  ``-inf`` logits keeps a boolean mask, not a second fp32 copy of the logits.

    python tests/cpu/trainers/test_distillation_softened_log_probs.py
"""

import pytest
import torch
from torch.nn.functional import kl_div, log_softmax, softmax

from src.args.mixins import DEFAULT_JSD_BETA
from src.trainers.distillation.losses import (
    call_divergence,
    forward_kl_loss,
    get_divergence,
    reverse_kl_loss,
    softened_log_probs,
    unnormalized_kl_loss,
)

BATCH, SEQ, VOCAB = 2, 5, 257
TEMPERATURES = (1.0, 2.0, 0.7, 1.3)
# A temperature whose divide rounds in bf16, so a bf16-side divide is visible.
INEXACT_TEMPERATURE = 0.7
# fp32 rounding between two spellings of one loss (q log q - q log p vs q (log q - log p), softmax vs
# exp(log_softmax)).
FP32_RTOL = 1e-5
FP32_ATOL = 1e-6
TEACHER_SOFTENED_LOSSES = ("kl_divergence", "soft_cross_entropy", "jensen_shannon")
# Trailing vocabulary columns padded with -inf logits, and the teacher support a top-k truncation keeps.
PADDED_COLUMNS = 7
TEACHER_TOP_K = 5


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
    loss_fn = get_divergence(name, jsd_beta=DEFAULT_JSD_BETA)
    hard_labels = torch.zeros(BATCH, SEQ, dtype=torch.long)
    return lambda student, teacher, temperature: call_divergence(loss_fn, student, teacher, temperature, hard_labels)


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
        (reverse_kl_loss, _reverse_kl),
        (forward_kl_loss, _forward_kl),
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("name", TEACHER_SOFTENED_LOSSES)
def test_teacher_softened_losses_match_their_reference_spelling(name, temperature, dtype):
    student, teacher = _logits(dtype, seed=1)
    value, grad = _value_and_grad(_teacher_loss(name), student, teacher, temperature)
    reference, reference_grad = _value_and_grad(SPELLINGS[name], student, teacher, temperature)
    torch.testing.assert_close(value.sum(-1), reference.sum(-1), rtol=FP32_RTOL, atol=FP32_ATOL)
    torch.testing.assert_close(grad.float(), reference_grad.float(), rtol=FP32_RTOL, atol=FP32_ATOL)


def _padded(logits):
    return torch.cat([logits, torch.full((*logits.shape[:-1], PADDED_COLUMNS), float("-inf"))], dim=-1)


def _top_k_truncated(logits):
    kept = logits.topk(TEACHER_TOP_K, dim=-1).indices
    return torch.full_like(logits, float("-inf")).scatter(-1, kept, logits.gather(-1, kept))


def _finite_value_and_grad(loss_fn, student, teacher, temperature):
    value, grad = _value_and_grad(loss_fn, student, teacher, temperature)
    assert torch.isfinite(value).all(), f"{loss_fn}: non-finite loss"
    assert torch.isfinite(grad).all(), f"{loss_fn}: non-finite student gradient"
    return value, grad


@pytest.mark.parametrize("temperature", (1.0, INEXACT_TEMPERATURE))
@pytest.mark.parametrize("loss_fn", [reverse_kl_loss, forward_kl_loss, unnormalized_kl_loss])
def test_opd_losses_ignore_a_vocabulary_padded_on_both_sides(loss_fn, temperature):
    """Self-distillation scores student and teacher with one model, so both pad the same columns."""
    student, teacher = _logits(torch.float32, seed=3)
    padded, padded_grad = _finite_value_and_grad(loss_fn, _padded(student), _padded(teacher), temperature)
    unpadded, unpadded_grad = _value_and_grad(loss_fn, student, teacher, temperature)
    torch.testing.assert_close(padded.sum(-1), unpadded.sum(-1), rtol=FP32_RTOL, atol=FP32_ATOL)
    torch.testing.assert_close(padded_grad[..., :VOCAB], unpadded_grad, rtol=FP32_RTOL, atol=FP32_ATOL)
    assert padded_grad[..., VOCAB:].eq(0).all()


@pytest.mark.parametrize("temperature", (1.0, INEXACT_TEMPERATURE))
@pytest.mark.parametrize("name", TEACHER_SOFTENED_LOSSES)
def test_teacher_losses_score_a_top_k_truncated_teacher(name, temperature):
    """The truncated entries add exactly what ``kl_div``/``softmax`` give a zero probability: 0."""
    student, teacher = _logits(torch.float32, seed=4)
    teacher = _top_k_truncated(teacher)
    value, grad = _finite_value_and_grad(_teacher_loss(name), student, teacher, temperature)
    reference, reference_grad = _value_and_grad(SPELLINGS[name], student, teacher, temperature)
    torch.testing.assert_close(value.sum(-1), reference.sum(-1), rtol=FP32_RTOL, atol=FP32_ATOL)
    torch.testing.assert_close(grad, reference_grad, rtol=FP32_RTOL, atol=FP32_ATOL)


def test_student_backward_keeps_one_fp32_vocabulary_plane():
    """At ``[tokens, vocab]`` scale every extra fp32 plane the graph keeps is gigabytes of peak memory."""
    student = _padded(_logits(torch.bfloat16)[0]).to(torch.bfloat16).requires_grad_(True)
    saved: list[torch.Tensor] = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor
    ):
        softened_log_probs(student, INEXACT_TEMPERATURE)
    fp32_planes = [tensor for tensor in saved if tensor.dtype == torch.float32 and tensor.shape == student.shape]
    assert len(fp32_planes) == 1, f"the backward keeps {len(fp32_planes)} fp32 [..., V] planes, expected 1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
