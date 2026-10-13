#!/usr/bin/env python
"""The divergence registry, the gold-token gathers, the token reduction and the SDPG beta schedule
(``src/trainers/distillation/losses.py``).

Pure tensor math, so the assertions are closed-form values, symmetry and bound properties, an
independent float64 spelling of the generalized JSD, the temperature convention and the reduction
semantics. The fp32 evaluation and the ``-inf`` handling
live in ``test_distillation_loss_precision.py`` and ``test_distillation_softened_log_probs.py``.

Run: ``pytest tests/cpu/trainers/test_distillation_losses.py``.
"""

import inspect
import math

import pytest
import torch
from torch.nn.functional import log_softmax, softmax

from src.args.mixins import DEFAULT_JSD_BETA
from src.trainers.distillation.losses import (
    DIVERGENCES,
    beta_warmup_decay,
    call_divergence,
    cosine_similarity_loss,
    forward_kl_loss,
    generalized_jsd_loss,
    get_divergence,
    hard_labels_coefficient,
    masked_token_mean,
    mse_loss_fn,
    reverse_kl_loss,
    slim_loss,
    soft_target_cross_entropy_loss,
    unnormalized_kl_loss,
)

BS, SEQ, VOCAB = 2, 4, 8
torch.manual_seed(42)
LOGITS_A = torch.randn(BS, SEQ, VOCAB)
LOGITS_B = torch.randn(BS, SEQ, VOCAB)
KL_FAMILY = (reverse_kl_loss, forward_kl_loss, unnormalized_kl_loss)

# Temperatures the gradient-magnitude sweep spans (an 8x range: without the T**2 factor a softened
# divergence's gradient falls 64x across it).
_TEMPERATURES = (1.0, 2.0, 4.0, 8.0)
# Small logits — the near-uniform-softmax regime Hinton's T**2 argument is derived in, and the
# converged regime distillation actually ends in.
_SMALL_LOGIT_SIGMA = 0.02
# Largest gradient-norm spread across _TEMPERATURES that still counts as invariant.
_INVARIANCE_SPREAD = 1.01
# fp32 loss against a float64 reference.
_REFERENCE_RTOL = 1e-5
_REFERENCE_ATOL = 1e-6
# β this close to an endpoint makes the JSD's first-order term dominate its O(β**2) remainder.
_NEAR_ENDPOINT = 1e-4


@pytest.mark.parametrize("loss_fn", KL_FAMILY)
def test_the_kl_family_is_zero_at_identity_and_non_negative(loss_fn):
    assert loss_fn(LOGITS_A, LOGITS_A, 1.0).sum(-1).abs().max() < 1e-5
    assert loss_fn(LOGITS_A, LOGITS_B, 1.0).sum(-1).min() > -1e-6


@pytest.mark.parametrize("loss_fn", KL_FAMILY)
def test_the_kl_family_stops_the_teacher_gradient(loss_fn):
    student = LOGITS_A.clone().requires_grad_(True)
    teacher = LOGITS_B.clone().requires_grad_(True)
    loss_fn(student, teacher, 1.0).sum().backward()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_forward_kl_closed_form_and_direction():
    """``KL(teacher || student)``: the reversed argument order gives a different number."""
    student = torch.tensor([[[2.0, 0.0]]])
    teacher = torch.tensor([[[0.0, 0.0]]])
    q, p = softmax(teacher[0, 0], dim=-1), softmax(student[0, 0], dim=-1)
    assert forward_kl_loss(student, teacher, 1.0).sum().item() == pytest.approx((q * (q / p).log()).sum().item())
    assert reverse_kl_loss(student, teacher, 1.0).sum().item() == pytest.approx((p * (p / q).log()).sum().item())
    assert not torch.allclose(
        forward_kl_loss(student, teacher, 1.0).sum(), reverse_kl_loss(student, teacher, 1.0).sum()
    )


def test_unnormalized_kl_is_non_negative_elementwise_and_sums_to_the_reverse_kl():
    ukl = unnormalized_kl_loss(LOGITS_A, LOGITS_B, 1.0)
    assert ukl.min() > -1e-6
    torch.testing.assert_close(ukl.sum(-1), reverse_kl_loss(LOGITS_A, LOGITS_B, 1.0).sum(-1), atol=1e-5, rtol=0)


def test_mse_known_value():
    a = torch.tensor([[[1.0, 0.0]]])
    b = torch.tensor([[[0.0, 3.0]]])
    assert mse_loss_fn(a, b).tolist() == [[[1.0, 9.0]]]


def test_soft_ce_is_the_entropy_at_identity_and_bounded_below_by_it():
    """Gibbs: ``H(q, p) >= H(q)``, with equality iff ``p == q``."""
    q = softmax(LOGITS_B, dim=-1)
    entropy = -(q * q.log()).sum(-1)
    torch.testing.assert_close(soft_target_cross_entropy_loss(LOGITS_B, LOGITS_B, 1.0).sum(-1), entropy)
    assert (soft_target_cross_entropy_loss(LOGITS_A, LOGITS_B, 1.0).sum(-1) - entropy).min() > -1e-5


def test_cosine_distance_values_and_scale_invariance():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    assert cosine_similarity_loss(a, torch.tensor([[0.0, 1.0, 0.0]])).item() == pytest.approx(1.0)
    assert cosine_similarity_loss(a, -a).item() == pytest.approx(2.0)
    assert cosine_similarity_loss(LOGITS_A, LOGITS_A).abs().max() < 1e-5
    torch.testing.assert_close(
        cosine_similarity_loss(LOGITS_A * 5.0, LOGITS_B), cosine_similarity_loss(LOGITS_A, LOGITS_B)
    )


def _symmetric_jsd(student, teacher):
    return generalized_jsd_loss(student, teacher, 1.0, jsd_beta=DEFAULT_JSD_BETA)


def test_the_default_jsd_is_symmetric_zero_at_identity_and_saturates_at_ln2():
    """Saturates at ln 2; the reversed-argument form ``KL(M||P)+KL(M||Q)`` is unbounded and fails here."""
    torch.testing.assert_close(_symmetric_jsd(LOGITS_A, LOGITS_B).sum(-1), _symmetric_jsd(LOGITS_B, LOGITS_A).sum(-1))
    assert _symmetric_jsd(LOGITS_A, LOGITS_A).sum(-1).abs().max() < 1e-5
    sharp = _symmetric_jsd(torch.tensor([[[50.0, -50.0]]]), torch.tensor([[[-50.0, 50.0]]])).sum()
    assert sharp.item() == pytest.approx(math.log(2), abs=1e-4)


def _gkd_reference(student, teacher, beta, temperature):
    """GKD's generalized JSD in TRL's spelling (a logsumexp over the stacked mixture), in float64."""
    student_logprobs = log_softmax(student.double() / temperature, dim=-1)
    teacher_logprobs = log_softmax(teacher.double() / temperature, dim=-1)
    mixture = torch.logsumexp(
        torch.stack([student_logprobs + math.log1p(-beta), teacher_logprobs + math.log(beta)]), dim=0
    )
    teacher_kl = (teacher_logprobs.exp() * (teacher_logprobs - mixture)).sum(-1)
    student_kl = (student_logprobs.exp() * (student_logprobs - mixture)).sum(-1)
    return (beta * teacher_kl + (1 - beta) * student_kl) * temperature**2


def test_the_generalized_jsd_endpoints_are_the_exact_kls():
    assert torch.equal(
        generalized_jsd_loss(LOGITS_A, LOGITS_B, 1.0, jsd_beta=0.0), forward_kl_loss(LOGITS_A, LOGITS_B, 1.0)
    )
    assert torch.equal(
        generalized_jsd_loss(LOGITS_A, LOGITS_B, 1.0, jsd_beta=1.0), reverse_kl_loss(LOGITS_A, LOGITS_B, 1.0)
    )


@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("beta", [0.1, 0.5, 0.9])
def test_the_generalized_jsd_matches_the_gkd_reference(beta, temperature):
    student = LOGITS_A.clone().requires_grad_(True)
    loss = generalized_jsd_loss(student, LOGITS_B, temperature, jsd_beta=beta).sum(-1)
    loss.sum().backward()

    reference_student = LOGITS_A.double().requires_grad_(True)
    reference = _gkd_reference(reference_student, LOGITS_B, beta, temperature)
    reference.sum().backward()

    torch.testing.assert_close(loss.double(), reference.detach(), rtol=_REFERENCE_RTOL, atol=_REFERENCE_ATOL)
    torch.testing.assert_close(
        student.grad.double(), reference_student.grad, rtol=_REFERENCE_RTOL, atol=_REFERENCE_ATOL
    )


def test_the_generalized_jsd_shrinks_toward_its_endpoints_and_jumps_at_them():
    """Near β = 0 the loss is ≈ β·KL(q‖p), near β = 1 ≈ (1 - β)·KL(p‖q); exactly at either it is the full KL."""
    forward = forward_kl_loss(LOGITS_A, LOGITS_B, 1.0).sum(-1)
    reverse = reverse_kl_loss(LOGITS_A, LOGITS_B, 1.0).sum(-1)
    near_zero = generalized_jsd_loss(LOGITS_A, LOGITS_B, 1.0, jsd_beta=_NEAR_ENDPOINT).sum(-1)
    near_one = generalized_jsd_loss(LOGITS_A, LOGITS_B, 1.0, jsd_beta=1.0 - _NEAR_ENDPOINT).sum(-1)
    torch.testing.assert_close(near_zero, _NEAR_ENDPOINT * forward, rtol=1e-2, atol=0.0)
    torch.testing.assert_close(near_one, _NEAR_ENDPOINT * reverse, rtol=1e-2, atol=0.0)


def test_the_generalized_jsd_stops_the_teacher_gradient():
    student = LOGITS_A.clone().requires_grad_(True)
    teacher = LOGITS_B.clone().requires_grad_(True)
    generalized_jsd_loss(student, teacher, 1.0, jsd_beta=0.3).sum().backward()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


@pytest.mark.parametrize("beta", [-0.1, 1.1, math.nan])
def test_the_generalized_jsd_refuses_beta_outside_the_unit_interval(beta):
    with pytest.raises(ValueError, match="jsd_beta must be in"):
        generalized_jsd_loss(LOGITS_A, LOGITS_B, 1.0, jsd_beta=beta)


def _gold(logits, label, temperature=1.0):
    return softmax(logits / temperature, dim=-1)[label]


def _slim_reference(student, teacher, labels, temperature):
    """SLIM spelled out per token: ``w * T**2 * sum_v q_T (log q_T - log p_T)``, ``w = 1 - exp(-q[y]/p[y])``."""
    expected = []
    for position, label in enumerate(labels[0].tolist()):
        s, t = student[0, position], teacher[0, position]
        weight = 1 - math.exp(-_gold(t, label).item() / _gold(s, label).item())
        q, p = softmax(t / temperature, dim=-1), softmax(s / temperature, dim=-1)
        expected.append(weight * temperature**2 * (q * (q.log() - p.log())).sum().item())
    return expected


@pytest.mark.parametrize("temperature", [1.0, 2.5])
def test_slim_closed_form_on_a_tiny_vocabulary(temperature):
    """The whole vocabulary's KL, not the gold column alone, weighted by teacher over student
    confidence on the gold token at ``T = 1`` (an inverted ratio is a different number)."""
    student = torch.tensor([[[1.0, -0.5, 0.3], [0.2, 0.9, -1.1]]])
    teacher = torch.tensor([[[-0.4, 1.2, 0.1], [0.7, -0.3, 0.5]]])
    labels = torch.tensor([[2, 0]])
    got = slim_loss(student, teacher, temperature, labels).sum(-1)[0].tolist()
    assert got == pytest.approx(_slim_reference(student, teacher, labels, temperature), rel=1e-5)


def test_slim_weight_is_detached():
    """The weight is read off the student itself, so a live one would add its own gradient."""
    labels = torch.randint(0, VOCAB, (BS, SEQ))
    student = LOGITS_A.clone().requires_grad_(True)
    slim_loss(student, LOGITS_B, 2.0, labels).sum().backward()
    weight = 1 - torch.exp(
        -softmax(LOGITS_B, -1).gather(-1, labels[..., None]) / softmax(LOGITS_A, -1).gather(-1, labels[..., None])
    )
    reference = LOGITS_A.clone().requires_grad_(True)
    (weight * forward_kl_loss(reference, LOGITS_B, 2.0)).sum().backward()
    torch.testing.assert_close(student.grad, reference.grad)


def test_hard_labels_coefficient_closed_form_and_ignored_labels():
    """``(1 - p[y]) * q[y]`` per token; a ``-100`` label must not raise in the gather."""
    student = torch.tensor([[[2.0, 0.0], [0.5, 0.5]]])
    teacher = torch.tensor([[[0.0, 2.0], [1.0, -1.0]]])
    coefficient = hard_labels_coefficient(student, teacher, torch.tensor([[0, -100]]))
    assert coefficient.shape == (1, 2)
    assert coefficient[0, 0].item() == pytest.approx(
        (1 - _gold(student[0, 0], 0).item()) * _gold(teacher[0, 0], 0).item()
    )
    assert torch.isfinite(coefficient).all()


def test_hard_labels_coefficient_is_a_detached_weight():
    """The gate scales the divergence's gradient; it must not push the student's gold probability itself."""
    student = LOGITS_A.clone().requires_grad_(True)
    assert not hard_labels_coefficient(student, LOGITS_B, torch.randint(0, VOCAB, (BS, SEQ))).requires_grad


def test_masked_token_mean_weights_each_sample_by_its_own_token_mean():
    """Distinct per-token values and a ragged mask, so an ignored mask or ignored weights lands on a
    different number: means ``1.5`` and ``25``, weighted by ``[2, 0.5]`` → ``7.75``."""
    loss = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    assert masked_token_mean(loss, mask, torch.tensor([2.0, 0.5])).item() == pytest.approx(7.75)
    assert masked_token_mean(loss.unsqueeze(-1).expand(2, 3, 4), mask).item() == pytest.approx(4 * 13.25)


def test_masked_token_mean_of_an_empty_mask_is_zero_not_nan():
    assert masked_token_mean(torch.ones(1, 4, 8), torch.zeros(1, 4)).item() == 0.0


def _gradient_norms_across_temperature(loss_fn, seed=0):
    """Student-gradient norm of ``loss_fn.sum()`` at each of :data:`_TEMPERATURES`."""
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.randn(BS, SEQ, VOCAB * 2, generator=generator) * _SMALL_LOGIT_SIGMA
    base_student = teacher + torch.randn(teacher.shape, generator=generator) * _SMALL_LOGIT_SIGMA * 0.5
    hard_labels = torch.randint(0, teacher.size(-1), teacher.shape[:2], generator=generator)

    norms = []
    for temperature in _TEMPERATURES:
        student = base_student.clone().requires_grad_(True)
        call_divergence(loss_fn, student, teacher, temperature, hard_labels).sum().backward()
        norms.append(student.grad.norm().item())
    return norms


SOFTENED = sorted(name for name, fn in DIVERGENCES.items() if "temperature" in inspect.signature(fn).parameters)


@pytest.mark.parametrize("name", SOFTENED)
def test_every_softened_divergence_holds_its_gradient_magnitude_across_temperature(name):
    """The point of the ``T**2`` factor: softening divides the logits by ``T``, shrinking a divergence
    and its gradient as ``1/T**2``; multiplying back holds the distillation term's pull — and its
    weight against the hard-label term — fixed as the temperature moves."""
    norms = _gradient_norms_across_temperature(get_divergence(name, jsd_beta=DEFAULT_JSD_BETA))
    spread = max(norms) / min(norms)
    assert spread < _INVARIANCE_SPREAD, (
        f"{name}: gradient magnitude moves {spread:.1f}x across T={_TEMPERATURES} ({norms}) — the T**2 "
        f"rescale is missing or doubled"
    )


def test_an_unknown_divergence_is_refused_by_name():
    with pytest.raises(ValueError, match="Unsupported distillation divergence 'nope'"):
        get_divergence("nope")


def test_the_teacher_arm_kl_is_the_forward_kl():
    assert get_divergence("kl_divergence") is get_divergence("forward_kl") is forward_kl_loss


def test_jensen_shannon_is_refused_without_a_beta():
    """The reference anchor resolves its loss with no β, so it cannot silently get a default one."""
    with pytest.raises(ValueError, match="needs a jsd_beta"):
        get_divergence("jensen_shannon")


def test_jensen_shannon_runs_at_the_beta_it_was_resolved_with():
    assert torch.equal(
        get_divergence("jensen_shannon", jsd_beta=0.3)(LOGITS_A, LOGITS_B, 1.0),
        generalized_jsd_loss(LOGITS_A, LOGITS_B, 1.0, jsd_beta=0.3),
    )


@pytest.mark.parametrize("name", sorted(name for name in DIVERGENCES if name != "jensen_shannon"))
def test_a_jsd_beta_never_reaches_a_divergence_without_one(name):
    assert get_divergence(name, jsd_beta=0.3) is DIVERGENCES[name]


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0.0), (5, 0.5), (10, 1.0), (50, 1.0), (95, 0.5), (100, 0.0)],
    ids=["warmup-start", "warmup-mid", "warmup-end", "hold", "decay-mid", "decay-end"],
)
def test_beta_warms_up_holds_and_decays(step, expected):
    assert beta_warmup_decay(step, 100, 1.0, 10, 10) == pytest.approx(expected)


def test_beta_is_the_constant_base_without_windows_or_a_total():
    assert beta_warmup_decay(7, 100, 0.3, 0, 0) == pytest.approx(0.3)
    assert beta_warmup_decay(7, 0, 0.3, 0, 10) == pytest.approx(0.3)
    assert beta_warmup_decay(50, 100, 0.0, 10, 10) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
