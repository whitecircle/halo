"""Off-policy teacher→student distillation losses (used by ``teacher_distillation``).

Eight loss types over teacher/student logits (KL, soft cross-entropy, MSE, SLIM, cosine,
Jensen-Shannon, earth-mover, alpha-beta) plus hard-label/attention masking helpers.

Every loss evaluates in fp32. The logits arrive bf16 (the toolkit's default storage dtype), and
these objectives all subtract nearly equal quantities — ``log p - log q``, ``1 - cos``,
``CDF_s - CDF_t`` — whose bf16 cancellation dominates the result exactly in the converged regime
distillation ends in, to the point of flipping the sign of a provably non-negative divergence. The
probability-space losses fold the upcast into their ``softmax``/``log_softmax`` via the ``dtype``
kwarg, which computes in fp32 (as those kernels already do internally) and writes an fp32 output
without an extra fp32 copy of the ``[B, S, V]`` input — distillation's peak allocation.
"""

import inspect
from collections.abc import Callable

import torch
from torch.nn.functional import cosine_similarity, kl_div, log_softmax, mse_loss, one_hot, softmax

from src.data.spans import LABEL_IGNORE_INDEX
from src.trainers.distillation.losses import masked_token_mean

# Shape of the alpha-beta divergence (Cichocki, Cruces & Amari 2011). Fixed rather than configurable:
# ``call_distillation_loss`` forwards only ``temperature``/``hard_labels``, so nothing dispatched
# through the registry can reach them. The family requires alpha, beta and alpha + beta all non-zero;
# this pair puts the higher exponent on the STUDENT's probabilities, which is where gradients flow.
_AB_DIVERGENCE_ALPHA = 1.0
_AB_DIVERGENCE_BETA = 2.0


def _temperature_rescaled(loss: torch.Tensor, temperature: float) -> torch.Tensor:
    """Hinton's ``T**2`` rescale, applied by every softened divergence in this module.

    Softening divides the logits by ``T``, which shrinks both the divergence and its gradient as
    ``1/T**2`` in the small-logit limit the convention is derived in; multiplying back by ``T**2``
    holds the gradient magnitude — and therefore the distillation term's weight against the hard-label
    term — fixed as ``distill_temperature`` moves. ``losses`` applies the same factor, so the two
    distillation modules share one convention. :func:`slim_loss` is the sole exception, for the reason
    stated at its call of the unscaled core.
    """
    return loss * (temperature**2)


def apply_hard_labels_mask(loss: torch.Tensor, hard_labels: torch.Tensor) -> torch.Tensor:
    """Mask out ignored-label positions on a [batch, seq, dim] loss tensor."""
    return masked_token_mean(loss, hard_labels != LABEL_IGNORE_INDEX)


def hard_labels_coefficient(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, hard_labels: torch.Tensor
) -> torch.Tensor:
    """Calculate hard labels coefficient for distillation loss."""
    student_probs = softmax(student_logits, dim=-1, dtype=torch.float32)
    teacher_probs = softmax(teacher_logits, dim=-1, dtype=torch.float32)

    # Ignored (-100) positions clamped to a valid index so gather doesn't raise; masked out later.
    safe_labels = hard_labels.clamp_min(0).unsqueeze(-1)

    student_coef = 1 - student_probs.gather(-1, safe_labels)
    teacher_coef = teacher_probs.gather(-1, safe_labels)
    coef = student_coef * teacher_coef

    return coef


def kl_divergence_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Forward KL ``D_KL(q_teacher || p_student)``, per (token, vocab).

    Same quantity as ``losses.forward_kl_opd_loss``, deliberately spelled differently: this one
    folds the fp32 upcast into ``softmax``/``log_softmax`` (see the module docstring) instead of
    materializing a second fp32 ``[B, S, V]``, which is the peak allocation of a full-vocab teacher
    forward. Merging the two would trade that peak — and ``kl_div``'s exact zero at ``target == 0``
    — for one fewer function.
    """
    student_logprobs = log_softmax(student_logits / temperature, dim=-1, dtype=torch.float32)
    teacher_probs = softmax(teacher_logits / temperature, dim=-1, dtype=torch.float32)
    return _temperature_rescaled(
        kl_div(student_logprobs, teacher_probs, reduction="none", log_target=False), temperature
    )


def mse_loss_fn(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """MSE loss for distillation."""
    return mse_loss(student_logits.float(), teacher_logits.float(), reduction="none")


def _soft_target_cross_entropy(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Soft target cross entropy at ``temperature``, WITHOUT the :func:`_temperature_rescaled` factor."""
    teacher_probs = softmax(teacher_logits / temperature, dim=-1, dtype=torch.float32)
    student_log_probs = log_softmax(student_logits / temperature, dim=-1, dtype=torch.float32)
    return -(teacher_probs * student_log_probs)


def soft_target_cross_entropy_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Soft target cross entropy loss for distillation (:func:`_temperature_rescaled`)."""
    return _temperature_rescaled(_soft_target_cross_entropy(student_logits, teacher_logits, temperature), temperature)


def slim_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float, hard_labels: torch.Tensor
) -> torch.Tensor:
    """SLIM loss for distillation."""
    student_probs = softmax(student_logits, dim=-1, dtype=torch.float32)
    teacher_probs = softmax(teacher_logits, dim=-1, dtype=torch.float32)
    # The UNSCALED core, alone among the temperature losses: SLIM's coefficient depends on the
    # student, so it multiplies the cross-entropy's teacher-entropy offset (which does NOT shrink as
    # 1/T**2) as well as the divergence. Rescaling that product by T**2 grows the gradient ~T**2
    # instead of holding it fixed — the opposite of what the convention is for.
    kd_loss = _soft_target_cross_entropy(student_logits, teacher_logits, temperature)
    # Ignored (-100) positions clamped to a valid index so one_hot doesn't raise; masked out later.
    hard_labels = one_hot(hard_labels.clamp_min(0), num_classes=student_logits.size(-1)).to(
        device=student_logits.device, dtype=student_probs.dtype
    )
    filtered_student_probs = hard_labels * student_probs
    filtered_teacher_probs = hard_labels * teacher_probs
    diff = filtered_teacher_probs / torch.clamp(filtered_student_probs, min=1e-9)
    coef = 1 - torch.exp(-diff)
    return coef * kd_loss


def cosine_similarity_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Cosine similarity loss for distillation."""
    return 1 - cosine_similarity(student_logits.float(), teacher_logits.float(), dim=-1)


def jensen_shannon_divergence(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Jensen-Shannon divergence: JSD = 0.5*(KL(P||M) + KL(Q||M)), M = 0.5*(P+Q).

    ``F.kl_div(input, target)`` computes ``KL(target||exp(input))``, so ``log M`` is the input and each
    distribution the target (swapping them gives the unbounded ``KL(M||P)+KL(M||Q)``). Log-space avoids
    ``log(0)`` underflow. Rescaled like every other softened divergence here
    (:func:`_temperature_rescaled`).
    """
    student_logprobs = log_softmax(student_logits / temperature, dim=-1, dtype=torch.float32)
    teacher_logprobs = log_softmax(teacher_logits / temperature, dim=-1, dtype=torch.float32)
    student_probs = student_logprobs.exp()
    teacher_probs = teacher_logprobs.exp()
    m = 0.5 * (teacher_probs + student_probs)
    log_m = m.clamp_min(1e-12).log()
    jsd = 0.5 * (kl_div(log_m, student_probs, reduction="none") + kl_div(log_m, teacher_probs, reduction="none"))
    return _temperature_rescaled(jsd, temperature)


def earth_mover_distance(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Earth mover (1-Wasserstein) distance: ``sum_i |CDF_s(i) - CDF_t(i)|`` over the ordered vocab.

    Returns the per-logit integrand ``[batch, seq, vocab]`` (``masked_token_mean`` sums over vocab).
    Not ``torch.cdist`` — that gives a position-coupled ``[batch, seq, seq]`` matrix.
    """
    student_cdf = torch.cumsum(softmax(student_logits, dim=-1, dtype=torch.float32), dim=-1)
    teacher_cdf = torch.cumsum(softmax(teacher_logits, dim=-1, dtype=torch.float32), dim=-1)
    return torch.abs(student_cdf - teacher_cdf)


def alpha_beta_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    alpha: float = _AB_DIVERGENCE_ALPHA,
    beta: float = _AB_DIVERGENCE_BETA,
) -> torch.Tensor:
    """Alpha-Beta divergence (Cichocki, Cruces & Amari 2011): generalised teacher-vs-student divergence.

    Returns the per-logit integrand ``[batch, seq, vocab]`` (``masked_token_mean`` sums over vocab).
    Requires ``alpha``, ``beta``, ``alpha + beta`` all non-zero.
    """
    if alpha == 0 or beta == 0 or (alpha + beta) == 0:
        raise ValueError(
            "alpha_beta_divergence_loss requires alpha != 0, beta != 0 and "
            f"alpha + beta != 0; got alpha={alpha}, beta={beta}."
        )
    student_probs = softmax(student_logits, dim=-1, dtype=torch.float32)
    teacher_probs = softmax(teacher_logits, dim=-1, dtype=torch.float32)

    cross = teacher_probs**alpha * student_probs**beta
    power = (alpha * teacher_probs ** (alpha + beta) + beta * student_probs ** (alpha + beta)) / (alpha + beta)
    return -(1.0 / (alpha * beta)) * (cross - power)


_DISTILLATION_LOSSES: dict[str, Callable] = {
    "kl_divergence": kl_divergence_loss,
    "mse": mse_loss_fn,
    "soft_cross_entropy": soft_target_cross_entropy_loss,
    "cosine_similarity": cosine_similarity_loss,
    "jensen_shannon": jensen_shannon_divergence,
    "earth_mover_distance": earth_mover_distance,
    "alpha_beta_divergence": alpha_beta_divergence_loss,
    "slim": slim_loss,
}


def get_distillation_loss_fn(loss_type: str) -> Callable:
    """Get distillation loss function by name."""
    if loss_type not in _DISTILLATION_LOSSES:
        raise ValueError(f"Unsupported distillation loss type: {loss_type}")
    return _DISTILLATION_LOSSES[loss_type]


def consumes_hard_labels(loss_fn: Callable) -> bool:
    """Whether a distillation loss takes the hard labels itself.

    Read off the signature, the same declaration :func:`call_distillation_loss` dispatches on: such a
    loss (``slim``) derives its own gold-token coefficient, so the shared
    :func:`hard_labels_coefficient` must not be applied on top of it.
    """
    return "hard_labels" in inspect.signature(loss_fn).parameters


def call_distillation_loss(
    loss_fn: Callable,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    hard_labels: torch.Tensor,
) -> torch.Tensor:
    """Invoke a distillation loss, forwarding only the optional args (``temperature``/``hard_labels``)
    its own signature declares."""
    params = inspect.signature(loss_fn).parameters
    optional = {}
    if "temperature" in params:
        optional["temperature"] = temperature
    if consumes_hard_labels(loss_fn):
        optional["hard_labels"] = hard_labels
    return loss_fn(student_logits, teacher_logits, **optional)
