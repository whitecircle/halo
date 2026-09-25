"""Shared distillation losses and the SDPG beta schedule (arXiv:2606.04036).

A single model acts as student ``p = pi(. | x, y_<t)`` and privileged teacher
``q = pi(. | c, x, y_<t)`` (``c`` = a hint stating the ground-truth answer) on the same response
tokens. The OPD loss is the full-vocab reverse KL ``D_KL(p || SG[q])`` (teacher detached). Loss
functions return un-reduced per-token, per-vocab tensors; :func:`masked_token_mean` reduces over
response tokens.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch.nn.functional import cross_entropy, log_softmax

from src.data.spans import LABEL_IGNORE_INDEX

# Floor for a -inf logit (vocab padding, a top-k-truncated teacher). Its probability stays an exact 0
# while its log-prob turns finite, so a term that probability weights is 0 instead of 0 * -inf = NaN.
_MASKED_LOGIT_FLOOR = torch.finfo(torch.float32).min


def logits_forward_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Model inputs for a forward that must return full-vocab logits.

    ``labels`` and ``skip_logits`` are dropped so a fused linear-cross-entropy forward keeps its
    logits instead of reducing them to a loss (Liger rejects ``skip_logits`` without labels).
    ``use_cache`` is forced off: a cache object on a teacher/reference forward reusing these inputs
    suppresses the packed-document mask.
    """
    return {**{k: v for k, v in inputs.items() if k not in ("labels", "skip_logits")}, "use_cache": False}


@contextmanager
def privileged_teacher_pass(model: torch.nn.Module) -> Iterator[None]:
    """Run the same model as the frozen privileged teacher, under ``no_grad`` and in eval mode.

    eval() as well as no_grad: train-mode dropout would perturb the teacher target. The training
    flag is restored on exit whether or not the forward raised.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()


def shifted_token_cross_entropy(shift_logits: torch.Tensor, shift_labels: torch.Tensor) -> torch.Tensor:
    """Per-token hard-label cross-entropy over already-shifted logits/labels → ``[B, S]``.

    Computed in fp32: a bf16 log-sum-exp over a 100k+ vocab rounds every log-prob. Left un-reduced
    (ignored positions are 0) so each caller applies its own denominator.
    """
    return cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=LABEL_IGNORE_INDEX,
        reduction="none",
    ).view(shift_labels.shape)


def temperature_rescaled(loss: torch.Tensor, temperature: float) -> torch.Tensor:
    """Hinton's ``T**2`` rescale, applied by every softened divergence.

    Softening by ``T`` shrinks the divergence and its gradient as ``1/T**2`` in the small-logit limit,
    so multiplying back by ``T**2`` keeps the distillation term's weight against the hard-label term
    fixed as the temperature varies. ``teacher_losses.slim_loss`` is the exception.
    """
    return loss * (temperature**2)


def softened_log_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """fp32 ``log_softmax(logits / T)``, divided after the upcast so a bf16 logit is rounded once.

    fp32 because a bf16 log-sum-exp over a 100k+ vocab plus a bf16 log-prob difference biases the
    distillation gradient. Folding the upcast into ``log_softmax``'s ``dtype`` would not save the fp32
    copy: torch casts a bf16 input to fp32 first (only fp16 has a fused path). ``-inf`` logits are
    floored in place on that copy (:data:`_MASKED_LOGIT_FLOOR`), a no-op on finite ones.
    """
    return log_softmax((logits.float() / temperature).clamp_min_(_MASKED_LOGIT_FLOOR), dim=-1)


def reverse_kl_opd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Student-to-teacher reverse KL ``D_KL(p || SG[q])`` (SDPG OPD objective), per (token, vocab).

    Teacher detached, so gradient flows only through student ``p``; :func:`temperature_rescaled`.
    """
    student_logprobs = softened_log_probs(student_logits, temperature)
    teacher_logprobs = softened_log_probs(teacher_logits.detach(), temperature)
    return temperature_rescaled(student_logprobs.exp() * (student_logprobs - teacher_logprobs), temperature)


def forward_kl_opd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Teacher-to-student forward KL ``D_KL(SG[q] || p)`` (mode-covering), per (token, vocab).

    Also the teacher-distillation ``kl_divergence`` loss. :func:`temperature_rescaled`.
    """
    student_logprobs = softened_log_probs(student_logits, temperature)
    teacher_logprobs = softened_log_probs(teacher_logits.detach(), temperature)
    return temperature_rescaled(teacher_logprobs.exp() * (teacher_logprobs - student_logprobs), temperature)


def unnormalized_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Unnormalized KL (UKL / Schulman k3) ``sum P log(P/Q) + (Q - P)``.

    Reverse-KL flavored (``P`` = student, ``Q`` = detached teacher); the ``(Q - P)`` mass-correction
    makes it non-negative element-wise and unbiased when summed. Returns ``[..., V]`` per-element.
    The fp32 evaluation also keeps bf16 cancellation in ``Q - P`` from breaking that non-negativity.
    """
    student_logprobs = softened_log_probs(student_logits, temperature)
    teacher_logprobs = softened_log_probs(teacher_logits.detach(), temperature)
    student_probs = student_logprobs.exp()
    log_ratio = student_logprobs - teacher_logprobs
    return temperature_rescaled(student_probs * log_ratio + (teacher_logprobs.exp() - student_probs), temperature)


_SELF_DISTILL_LOSSES = {
    "reverse_kl": reverse_kl_opd_loss,
    "forward_kl": forward_kl_opd_loss,
    "unnormalized_kl": unnormalized_kl_loss,
}


def get_self_distillation_loss_fn(loss_type: str) -> Callable:
    """Resolve a self-distillation loss by name (``reverse_kl``/``forward_kl``/``unnormalized_kl``)."""
    if loss_type not in _SELF_DISTILL_LOSSES:
        raise ValueError(
            f"Unsupported self-distillation loss '{loss_type}'. Available: {sorted(_SELF_DISTILL_LOSSES)}"
        )
    return _SELF_DISTILL_LOSSES[loss_type]


def positive_advantage_gate(
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """OPD token gate: response tokens, optionally restricted to strictly-positive-advantage rows.

    Strict ``> 0``: a zero advantage means a tied or unscorable group, so those rows must not pull
    the student toward the teacher.

    Args:
        completion_mask: ``[B, C]`` mask, 1 on completion tokens.
        advantages: ``[B]`` or ``[B, 1]`` per-sample advantages.
        enabled: when False the gate is the completion mask alone.
    """
    if not enabled:
        return completion_mask
    adv = advantages.unsqueeze(1) if advantages.dim() == 1 else advantages
    return completion_mask * (adv > 0).to(completion_mask.dtype)


def masked_token_mean(
    per_token_vocab_loss: torch.Tensor,
    response_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce a per-(token, vocab) loss to a scalar over response tokens.

    Sums over vocab, masks to response tokens, optionally scales each sample by ``sample_weights``,
    then per-sample token-means and batch-means.

    Args:
        per_token_vocab_loss: ``[B, S, V]`` (vocab-summed internally) or ``[B, S]``.
        response_mask: ``[B, S]`` mask, 1 on response tokens to distill.
        sample_weights: optional ``[B]`` per-sample weights (``None`` ≡ all ones).
    """
    if per_token_vocab_loss.dim() == response_mask.dim() + 1:
        per_token_loss = per_token_vocab_loss.sum(-1)
    else:
        per_token_loss = per_token_vocab_loss
    token_sum = (per_token_loss * response_mask.to(per_token_loss.dtype)).sum(-1)
    # fp32 count and division regardless of loss dtype: bf16's 8 mantissa bits round a >256-token
    # denominator to a multiple of 8, so the divide must not be pulled back to the loss dtype.
    token_count = response_mask.sum(-1, dtype=torch.float32).clamp(min=1e-9)
    per_sample = token_sum.float() / token_count
    if sample_weights is not None:
        per_sample = per_sample * sample_weights.to(per_sample.dtype)
    return per_sample.mean()


def beta_warmup_decay(
    step: int,
    total_steps: int,
    beta_base: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """SDPG distillation-coefficient schedule (warmup → hold → linear decay).

    ``beta(k) = beta_base * min(1, k / T_warm) * min(1, (T - k) / T_decay)``. Constant ``beta_base``
    when ``warmup_steps=decay_steps=0``.
    """
    if beta_base == 0.0:
        return 0.0
    warm = 1.0 if warmup_steps <= 0 else min(1.0, step / warmup_steps)
    # An unset total (<=0) must not zero OPD for the whole run.
    decay_inactive = decay_steps <= 0 or total_steps <= 0
    decay = 1.0 if decay_inactive else min(1.0, max(0.0, (total_steps - step) / decay_steps))
    return beta_base * warm * decay
