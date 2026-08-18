"""Distillation losses shared across the package + the SDPG beta schedule (arXiv:2606.04036).

A single model plays student ``p = pi(. | x, y_<t)`` and privileged teacher ``q = pi(. | c, x, y_<t)``
(``c`` = a hint stating the ground-truth answer) on the same response tokens. The OPD loss is the
full-vocab reverse KL ``D_KL(p || SG[q])`` (teacher detached). Loss fns return un-reduced per-token,
per-vocab tensors; :func:`masked_token_mean` reduces over response tokens. Shared by
``self_distillation`` and ``sdpg``, along with the two primitives every distillation objective in the
package builds on: :func:`privileged_teacher_pass` (the same-model teacher bracket) and
:func:`shifted_token_cross_entropy` (the hard-label CE each trainer then reduces its own way).
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch.nn.functional import cross_entropy, log_softmax

from src.data.spans import LABEL_IGNORE_INDEX


def logits_forward_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Model inputs for a forward that must return full-vocab logits.

    ``labels`` are dropped so a fused linear-cross-entropy forward keeps its logits instead of
    reducing them to a loss, and ``skip_logits`` with them — the HF trainer injects it into loss-only
    eval steps under Liger, and Liger's forward refuses it without labels. ``use_cache`` is forced
    off: the Trainer writes it on the student only, and a cache object on the teacher / reference
    forward that reuses these inputs suppresses the packed-document mask.
    """
    return {**{k: v for k, v in inputs.items() if k not in ("labels", "skip_logits")}, "use_cache": False}


@contextmanager
def privileged_teacher_pass(model: torch.nn.Module) -> Iterator[None]:
    """Run the SAME model as the frozen privileged teacher: ``no_grad`` AND eval mode.

    eval() as well as no_grad: train-mode dropout — including a config-float attention dropout —
    would perturb the teacher target the student is distilled toward. The training flag is restored
    on the way out whether or not the forward raised.
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
    """Per-token hard-label cross-entropy over ALREADY-shifted logits/labels → ``[B, S]``.

    fp32: a bf16 log-sum-exp over a 100k+ vocab rounds every log-prob, then sums the error. Left
    un-reduced (ignored positions are 0) because the objectives disagree on the denominator — a
    per-sample token mean for the OPD trainers, one global token mean for teacher distillation.
    """
    return cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=LABEL_IGNORE_INDEX,
        reduction="none",
    ).view(shift_labels.shape)


def reverse_kl_opd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Student-to-teacher reverse KL ``D_KL(p || SG[q])`` (SDPG OPD objective).

    Teacher detached → gradient flows only through student ``p``. Per (token, vocab); scaled by
    ``temperature ** 2`` to keep gradient magnitudes comparable across temperatures.

    Logits are upcast to fp32: under autocast they arrive bf16, and a bf16 log-sum-exp over a
    100k+ vocab plus a bf16 log-prob difference biases the distillation gradient.
    """
    student_logprobs = log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logprobs = log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    student_probs = student_logprobs.exp()
    return student_probs * (student_logprobs - teacher_logprobs) * (temperature**2)


def forward_kl_opd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Teacher-to-student forward KL ``D_KL(SG[q] || p)`` (mode-covering variant). Per (token, vocab).

    fp32 upcast for the same reason as :func:`reverse_kl_opd_loss`."""
    student_logprobs = log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logprobs = log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    teacher_probs = teacher_logprobs.exp()
    return teacher_probs * (teacher_logprobs - student_logprobs) * (temperature**2)


def unnormalized_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Unnormalized KL (UKL / Schulman k3) ``sum P log(P/Q) + (Q - P)``.

    Reverse-KL flavored (``P`` = student, ``Q`` = detached teacher); the ``(Q - P)`` mass-correction
    makes it non-negative element-wise and unbiased when summed. Returns ``[..., V]`` per-element.

    fp32 upcast for the same reason as :func:`reverse_kl_opd_loss`, and additionally because
    ``teacher_probs - student_probs`` subtracts two nearly equal probabilities — in bf16 that
    cancellation destroys the element-wise non-negativity the mass correction exists to provide.
    """
    student_logprobs = log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logprobs = log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    student_probs = student_logprobs.exp()
    teacher_probs = teacher_logprobs.exp()
    log_ratio = student_logprobs - teacher_logprobs
    return (student_probs * log_ratio + (teacher_probs - student_probs)) * (temperature**2)


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
    """The OPD token gate: response tokens, optionally restricted to strictly-positive-advantage rows.

    Strict ``> 0``: a zero advantage means a tied or unscorable group, where the verifier expressed
    no preference, so those rows must not pull the student toward the teacher.

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
