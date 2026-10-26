"""The divergences of the teacher, self-distillation and SDPG trainers (arXiv:2606.04036): one registry
(:data:`DIVERGENCES`, dispatched by :func:`call_divergence` on the signature), the token reductions and the
frozen scorer's vocabulary checks and placement. Every divergence evaluates in fp32: bf16 cancellation in
``log p - log q`` can flip a divergence's sign.
"""

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch.nn.functional import cosine_similarity, cross_entropy, kl_div, log_softmax, mse_loss

from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.parallel.frozen_models import place_and_freeze

# Floor for a -inf logit (vocab padding, a top-k-truncated teacher). Its probability stays an exact 0
# while its log-prob turns finite, so a term that probability weights is 0 instead of 0 * -inf = NaN.
_MASKED_LOGIT_FLOOR = torch.finfo(torch.float32).min


def logits_forward_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Model inputs for a forward that must return full-vocab logits: without ``labels`` and ``skip_logits``,
    which turn a fused linear-cross-entropy forward into a loss, and with ``use_cache`` off, since a cache
    object on a forward reusing these inputs suppresses the packed-document mask."""
    return {**{k: v for k, v in inputs.items() if k not in ("labels", "skip_logits")}, "use_cache": False}


def shifted_token_cross_entropy(shift_logits: torch.Tensor, shift_labels: torch.Tensor) -> torch.Tensor:
    """Per-token fp32 hard-label cross-entropy over already-shifted logits and labels ``[B, S]``, ignored
    positions 0, left un-reduced for each caller's own denominator."""
    return cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=LABEL_IGNORE_INDEX,
        reduction="none",
    ).view(shift_labels.shape)


def temperature_rescaled(loss: torch.Tensor, temperature: float) -> torch.Tensor:
    """Hinton's ``T**2`` rescale of every softened divergence: softening shrinks the gradient as ``1/T**2``,
    so the rescale keeps the term's weight against the hard-label term fixed across temperatures."""
    return loss * (temperature**2)


def softened_log_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """fp32 ``log_softmax(logits / T)``, divided after the upcast so a bf16 logit is rounded once. ``-inf``
    logits are floored through a boolean mask, which is all the backward keeps of that step; a clamp would
    keep a second fp32 ``[tokens, vocab]`` copy."""
    scaled = logits.float() / temperature
    return log_softmax(scaled.masked_fill(torch.isneginf(scaled), _MASKED_LOGIT_FLOOR), dim=-1)


@torch.no_grad()
def gold_token_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Detached ``softmax(logits)`` at each position's label ``[B, S]``, through a log-sum-exp so no
    probability plane is built; ignored labels gather a valid id, for the caller to mask out."""
    logits = logits.float()
    gold = logits.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (gold - logits.logsumexp(-1)).exp()


def reverse_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Student-to-teacher reverse KL ``D_KL(p || SG[q])`` (the SDPG OPD objective), per (token, vocab)."""
    student_logprobs = softened_log_probs(student_logits, temperature)
    teacher_logprobs = softened_log_probs(teacher_logits.detach(), temperature)
    return temperature_rescaled(student_logprobs.exp() * (student_logprobs - teacher_logprobs), temperature)


def forward_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Teacher-to-student forward KL ``D_KL(SG[q] || p)`` (mode-covering), per (token, vocab)."""
    student_logprobs = softened_log_probs(student_logits, temperature)
    teacher_logprobs = softened_log_probs(teacher_logits.detach(), temperature)
    return temperature_rescaled(teacher_logprobs.exp() * (teacher_logprobs - student_logprobs), temperature)


def unnormalized_kl_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Unnormalized KL (Schulman k3) ``P log(P/Q) + (Q - P)``, ``P`` the student and ``Q`` the detached
    teacher, per (token, vocab): the mass correction makes it non-negative element-wise."""
    student_logprobs = softened_log_probs(student_logits, temperature)
    teacher_logprobs = softened_log_probs(teacher_logits.detach(), temperature)
    student_probs = student_logprobs.exp()
    log_ratio = student_logprobs - teacher_logprobs
    return temperature_rescaled(student_probs * log_ratio + (teacher_logprobs.exp() - student_probs), temperature)


def mse_loss_fn(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """MSE between the raw logits, per (token, vocab)."""
    return mse_loss(student_logits.float(), teacher_logits.float(), reduction="none")


def soft_target_cross_entropy_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Soft target cross entropy, per (token, vocab) (:func:`temperature_rescaled`)."""
    teacher_probs = softened_log_probs(teacher_logits, temperature).exp()
    return temperature_rescaled(-(teacher_probs * softened_log_probs(student_logits, temperature)), temperature)


def slim_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float, hard_labels: torch.Tensor
) -> torch.Tensor:
    """Forward KL ``[B, S, V]`` weighted per token by the detached ``1 - exp(-q[y] / p[y])`` at ``T = 1``,
    which grows toward 1 where the teacher is surer of the gold token than the student. This is the weighting
    SLIM's text describes (Raman et al., 2023), not its Eq. 4, which prints the inverse ratio."""
    student_gold = gold_token_probs(student_logits, hard_labels)
    weight = 1 - torch.exp(-gold_token_probs(teacher_logits, hard_labels) / student_gold.clamp(min=1e-9))
    return weight.unsqueeze(-1) * forward_kl_loss(student_logits, teacher_logits, temperature)


def cosine_similarity_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """``1 - cos`` between the raw logit rows → ``[B, S]``."""
    return 1 - cosine_similarity(student_logits.float(), teacher_logits.float(), dim=-1)


def jensen_shannon_divergence(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Jensen-Shannon divergence ``0.5 * (KL(P||M) + KL(Q||M))``, ``M = 0.5 * (P + Q)``. ``kl_div(input,
    target)`` is ``KL(target||exp(input))``, so ``log M`` is the input; swapped, it would be the unbounded
    ``KL(M||P) + KL(M||Q)``."""
    student_probs = softened_log_probs(student_logits, temperature).exp()
    teacher_probs = softened_log_probs(teacher_logits, temperature).exp()
    m = 0.5 * (teacher_probs + student_probs)
    log_m = m.clamp_min(1e-12).log()
    jsd = 0.5 * (kl_div(log_m, student_probs, reduction="none") + kl_div(log_m, teacher_probs, reduction="none"))
    return temperature_rescaled(jsd, temperature)


# Pinned by a test against each arm's config Literal (SelfDistillationLoss, DistillationConfig.distill_loss):
# the config layer imports no trainer.
DIVERGENCES: dict[str, Callable[..., torch.Tensor]] = {
    "reverse_kl": reverse_kl_loss,
    "forward_kl": forward_kl_loss,
    "unnormalized_kl": unnormalized_kl_loss,
    "kl_divergence": forward_kl_loss,
    "soft_cross_entropy": soft_target_cross_entropy_loss,
    "jensen_shannon": jensen_shannon_divergence,
    "slim": slim_loss,
    "mse": mse_loss_fn,
    "cosine_similarity": cosine_similarity_loss,
}


def require_divergence(name: str) -> Callable[..., torch.Tensor]:
    """The :data:`DIVERGENCES` entry ``name``; an unknown name raises."""
    if name not in DIVERGENCES:
        raise ValueError(f"Unsupported distillation divergence {name!r}. Available: {sorted(DIVERGENCES)}")
    return DIVERGENCES[name]


def uses_hard_labels(loss_fn: Callable) -> bool:
    """Whether a divergence takes the hard labels itself (``slim``), read off the signature
    :func:`call_divergence` dispatches on: it weighs the gold token on its own, so no
    :func:`hard_labels_coefficient` goes on top."""
    return "hard_labels" in inspect.signature(loss_fn).parameters


def call_divergence(
    loss_fn: Callable,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    hard_labels: torch.Tensor,
) -> torch.Tensor:
    """Invoke a divergence, forwarding only the optional args (``temperature``/``hard_labels``) its
    own signature declares."""
    params = inspect.signature(loss_fn).parameters
    optional = {}
    if "temperature" in params:
        optional["temperature"] = temperature
    if uses_hard_labels(loss_fn):
        optional["hard_labels"] = hard_labels
    return loss_fn(student_logits, teacher_logits, **optional)


def hard_labels_coefficient(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, hard_labels: torch.Tensor
) -> torch.Tensor:
    """Detached gold-token gate ``(1 - p[y]) * q[y]`` at ``T = 1``, ``[B, S]``."""
    return (1 - gold_token_probs(student_logits, hard_labels)) * gold_token_probs(teacher_logits, hard_labels)


def masked_token_mean(
    per_token_vocab_loss: torch.Tensor,
    response_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """The batch mean of each sample's token mean over ``response_mask`` ``[B, S]`` of a ``[B, S, V]``
    (vocab-summed) or ``[B, S]`` loss, each sample scaled by the optional ``sample_weights`` ``[B]``."""
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
    # No sample at all (an eval rank holding only a split's final-round padding) means 0, not NaN.
    return per_sample.mean() if per_sample.numel() else per_sample.sum()


def global_token_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """A per-(token, vocab) or per-token loss averaged over every masked token of the batch, one denominator
    for all as in a causal-LM loss; an all-masked micro-batch gives 0, not a NaN the gradient reduce spreads."""
    per_token = loss.sum(-1) if loss.dim() == mask.dim() + 1 else loss
    total = (per_token * mask.to(per_token.dtype)).sum().float()
    return total / mask.sum(dtype=torch.float32).clamp(min=1.0)


def verify_same_token_ids(tokenizer, scorer_tokenizer) -> None:
    """Raise unless the two tokenizers map every token to the same id, both ways: a divergence pairs logit
    rows by token id."""
    differing = set(tokenizer.get_vocab().items()) ^ set(scorer_tokenizer.get_vocab().items())
    if differing:
        raise ValueError(
            f"The frozen scorer's tokenizer and the policy's disagree on {len(differing)} (token, id) pair(s) "
            f"(e.g. {sorted(differing)[:3]}), so the divergence would compare logits of different tokens. "
            f"Use a scorer of the same tokenizer, with no tokens added to one side only."
        )


def resolve_shared_vocab_width(config, scorer_config, tokenizer, scorer_tokenizer) -> int | None:
    """The logit width a divergence between a model and a frozen scorer runs over, ``None`` for the whole row,
    after :func:`verify_same_token_ids` (skipped for one tokenizer object). Rows past ``len(tokenizer)`` are
    embedding padding no token reaches. Reads only configs and tokenizers, so a script can call it before any
    scorer weight loads."""
    if scorer_tokenizer is not tokenizer:
        verify_same_token_ids(tokenizer, scorer_tokenizer)
    model_rows = config.get_text_config().vocab_size
    scorer_rows = scorer_config.get_text_config().vocab_size
    if model_rows == scorer_rows:
        return None
    if min(model_rows, scorer_rows) < len(tokenizer):
        raise ValueError(
            f"Policy vocab_size={model_rows} vs frozen scorer vocab_size={scorer_rows}: one has fewer logit "
            f"rows than the tokenizer's {len(tokenizer)} ids, so some token has no logit to compare."
        )
    return len(tokenizer)


def place_frozen_scorer(
    scorer: torch.nn.Module, model: torch.nn.Module, tokenizer, scorer_tokenizer
) -> tuple[int | None, torch.device]:
    """``(width, device)``: the logit width a divergence against ``scorer`` runs over
    (:func:`resolve_shared_vocab_width`), and the device of ``model`` the scorer is then placed on, frozen."""
    width = resolve_shared_vocab_width(model.config, scorer.config, tokenizer, scorer_tokenizer)
    return width, place_and_freeze(scorer, model)
