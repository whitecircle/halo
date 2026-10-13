"""The distillation objectives of all three trainers (teacher, self-distillation, SDPG) and the primitives
they build on: one divergence registry, the token reductions, and the SDPG beta schedule (arXiv:2606.04036).

A divergence takes ``(student_logits, teacher_logits)`` plus whichever of ``temperature`` /
``hard_labels`` its signature declares (:func:`call_divergence` dispatches on it) and returns an
un-reduced per-(token, vocab) or per-token tensor, reduced by :func:`masked_token_mean` (per-sample
means, the OPD arms) or :func:`global_token_mean` (one batch mean, the teacher arm). Each arm's
config ``Literal`` admits its own names from :data:`DIVERGENCES`; the OPD arms default to the reverse
KL ``D_KL(p || SG[q])``, the teacher arm to the forward KL under its own name, ``kl_divergence``.
:func:`get_divergence` binds the generalized JSD's β and wraps the teacher top-k + tail-bin transform.

Every divergence evaluates in fp32: the logits arrive bf16 and these objectives subtract nearly equal
quantities (``log p - log q``, ``1 - cos``), where bf16 cancellation can flip the sign of a
non-negative divergence.
"""

import inspect
import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
from torch.nn.functional import cosine_similarity, cross_entropy, log_softmax, mse_loss

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
    fixed as the temperature varies.
    """
    return loss * (temperature**2)


def softened_log_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """fp32 ``log_softmax(logits / T)``, divided after the upcast so a bf16 logit is rounded once.

    fp32 because a bf16 log-sum-exp over a 100k+ vocab plus a bf16 log-prob difference biases the
    distillation gradient. Folding the upcast into ``log_softmax``'s ``dtype`` would not save the fp32
    copy: torch casts a bf16 input to fp32 first (only fp16 has a fused path). ``-inf`` logits are
    floored (:data:`_MASKED_LOGIT_FLOOR`) through a boolean mask, all the backward keeps of that step;
    a clamp would keep a second fp32 ``[tokens, vocab]`` copy for its own backward.
    """
    scaled = logits.float() / temperature
    return log_softmax(scaled.masked_fill(torch.isneginf(scaled), _MASKED_LOGIT_FLOOR), dim=-1)


@torch.no_grad()
def gold_token_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Detached ``softmax(logits)`` at each position's label → ``[B, S]``, read through a log-sum-exp so
    no probability plane is materialized.

    Ignored (``-100``) labels are clamped to a valid id so the gather does not raise; callers mask
    those positions out.
    """
    logits = logits.float()
    gold = logits.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (gold - logits.logsumexp(-1)).exp()


def reverse_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Student-to-teacher reverse KL ``D_KL(p || SG[q])`` (mode-seeking), per (token, vocab).

    Teacher detached, so gradient flows only through student ``p``; :func:`temperature_rescaled`.
    """
    student_logprobs = softened_log_probs(student_logits, temperature)
    teacher_logprobs = softened_log_probs(teacher_logits.detach(), temperature)
    return temperature_rescaled(student_logprobs.exp() * (student_logprobs - teacher_logprobs), temperature)


def forward_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Teacher-to-student forward KL ``D_KL(SG[q] || p)`` (mode-covering), per (token, vocab).

    :func:`temperature_rescaled`.
    """
    student_logprobs = softened_log_probs(student_logits, temperature)
    teacher_logprobs = softened_log_probs(teacher_logits.detach(), temperature)
    return temperature_rescaled(teacher_logprobs.exp() * (teacher_logprobs - student_logprobs), temperature)


def unnormalized_kl_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
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
    """Confidence-weighted forward KL: the KL over the whole vocabulary, weighted per token by
    ``1 - exp(-q[y] / p[y])`` → ``[B, S, V]``.

    The weight reads the gold token at ``T = 1`` and grows toward 1 where the teacher is more confident
    in it than the student. It is detached, so it scales the KL's gradient without one of its own, and
    the KL keeps the :func:`temperature_rescaled` convention. This is the weighting SLIM's text
    describes (Raman et al., NeurIPS 2023 workshop), not its loss: the paper's Eq. 4 prints the inverse
    ratio and weights a soft cross-entropy over a top-5% teacher beside a unit-weight CE term.
    """
    student_gold = gold_token_probs(student_logits, hard_labels)
    weight = 1 - torch.exp(-gold_token_probs(teacher_logits, hard_labels) / student_gold.clamp(min=1e-9))
    return weight.unsqueeze(-1) * forward_kl_loss(student_logits, teacher_logits, temperature)


def cosine_similarity_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """``1 - cos`` between the raw logit rows → ``[B, S]``."""
    return 1 - cosine_similarity(student_logits.float(), teacher_logits.float(), dim=-1)


def generalized_jsd_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float, *, jsd_beta: float
) -> torch.Tensor:
    """Generalized Jensen-Shannon divergence (GKD, arXiv:2306.13649 Eq. 1), per (token, vocab).

    ``JSD_β = β·KL(q || M) + (1 - β)·KL(p || M)`` with ``M = β·q + (1 - β)·p``, ``q`` the detached
    teacher and ``p`` the student. The mixture vanishes into either side at the endpoints, so
    ``β = 0`` and ``β = 1`` resolve to the exact :func:`forward_kl_loss` and :func:`reverse_kl_loss`;
    ``β = 0.5`` is the symmetric JSD. ``log M`` is a ``logaddexp`` of the two log-probs, exact where
    either probability underflows. :func:`temperature_rescaled`.
    """
    if not 0.0 <= jsd_beta <= 1.0:
        raise ValueError(f"jsd_beta must be in [0, 1], got {jsd_beta}")
    if jsd_beta == 0.0:
        return forward_kl_loss(student_logits, teacher_logits, temperature)
    if jsd_beta == 1.0:
        return reverse_kl_loss(student_logits, teacher_logits, temperature)
    student_logprobs = softened_log_probs(student_logits, temperature)
    teacher_logprobs = softened_log_probs(teacher_logits.detach(), temperature)
    mixture_logprobs = torch.logaddexp(teacher_logprobs + math.log(jsd_beta), student_logprobs + math.log1p(-jsd_beta))
    teacher_term = teacher_logprobs.exp() * (teacher_logprobs - mixture_logprobs)
    student_term = student_logprobs.exp() * (student_logprobs - mixture_logprobs)
    return temperature_rescaled(jsd_beta * teacher_term + (1.0 - jsd_beta) * student_term, temperature)


def _with_tail_bin(logprobs: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """``logprobs`` on ``support`` plus one bin holding the log-mass off it.

    The tail is summed over the off-support tokens, not taken as ``log(1 - support mass)``: once a
    student's support mass rounds to 1 in fp32, that spelling leaves its tail constant, and the
    student gets no gradient toward the teacher's tail mass.
    """
    tail_log_mass = logprobs.scatter(-1, support, float("-inf")).logsumexp(dim=-1, keepdim=True)
    return torch.cat([logprobs.gather(-1, support), tail_log_mass], dim=-1)


def teacher_topk_log_probs(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, topk: int, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student and teacher softened log-probs on the teacher's top-``topk`` tokens plus a tail bin.

    Both returned ``[..., topk + 1]`` rows are proper distributions over the same bins, so any
    probability-space divergence applies to them. Lumping the off-support tokens into one bin never
    increases a divergence, so the result lower-bounds the full-vocab one.
    """
    student_logprobs = softened_log_probs(student_logits, temperature)
    teacher_logprobs = softened_log_probs(teacher_logits.detach(), temperature)
    support = teacher_logprobs.topk(topk, dim=-1).indices
    return _with_tail_bin(student_logprobs, support), _with_tail_bin(teacher_logprobs, support)


def teacher_topk_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    *,
    loss_fn: Callable[..., torch.Tensor],
    topk: int,
) -> torch.Tensor:
    """``loss_fn`` on the teacher's top-``topk`` + tail bins (:func:`teacher_topk_log_probs`).

    The binned log-probs are valid logits of the binned distributions, so ``loss_fn`` scores them
    at ``T = 1`` (the softening already ran over the full vocab), and the ``T**2`` rescale is
    applied here instead.
    """
    student_logprobs, teacher_logprobs = teacher_topk_log_probs(student_logits, teacher_logits, topk, temperature)
    return temperature_rescaled(loss_fn(student_logprobs, teacher_logprobs, 1.0), temperature)


# Pinned against each arm's config Literal (SelfDistillationLoss, DistillationConfig.distill_loss) by
# tests/cpu/config/test_post_override_validation.py: the args layer imports no trainer.
DIVERGENCES: dict[str, Callable[..., torch.Tensor]] = {
    "reverse_kl": reverse_kl_loss,
    "forward_kl": forward_kl_loss,
    "unnormalized_kl": unnormalized_kl_loss,
    "kl_divergence": forward_kl_loss,
    "soft_cross_entropy": soft_target_cross_entropy_loss,
    "jensen_shannon": generalized_jsd_loss,
    "slim": slim_loss,
    "mse": mse_loss_fn,
    "cosine_similarity": cosine_similarity_loss,
}


def bind_jsd_beta(loss_fn: Callable[..., torch.Tensor], jsd_beta: float | None) -> Callable[..., torch.Tensor]:
    """Bind ``jsd_beta`` into ``loss_fn`` if it takes one; return any other loss as is.

    A loss that needs β but gets none raises, so it never quietly runs on a β nobody configured.
    """
    if "jsd_beta" not in inspect.signature(loss_fn).parameters:
        return loss_fn
    if jsd_beta is None:
        raise ValueError(f"{loss_fn.__name__} needs a jsd_beta, and none was given")
    return partial(loss_fn, jsd_beta=jsd_beta)


def get_divergence(
    name: str, *, jsd_beta: float | None = None, topk: int | None = None
) -> Callable[..., torch.Tensor]:
    """Resolve a divergence from :data:`DIVERGENCES` by name, with its β bound (:func:`bind_jsd_beta`)
    and, given ``topk``, scored on the teacher's top-k + tail bins (:func:`teacher_topk_loss`)."""
    if name not in DIVERGENCES:
        raise ValueError(f"Unsupported distillation divergence {name!r}. Available: {sorted(DIVERGENCES)}")
    loss_fn = bind_jsd_beta(DIVERGENCES[name], jsd_beta)
    if topk is None:
        return loss_fn
    return partial(teacher_topk_loss, loss_fn=loss_fn, topk=topk)


def consumes_hard_labels(loss_fn: Callable) -> bool:
    """Whether a divergence takes the hard labels itself.

    Read off the signature, the same declaration :func:`call_divergence` dispatches on: such a loss
    (``slim``) derives its own gold-token weight, so :func:`hard_labels_coefficient` must not be
    applied on top of it.
    """
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
    if consumes_hard_labels(loss_fn):
        optional["hard_labels"] = hard_labels
    return loss_fn(student_logits, teacher_logits, **optional)


def hard_labels_coefficient(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, hard_labels: torch.Tensor
) -> torch.Tensor:
    """Gold-token gate ``(1 - p[y]) * q[y]`` at ``T = 1`` → ``[B, S]``.

    Detached: it weights the divergence it multiplies and adds no gradient of its own.
    """
    return (1 - gold_token_probs(student_logits, hard_labels)) * gold_token_probs(teacher_logits, hard_labels)


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
    # No sample at all (an eval rank holding only a split's final-round padding) means 0, not NaN.
    return per_sample.mean() if per_sample.numel() else per_sample.sum()


def global_token_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reduce a per-(token, vocab) or per-token loss to its mean over every masked token of the batch.

    One denominator for the whole batch, unlike :func:`masked_token_mean`: every supervised token
    weighs the same, as in a causal-LM loss. The fp32 count is clamped, so an all-masked micro-batch
    gives 0 rather than a NaN the gradient reduce would carry to every rank.
    """
    per_token = loss.sum(-1) if loss.dim() == mask.dim() + 1 else loss
    total = (per_token * mask.to(per_token.dtype)).sum().float()
    return total / mask.sum(dtype=torch.float32).clamp(min=1.0)


def require_same_token_ids(tokenizer, scorer_tokenizer) -> None:
    """Raise unless the two tokenizers map every token to the same id, both ways.

    A divergence pairs two logit rows by token id, so a token one side lacks or spells at another id
    compares logits of different tokens.
    """
    differing = set(tokenizer.get_vocab().items()) ^ set(scorer_tokenizer.get_vocab().items())
    if differing:
        raise ValueError(
            f"The frozen scorer's tokenizer and the policy's disagree on {len(differing)} (token, id) pair(s) "
            f"(e.g. {sorted(differing)[:3]}), so the divergence would compare logits of different tokens. "
            f"Use a scorer of the same tokenizer, with no tokens added to one side only."
        )


def shared_vocab_width(config, scorer_config, tokenizer, scorer_tokenizer) -> int | None:
    """The logit width a divergence between a model and a frozen scorer runs over, from their configs;
    ``None`` is the whole row.

    Checks the two tokenizers' ids (:func:`require_same_token_ids`; skipped when they are one object).
    Rows past ``len(tokenizer)`` are embedding padding no token reaches, so models that differ only
    there are compared over the tokenizer's ids. Reads only configs and tokenizer files, so every rank
    reaches the same verdict, and a script can call it before any scorer weight loads.
    """
    if scorer_tokenizer is not tokenizer:
        require_same_token_ids(tokenizer, scorer_tokenizer)
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
