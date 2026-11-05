"""The divergences of the teacher, self-distillation and SDPG trainers (arXiv:2606.04036). Every divergence evaluates
in fp32: bf16 cancellation in ``log p - log q`` can flip its sign."""

import functools
import inspect
import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch.nn.functional import cosine_similarity, cross_entropy, log_softmax, mse_loss

from halo.kernels.logprobs import LABEL_IGNORE_INDEX, at_least_fp32, map_token_chunks
from halo.model_loading.parallel.frozen_models import place_and_freeze

# Floor for a -inf logit (vocab padding, a top-k-truncated teacher). Its probability stays an exact 0
# while its log-prob turns finite, so a term that probability weights is 0 instead of 0 * -inf = NaN.
_MASKED_LOGIT_FLOOR = torch.finfo(torch.float32).min


def logits_forward_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Without ``labels`` and ``skip_logits``, which turn a fused linear-cross-entropy forward into a loss, and with
    ``use_cache`` off, since a cache object on a forward reusing these inputs suppresses the packed-document mask."""
    return {**{k: v for k, v in inputs.items() if k not in ("labels", "skip_logits")}, "use_cache": False}


def _token_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Hard-label cross-entropy of ``[rows, V]`` logits (:func:`at_least_fp32`), ``[rows]``, ignored labels 0."""
    return cross_entropy(at_least_fp32(logits), labels, ignore_index=LABEL_IGNORE_INDEX, reduction="none")


def shifted_token_cross_entropy(shift_logits: torch.Tensor, shift_labels: torch.Tensor) -> torch.Tensor:
    """Over already-shifted ``[B, S]``, ignored positions 0, un-reduced for each caller's own denominator."""
    return map_token_chunks(_token_cross_entropy, shift_logits, shift_labels)


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
    return mse_loss(student_logits.float(), teacher_logits.float(), reduction="none")


def soft_target_cross_entropy_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    teacher_probs = softened_log_probs(teacher_logits, temperature).exp()
    return temperature_rescaled(-(teacher_probs * softened_log_probs(student_logits, temperature)), temperature)


def slim_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float, hard_labels: torch.Tensor
) -> torch.Tensor:
    """Forward KL ``[B, S, V]`` weighted per token by the detached ``1 - exp(-q[y] / p[y])`` at ``T = 1``. This is the
    weighting SLIM's text describes (Raman et al., 2023); its Eq. 4 prints the inverse ratio."""
    student_gold = gold_token_probs(student_logits, hard_labels)
    weight = 1 - torch.exp(-gold_token_probs(teacher_logits, hard_labels) / student_gold.clamp(min=1e-9))
    return weight.unsqueeze(-1) * forward_kl_loss(student_logits, teacher_logits, temperature)


def cosine_similarity_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """``1 - cos`` between the raw logit rows → ``[B, S]``."""
    return 1 - cosine_similarity(student_logits.float(), teacher_logits.float(), dim=-1)


def generalized_jsd_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float, *, jsd_beta: float
) -> torch.Tensor:
    """Generalized JSD (GKD, arXiv:2306.13649 Eq. 1) ``β·KL(q || M) + (1 - β)·KL(p || M)`` with
    ``M = β·q + (1 - β)·p``, ``q`` the detached teacher, per (token, vocab). ``β = 0`` and ``β = 1`` are the exact
    forward and reverse KL, ``β = 0.5`` the symmetric JSD. ``log M`` is a ``logaddexp`` of the log-probs, exact where a
    probability underflows."""
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
    """``logprobs`` on ``support`` plus one bin holding the log-mass off it. The tail sums the off-support tokens
    rather than taking ``log(1 - support mass)``: once a student's support mass rounds to 1 in fp32, that form
    leaves the tail constant and the student gets no gradient toward the teacher's tail mass."""
    tail_log_mass = logprobs.scatter(-1, support, float("-inf")).logsumexp(dim=-1, keepdim=True)
    return torch.cat([logprobs.gather(-1, support), tail_log_mass], dim=-1)


def teacher_topk_log_probs(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, topk: int, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student and teacher softened log-probs ``[..., topk + 1]`` on the teacher's top-``topk`` tokens plus a tail
    bin, each row a distribution over the same bins. Merging tokens never increases a divergence, so a divergence
    of these bins lower-bounds the full-vocab one."""
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
    """``loss_fn`` on :func:`teacher_topk_log_probs`, scored at ``T = 1`` since the softening already ran over the
    full vocabulary, so the ``T**2`` rescale is applied here instead."""
    student_logprobs, teacher_logprobs = teacher_topk_log_probs(student_logits, teacher_logits, topk, temperature)
    return temperature_rescaled(loss_fn(student_logprobs, teacher_logprobs, 1.0), temperature)


# Pinned by a test against each arm's config Literal (SelfDistillationLoss, TeacherDistillationConfig.distill_loss):
# the config layer imports no trainer.
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


@functools.cache
def _declared_parameters(loss_fn: Callable) -> frozenset[str]:
    """The parameter names ``loss_fn`` declares, read once per function: the dispatch runs per token chunk."""
    return frozenset(inspect.signature(loss_fn).parameters)


def require_divergence(
    name: str, *, jsd_beta: float | None = None, topk: int | None = None
) -> Callable[..., torch.Tensor]:
    """The ``name`` divergence with its ``jsd_beta`` bound, scored on the teacher's top-``topk`` + tail bins
    (:func:`teacher_topk_loss`) when ``topk`` is set. A divergence that takes a β raises without one, so it never
    runs on a β nobody configured."""
    if name not in DIVERGENCES:
        raise ValueError(f"Unsupported distillation divergence {name!r}. Available: {sorted(DIVERGENCES)}")
    loss_fn = DIVERGENCES[name]
    if "jsd_beta" in _declared_parameters(loss_fn):
        if jsd_beta is None:
            raise ValueError(f"distillation divergence {name!r} takes a jsd_beta, and none was given.")
        loss_fn = functools.partial(loss_fn, jsd_beta=jsd_beta)
    if topk is None:
        return loss_fn
    return functools.partial(teacher_topk_loss, loss_fn=loss_fn, topk=topk)


def uses_hard_labels(loss_fn: Callable) -> bool:
    """Read off the signature :func:`call_divergence` dispatches on. Such a divergence weighs the gold token on its
    own, so no :func:`hard_labels_coefficient` goes on top."""
    return "hard_labels" in _declared_parameters(loss_fn)


def call_divergence(
    loss_fn: Callable,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    hard_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward only the optional args its signature declares; one that weighs the gold token raises without
    ``hard_labels``."""
    params = _declared_parameters(loss_fn)
    optional = {}
    if "temperature" in params:
        optional["temperature"] = temperature
    if "hard_labels" in params:
        if hard_labels is None:
            raise ValueError(f"{loss_fn.__name__} weighs each token's gold label and needs hard_labels.")
        optional["hard_labels"] = hard_labels
    return loss_fn(student_logits, teacher_logits, **optional)


def _vocab_summed_divergence(
    loss_fn: Callable,
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float,
    hard_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = call_divergence(loss_fn, student, teacher, temperature, hard_labels)
    return loss.sum(-1) if loss.dim() == student.dim() else loss


def token_divergence(
    loss_fn: Callable,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    hard_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """``loss_fn`` per token, summed over the vocabulary. Each token chunk runs under a non-reentrant checkpoint, so
    backward holds the two logits planes and no fp32 plane of the divergence."""

    def per_token(student: torch.Tensor, teacher: torch.Tensor, *labels: torch.Tensor) -> torch.Tensor:
        return _vocab_summed_divergence(loss_fn, student, teacher, temperature, *labels)

    labels = (hard_labels,) if hard_labels is not None and uses_hard_labels(loss_fn) else ()
    return map_token_chunks(per_token, student_logits, teacher_logits, *labels)


def token_divergence_and_cross_entropy(
    loss_fn: Callable,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    hard_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both terms from one chunked pass: each chunk's backward builds the student gradient of both, where a pass per
    term builds a whole-plane gradient each and holds the first while it builds the second."""

    def per_token(student: torch.Tensor, teacher: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        divergence = _vocab_summed_divergence(loss_fn, student, teacher, temperature, labels)
        return torch.stack([divergence, _token_cross_entropy(student, labels)], dim=-1)

    both = map_token_chunks(per_token, student_logits, teacher_logits, hard_labels)
    return both[..., 0], both[..., 1]


def _hard_labels_gate(student: torch.Tensor, teacher: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (1 - gold_token_probs(student, labels)) * gold_token_probs(teacher, labels)


@torch.no_grad()
def hard_labels_coefficient(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, hard_labels: torch.Tensor
) -> torch.Tensor:
    """Detached gold-token gate ``(1 - p[y]) * q[y]`` at ``T = 1``, one token chunk at a time."""
    return map_token_chunks(_hard_labels_gate, student_logits, teacher_logits, hard_labels)


def masked_token_mean(
    per_token_loss: torch.Tensor,
    response_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """The batch mean of each sample's token mean over ``response_mask``, each sample scaled by ``sample_weights``."""
    token_sum = (per_token_loss * response_mask.to(per_token_loss.dtype)).sum(-1)
    # fp32 count and division regardless of loss dtype: bf16's 8 mantissa bits round a >256-token
    # denominator to a multiple of 8, so the divide must not be pulled back to the loss dtype.
    token_count = response_mask.sum(-1, dtype=torch.float32).clamp(min=1e-9)
    per_sample = token_sum.float() / token_count
    if sample_weights is not None:
        per_sample = per_sample * sample_weights.to(per_sample.dtype)
    # No sample at all (an eval rank holding only a split's final-round padding) means 0, not NaN.
    return per_sample.mean() if per_sample.numel() else per_sample.sum()


def pooled_token_mean(per_token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """One denominator for all tokens, as in a causal-LM loss; an all-masked micro-batch gives 0, not a NaN the
    gradient reduce spreads."""
    total = (per_token_loss * mask.to(per_token_loss.dtype)).sum().float()
    return total / mask.sum(dtype=torch.float32).clamp(min=1.0)


def verify_same_token_ids(tokenizer, scorer_tokenizer) -> None:
    """A divergence pairs logit rows by token id."""
    differing = set(tokenizer.get_vocab().items()) ^ set(scorer_tokenizer.get_vocab().items())
    if differing:
        raise ValueError(
            f"The frozen scorer's tokenizer and the policy's disagree on {len(differing)} (token, id) pair(s) "
            f"(e.g. {sorted(differing)[:3]}), so the divergence would compare logits of different tokens. "
            f"Use a scorer of the same tokenizer, with no tokens added to one side only."
        )


def resolve_shared_vocab_width(config, scorer_config, tokenizer, scorer_tokenizer) -> int | None:
    """``None`` for the whole row; rows past ``len(tokenizer)`` are embedding padding no token reaches. Reads only
    configs and tokenizers, so a script can call it before any scorer weight loads."""
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


def reject_topk_covering_vocab(topk: int | None, vocab_width: int | None, model) -> None:
    """Refuse a ``topk`` that leaves no token for the tail bin. ``vocab_width`` is the compared width
    (:func:`resolve_shared_vocab_width`), ``None`` for the whole row of ``model``'s vocabulary."""
    if topk is None:
        return
    width = model.config.get_text_config().vocab_size if vocab_width is None else vocab_width
    if topk >= width:
        raise ValueError(
            f"distill_topk={topk} covers the whole compared vocabulary ({width} tokens), so no tail is left to bin. "
            f"Set distill_topk: null for the full-vocab loss, or a k below {width}."
        )


def place_frozen_scorer(
    scorer: torch.nn.Module, model: torch.nn.Module, tokenizer, scorer_tokenizer
) -> tuple[int | None, torch.device]:
    """``(width, device)`` of the divergence against ``scorer``, which is placed on ``model``'s device, frozen."""
    width = resolve_shared_vocab_width(model.config, scorer.config, tokenizer, scorer_tokenizer)
    return width, place_and_freeze(scorer, model)
