"""The offline-GRPO objective, shared by the non-PP loss, its pipeline counterpart and CP scoring.

Pure tensor math over the completion log-probs of one batch or pipeline microbatch, as complete rows
or as one rank's CP token shard of them: the negative-advantage ``min_log_prob`` floor applied to the
policy and the reference, the policy term, the capped k3 KL against the reference, the per-token
quantities buffered as diagnostics, and the loss type's reduction. The reduction comes in two halves
because the pipeline divides once per step: :func:`offline_loss_numerator` is row-local, so
microbatch numerators sum to the batch's, and :func:`offline_loss_normalizer` reads batch metadata
alone. Under CP the numerator sums its partials across the shards first. :func:`offline_loss` is
their quotient.
"""

from __future__ import annotations

from typing import get_args

import torch

from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.context_parallel.autograd import cp_sum_rows
from src.distributed.context_parallel.config import CPConfig
from src.trainers.grpo.objective.logratio import clamp_ref_logps

# Derived from the config's own Literal annotations, so the dispatches here and the parse-time gate
# can never drift apart.
LOSS_TYPES: tuple[str, ...] = get_args(OfflineGRPOConfig.__annotations__["loss_type"])
PG_FORMULATIONS: tuple[str, ...] = get_args(OfflineGRPOConfig.__annotations__["policy_gradient_formulation"])


def clamp_negative_advantage_logps(
    token_logps: torch.Tensor, advantages: torch.Tensor, min_log_prob: float | None
) -> torch.Tensor:
    """``token_logps`` ([B, T]) floored at ``min_log_prob`` on the rows whose advantage ([B]) is negative.

    Below the floor the clamp passes no gradient, so a negative-advantage row stops pushing a token
    it already rates that unlikely further toward zero probability. Returns ``token_logps`` itself when
    no floor is configured.
    """
    if min_log_prob is None:
        return token_logps
    return torch.where((advantages < 0).unsqueeze(1), token_logps.clamp(min=min_log_prob), token_logps)


def offline_token_objective(
    token_logps: torch.Tensor,
    token_logps_unclamped: torch.Tensor,
    advantages: torch.Tensor,
    *,
    policy_gradient_formulation: str,
    beta: float = 0.0,
    ref_logps: torch.Tensor | None = None,
    ref_logps_unclamped: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The per-token loss (unmasked and unweighted) plus the per-token diagnostics to buffer.

    ``prob_weighted`` L = -(π·A) weights high-prob tokens more; ``reinforce`` L = -(log π·A) is
    uniform. At ``beta != 0`` the capped k3 KL ``exp(Δ) - Δ - 1`` is added on top of the reward term,
    so the diagnostics capture the reward term before the KL lands.

    ``clamp_ref_logps`` is fed the detached policy log-probs: the ceiling is ``policy + the clamp``,
    so a grad-carrying policy tensor would make ``ref_clamped - logp`` constant on every clamped
    token, zeroing the KL gradient there instead of bounding it. The clamped fraction is dropped
    rather than logged, since reading it needs an ``.item()`` host sync per microbatch.
    """
    weights = torch.exp(token_logps) if policy_gradient_formulation == "prob_weighted" else token_logps
    per_token_loss = -(weights * advantages.unsqueeze(1))
    sample_values = {
        "logps": token_logps,
        "logps_unclamped": token_logps_unclamped,
        "rewards": -per_token_loss,
    }
    if beta != 0.0:
        ref_logps, _ = clamp_ref_logps(ref_logps, token_logps.detach())
        per_token_kl = torch.exp(ref_logps - token_logps) - (ref_logps - token_logps) - 1
        per_token_loss = per_token_loss + beta * per_token_kl
        sample_values |= {"kl": per_token_kl, "ref_logps": ref_logps, "ref_logps_unclamped": ref_logps_unclamped}
    return per_token_loss, sample_values


def offline_loss_numerator(
    per_token_loss: torch.Tensor,
    supervised_mask: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    loss_type: str,
    cp_config: CPConfig | None = None,
) -> torch.Tensor:
    """The per-token loss summed over the supervised tokens, each row weighted ``1/group_size`` so every
    source group counts equally; ``grpo`` first averages each row over its own tokens. Row-local, so
    pipeline microbatch numerators sum to the batch's."""
    if per_token_loss.shape != supervised_mask.shape or per_token_loss.shape[0] != group_sizes.numel():
        raise ValueError("Offline loss needs one supervision mask and group size per token row")
    group_weights = 1.0 / group_sizes.float()
    weighted = per_token_loss * group_weights.unsqueeze(1) * supervised_mask
    if loss_type == "grpo":
        rows = cp_sum_rows(weighted.sum(dim=1, dtype=torch.float32), cp_config)
        counts = cp_sum_rows(supervised_mask.sum(dim=1, dtype=torch.float32), cp_config)
        return (rows / counts.clamp(min=1)).sum()
    return cp_sum_rows(weighted.sum(), cp_config)


def offline_loss_normalizer(
    token_counts: torch.Tensor | None,
    group_sizes: torch.Tensor,
    *,
    loss_type: str,
    max_completion_length: int | None,
) -> torch.Tensor:
    """The loss type's whole-batch denominator over the ``1/group_size`` row weights: their sum
    (``grpo``), the weighted completion-token count (``bnpo``; ``token_counts`` holds complete per-row
    counts), or their sum times ``max_completion_length`` (``dr_grpo``). Group metadata is replicated
    on every CP peer, so no collective runs here."""
    group_weights = 1.0 / group_sizes.float()
    if loss_type == "grpo":
        return group_weights.sum()
    if loss_type == "bnpo":
        return (token_counts * group_weights).sum().clamp(min=1)
    if loss_type == "dr_grpo":
        if max_completion_length is None or max_completion_length <= 0:
            raise ValueError("dr_grpo needs a positive max_completion_length")
        return group_weights.sum() * max_completion_length
    # A fallthrough would substitute another loss type's denominator instead of failing.
    raise ValueError(f"Unknown loss type: {loss_type!r}. Supported types: {list(LOSS_TYPES)}")


def offline_loss(
    per_token_loss: torch.Tensor,
    supervised_mask: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    loss_type: str,
    max_completion_length: int | None,
    cp_config: CPConfig | None = None,
) -> torch.Tensor:
    """Offline objective on complete rows or CP-owned token shards of the same rows."""
    numerator = offline_loss_numerator(
        per_token_loss, supervised_mask, group_sizes, loss_type=loss_type, cp_config=cp_config
    )
    counts = cp_sum_rows(supervised_mask.sum(dim=1, dtype=torch.float32), cp_config) if loss_type == "bnpo" else None
    return numerator / offline_loss_normalizer(
        counts, group_sizes, loss_type=loss_type, max_completion_length=max_completion_length
    )
