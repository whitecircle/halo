"""The offline-GRPO objective, shared by the non-PP loss and its pipeline counterpart.

Pure tensor math over one batch, or one pipeline microbatch, of completion log-probs: the
negative-advantage ``min_log_prob`` floor both paths apply to the policy and the reference, the
policy term, the capped k3 KL against the reference, the per-token quantities both paths buffer as
diagnostics, and the loss type's reduction. The reduction comes in two halves because the pipeline
divides once per step: :func:`loss_numerator` is row-local, so microbatch numerators sum to the
batch's, and :func:`loss_normalizer` reads batch metadata alone. Off PP the loss is their quotient.
"""

from __future__ import annotations

from typing import get_args

import torch

from src.configs.offline_grpo_config import OfflineGRPOConfig
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


def loss_numerator(
    per_token_loss: torch.Tensor, mask: torch.Tensor, group_sizes: torch.Tensor, loss_type: str
) -> torch.Tensor:
    """The per-token loss summed over the completion tokens, each row weighted ``1/group_size`` so every
    source group counts equally; ``grpo`` first averages each row over its own tokens."""
    weighted = per_token_loss * mask * (1.0 / group_sizes.float()).unsqueeze(1)
    if loss_type == "grpo":
        return (weighted.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).sum()
    return weighted.sum()


def loss_normalizer(
    group_sizes: torch.Tensor, token_counts: torch.Tensor, loss_type: str, max_completion_length: int | None
) -> torch.Tensor:
    """The loss type's whole-batch denominator over the ``1/group_size`` row weights: their sum
    (``grpo``), the weighted completion-token count (``bnpo``; ``token_counts`` per row), or their sum
    times ``max_completion_length`` (``dr_grpo``)."""
    group_weights = 1.0 / group_sizes.float()
    if loss_type == "grpo":
        return group_weights.sum()
    if loss_type == "bnpo":
        return (group_weights * token_counts).sum().clamp(min=1.0)
    if loss_type == "dr_grpo":
        return group_weights.sum() * max_completion_length
    # A fallthrough would substitute another loss type's denominator instead of failing.
    raise ValueError(f"Unknown loss type: {loss_type!r}. Supported types: {list(LOSS_TYPES)}")
