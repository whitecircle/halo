"""The offline-GRPO objective, pure tensor math. The reduction splits because the pipeline divides once per step:
:func:`offline_loss_numerator` is row-local, :func:`offline_loss_normalizer` reads batch metadata alone."""

from __future__ import annotations

from typing import get_args

import torch

from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.sequence_chunks import cp_sum_rows
from halo.trainers.grpo.objective.logratio import clamp_ref_logps

# From the config's own Literal annotations, so these dispatches and the parse-time gate cannot drift.
LOSS_TYPES: tuple[str, ...] = get_args(OfflineGRPOConfig.__annotations__["loss_type"])
PG_FORMULATIONS: tuple[str, ...] = get_args(OfflineGRPOConfig.__annotations__["policy_gradient_formulation"])


def clamp_negative_advantage_logps(
    token_logps: torch.Tensor, advantages: torch.Tensor, min_log_prob: float | None
) -> torch.Tensor:
    """Floored on negative-advantage rows. Below the floor the clamp passes no gradient, so such a row stops pushing a
    token it already rates that unlikely further toward zero probability."""
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
    """``prob_weighted`` L = -(π·A) weights high-prob tokens more; ``reinforce`` L = -(log π·A) is uniform.
    ``clamp_ref_logps`` takes the detached policy log-probs, since a grad-carrying ceiling would zero the KL gradient
    on every clamped token."""
    weights = torch.exp(token_logps) if policy_gradient_formulation == "prob_weighted" else token_logps
    per_token_loss = -(weights * advantages.unsqueeze(1))
    sample_values = {"logps": token_logps, "logps_unclamped": token_logps_unclamped, "pg_objective": -per_token_loss}
    if beta != 0.0:
        ref_logps, _ = clamp_ref_logps(ref_logps, token_logps.detach())
        per_token_kl = torch.exp(ref_logps - token_logps) - (ref_logps - token_logps) - 1
        per_token_loss = per_token_loss + beta * per_token_kl
        sample_values |= {"kl": per_token_kl, "ref_logps": ref_logps, "ref_logps_unclamped": ref_logps_unclamped}
    return per_token_loss, sample_values


def _complete_row_token_counts(
    supervised_mask: torch.Tensor, row_token_counts: torch.Tensor | None, cp_config: CPConfig | None
) -> torch.Tensor:
    """Each row's supervised-token count in fp32: ``row_token_counts`` when given, else the mask's own,
    which a CP shard holds only part of."""
    if row_token_counts is not None:
        if row_token_counts.shape != supervised_mask.shape[:1]:
            raise ValueError("Offline loss needs one supervised-token count per token row")
        return row_token_counts.float()
    if cp_config is not None and cp_config.cp_size > 1:
        raise ValueError("A CP shard's mask holds part of each row: pass the complete row_token_counts")
    return supervised_mask.sum(dim=1, dtype=torch.float32)


def offline_loss_numerator(
    per_token_loss: torch.Tensor,
    supervised_mask: torch.Tensor,
    group_sizes: torch.Tensor,
    *,
    loss_type: str,
    row_token_counts: torch.Tensor | None = None,
    cp_config: CPConfig | None = None,
) -> torch.Tensor:
    """Summed over supervised tokens, each row weighted ``1/group_size`` so every source group counts equally; ``grpo``
    first averages each row over its own tokens. Row-local, so pipeline microbatch numerators sum to the batch's."""
    if per_token_loss.shape != supervised_mask.shape or per_token_loss.shape[0] != group_sizes.numel():
        raise ValueError("Offline loss needs one supervision mask and group size per token row")
    group_weights = 1.0 / group_sizes.float()
    weighted = per_token_loss * group_weights.unsqueeze(1) * supervised_mask
    if loss_type == "grpo":
        rows = cp_sum_rows(weighted.sum(dim=1, dtype=torch.float32), cp_config)
        counts = _complete_row_token_counts(supervised_mask, row_token_counts, cp_config)
        return (rows / counts.clamp(min=1)).sum()
    return cp_sum_rows(weighted.sum(), cp_config)


def offline_loss_normalizer(
    token_counts: torch.Tensor | None, group_sizes: torch.Tensor, *, loss_type: str, max_completion_length: int | None
) -> torch.Tensor:
    """``grpo``: the ``1/group_size`` weights' sum; ``bnpo``: the weighted completion-token count; ``dr_grpo``: the sum
    times ``max_completion_length``. Group metadata is replicated on CP peers, so no collective runs."""
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
    row_token_counts: torch.Tensor | None = None,
    cp_config: CPConfig | None = None,
) -> torch.Tensor:
    """A CP token shard passes each row's complete ``row_token_counts``."""
    counts = _complete_row_token_counts(supervised_mask, row_token_counts, cp_config)
    numerator = offline_loss_numerator(
        per_token_loss, supervised_mask, group_sizes, loss_type=loss_type, row_token_counts=counts, cp_config=cp_config
    )
    return numerator / offline_loss_normalizer(
        counts, group_sizes, loss_type=loss_type, max_completion_length=max_completion_length
    )
