"""SMPO's objective: the per-pair margin loss, the log-prob clip, CP-global reductions and the metric folds."""

from collections.abc import Callable
from dataclasses import dataclass, fields

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.distributed.context_parallel.sequence_chunks import cp_sum_rows

# Per-pair margin losses of ``beta * (logratio - margin)``, positive when the margin holds; keyed by
# ``SmoothMarginPOConfig.loss_type``.
MARGIN_LOSSES: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "sigmoid": lambda scaled: -F.logsigmoid(scaled),
    "hinge": lambda scaled: torch.relu(-scaled),
    "ipo": lambda scaled: scaled.pow(2),
    "smooth_lower_bound": lambda scaled: torch.relu(-scaled).pow(2),
}


@dataclass(frozen=True)
class SMPOStepConstants:
    """A step's metric and loss divisors and its live margin: whole-batch counts under PP, units off it."""

    pair_count: int = 1
    chosen_token_count: torch.Tensor | float = 1.0
    rejected_token_count: torch.Tensor | float = 1.0
    target_margin: float = 0.0


def cp_global_mean(sums: torch.Tensor, counts: torch.Tensor, cp_config) -> torch.Tensor:
    """``sums / counts`` over whole sequences: under CP both are summed over the group first.

    The ``sums`` reduce is autograd-aware (its backward sums the gradient over the group); an in-place
    ``dist.all_reduce`` reaches autograd only through the deprecated c10d fallback, as the identity.
    """
    if cp_config is not None and cp_config.cp_size > 1:
        sums = cp_sum_rows(sums, cp_config)
        counts = cp_sum_rows(counts, cp_config)
    return sums / counts.clamp(min=1)


def masked_logit_mean(logits: torch.Tensor, loss_mask: torch.Tensor, cp_config=None) -> torch.Tensor:
    """``logits[loss_mask].mean()`` without the masked copy, fp32-accumulated; an empty mask yields 0.

    Under CP the sum and token count are summed over the group, so every rank reports the sequence-global
    mean. A logged metric, computed without grad.
    """
    with torch.no_grad():
        stats = torch.stack(
            [(logits.sum(dim=-1, dtype=torch.float32) * loss_mask).sum(), loss_mask.sum(dtype=torch.float32)]
        )
        if cp_config is not None and cp_config.cp_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=cp_config.process_group)
        total, tokens = stats.unbind()
        return total / (tokens * logits.size(-1)).clamp(min=1)


def cp_quantile(values: torch.Tensor, q: float, cp_config) -> torch.Tensor | None:
    """Quantile of ``values`` over the whole sequence, not this CP rank's slice; ``None`` when no rank holds one.

    A local quantile would give each rank its own clip bound, so the CP sum would drift with ``cp_size``.
    Detached: the bound is a clamp limit, not a term of the objective; left differentiable, every clamped
    token would route its gradient into the one element that is the quantile. It also keeps the
    all-gather, which has no autograd kernel, off the graph.
    """
    values = values.detach().float()
    group = cp_config.process_group if cp_config else None
    if group is None or dist.get_world_size(group) == 1:
        return torch.quantile(values, q) if values.numel() > 0 else None

    world = dist.get_world_size(group)
    counts = torch.zeros(world, dtype=torch.long, device=values.device)
    dist.all_gather_into_tensor(
        counts, torch.tensor([values.numel()], dtype=torch.long, device=values.device), group=group
    )
    if int(counts.sum()) == 0:
        return None

    width = int(counts.max())
    padded = values.new_zeros(width)
    padded[: values.numel()] = values
    gathered = padded.new_zeros(world * width)
    dist.all_gather_into_tensor(gathered, padded, group=group)
    gathered = gathered.view(world, width)
    real = torch.cat([gathered[r, : int(counts[r])] for r in range(world)])
    return torch.quantile(real, q)


@dataclass(frozen=True)
class SMPOObjective:
    """``L_margin + L_sft``: a margin loss pushing ``logp(chosen) - logp(rejected)`` past the target margin,
    plus a ``chosen_sft_ratio``-weighted cross-entropy on both completions."""

    beta: float
    loss_type: str
    chosen_sft_ratio: float
    lower_clip_percentile: float | None
    upper_clip_percentile: float | None
    min_log_prob: float | None

    @classmethod
    def from_args(cls, args) -> "SMPOObjective":
        return cls(**{field.name: getattr(args, field.name) for field in fields(cls)})

    def margin_loss(
        self, chosen_logps: torch.Tensor, rejected_logps: torch.Tensor, target_margin: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(per-pair losses, chosen_rewards, rejected_rewards)``; the rewards are beta-scaled log-probs,
        detached, for metrics."""
        losses = MARGIN_LOSSES[self.loss_type](self.beta * (chosen_logps - rejected_logps - target_margin))
        return losses, (self.beta * chosen_logps).detach(), (self.beta * rejected_logps).detach()

    def clip_log_probs(
        self, per_token_logps: torch.Tensor, loss_mask: torch.Tensor, is_chosen: torch.Tensor, cp_config=None
    ) -> torch.Tensor:
        """Percentile-clip the rejected low tail and the chosen high tail, then floor the rejected side.

        ``is_chosen`` selects the chosen side in any batch shape (a ``[rows, 1]`` column, or the padding-free
        row's per-token mask). The lower percentile runs before ``min_log_prob``, so the floor can only raise
        the tail further; masked positions are never touched and the input is left unmodified.
        """
        if self.lower_clip_percentile is None and self.upper_clip_percentile is None and self.min_log_prob is None:
            return per_token_logps
        clipped = per_token_logps
        chosen_valid = loss_mask & is_chosen
        rejected_valid = loss_mask & ~is_chosen

        if self.lower_clip_percentile is not None:
            lower_bound = cp_quantile(clipped[rejected_valid], self.lower_clip_percentile, cp_config)
            if lower_bound is not None:
                clipped = torch.where(rejected_valid & (clipped < lower_bound), lower_bound.to(clipped.dtype), clipped)

        if self.upper_clip_percentile is not None:
            upper_bound = cp_quantile(clipped[chosen_valid], self.upper_clip_percentile, cp_config)
            if upper_bound is not None:
                clipped = torch.where(chosen_valid & (clipped > upper_bound), upper_bound.to(clipped.dtype), clipped)

        if self.min_log_prob is not None:
            clipped = torch.where(rejected_valid & (clipped < self.min_log_prob), self.min_log_prob, clipped)

        return clipped


def metric_sums(
    *,
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    pair_valid: torch.Tensor,
    chosen_logits_sum: torch.Tensor,
    rejected_logits_sum: torch.Tensor,
    chosen_logit_tokens: torch.Tensor,
    rejected_logit_tokens: torch.Tensor,
    chosen_sft_sum: torch.Tensor,
    rejected_sft_sum: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """One batch's detached metric sum contributions, the definition both loss paths build from.

    ``pair_valid`` is a per-pair 0/1 mask, so the PP eval's filler pairs contribute nothing. Nothing is
    re-reduced: under CP the log-probs, logit means and SFT losses arrive as CP-global means.
    """
    return {
        key: value.detach()
        for key, value in (
            ("rewards/chosen", (chosen_rewards * pair_valid).sum()),
            ("rewards/rejected", (rejected_rewards * pair_valid).sum()),
            ("rewards/accuracies", ((chosen_rewards > rejected_rewards).float() * pair_valid).sum()),
            ("rewards/margins", ((chosen_rewards - rejected_rewards) * pair_valid).sum()),
            ("logps/chosen", (chosen_logps * pair_valid).sum()),
            ("logps/rejected", (rejected_logps * pair_valid).sum()),
            ("logits/chosen", chosen_logits_sum),
            ("logits/rejected", rejected_logits_sum),
            ("logit_tokens/chosen", chosen_logit_tokens),
            ("logit_tokens/rejected", rejected_logit_tokens),
            ("sft_loss/chosen", chosen_sft_sum),
            ("sft_loss/rejected", rejected_sft_sum),
        )
    }


def metrics_from_sums(
    sums: dict[str, torch.Tensor], constants: SMPOStepConstants, *, report_margin: bool
) -> dict[str, torch.Tensor]:
    """Fold metric sums into the batch's unprefixed means over ``constants``' divisors.

    The key set is fixed here, never by what accumulated: a missing key reads as zero, so a pipeline stage
    that ran no microbatch reports the same names as one that did.
    """
    # On the device the step's own constants live on: a stage that accumulated nothing would otherwise
    # divide a CPU zero by a device token count.
    device = next(
        (
            value.device
            for value in (*sums.values(), constants.chosen_token_count, constants.rejected_token_count)
            if isinstance(value, torch.Tensor)
        ),
        None,
    )
    zero = torch.zeros((), dtype=torch.float32, device=device)

    def total(key: str) -> torch.Tensor:
        return sums.get(key, zero)

    pairs = max(float(constants.pair_count), 1.0)
    metrics = {
        key: total(key) / pairs
        for key in (
            "rewards/chosen",
            "rewards/rejected",
            "rewards/accuracies",
            "rewards/margins",
            "logps/chosen",
            "logps/rejected",
        )
    }
    for side in ("chosen", "rejected"):
        metrics[f"logits/{side}"] = total(f"logits/{side}") / total(f"logit_tokens/{side}").clamp(min=1)
    metrics["sft_loss/chosen"] = total("sft_loss/chosen") / constants.chosen_token_count
    metrics["sft_loss/rejected"] = total("sft_loss/rejected") / constants.rejected_token_count
    if report_margin:
        metrics["target_margin"] = zero + constants.target_margin
    return metrics
