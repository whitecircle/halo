"""Group-relative advantages for GRPO: baseline, scaling, and negative-side surgery.

Pure functions over a rank-LOCAL reward tensor: ``RepeatSampler`` keeps each prompt's completions
together on one rank, so the baseline needs no cross-rank gather (except ``scale_rewards="batch"``,
whose divisor must be identical on every rank).
"""

import torch
from accelerate.utils import gather

from src.args.mixins import AdvantageShaping

# Matches TRL, so env-, online- and offline-GRPO agree on near-degenerate groups.
STD_EPS = 1e-4
# Reward spread below this counts as "all completions scored alike". NOT shared with offline GRPO's
# drop_degenerate_groups, whose rank-based advantage methods zero only EXACT ties — a near-tie
# within this spread still trains there with full-scale advantages.
_DEGENERATE_SPREAD = 1e-6


def _grouped(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    """``rewards`` as ``(num_groups, num_generations)``; raises if generations were split across ranks."""
    if rewards.numel() % num_generations != 0:
        raise ValueError(
            f"rewards size ({rewards.numel()}) is not a multiple of num_generations "
            f"({num_generations}); group-relative advantages require each prompt's "
            f"generations to stay together on one rank."
        )
    return rewards.view(-1, num_generations)


def group_relative_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    scale_rewards: str | bool | None,
    valid_mask: torch.Tensor | None = None,
    shaping: AdvantageShaping | None = None,
    gate_rewards: torch.Tensor | None = None,
    already_gathered: bool = False,
    std_floor: float = 0.0,
) -> torch.Tensor:
    """GRPO advantage: reward minus its group mean, optionally rescaled.

    ``valid_mask`` (True = real completion) excludes infra-failed placeholder rows from the group-mean
    baseline so they cannot poison the advantages of their valid siblings; a group with no valid member
    falls back to the plain mean.

    ``scale_rewards`` is ``"group"`` / ``"batch"`` / ``"none"`` (or a bool). ``"none"`` is TRUTHY, so the
    disabled values are matched explicitly rather than via ``if scale_rewards``. ``"group"`` divides by
    the per-group std (dangerous on sparse reward — degenerate groups have std → 0); ``"batch"`` divides
    by the global-batch std, keeping degenerate groups near 0 while holding the gradient scale steady.

    ``std_floor`` bounds the scaling amplification: the divisor is ``max(std, std_floor)``. A
    behaviorally-degenerate batch/group (every reward within a few hundredths) otherwise divides its own
    shaping noise by a near-zero std, inflating it to full-scale advantages; batches with real spread
    (std above the floor) are unaffected.

    ``shaping`` (default None) selects :class:`AdvantageShaping`; ``gate_rewards`` supplies the
    objective-component gate for ``neg_mask_hard``.
    """
    grouped = _grouped(rewards, num_generations)
    if shaping is not None and shaping.mode == "qae" and num_generations > 1:
        vals = grouped.float()
        if valid_mask is not None:
            vals = vals.masked_fill(~_grouped(valid_mask, num_generations).bool(), float("nan"))
        baseline = vals.nanquantile(shaping.quantile, dim=1, keepdim=True)
        baseline = torch.where(baseline.isnan(), grouped.mean(dim=1, keepdim=True), baseline)
        group_baseline = baseline.to(grouped.dtype).expand_as(grouped).flatten()
    elif valid_mask is not None:
        valid = _grouped(valid_mask.to(grouped.dtype), num_generations)
        valid_count = valid.sum(dim=1, keepdim=True)
        safe_mean = torch.where(
            valid_count > 0,
            (grouped * valid).sum(dim=1, keepdim=True) / valid_count.clamp_min(1.0),
            grouped.mean(dim=1, keepdim=True),
        )
        group_baseline = safe_mean.expand_as(grouped).flatten()
    else:
        group_baseline = grouped.mean(dim=1, keepdim=True).expand_as(grouped).flatten()
    advantages = rewards - group_baseline

    if scale_rewards not in (None, False, "none"):
        # The std divisor honors ``valid_mask`` like the baseline: a placeholder would bias every valid row.
        if scale_rewards == "batch":
            # GLOBAL std: a rank-local one would scale each DP rank's advantages differently.
            batch_rewards = rewards if already_gathered else gather(rewards)
            if valid_mask is not None:
                batch_valid = valid_mask if already_gathered else gather(valid_mask)
                batch_rewards = batch_rewards[batch_valid.to(torch.bool)]
            std = batch_rewards.std() if batch_rewards.numel() > 1 else torch.zeros_like(rewards[:1])
            advantages = advantages / (std.clamp_min(std_floor) + STD_EPS)
        elif num_generations > 1:
            if valid_mask is not None:
                # Unbiased std over valid members; groups with < 2 of them get std 0 (advantage already 0).
                valid = _grouped(valid_mask.to(grouped.dtype), num_generations)
                n = valid.sum(dim=1, keepdim=True)
                vmean = (grouped * valid).sum(dim=1, keepdim=True) / n.clamp_min(1.0)
                var = (((grouped - vmean) ** 2) * valid).sum(dim=1, keepdim=True) / (n - 1.0).clamp_min(1.0)
                std = torch.where(n > 1, var.sqrt(), torch.zeros_like(var)).expand_as(grouped).flatten()
            else:
                std = grouped.std(dim=1, keepdim=True).expand_as(grouped).flatten()
            advantages = advantages / (std.clamp_min(std_floor) + STD_EPS)
        # num_generations == 1: singleton groups have advantage 0; leave unscaled rather than /NaN.

    if shaping is not None and shaping.mode in ("asymmetric", "neg_mask_hard"):
        advantages = _negative_side_surgery(advantages, num_generations, shaping, gate_rewards, rewards)
    # Fail safe: a NaN advantage would propagate straight into the optimizer.
    return torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)


def _negative_side_surgery(
    advantages: torch.Tensor,
    num_generations: int,
    shaping: AdvantageShaping,
    gate_rewards: torch.Tensor | None,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """Post-baseline negative-advantage treatment (see :class:`AdvantageShaping`).

    ``asymmetric`` scales each sign unconditionally; ``neg_mask_hard`` zeroes negatives only in groups
    whose best ``gate_rewards`` member stayed below ``hard_group_threshold``.
    """
    if shaping.mode == "asymmetric":
        return torch.where(advantages >= 0, advantages * shaping.pos_scale, advantages * shaping.neg_scale)
    gate = gate_rewards if gate_rewards is not None else rewards
    grouped_gate = _grouped(gate, num_generations)
    hard = (grouped_gate.max(dim=1, keepdim=True).values < shaping.hard_group_threshold).expand_as(grouped_gate)
    return torch.where(hard.flatten() & (advantages < 0), torch.zeros_like(advantages), advantages)


def degenerate_group_mask(
    rewards: torch.Tensor, num_generations: int, valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-trajectory mask (True = drop) of groups whose completions ALL scored the same reward.

    A zero-spread group contributes no policy gradient yet its tokens inflate the DAPO loss normalizer,
    diluting the groups that carry signal; dropping them restores the effective batch size (the cheap
    half of DAPO dynamic sampling — drop dead groups without resampling).

    ``valid_mask`` (True = real completion) restricts degeneracy to VALID members: an infra-failed
    placeholder's differing reward must not let an all-alike group escape detection. A group with
    fewer than two valid members carries no intra-group comparison signal, so it is degenerate by
    definition.
    """
    if num_generations <= 1 or rewards.numel() % num_generations != 0:
        return torch.zeros_like(rewards, dtype=torch.bool)
    grouped = _grouped(rewards, num_generations)
    if valid_mask is None:
        spread = grouped.max(dim=1).values - grouped.min(dim=1).values
        degenerate = spread <= _DEGENERATE_SPREAD
    else:
        valid = _grouped(valid_mask, num_generations).to(torch.bool)
        vmax = grouped.masked_fill(~valid, float("-inf")).max(dim=1).values
        vmin = grouped.masked_fill(~valid, float("inf")).min(dim=1).values
        degenerate = (vmax - vmin <= _DEGENERATE_SPREAD) | (valid.sum(dim=1) < 2)
    return degenerate.unsqueeze(1).expand_as(grouped).flatten()
