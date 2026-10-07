"""Group-relative advantages for GRPO: baseline, scaling, and the degenerate-group mask.

Pure functions over a rank-local reward tensor: ``RepeatSampler`` keeps each prompt's completions
together on one rank, so the baseline needs no cross-rank gather (except ``scale_rewards="batch"``,
whose divisor must be identical on every rank, and the non-finite check, which every rank must agree
on before any of them raises). Offline GRPO's stored groups take their advantages at tokenization
instead, one group at a time (:func:`compute_group_advantages`).
"""

from collections.abc import Callable

import numpy as np
import torch
from accelerate.utils import gather
from scipy import stats

from src.distributed.runtime import rank_consensus

# Matches TRL, so env-, online- and offline-GRPO agree on near-degenerate groups.
STD_EPS = 1e-4
# Reward spread below this counts as all completions scoring alike. Not shared with offline GRPO's
# drop_degenerate_groups, whose rank-based advantage methods zero only exact ties: a near-tie within
# this spread still trains there with full-scale advantages.
_DEGENERATE_SPREAD = 1e-6


def _z_norm(rewards: np.ndarray) -> np.ndarray:
    # ddof=1 explicitly: numpy defaults to 0 while torch's .std() is correction=1, and the online and
    # environmental z-norms take the torch path; an implicit default would split the two by sqrt((n-1)/n).
    reward_std = np.std(rewards, ddof=1) if len(rewards) > 1 else 1.0
    return (rewards - np.mean(rewards)) / (reward_std + STD_EPS)


def _minmax(rewards: np.ndarray) -> np.ndarray:
    reward_min, reward_max = np.min(rewards), np.max(rewards)
    if reward_max == reward_min:
        return np.zeros_like(rewards)
    return 2 * (rewards - reward_min) / (reward_max - reward_min) - 1


def _quantile_norm(rewards: np.ndarray) -> np.ndarray:
    # (ranks - 0.5)/n: uniform on [0, 1] without its boundaries, then to the normal.
    uniform_scores = (stats.rankdata(rewards) - 0.5) / len(rewards)
    return stats.norm.ppf(uniform_scores)


def _quantile_uniform(rewards: np.ndarray) -> np.ndarray:
    # A single or all-equal group has no spread to rank, and n-1 == 0 would divide by zero.
    if len(rewards) == 1 or np.all(rewards == rewards[0]):
        return np.zeros(len(rewards))
    uniform_scores = (stats.rankdata(rewards) - 1) / (len(rewards) - 1)
    return 2 * uniform_scores - 1


def _robust(rewards: np.ndarray) -> np.ndarray:
    q75, q25 = np.percentile(rewards, [75, 25])
    iqr = q75 - q25
    return np.zeros_like(rewards) if iqr == 0 else (rewards - np.median(rewards)) / iqr


# Each OfflineGRPOConfig.advantage_method spelling to the map it names.
GROUP_ADVANTAGE_METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "z_norm": _z_norm,
    "minmax": _minmax,
    "quantile_norm": _quantile_norm,
    "quantile_uniform": _quantile_uniform,
    "robust": _robust,
}


def compute_group_advantages(
    rewards_list: list[float],
    method: str,
    best_completion_emphasis: float | str,
) -> list[float]:
    """Advantages from one stored group's rewards via ``method`` (a :data:`GROUP_ADVANTAGE_METHODS` key,
    refused otherwise by the offline trainer's constructor), with optional best-completion emphasis,
    clipped to [-10, 10]."""
    # float64 explicitly: integral rewards (0/1 verifiable) would keep an int64 dtype through the
    # `np.zeros_like` degenerate-group branches, and the in-place emphasis multiply below then raises
    # UFuncTypeError inside datasets.map.
    rewards_array = np.asarray(rewards_list, dtype=np.float64)
    if not np.all(np.isfinite(rewards_array)):
        # Every method divides by a spread derived from these rewards, so one NaN/Inf reaches the
        # whole group's advantages and from there the micro-batch gradient.
        raise ValueError(
            f"Non-finite reward in a completion group: {rewards_list}. Fix the reward column — "
            f"training through it silently either zeroes the row's advantage or NaNs the batch, "
            f"depending only on the group size."
        )
    advantages = GROUP_ADVANTAGE_METHODS[method](rewards_array)

    if len(rewards_array) > 1:
        if best_completion_emphasis == "auto":
            # Scale emphasis with std: 3.0 at std=0 → 5.0 at std→∞. Population std (numpy's default),
            # unlike the z_norm divisor: this heuristic has no torch counterpart to match, and it
            # multiplies the best row under every method, so aligning it would also move the advantages
            # of the rank/minmax methods.
            reward_std = np.std(rewards_array)
            emphasis_factor = 3.0 + 2.0 * reward_std / (1.0 + reward_std)
        else:
            emphasis_factor = float(best_completion_emphasis)

        if emphasis_factor > 1.0:
            advantages[rewards_array == np.max(rewards_array)] *= emphasis_factor

    return np.clip(advantages, -10.0, 10.0).tolist()


def _grouped(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    """``rewards`` as ``(num_groups, num_generations)``; raises if generations were split across ranks."""
    if rewards.numel() % num_generations != 0:
        raise ValueError(
            f"rewards size ({rewards.numel()}) is not a multiple of num_generations "
            f"({num_generations}); group-relative advantages require each prompt's "
            f"generations to stay together on one rank."
        )
    return rewards.view(-1, num_generations)


def scales_rewards(scale_rewards: str | bool | None) -> bool:
    """Whether ``scale_rewards`` (``"group"`` / ``"batch"`` / ``"none"``, or a bool) divides the advantages
    by a std. ``"none"`` is TRUTHY, so the off values are matched explicitly."""
    return scale_rewards not in (None, False, "none")


def reject_inert_std_floor(scale_rewards: str | bool | None, std_floor: float) -> None:
    """Refuse ``scale_rewards_std_floor`` on a run whose advantages divide by no std: it floors nothing."""
    if std_floor > 0 and not scales_rewards(scale_rewards):
        raise ValueError(
            f"scale_rewards_std_floor={std_floor} with scale_rewards={scale_rewards!r}: the floor bounds the std "
            "the advantages divide by, and this run divides by none. Set scale_rewards to 'group' or 'batch', "
            "or drop the floor."
        )


def valid_group_stats(
    rewards: torch.Tensor, num_generations: int, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each group's reward mean and unbiased std over its valid members, both ``[groups, 1]``.

    A group with no valid member takes its plain mean; one with fewer than two has std 0, the spread
    a group-scaled advantage of 0 divides by.
    """
    grouped = _grouped(rewards, num_generations)
    valid = _grouped(valid_mask.to(grouped.dtype), num_generations)
    valid_count = valid.sum(dim=1, keepdim=True)
    group_mean = torch.where(
        valid_count > 0,
        (grouped * valid).sum(dim=1, keepdim=True) / valid_count.clamp_min(1.0),
        grouped.mean(dim=1, keepdim=True),
    )
    squares = (((grouped - group_mean) ** 2) * valid).sum(dim=1, keepdim=True)
    var = squares / (valid_count - 1.0).clamp_min(1.0)
    return group_mean, torch.where(valid_count > 1, var.sqrt(), torch.zeros_like(var))


def group_relative_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    scale_rewards: str | bool | None,
    valid_mask: torch.Tensor | None = None,
    already_gathered: bool = False,
    std_floor: float = 0.0,
) -> torch.Tensor:
    """GRPO advantage: reward minus its group mean, optionally rescaled.

    ``valid_mask`` (True = real completion) excludes infra-failed placeholder rows from the group-mean
    baseline so they cannot poison the advantages of their valid siblings; a group with no valid member
    falls back to the plain mean.

    ``scale_rewards`` is read by :func:`scales_rewards`. ``"group"`` divides by the per-group std
    (dangerous on sparse reward — degenerate groups have std → 0); ``"batch"`` divides by the
    global-batch std, keeping degenerate groups near 0 while holding the gradient scale steady.

    ``std_floor`` bounds the scaling amplification: the divisor is ``max(std, std_floor)``. A
    behaviorally-degenerate batch/group (every reward within a few hundredths) otherwise divides its own
    shaping noise by a near-zero std, inflating it to full-scale advantages; batches with real spread
    (std above the floor) are unaffected.
    """
    grouped = _grouped(rewards, num_generations)
    if valid_mask is not None:
        group_mean, valid_std = valid_group_stats(rewards, num_generations, valid_mask)
    else:
        group_mean = grouped.mean(dim=1, keepdim=True)
    advantages = rewards - group_mean.expand_as(grouped).flatten()

    if scales_rewards(scale_rewards):
        # The std divisor honors ``valid_mask`` like the baseline: a placeholder would bias every valid row.
        if scale_rewards == "batch":
            # Global std: a rank-local one would scale each DP rank's advantages differently.
            batch_rewards = rewards if already_gathered else gather(rewards)
            if valid_mask is not None:
                batch_valid = valid_mask if already_gathered else gather(valid_mask)
                batch_rewards = batch_rewards[batch_valid.to(torch.bool)]
            std = batch_rewards.std() if batch_rewards.numel() > 1 else torch.zeros_like(rewards[:1])
            advantages = advantages / (std.clamp_min(std_floor) + STD_EPS)
        elif num_generations > 1:
            group_std = valid_std if valid_mask is not None else grouped.std(dim=1, keepdim=True)
            advantages = advantages / (group_std.expand_as(grouped).flatten().clamp_min(std_floor) + STD_EPS)
        # num_generations == 1: singleton groups have advantage 0; leave unscaled rather than /NaN.

    _require_finite(rewards, advantages)
    return advantages


def _require_finite(rewards: torch.Tensor, advantages: torch.Tensor) -> None:
    """Raise on every rank when any rank's rewards or advantages are non-finite.

    The verdict is agreed across ranks: a rank raising alone leaves its peers in the next collective.
    """
    if rank_consensus(bool(torch.isfinite(rewards).all() & torch.isfinite(advantages).all()))[0]:
        return
    raise ValueError(
        f"Non-finite GRPO rewards or advantages on at least one rank (this rank: "
        f"{int((~torch.isfinite(rewards)).sum())} of {rewards.numel()} rewards and "
        f"{int((~torch.isfinite(advantages)).sum())} advantages non-finite). Fix the reward function or "
        f"environment returning NaN/Inf; under scale_rewards='batch' a single one makes every advantage "
        f"of the step non-finite."
    )


def degenerate_group_mask(
    rewards: torch.Tensor, num_generations: int, valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-trajectory mask (True = drop) of groups whose completions all scored the same reward.

    A zero-spread group contributes no policy gradient yet its tokens inflate the DAPO loss normalizer,
    diluting the groups that carry signal; dropping them restores the effective batch size (DAPO
    dynamic sampling without the resampling half).

    ``valid_mask`` (True = real completion) restricts degeneracy to valid members, so an infra-failed
    placeholder's differing reward cannot let an all-alike group escape detection. A group with fewer
    than two valid members carries no intra-group comparison signal and counts as degenerate.
    """
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
