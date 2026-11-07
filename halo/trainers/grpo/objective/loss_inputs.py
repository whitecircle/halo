"""The loss inputs the on-policy GRPO advantages shape: degenerate-group drops, the gathered-global DAPO
normalizer, the token-mass balance, and per-trajectory values laid over turn rows."""

import math
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass

import torch

DEGENERATE_GROUP_FRAC_KEY = "sampling/degenerate_group_frac"
NET_TOKEN_MASS_KEY = "advantage/net_token_mass"
TOKEN_MASS_SCALE_KEY = "advantage/token_mass_scale"
NEGATIVE_ONLY_MASS_KEY = "advantage/negative_only_mass"
# Batch keys of the importance weight's inputs when the loss forward builds the weight: the engine's log-probs, the
# tokens it corrects, the forced closes it zeroes, the untrainable-turn rows, and the reference log-probs before the
# loss forward clamps them against the policy's.
IMPORTANCE_RATIO_KEY = "importance_sampling_ratio"
REF_LOGPS_KEY = "ref_per_token_logps"
SAMPLING_LOGPS_KEY = "sampling_per_token_logps"
IS_SAMPLING_LOGPS_KEY = "is_sampling_logps"
IS_CORRECTED_MASK_KEY = "is_corrected_mask"
IS_FORCED_ZERO_KEY = "is_forced_zero_mask"
NEGATIVE_ONLY_ROWS_KEY = "negative_only_rows"
UNCLAMPED_REF_LOGPS_KEY = "unclamped_ref_per_token_logps"
# Metric sums a micro-batch adds to the round's token masses: trainable positive, trainable negative, and the
# negative-only rows' negative mass.
TOKEN_MASS_SUM_KEYS = ("loss_forward/positive_mass", "loss_forward/negative_mass", "loss_forward/negative_only_mass")
# TRL loss types whose loss tokens of a step share one normalizer, so a row pulls with its advantage times its
# token weight: the only losses the token-mass balance is exact for.
TOKEN_SUM_LOSS_TYPES = ("cispo", "dapo", "dr_grpo")


@dataclass(frozen=True)
class TokenMassBalance:
    """``net`` is ``(P - N) / (P + N)`` before balancing, over the summed positive and negative ``advantage x token
    weight`` of the round's trainable rows."""

    net: float
    positive_scale: float = 1.0
    negative_scale: float = 1.0

    @property
    def scale(self) -> float:
        """The factor the heavier sign takes (1.0 when nothing is balanced)."""
        return min(self.positive_scale, self.negative_scale)

    def apply(self, advantages: torch.Tensor, negative_only: torch.Tensor | None = None) -> torch.Tensor:
        """``advantages`` with each sign scaled; the ``negative_only`` rows, outside the balance, keep theirs."""
        scaled = torch.where(advantages > 0, advantages * self.positive_scale, advantages * self.negative_scale)
        if negative_only is None:
            return scaled
        return torch.where(negative_only.to(advantages.device), advantages, scaled)


def _signed_masses(advantages: torch.Tensor, token_weights: torch.Tensor) -> torch.Tensor:
    """This rank's ``[positive, negative]`` token-weighted advantage mass, both non-negative."""
    weights = token_weights.to(advantages.dtype)
    return torch.stack([(advantages.clamp_min(0) * weights).sum(), (advantages.clamp_max(0).neg() * weights).sum()])


def _check_finite_masses(masses: list[float]) -> None:
    """A non-finite sum raises on every rank alike: a NaN or infinite advantage or IS ratio reached it, and so the
    loss."""
    if not all(math.isfinite(mass) for mass in masses):
        raise RuntimeError(
            f"The round's token-weighted advantage masses are not finite ({masses}): a NaN or infinite "
            "advantage or IS ratio reached the loss."
        )


def _balance(positive: float, negative: float) -> TokenMassBalance:
    total = positive + negative
    if total <= 0:
        return TokenMassBalance(net=0.0)
    net = (positive - negative) / total
    if negative > positive:
        return TokenMassBalance(net=net, negative_scale=positive / negative)
    return TokenMassBalance(net=net, positive_scale=negative / positive)


def token_mass_sums(
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    importance_ratio: torch.Tensor | None,
    negative_only: torch.Tensor | None = None,
) -> torch.Tensor:
    """This rank's additive token-weighted masses: ``[positive, negative]``, and with ``negative_only`` rows (outside
    the balance) a third, theirs."""
    weights = (loss_mask if importance_ratio is None else loss_mask * importance_ratio).sum(dim=1)
    if negative_only is None:
        return _signed_masses(advantages, weights)
    flagged = negative_only.to(advantages.device)
    trainable = _signed_masses(advantages[~flagged], weights[~flagged])
    return torch.cat([trainable, _signed_masses(advantages[flagged], weights[flagged])[1:]])


def record_world_token_mass(
    masses: list[float], metrics: MutableMapping[str, list[float]], enabled: bool, with_negative_only: bool
) -> TokenMassBalance | None:
    """The world's summed ``masses`` of :func:`token_mass_sums` as metrics, and the balance when ``enabled``. Raises on
    a non-finite mass."""
    _check_finite_masses(masses)
    result = _balance(masses[0], masses[1])
    metrics[NET_TOKEN_MASS_KEY].append(result.net)
    if with_negative_only:
        scales = result if enabled else TokenMassBalance(net=result.net)
        trained = masses[0] * scales.positive_scale + masses[1] * scales.negative_scale + masses[2]
        metrics[NEGATIVE_ONLY_MASS_KEY].append(masses[2] / trained if trained > 0 else 0.0)
    if not enabled:
        return None
    metrics[TOKEN_MASS_SCALE_KEY].append(result.scale)
    return result


def record_token_mass(
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    importance_ratio: torch.Tensor | None,
    gather_fn: Callable[[torch.Tensor], torch.Tensor],
    metrics: MutableMapping[str, list[float]],
    enabled: bool,
    negative_only: torch.Tensor | None = None,
) -> TokenMassBalance | None:
    """COLLECTIVE. A group's advantages sum to zero but their token-weighted sum does not: longer failures push down
    the tokens the policy sampled (entropy rises), longer solves sharpen it. ``negative_only`` rows are neither weighed
    nor scaled; pass them to :meth:`TokenMassBalance.apply` too."""
    local = token_mass_sums(advantages, loss_mask, importance_ratio, negative_only)
    masses = gather_fn(local.unsqueeze(0)).sum(dim=0).tolist()
    return record_world_token_mass(masses, metrics, enabled, negative_only is not None)


def narrow_loss_masks(drop_rows: torch.Tensor, *masks: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Masking rather than deleting, so the rows still run the forward and every rank issues an identical collective
    sequence."""
    if not drop_rows.any():
        return masks
    keep = ~drop_rows.to(masks[0].device).unsqueeze(1)
    return tuple(mask & keep for mask in masks)


def gathered_num_items(loss_mask: torch.Tensor, gather_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """COLLECTIVE. A local count mis-scales loss and effective LR (TRL divides the gathered total by num_processes);
    ``clamp(min=1)`` guards an all-degenerate global batch."""
    return gather_fn(loss_mask.sum(dim=1)).sum().clamp(min=1)


def expand_traj_to_rows(
    values: torch.Tensor,
    turns_per_traj: list[int],
    num_dummy_rows: int,
    expand: bool,
    dummy_fill: float = 0,
) -> torch.Tensor:
    """``expand`` repeats each value over its turn rows (``train_on_sampled_tokens``); ``False`` keeps one row per
    trajectory. Dummy rows are appended either way."""
    rows = values.repeat_interleave(torch.tensor(turns_per_traj, device=values.device)) if expand else values
    if num_dummy_rows:
        rows = torch.cat([rows, torch.full((num_dummy_rows,), dummy_fill, device=rows.device, dtype=rows.dtype)])
    return rows
