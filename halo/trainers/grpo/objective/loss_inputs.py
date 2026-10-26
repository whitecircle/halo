"""The loss inputs the on-policy GRPO advantages shape: degenerate-group drops, the gathered-global DAPO
normalizer, the token-mass balance, and per-trajectory values laid over turn rows."""

import math
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass

import torch

# Metric keys both GRPO trainers log under.
DEGENERATE_GROUP_FRAC_KEY = "sampling/degenerate_group_frac"
NET_TOKEN_MASS_KEY = "advantage/net_token_mass"
TOKEN_MASS_SCALE_KEY = "advantage/token_mass_scale"
NEGATIVE_ONLY_MASS_KEY = "advantage/negative_only_mass"
# TRL loss types whose loss tokens of a step share one normalizer, so a row pulls with its advantage times its
# token weight: the only losses the token-mass balance is exact for.
TOKEN_SUM_LOSS_TYPES = ("cispo", "dapo", "dr_grpo")


@dataclass(frozen=True)
class TokenMassBalance:
    """A generation round's token-weighted advantage mass and the per-sign scales that cancel it.

    ``net`` is ``(P - N) / (P + N)`` before balancing, where ``P`` and ``N`` are the summed positive and
    negative ``advantage x token weight`` of the round's trainable rows (all but the negative-only ones):
    the share of their push that raises (``net > 0``) or lowers (``net < 0``) the probability of the
    tokens the policy sampled.
    """

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


def _world_masses(local: torch.Tensor, gather_fn: Callable[[torch.Tensor], torch.Tensor]) -> list[float]:
    """``local`` summed over every rank (the gather is collective). A sum that is not finite raises on every
    rank alike: a NaN or infinite advantage or IS ratio reached it, and so the loss."""
    masses = gather_fn(local.unsqueeze(0)).sum(dim=0).tolist()
    if not all(math.isfinite(mass) for mass in masses):
        raise RuntimeError(
            f"The round's token-weighted advantage masses are not finite ({masses}): a NaN or infinite "
            "advantage or IS ratio reached the loss."
        )
    return masses


def _balance(positive: float, negative: float) -> TokenMassBalance:
    """The balance of a round whose world-summed masses are ``positive`` and ``negative``."""
    total = positive + negative
    if total <= 0:
        return TokenMassBalance(net=0.0)
    net = (positive - negative) / total
    if negative > positive:
        return TokenMassBalance(net=net, negative_scale=positive / negative)
    return TokenMassBalance(net=net, positive_scale=negative / positive)


def record_token_mass(
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    is_ratio: torch.Tensor | None,
    gather_fn: Callable[[torch.Tensor], torch.Tensor],
    metrics: MutableMapping[str, list[float]],
    enabled: bool,
    negative_only: torch.Tensor | None = None,
) -> TokenMassBalance | None:
    """COLLECTIVE: log a generation round's net token mass and return the balance to apply, ``None`` when off.

    A group's advantages sum to zero but their token-weighted sum does not: longer failures push down the
    tokens the policy sampled (entropy rises), longer solves sharpen it. The balance scales the heavier sign
    down until the world-summed net push is zero, keeping every row's sign and order within it. A token weighs
    its ``loss_mask`` place times its IS ratio (``is_ratio``, ``None`` when the loss applies none); the net
    share is logged either way, as the early sign of an entropy drift. ``negative_only`` rows (an untrainable
    turn's) are neither weighed nor scaled (pass them to :meth:`TokenMassBalance.apply` too); their mass's
    share of the trained mass is logged.
    """
    weights = (loss_mask if is_ratio is None else loss_mask * is_ratio).sum(dim=1)
    if negative_only is None:
        local = _signed_masses(advantages, weights)
    else:
        flagged = negative_only.to(advantages.device)
        trainable = _signed_masses(advantages[~flagged], weights[~flagged])
        local = torch.cat([trainable, _signed_masses(advantages[flagged], weights[flagged])[1:]])
    masses = _world_masses(local, gather_fn)
    result = _balance(masses[0], masses[1])
    metrics[NET_TOKEN_MASS_KEY].append(result.net)
    if negative_only is not None:
        scales = result if enabled else TokenMassBalance(net=result.net)
        trained = masses[0] * scales.positive_scale + masses[1] * scales.negative_scale + masses[2]
        metrics[NEGATIVE_ONLY_MASS_KEY].append(masses[2] / trained if trained > 0 else 0.0)
    if not enabled:
        return None
    metrics[TOKEN_MASS_SCALE_KEY].append(result.scale)
    return result


def narrow_loss_masks(drop_rows: torch.Tensor, *masks: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Zero the dropped rows out of each per-row mask (pass-through when nothing is dropped).

    Masking, not deleting: the rows still run the forward so every rank issues an identical
    collective sequence.
    """
    if not drop_rows.any():
        return masks
    keep = ~drop_rows.to(masks[0].device).unsqueeze(1)
    return tuple(mask & keep for mask in masks)


def gathered_num_items(loss_mask: torch.Tensor, gather_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Gathered-global DAPO loss normalizer: total loss tokens across ranks from the post-drop mask.

    A local count mis-scales loss and effective LR (TRL divides the gathered total by num_processes);
    ``clamp(min=1)`` guards an all-degenerate global batch dividing by zero. The gather is collective,
    so every rank must call this.
    """
    return gather_fn(loss_mask.sum(dim=1)).sum().clamp(min=1)


def expand_traj_to_rows(
    values: torch.Tensor,
    turns_per_traj: list[int],
    num_dummy_rows: int,
    expand: bool,
    dummy_fill: float = 0,
) -> torch.Tensor:
    """Per-trajectory values expanded to per-turn rows plus ``dummy_fill`` padding rows.

    ``expand`` repeats each trajectory's value over its turn rows (``train_on_sampled_tokens``);
    ``False`` keeps one row per trajectory. Dummy padding rows are appended either way.
    """
    rows = values.repeat_interleave(torch.tensor(turns_per_traj, device=values.device)) if expand else values
    if num_dummy_rows:
        rows = torch.cat([rows, torch.full((num_dummy_rows,), dummy_fill, device=rows.device, dtype=rows.dtype)])
    return rows
