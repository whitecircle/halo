"""Applying advantages and drops to the loss inputs of the online and environmental GRPO trainers.

Once rewards exist, both trainers mask the rows of degenerate (all-equal-reward) groups out of the
loss, recompute the gathered-global DAPO normalizer from the post-drop loss mask, and balance the
round's token-weighted advantage mass: the online trainer on TRL's result dict, the environmental
trainer on the tensors it builds itself. The mask, normalizer and token-mass helpers serve both, so
their numerics match; :func:`expand_traj_to_rows` is the environmental trainer's layout of
per-trajectory values over its per-turn rows.
"""

import math
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass

import torch

# Metric keys both GRPO trainers log under.
DEGENERATE_GROUP_FRAC_KEY = "sampling/degenerate_group_frac"
NET_TOKEN_MASS_KEY = "advantage/net_token_mass"
TOKEN_MASS_SCALE_KEY = "advantage/token_mass_scale"
NEGATIVE_ONLY_MASS_KEY = "advantage/negative_only_mass"
# TRL loss types whose every loss token of a step shares one normalizer, so a row pulls with its
# advantage times its trained token weight: the losses the token-mass balance is exact for. ``grpo`` /
# ``sapo`` average each completion over its own length, ``bnpo`` normalizes per micro-batch, ``luspo``
# weighs every token of a completion by one clipped sequence-level ratio and, under the vLLM IS correction,
# averages over every position, padding included, and ``vespo`` weighs each sequence by its advantage's sign.
TOKEN_SUM_LOSS_TYPES = ("cispo", "dapo", "dr_grpo")


@dataclass(frozen=True)
class TokenMassBalance:
    """A generation round's token-weighted advantage mass and the per-sign scales that cancel it.

    ``net`` is ``(P - N) / (P + N)`` before balancing, where ``P`` and ``N`` are the summed positive and
    negative ``advantage x token weight`` of the whole round: the share of the round's push that raises
    (``net > 0``) or lowers (``net < 0``) the probability of the tokens the policy sampled.
    """

    net: float
    positive_scale: float = 1.0
    negative_scale: float = 1.0

    @property
    def scale(self) -> float:
        """The factor the heavier sign takes (1.0 when nothing is balanced)."""
        return min(self.positive_scale, self.negative_scale)

    def apply(self, advantages: torch.Tensor) -> torch.Tensor:
        return torch.where(advantages > 0, advantages * self.positive_scale, advantages * self.negative_scale)


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
    """Log a generation round's net token mass and return the balance to apply, or ``None`` when off.

    Under a token-sum loss a row pulls with its advantage times its trained token weight. The advantages
    of a group sum to zero, their token-weighted sum does not: where failures run longer than solves the
    round pushes down the tokens the policy itself sampled, which flattens it (entropy rises), and where
    solves run longer it sharpens it. The balance scales the heavier sign down, never up, until that net
    push is zero, keeping every row's sign and its order within its sign; its scale falls continuously to 0
    as the lighter side's mass does, so a round with mass on one sign only trains nothing on its advantages.
    The masses are summed over every rank (the gather is collective), so all ranks take the same scales.

    A token weighs what the policy gradient multiplies it by: its place in the loss (``loss_mask``, every
    drop already in it) times its truncated, masked IS ratio (``None`` when the loss applies none). The
    net share is logged whether or not the balance is on, as the early sign of an entropy drift.

    ``negative_only`` flags the rows that train only on a negative advantage, and their share of the
    round's trained mass is logged too, after the balance when it is on. The balance nets the whole round
    to zero, so under it that share is the net push left on every other row, raising the tokens it sampled.
    """
    weights = (loss_mask if is_ratio is None else loss_mask * is_ratio).sum(dim=1)
    local = _signed_masses(advantages, weights)
    if negative_only is not None:
        flagged = negative_only.to(advantages.device)
        local = torch.cat([local, _signed_masses(advantages[flagged], weights[flagged])[1:]])
    masses = _world_masses(local, gather_fn)
    result = _balance(masses[0], masses[1])
    metrics[NET_TOKEN_MASS_KEY].append(result.net)
    if negative_only is not None:
        scales = result if enabled else TokenMassBalance(net=result.net)
        trained = masses[0] * scales.positive_scale + masses[1] * scales.negative_scale
        metrics[NEGATIVE_ONLY_MASS_KEY].append(masses[2] * scales.negative_scale / trained if trained > 0 else 0.0)
    if not enabled:
        return None
    metrics[TOKEN_MASS_SCALE_KEY].append(result.scale)
    return result


def validate_token_mass_balance(args) -> None:
    """Refuse ``balance_token_mass`` under a TRL ``GRPOConfig`` whose loss does not pull with token mass.

    Two masks inside TRL's loss drop tokens after the balance has weighed them: the entropy quantile and
    the off-policy sequence mask, which drops negative sequences alone and so turns the net push positive.
    """
    if args.loss_type not in TOKEN_SUM_LOSS_TYPES:
        raise ValueError(
            f"balance_token_mass needs a loss whose loss tokens share one normalizer ({', '.join(TOKEN_SUM_LOSS_TYPES)}), "
            f"got loss_type={args.loss_type!r}: there a row does not pull with its token count, so the balance "
            "would add a push instead of cancelling one."
        )
    if args.top_entropy_quantile < 1.0:
        raise ValueError(
            f"balance_token_mass with top_entropy_quantile={args.top_entropy_quantile}: the entropy mask drops tokens "
            "inside the loss, after the balance has weighed them. Set top_entropy_quantile: 1.0."
        )
    if args.off_policy_mask_threshold is not None:
        raise ValueError(
            f"balance_token_mass with off_policy_mask_threshold={args.off_policy_mask_threshold}: the mask drops "
            "negative-advantage sequences inside the loss, after the balance has weighed them, so the round's net "
            "push turns positive. Unset one of them."
        )


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
