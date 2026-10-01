"""Applying advantages and drops to the GRPO loss inputs (shared by online and environmental GRPO).

Once rewards exist, both trainers run the same steps: mask the rows of degenerate (all-equal-reward)
groups out of the loss, recompute the gathered-global DAPO normalizer from the post-drop loss mask, and
balance the step's token-weighted advantage mass. They differ only in framing (the online trainer
mutates TRL's result dict; the environmental trainer builds its tensors directly), so the tensor math
lives here.
"""

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass

import torch

from src.trainers.grpo.objective.advantages import degenerate_group_mask

# Metric keys both GRPO trainers log under.
DEGENERATE_GROUP_FRAC_KEY = "sampling/degenerate_group_frac"
NET_TOKEN_MASS_KEY = "advantage/net_token_mass"
TOKEN_MASS_SCALE_KEY = "advantage/token_mass_scale"
# TRL loss types whose every loss token of a step shares one normalizer, so a row pulls with its
# advantage times its trained token weight: the losses the token-mass balance is exact for. ``grpo`` /
# ``sapo`` average each completion over its own length, ``bnpo`` normalizes per micro-batch, and
# ``vespo`` weighs each sequence by its advantage's sign.
TOKEN_SUM_LOSS_TYPES = ("cispo", "dapo", "dr_grpo")


@dataclass(frozen=True)
class TokenMassBalance:
    """A step's token-weighted advantage mass and the per-sign scales that cancel it.

    ``net`` is ``(P - N) / (P + N)`` before balancing, where ``P`` and ``N`` are the summed positive and
    negative ``advantage x token weight`` of the whole step: the share of the step's push that raises
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


def token_mass_balance(
    advantages: torch.Tensor, token_weights: torch.Tensor, gather_fn: Callable[[torch.Tensor], torch.Tensor]
) -> TokenMassBalance:
    """The scales that shrink the heavier sign of a step's advantages until its token mass nets to zero.

    Under a token-sum loss a row pulls with its advantage times its trained token weight. The advantages
    of a group sum to zero, their token-weighted sum does not: where failures run longer than solves the
    step pushes down the tokens the policy itself sampled, which flattens it (entropy rises), and where
    solves run longer it sharpens it. Scaling down the heavier side, never up, removes that net push and
    keeps every row's sign and its order within its sign. ``advantages`` and ``token_weights`` are
    per-row and rank-local; the masses are summed over every rank (the gather is collective), so all
    ranks take the same scales. A step with only one sign has nothing to balance against.
    """
    weights = token_weights.to(advantages.dtype)
    local = torch.stack([(advantages.clamp_min(0) * weights).sum(), (advantages.clamp_max(0).neg() * weights).sum()])
    positive, negative = gather_fn(local.unsqueeze(0)).sum(dim=0).tolist()
    total = positive + negative
    if total <= 0:
        return TokenMassBalance(net=0.0)
    net = (positive - negative) / total
    if positive == 0 or negative == 0:
        return TokenMassBalance(net=net)
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
) -> TokenMassBalance | None:
    """Log a training step's net token mass and return the balance to apply, or ``None`` when off.

    A token weighs what the policy gradient multiplies it by: its place in the loss (``loss_mask``, every
    drop already in it) times its truncated, masked IS ratio (``None`` when the loss applies none). The
    net share is logged whether or not the balance is on, as the early sign of an entropy drift.
    """
    weights = loss_mask if is_ratio is None else loss_mask * is_ratio
    result = token_mass_balance(advantages, weights.sum(dim=1), gather_fn)
    metrics[NET_TOKEN_MASS_KEY].append(result.net)
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
            "negative-advantage sequences inside the loss, after the balance has weighed them, so the step's net "
            "push turns positive. Unset one of them."
        )


def degenerate_drop_rows(
    rewards: torch.Tensor, num_generations: int, valid_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, float]:
    """Per-row drop mask for all-equal-reward groups plus the dropped fraction (the logged metric).

    A zero-spread group's advantage is already 0 (no gradient), but its tokens would still swell the
    DAPO normalizer and dilute the groups that carry signal. ``valid_mask`` restricts degeneracy to
    valid members (see :func:`degenerate_group_mask`).
    """
    drop = degenerate_group_mask(rewards, num_generations, valid_mask=valid_mask)
    return drop, drop.float().mean().item()


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
