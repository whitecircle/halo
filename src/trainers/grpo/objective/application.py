"""Applying advantages and drops to the GRPO loss inputs (shared by online + environmental).

Once rewards exist, both GRPO trainers run the same application steps — mask the rows of degenerate
(all-equal-reward) groups out of the loss and recompute the gathered-global DAPO normalizer from the
post-drop loss mask — while framing them differently: the online trainer mutates TRL's result dict,
the environmental trainer builds its tensors directly (expanding per-trajectory values to per-turn
rows first). These pure tensor helpers keep the numerics identical across both.
"""

from collections.abc import Callable

import torch

from src.trainers.grpo.objective.advantages import degenerate_group_mask

# Metric key both GRPO trainers log the dropped-group fraction under.
DEGENERATE_GROUP_FRAC_KEY = "sampling/degenerate_group_frac"


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
    """Gathered-global DAPO loss normalizer: total LOSS tokens across ranks from the post-drop mask.

    A local count mis-scales loss/effective-LR (TRL divides the gathered total by num_processes);
    ``clamp(min=1)`` guards an all-degenerate global batch dividing by zero. The gather is
    collective — call on every rank (rank-uniform gate).
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
