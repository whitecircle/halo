#!/usr/bin/env python
"""A GRPO group whose members tie on the environment's reward leaves the loss.

The environmental trainer charges a reasoning-length price per episode on top of the reward the
environment settled, so the totals it trains on almost never tie exactly: judged on them, an
all-solved group would train on the price alone. Degeneracy is judged on each episode's settled
environment reward instead, which the drop reads off the rollouts themselves — every other contrast
the environment prices (a tool error, a cut turn, a resubmission) still keeps a group alive.

    python tests/cpu/grpo/test_env_degenerate_groups.py
"""

import pytest
import torch

from tests.cpu.grpo.test_grpo_advantage_application_equivalence import _env_application


def _narrowed(env_rewards: list[float]):
    """Drive the real narrow phase over two groups of three single-row trajectories settled at
    ``env_rewards``; returns per-trajectory kept-token counts, the normalizer and the dropped share."""
    _adv, comp, _tool, num_items, frac, _metrics = _env_application(
        torch.zeros(6),
        torch.tensor(env_rewards),
        num_generations=3,
        truncated=torch.zeros(6, dtype=torch.bool),
        completion_mask=torch.ones(6, 4, dtype=torch.bool),
        tool_mask=torch.ones(6, 4, dtype=torch.bool),
        turns_per_traj=[1] * 6,
        num_dummy_rows=0,
        train_on_sampled_tokens=True,
        drop_degenerate_groups=True,
        mask_truncated_completions=False,
    )
    return comp.sum(dim=1).tolist(), num_items.item(), frac


def test_a_group_tied_on_the_environment_reward_leaves_the_loss():
    # Group A: three first-try solves (the trainer's length price, applied later, would split them).
    # Group B: one solve in three.
    kept, num_items, frac = _narrowed([1.1, 1.1, 1.1, 1.1, 0.1, 0.1])
    assert kept == [0, 0, 0, 4, 4, 4]
    assert (num_items, frac) == (12, pytest.approx(0.5))


def test_a_contrast_the_environment_prices_keeps_the_group():
    # All three solve, one after a resubmission: the price is a real contrast between them.
    kept, _, frac = _narrowed([1.1, 1.1, 0.9, 0.1, 0.1, 0.1])
    assert kept == [4, 4, 4, 0, 0, 0] and frac == pytest.approx(0.5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
