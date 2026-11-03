#!/usr/bin/env python
"""A GRPO group tied on the environment's reward leaves the loss. Judged on the totals instead, an all-solved group
would train on its reasoning-floor shortfall alone."""

import pytest
import torch

from tests.cpu.grpo.test_grpo_advantage_application_equivalence import _env_application


def _narrowed(env_rewards: list[float]):
    """The real narrow phase over two groups of three single-row trajectories."""
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
