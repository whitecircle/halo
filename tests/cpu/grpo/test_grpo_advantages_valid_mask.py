#!/usr/bin/env python
"""group_relative_advantages: an infra-errored rollout's placeholder 0.0 must not drag its group's baseline down and
inflate its valid siblings' advantages."""

import pytest
import torch

from halo.trainers.grpo.objective.advantages import group_relative_advantages


def test_default_no_mask_unchanged():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    adv = group_relative_advantages(rewards, num_generations=4, scale_rewards="none")
    assert torch.allclose(adv, rewards - rewards.mean())


def test_errored_member_excluded_from_baseline():
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0])
    valid = torch.tensor([True, True, True, False])

    poisoned = group_relative_advantages(rewards, 4, "none")  # baseline = 0.75 (0.0 included)
    fixed = group_relative_advantages(rewards, 4, "none", valid_mask=valid)  # baseline = 1.0 (valid only)

    assert torch.allclose(fixed[:3], torch.zeros(3))
    assert (fixed[:3] < poisoned[:3]).all()
    assert abs(fixed[0].item() - (1.0 - 1.0)) < 1e-6


def test_all_errored_group_falls_back_to_plain_mean():
    rewards = torch.tensor([0.0, 0.0])
    valid = torch.tensor([False, False])
    adv = group_relative_advantages(rewards, 2, "none", valid_mask=valid)
    # Dividing by an empty count would give NaN.
    assert torch.equal(adv, torch.zeros(2))


def test_partial_valid_across_two_groups():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    valid = torch.tensor([True, False, True, True])
    adv = group_relative_advantages(rewards, 2, "none", valid_mask=valid)
    # Group A baseline over valid = 1.0 → [0.0, -1.0]; Group B baseline = 0.5 → [0.5, -0.5].
    assert torch.allclose(adv, torch.tensor([0.0, -1.0, 0.5, -0.5]))


def test_group_std_excludes_invalid_members():
    """The scale divisor honors valid_mask like the baseline."""
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    valid = torch.tensor([True, True, True, False])
    adv = group_relative_advantages(rewards, 4, "group", valid_mask=valid)
    # Valid-only: baseline = mean([1,1,0]) = 2/3; std = std([1,1,0], unbiased) = sqrt(1/3).
    expected_std = torch.tensor([1.0, 1.0, 0.0]).std()
    expected = (torch.tensor([1.0, 1.0, 0.0, 0.0]) - 2.0 / 3.0) / (expected_std + 1e-4)
    assert torch.allclose(adv[:3], expected[:3], atol=1e-4), f"{adv=} vs {expected=}"
    all_valid = group_relative_advantages(rewards, 4, "group", valid_mask=torch.ones(4, dtype=torch.bool))
    plain = group_relative_advantages(rewards, 4, "group")
    assert torch.allclose(all_valid, plain)


def test_batch_std_excludes_invalid_members():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    valid = torch.tensor([True, True, True, False])
    adv = group_relative_advantages(rewards, 2, "batch", valid_mask=valid, already_gathered=True)
    expected_std = torch.tensor([1.0, 0.0, 1.0]).std()
    # Group A (both valid): baseline 0.5 → ±0.5 / std; Group B: baseline over valid = 1.0.
    expected = torch.tensor([0.5, -0.5, 0.0, -1.0]) / (expected_std + 1e-4)
    assert torch.allclose(adv, expected, atol=1e-4), f"{adv=} vs {expected=}"


def test_group_std_singleton_valid_member_is_safe():
    # The unbiased std over a single element is NaN unless the divisor is guarded.
    rewards = torch.tensor([1.0, 0.0])
    valid = torch.tensor([True, False])
    adv = group_relative_advantages(rewards, 2, "group", valid_mask=valid)
    assert torch.isfinite(adv).all()
    assert adv[0].abs() < 1e-6


def test_std_floor_caps_degenerate_batch_amplification():
    # Without the floor this 0.02-spread batch amplifies its own noise ~50x.
    rewards = torch.tensor([-0.10, -0.12, -0.10, -0.12])
    unfloored = group_relative_advantages(rewards, 2, "batch", already_gathered=True)
    floored = group_relative_advantages(rewards, 2, "batch", already_gathered=True, std_floor=0.2)
    expected = (rewards - rewards.view(-1, 2).mean(dim=1, keepdim=True).expand(-1, 2).flatten()) / (0.2 + 1e-4)
    assert torch.allclose(floored, expected, atol=1e-5), f"{floored=} vs {expected=}"
    assert floored.abs().max() < unfloored.abs().max() / 5


def test_std_floor_inert_above_floor():
    rewards = torch.tensor([1.0, 0.0, 0.5, -0.5])
    for mode in ("batch", "group"):
        base = group_relative_advantages(rewards, 2, mode, already_gathered=True)
        floored = group_relative_advantages(rewards, 2, mode, already_gathered=True, std_floor=0.05)
        assert torch.allclose(base, floored), f"{mode}: {base=} vs {floored=}"


def test_std_floor_group_mode():
    rewards = torch.tensor([0.01, -0.01, 1.0, -1.0])
    floored = group_relative_advantages(rewards, 2, "group", std_floor=0.5)
    healthy_std = torch.tensor([1.0, -1.0]).std()
    assert torch.allclose(floored[:2], torch.tensor([0.01, -0.01]) / (0.5 + 1e-4), atol=1e-5)
    assert torch.allclose(floored[2:], torch.tensor([1.0, -1.0]) / (healthy_std + 1e-4), atol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
