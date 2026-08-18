#!/usr/bin/env python
"""Equivalence pins for the GRPO advantage-application seam (online + environmental).

Both GRPO trainers apply the same post-reward bookkeeping: drop degenerate (all-equal-reward)
groups out of the loss masks, recompute the gathered-global DAPO normalizer from the post-drop
loss mask (``clamp(min=1)``), and — on the environmental side — expand per-trajectory tensors to
per-turn rows (``turns_per_traj`` + zero-filled dummy rows). These tests pin the exact numerics of
that seam on fixed reward/mask tensors:

* the ONLINE path drives the real ``DistributedGRPOTrainer._apply_degenerate_group_drop`` hook
  (with ``_gathered_rewards`` / ``_local_slice`` bound, gather = identity);
* the ENVIRONMENTAL path drives the real ``_narrow_masks_and_normalizer`` (and ``_compute_advantages``)
  on a fake self — a transcription of that block cannot catch it drifting away from the transcription;
* every scenario asserts hand-derived constants (masks, num_items, advantages), so any refactor of
  the shared seam must reproduce them bit-for-bit.

    python tests/cpu/grpo/test_grpo_advantage_application_equivalence.py
"""

import types
from collections import defaultdict

import pytest
import torch

from src.trainers.grpo.environmental import (
    BatchRows,
    DistributedAsyncEnvironmentalGRPOTrainer,
    rollout_valid_mask,
)
from src.trainers.grpo.online import DistributedGRPOTrainer

# --- Online trainer: fake-self driving the real _apply_degenerate_group_drop hook ---


def _online_fake_self(rewards_per_func: torch.Tensor, num_generations: int, drop: bool = True):
    """Stand-in exposing exactly what the online degenerate-drop hook reads.

    ``accelerator.gather`` is identity (single process, process_index 0); the real
    ``_gathered_rewards`` / ``_local_slice`` helpers are bound so the production
    stash-aggregation and group-integrity code runs, not a mock.
    """
    me = types.SimpleNamespace(
        _drop_degenerate_groups=drop,
        _last_rewards_per_func=rewards_per_func,
        reward_weights=torch.ones(rewards_per_func.shape[1]),
        num_generations=num_generations,
        model=types.SimpleNamespace(training=True),
        args=types.SimpleNamespace(multi_objective_aggregation="sum_then_normalize"),
        accelerator=types.SimpleNamespace(process_index=0, gather=lambda x: x),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )
    me._gathered_rewards = types.MethodType(DistributedGRPOTrainer._gathered_rewards, me)
    me._local_slice = types.MethodType(DistributedGRPOTrainer._local_slice, me)
    return me


def _mask_of_lengths(lengths: list[int], max_len: int) -> torch.Tensor:
    mask = torch.zeros(len(lengths), max_len, dtype=torch.long)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask


def test_online_drop_pins_masks_and_num_items():
    # 2 groups of 3: group A mixed rewards (kept), group B all-equal (dropped).
    rewards_per_func = torch.tensor([[1.0], [0.0], [1.0], [0.5], [0.5], [0.5]])
    me = _online_fake_self(rewards_per_func, num_generations=3)
    result = {
        "completion_mask": _mask_of_lengths([4, 2, 3, 5, 1, 2], max_len=5),
        "num_items_in_batch": torch.tensor(17),  # stale value the hook must recompute
    }
    DistributedGRPOTrainer._apply_degenerate_group_drop(me, result)

    expected_mask = _mask_of_lengths([4, 2, 3, 0, 0, 0], max_len=5)
    assert torch.equal(result["completion_mask"], expected_mask)
    assert result["num_items_in_batch"].item() == 9  # 4 + 2 + 3 surviving loss tokens
    assert me._metrics["train"]["sampling/degenerate_group_frac"] == [0.5]


def test_online_drop_with_tool_mask_narrows_both_and_normalizes_by_product():
    # TRL counts num_items_in_batch over completion_mask * tool_mask, so the hook must narrow both
    # masks and renormalize from the product.
    rewards_per_func = torch.tensor([[1.0], [0.0], [1.0], [0.5], [0.5], [0.5]])
    me = _online_fake_self(rewards_per_func, num_generations=3)
    tool_mask = torch.ones(6, 5, dtype=torch.long)
    tool_mask[0, 1] = 0  # tool-output token inside a KEPT row's completion span
    result = {
        "completion_mask": _mask_of_lengths([4, 2, 3, 5, 1, 2], max_len=5),
        "tool_mask": tool_mask,
        "num_items_in_batch": torch.tensor(17),
    }
    DistributedGRPOTrainer._apply_degenerate_group_drop(me, result)

    expected_comp = _mask_of_lengths([4, 2, 3, 0, 0, 0], max_len=5)
    assert torch.equal(result["completion_mask"], expected_comp)
    expected_tool = torch.ones(6, 5, dtype=torch.long)
    expected_tool[0, 1] = 0
    expected_tool[3:] = 0
    assert torch.equal(result["tool_mask"], expected_tool)
    # (4 - 1) + 2 + 3 over completion ∧ tool — 9 would mean completion_mask alone.
    assert result["num_items_in_batch"].item() == 8


def test_online_drop_all_degenerate_clamps_num_items_to_one():
    rewards_per_func = torch.tensor([[0.5], [0.5], [0.5], [0.5]])
    me = _online_fake_self(rewards_per_func, num_generations=4)
    result = {
        "completion_mask": _mask_of_lengths([3, 3, 3, 3], max_len=4),
        "num_items_in_batch": torch.tensor(12),
    }
    DistributedGRPOTrainer._apply_degenerate_group_drop(me, result)

    assert result["completion_mask"].sum().item() == 0
    assert result["num_items_in_batch"].item() == 1  # clamp(min=1) guards the 0/0 loss
    assert me._metrics["train"]["sampling/degenerate_group_frac"] == [1.0]


def test_online_drop_no_degenerate_groups_recomputes_but_keeps_masks():
    rewards_per_func = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
    me = _online_fake_self(rewards_per_func, num_generations=2)
    before = _mask_of_lengths([2, 4, 1, 3], max_len=4)
    result = {"completion_mask": before.clone(), "num_items_in_batch": torch.tensor(99)}
    DistributedGRPOTrainer._apply_degenerate_group_drop(me, result)

    assert torch.equal(result["completion_mask"], before)
    # The normalizer recompute is unconditional (rank-uniform gate → collective-safe).
    assert result["num_items_in_batch"].item() == 10
    assert me._metrics["train"]["sampling/degenerate_group_frac"] == [0.0]


def test_online_drop_disabled_is_noop():
    rewards_per_func = torch.tensor([[0.5], [0.5]])
    me = _online_fake_self(rewards_per_func, num_generations=2, drop=False)
    before = _mask_of_lengths([2, 2], max_len=3)
    result = {"completion_mask": before.clone(), "num_items_in_batch": torch.tensor(4)}
    DistributedGRPOTrainer._apply_degenerate_group_drop(me, result)
    assert torch.equal(result["completion_mask"], before)
    assert result["num_items_in_batch"].item() == 4


# --- Environmental trainer: the REAL _narrow_masks_and_normalizer on a fake self ---


def _rollouts(truncated: list[bool], valid: list[bool] | None = None):
    """Rollout stand-ins exposing exactly what the narrow phase reads off them."""
    valid = [True] * len(truncated) if valid is None else valid
    return [
        types.SimpleNamespace(
            trajectory=types.SimpleNamespace(truncated=t, episode_invalid=not v),
            error=None,
        )
        for t, v in zip(truncated, valid, strict=True)
    ]


def _env_application(
    traj_advantages: torch.Tensor,
    rewards: torch.Tensor,
    *,
    num_generations: int,
    truncated: torch.Tensor,
    completion_mask: torch.Tensor,
    tool_mask: torch.Tensor,
    turns_per_traj: list[int],
    num_dummy_rows: int,
    train_on_sampled_tokens: bool,
    drop_degenerate_groups: bool,
    mask_truncated_completions: bool,
    valid: list[bool] | None = None,
):
    """Drive the REAL ``_narrow_masks_and_normalizer`` (plus the caller's own row expansion) on a
    fake self, so the constants below pin production rather than a transcription of it.

    Returns ``(local_advantages, completion_mask, tool_mask, num_items_in_batch, degenerate_frac)``;
    ``gather`` is identity (single process).
    """
    rollout_results = _rollouts(truncated.tolist(), valid)
    rows = BatchRows(rollout_results, turns_per_traj, num_dummy_rows, train_on_sampled_tokens)
    me = types.SimpleNamespace(
        drop_degenerate_groups=drop_degenerate_groups,
        args=types.SimpleNamespace(mask_truncated_completions=mask_truncated_completions),
        accelerator=types.SimpleNamespace(gather=lambda x: x),
        model=types.SimpleNamespace(training=True),
        _empty_rollout_steps=0,
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )
    me._check_step_has_valid_episodes = types.MethodType(
        DistributedAsyncEnvironmentalGRPOTrainer._check_step_has_valid_episodes, me
    )

    local_advantages = rows.to_rows(traj_advantages)
    comp, tool, _loss_mask, num_items = DistributedAsyncEnvironmentalGRPOTrainer._narrow_masks_and_normalizer(
        me,
        rows,
        rewards,
        rollout_valid_mask(rollout_results, rewards.device),
        num_generations,
        completion_mask,
        tool_mask,
        rewards.device,
        "train",
    )
    fracs = me._metrics["train"]["sampling/degenerate_group_frac"]
    return local_advantages, comp, tool, num_items, (fracs[0] if fracs else None)


def _env_scenario():
    """6 trajectories (2 groups of 3), multi-turn rows + dummy padding rows.

    Group A rewards [1, 0, 1] (kept), group B [0.5, 0.5, 0.5] (degenerate → dropped);
    trajectory 1 truncated (→ dropped). turns_per_traj [1, 2, 1, 1, 1, 1] → 7 rows, +2 dummy = 9.
    """
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.5, 0.5, 0.5])
    truncated = torch.tensor([False, True, False, False, False, False])
    turns_per_traj = [1, 2, 1, 1, 1, 1]
    completion_mask = _mask_of_lengths([3, 2, 2, 4, 5, 1, 2, 3, 3], max_len=5).bool()
    tool_mask = torch.ones(9, 5, dtype=torch.bool)
    tool_mask[0, 1] = False
    return rewards, truncated, turns_per_traj, completion_mask, tool_mask


def _env_compute_advantages(rewards, num_generations, valid_mask=None):
    """Drive the REAL env ``_compute_advantages`` (group_relative_advantages framing) on a fake self."""
    me = types.SimpleNamespace(
        args=types.SimpleNamespace(scale_rewards="none"),
        _advantage_shaping=None,
        _scale_rewards_std_floor=0.0,
    )
    return DistributedAsyncEnvironmentalGRPOTrainer._compute_advantages(
        me, rewards, num_generations, valid_mask=valid_mask
    )


def test_env_application_pins_advantages_masks_and_num_items():
    rewards, truncated, turns_per_traj, completion_mask, tool_mask = _env_scenario()
    traj_advantages = _env_compute_advantages(rewards, num_generations=3)

    # Group-mean baseline, scaling off: A mean=2/3 → [1/3, -2/3, 1/3]; B mean=0.5 → zeros.
    expected_traj = torch.tensor([1 / 3, -2 / 3, 1 / 3, 0.0, 0.0, 0.0])
    assert torch.allclose(traj_advantages, expected_traj, atol=1e-6)

    adv, comp, tool, num_items, frac = _env_application(
        traj_advantages,
        rewards,
        num_generations=3,
        truncated=truncated,
        completion_mask=completion_mask,
        tool_mask=tool_mask,
        turns_per_traj=turns_per_traj,
        num_dummy_rows=2,
        train_on_sampled_tokens=True,
        drop_degenerate_groups=True,
        mask_truncated_completions=True,
    )

    # Advantages expanded to rows (traj 1 spans rows 1-2) + two zero dummy rows.
    expected_adv = torch.tensor([1 / 3, -2 / 3, -2 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(adv, expected_adv, atol=1e-6)

    # Dropped rows: traj 1 (truncated → rows 1, 2) and trajs 3-5 (degenerate → rows 4, 5, 6).
    expected_comp = _mask_of_lengths([3, 0, 0, 4, 0, 0, 0, 3, 3], max_len=5).bool()
    assert torch.equal(comp, expected_comp)
    expected_tool = torch.ones(9, 5, dtype=torch.bool)
    expected_tool[0, 1] = False
    expected_tool[[1, 2, 4, 5, 6]] = False
    assert torch.equal(tool, expected_tool)

    # Loss tokens: row 0 = 3 comp tokens minus the tool-masked one = 2; row 3 = 4; rows 7, 8 = 3 + 3.
    assert num_items.item() == 12
    assert frac == 0.5


def test_env_application_all_dropped_clamps_num_items():
    rewards = torch.tensor([0.7, 0.7, 0.7])
    traj_advantages = _env_compute_advantages(rewards, num_generations=3)
    assert torch.equal(traj_advantages, torch.zeros(3))

    adv, comp, _tool, num_items, frac = _env_application(
        traj_advantages,
        rewards,
        num_generations=3,
        truncated=torch.zeros(3, dtype=torch.bool),
        completion_mask=_mask_of_lengths([2, 3, 4], max_len=4).bool(),
        tool_mask=torch.ones(3, 4, dtype=torch.bool),
        turns_per_traj=[1, 1, 1],
        num_dummy_rows=0,
        train_on_sampled_tokens=False,
        drop_degenerate_groups=True,
        mask_truncated_completions=False,
    )
    assert torch.equal(adv, torch.zeros(3))
    assert comp.sum() == 0
    assert num_items.item() == 1
    assert frac == 1.0


def test_env_application_no_drops_keeps_masks_and_counts_loss_tokens():
    rewards = torch.tensor([1.0, 0.0])
    traj_advantages = _env_compute_advantages(rewards, num_generations=2)
    assert torch.allclose(traj_advantages, torch.tensor([0.5, -0.5]))

    completion_mask = _mask_of_lengths([3, 2], max_len=4).bool()
    tool_mask = torch.ones(2, 4, dtype=torch.bool)
    adv, comp, tool, num_items, frac = _env_application(
        traj_advantages,
        rewards,
        num_generations=2,
        truncated=torch.zeros(2, dtype=torch.bool),
        completion_mask=completion_mask.clone(),
        tool_mask=tool_mask.clone(),
        turns_per_traj=[1, 1],
        num_dummy_rows=0,
        train_on_sampled_tokens=False,
        drop_degenerate_groups=True,
        mask_truncated_completions=True,
    )
    assert torch.equal(adv, torch.tensor([0.5, -0.5]))
    assert torch.equal(comp, completion_mask)
    assert torch.equal(tool, tool_mask)
    assert num_items.item() == 5
    assert frac == 0.0


def test_an_invalid_episode_is_dropped_and_excluded_from_degeneracy():
    """The two things a transcription of this block kept missing: the drop set is SEEDED with the
    invalid episodes, and degeneracy is judged over the VALID members only.

    Rewards [1, 1, 0] with the last episode invalid: counting it would make the group non-degenerate
    (spread 1) and keep every row; excluding it leaves the two valid members tied, so the whole group
    drops. Its own rows go either way — invalid or degenerate.
    """
    rewards = torch.tensor([1.0, 1.0, 0.0])
    traj_advantages = _env_compute_advantages(rewards, num_generations=3, valid_mask=torch.tensor([True, True, False]))

    _adv, comp, _tool, num_items, frac = _env_application(
        traj_advantages,
        rewards,
        num_generations=3,
        truncated=torch.zeros(3, dtype=torch.bool),
        completion_mask=_mask_of_lengths([2, 3, 4], max_len=4).bool(),
        tool_mask=torch.ones(3, 4, dtype=torch.bool),
        turns_per_traj=[1, 1, 1],
        num_dummy_rows=0,
        train_on_sampled_tokens=False,
        drop_degenerate_groups=True,
        mask_truncated_completions=False,
        valid=[True, True, False],
    )
    assert comp.sum() == 0, "the two tied valid members are degenerate; the invalid one is dropped outright"
    assert num_items.item() == 1
    assert frac == 1.0


def test_shared_seam_traj_row_ids_dummy_fill():
    """expand_traj_to_rows with dummy_fill=-1 reproduces the traj_row_ids construction."""
    from src.trainers.grpo.objective.application import expand_traj_to_rows

    ids = expand_traj_to_rows(torch.arange(3), [2, 1, 3], num_dummy_rows=2, expand=True, dummy_fill=-1)
    assert torch.equal(ids, torch.tensor([0, 0, 1, 2, 2, 2, -1, -1]))
    ids = expand_traj_to_rows(torch.arange(3), [2, 1, 3], num_dummy_rows=0, expand=False, dummy_fill=-1)
    assert torch.equal(ids, torch.arange(3))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
