#!/usr/bin/env python
"""Tests for RLRR relative-reward shaping (arXiv:2601.23058).

Run: python tests/cpu/grpo/test_relative_rewards.py
"""

import math

import numpy as np
import pytest

from src.args.mixins import RLRRConfig
from src.trainers.grpo.objective.relative_rewards import (
    dense_ranks,
    hierarchical_ranks,
    relative_advantages,
    relative_advantages_grouped,
    shape_rewards,
)


def test_dense_ranks_basic():
    keys = [(0, 0, -0.9), (0, 0, -0.1), (0, 0, -0.5)]
    ranks = dense_ranks(keys)
    # reward 0.9 (key -0.9) best -> rank 1; 0.5 -> 2; 0.1 -> 3
    assert list(ranks) == [1, 3, 2], list(ranks)


def test_dense_ranks_ties_share_rank():
    keys = [(0,), (0,), (1,)]
    ranks = dense_ranks(keys)
    assert list(ranks) == [1, 1, 2], list(ranks)
    assert ranks.max() == 2  # number of distinct keys


def test_rerank_correct_outranks_incorrect():
    cfg = RLRRConfig()
    ranks = hierarchical_ranks(rewards=[5.0, 0.1], correct=[False, True], lengths=None, config=cfg)
    assert ranks[1] < ranks[0], ranks


def test_rerank_shorter_correct_ranks_higher():
    cfg = RLRRConfig(lam=100.0)
    # equal reward; length bins 0 vs 5 decide
    ranks = hierarchical_ranks(rewards=[1.0, 1.0], correct=[True, True], lengths=[10, 550], config=cfg)
    assert ranks[0] < ranks[1], ranks


def test_rerank_reward_breaks_remaining_ties():
    cfg = RLRRConfig(lam=1000.0)
    # same length bin -> reward is the only remaining key
    ranks = hierarchical_ranks(rewards=[0.2, 0.9], correct=[True, True], lengths=[10, 10], config=cfg)
    assert ranks[1] < ranks[0], ranks


def test_rerank_length_disabled():
    cfg = RLRRConfig(length_rerank=False)
    ranks = hierarchical_ranks(rewards=[1.0, 1.0], correct=[True, True], lengths=[10, 9999], config=cfg)
    assert ranks[0] == ranks[1], ranks


def test_hrr_formula_matches_paper():
    cfg = RLRRConfig(mode="hrr", tau=0.1)
    ranks = np.array([1, 2, 3])
    rule = np.array([1.0, 1.0, 1.0])
    shaped = shape_rewards(ranks, rule, cfg)
    r_max = 3.0
    expected = rule + 0.1 * np.tanh(r_max / ranks - 1.0)
    assert np.allclose(shaped, expected), (shaped, expected)


def test_hrr_best_rank_gets_largest_boost():
    cfg = RLRRConfig(mode="hrr", tau=0.1)
    ranks = np.array([1, 2, 3])
    shaped = shape_rewards(ranks, np.ones(3), cfg)
    assert shaped[0] > shaped[1] > shaped[2]
    assert math.isclose(shaped[2], 1.0, abs_tol=1e-9)  # tanh(r_max/r_max - 1)=0


def test_hrr_preserves_correctness_ordering():
    # Small tau must not let a best-ranked incorrect (rule=0) outscore a worst-ranked correct.
    cfg = RLRRConfig(mode="hrr", tau=0.1)
    ranks = np.array([1, 2])
    rule = np.array([0.0, 1.0])
    shaped = shape_rewards(ranks, rule, cfg)
    assert shaped[1] > shaped[0], shaped


def test_prr_linear_map_to_unit_interval():
    cfg = RLRRConfig(mode="prr")
    ranks = np.array([1, 2, 3, 4])
    shaped = shape_rewards(ranks, np.zeros(4), cfg)
    assert math.isclose(shaped[0], 1.0)
    assert math.isclose(shaped[-1], 0.0)
    assert np.allclose(np.diff(shaped), -1.0 / 3.0)


def test_prr_single_rank_degenerate():
    cfg = RLRRConfig(mode="prr")
    ranks = np.array([1, 1, 1])
    shaped = shape_rewards(ranks, np.zeros(3), cfg)
    assert np.allclose(shaped, 0.0)


def test_advantage_centered_zero_mean():
    cfg = RLRRConfig(mode="prr", correctness_clip=False, correctness_threshold=0.0)
    adv = relative_advantages([0.1, 0.5, 0.9], cfg)
    assert abs(adv.mean()) < 1e-9, adv


def test_all_correct_group_still_has_signal():
    # The RLRR property: zero binary-reward variance, yet length re-ranking still yields signal.
    cfg = RLRRConfig(mode="hrr", lam=100.0, correctness_clip=False)
    adv = relative_advantages(
        [1.0, 1.0, 1.0],
        cfg,
        lengths=[10, 300, 600],  # distinct length bins -> distinct ranks
    )
    assert adv.std() > 0, adv
    assert adv[0] > adv[1] > adv[2], adv


def test_truly_identical_group_zero_advantage():
    # Tied rank must stay at zero advantage — the flip side of the test above (no fabricated signal).
    cfg = RLRRConfig(mode="hrr", lam=100.0, correctness_clip=False)
    adv = relative_advantages([1.0, 1.0, 1.0], cfg, lengths=[10, 10, 10])
    assert np.allclose(adv, 0.0), adv


def test_correctness_clip_floors_correct():
    # A last-ranked correct response earns a negative raw advantage; the clip floors it at xi_neg.
    cfg = RLRRConfig(mode="prr", xi_neg=-1e-3, xi_pos=1e-3, correctness_threshold=0.0)
    adv = relative_advantages([0.0, 1.0, 1.0, 1.0], cfg)
    assert (adv >= -1e-3 - 1e-12).all(), adv


def test_correctness_clip_caps_incorrect():
    cfg = RLRRConfig(mode="prr", xi_neg=-1e-3, xi_pos=1e-3, correctness_threshold=10.0)
    # idx 0 is incorrect yet best-ranked, so its raw advantage is positive; the cap holds it at xi_pos.
    adv = relative_advantages([9.0, 0.0, 0.0, 0.0], cfg)
    assert (adv <= 1e-3 + 1e-12).all(), adv


def test_correctness_clip_mixed_group():
    cfg = RLRRConfig(mode="hrr", tau=0.1, xi_neg=-1e-3, xi_pos=1e-3)
    # rewards straddle the 0.5 threshold: the first two are correct, the last two are not.
    adv = relative_advantages([1.0, 1.0, 0.0, 0.0], cfg, lengths=[10, 20, 30, 40])
    adv = np.asarray(adv)
    assert (adv[:2] >= -1e-3 - 1e-12).all(), adv
    assert (adv[2:] <= 1e-3 + 1e-12).all(), adv


def test_std_normalize_scales():
    cfg_raw = RLRRConfig(mode="prr", std_normalize=False, correctness_clip=False, correctness_threshold=0.0)
    cfg_std = RLRRConfig(mode="prr", std_normalize=True, correctness_clip=False, correctness_threshold=0.0)
    rewards = [0.1, 0.4, 0.6, 0.9]
    a_raw = relative_advantages(rewards, cfg_raw)
    a_std = relative_advantages(rewards, cfg_std)
    # 0.2 slack: 4 discrete ranks give a sample std only approximately 1.
    assert abs(a_std.std() - 1.0) < 0.2, a_std.std()
    assert not np.allclose(a_raw, a_std)


def test_correctness_from_threshold():
    cfg = RLRRConfig(mode="hrr", correctness_threshold=0.5, correctness_clip=True)
    # No explicit `correct` labels: the first two must be inferred correct from the 0.5 threshold.
    adv = relative_advantages([0.9, 0.8, 0.2, 0.1], cfg)
    adv = np.asarray(adv)
    assert (adv[:2] >= -1e-3 - 1e-12).all(), adv
    assert (adv[2:] <= 1e-3 + 1e-12).all(), adv


def test_grouped_matches_per_group():
    cfg = RLRRConfig(mode="hrr", lam=100.0)
    rewards = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    lengths = [10, 200, 5, 5, 50, 50]
    flat = relative_advantages_grouped(rewards, group_size=2, config=cfg, lengths=lengths)
    g0 = relative_advantages(rewards[:2], cfg, lengths=lengths[:2])
    g2 = relative_advantages(rewards[4:], cfg, lengths=lengths[4:])
    assert np.allclose(flat[:2], g0), (flat[:2], g0)
    assert np.allclose(flat[4:], g2), (flat[4:], g2)


def test_grouped_rejects_bad_size():
    cfg = RLRRConfig()
    try:
        relative_advantages_grouped([1.0, 2.0, 3.0], group_size=2, config=cfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError for indivisible group size")


def test_config_rejects_bad_mode():
    try:
        RLRRConfig(mode="bogus")  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad mode")


def test_config_rejects_bad_xi():
    try:
        RLRRConfig(xi_neg=1.0, xi_pos=-1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for xi_neg > xi_pos")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
