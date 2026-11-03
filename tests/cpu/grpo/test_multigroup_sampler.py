"""TP/ETP/CP siblings share one DP slice and must iterate it identically, or they feed different sequences into one
sharded forward. Every shuffle seeds from ``random.Random(seed + epoch)``, never the unsynchronized global module."""

import random

import pytest

from halo.trainers.grpo.offline.group_sampler import MultiGroupSampler


def _group_ids(num_groups=24, group_size=5):
    return [g for g in range(num_groups) for _ in range(group_size)]


def test_global_random_state_does_not_perturb_order():
    """Shuffling with the global module desyncs ranks that made an unequal number of prior ``random.*`` calls."""
    gids = _group_ids()
    random.seed(0)
    a = MultiGroupSampler(gids, rank=0, world_size=2, shuffle=True, seed=42)
    order_a = list(a)

    random.seed(123)
    for _ in range(17):
        random.random()
    b = MultiGroupSampler(gids, rank=0, world_size=2, shuffle=True, seed=42)
    order_b = list(b)

    assert a.indices_sequence == b.indices_sequence
    assert order_a == order_b, "global random state leaked into the sampler order (divergence bug)"


def test_different_ranks_get_disjoint_slices():
    gids = _group_ids()
    r0 = MultiGroupSampler(gids, rank=0, world_size=2, shuffle=True, seed=42)
    r1 = MultiGroupSampler(gids, rank=1, world_size=2, shuffle=True, seed=42)
    s0, s1 = set(r0.indices_sequence), set(r1.indices_sequence)
    assert s0.isdisjoint(s1), "DP ranks overlap in their assigned indices"
    assert s0 | s1 == set(range(len(gids)))


def test_set_epoch_reshuffles_consistently():
    gids = _group_ids()
    a = MultiGroupSampler(gids, rank=0, world_size=2, shuffle=True, seed=42)
    b = MultiGroupSampler(gids, rank=0, world_size=2, shuffle=True, seed=42)

    epoch0 = list(a)
    a.set_epoch(1)
    b.set_epoch(1)
    epoch1_a = list(a)
    epoch1_b = list(b)

    assert epoch1_a == epoch1_b, "siblings diverged at epoch 1"
    assert epoch1_a != epoch0, "set_epoch did not change the iteration order"
    assert sorted(epoch1_a) == sorted(epoch0)


def test_no_shuffle_is_deterministic_and_seed_independent():
    """``shuffle=False`` is the eval path."""
    gids = _group_ids()
    a = MultiGroupSampler(gids, rank=0, world_size=2, shuffle=False, seed=1)
    b = MultiGroupSampler(gids, rank=0, world_size=2, shuffle=False, seed=999)
    assert list(a) == list(b) == a.indices_sequence


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
