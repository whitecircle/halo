"""CPU tests for offline-GRPO MultiGroupSampler cross-rank determinism.

Under TP/CP the sibling ranks share one DP slice (same dp_rank) and MUST iterate it in the IDENTICAL
order — else they feed different sequences into the same sharded forward (silent divergence), and the
global-min truncation in get_train_dataloader would even keep DIFFERENT indices per sibling. Every
shuffle is seeded from a rank-independent ``random.Random(seed + epoch)`` rather than the process-
global ``random`` module (which is not synchronized across ranks). These tests fail on a
global-``random.shuffle`` implementation.
"""

import random

import pytest
from accelerate import PartialState

# MultiGroupSampler.__init__ logs via the accelerate logger, which requires an initialized state.
PartialState()

from src.trainers.grpo.mixins.dataloader import MultiGroupSampler


def _group_ids(num_groups=24, group_size=5):
    """Flat list mapping example index -> group id (group_size consecutive examples per group)."""
    return [g for g in range(num_groups) for _ in range(group_size)]


def test_global_random_state_does_not_perturb_order():
    """The KEY guard: consuming the process-global ``random`` stream between constructing two
    siblings must NOT change their order. Shuffling with the global module desyncs ranks that made
    an unequal number of prior ``random.*`` calls; the sampler's own seeded RNG is immune."""
    gids = _group_ids()
    random.seed(0)
    a = MultiGroupSampler(gids, rank=0, world_size=2, shuffle=True, seed=42)
    order_a = list(a)

    # Perturb the global RNG by a different amount (as a divergent sibling rank would).
    random.seed(123)
    for _ in range(17):
        random.random()
    b = MultiGroupSampler(gids, rank=0, world_size=2, shuffle=True, seed=42)
    order_b = list(b)

    assert a.indices_sequence == b.indices_sequence
    assert order_a == order_b, "global random state leaked into the sampler order (divergence bug)"


def test_different_ranks_get_disjoint_slices():
    """Distinct DP ranks must receive disjoint index slices (data parallelism), regardless of seed."""
    gids = _group_ids()
    r0 = MultiGroupSampler(gids, rank=0, world_size=2, shuffle=True, seed=42)
    r1 = MultiGroupSampler(gids, rank=1, world_size=2, shuffle=True, seed=42)
    s0, s1 = set(r0.indices_sequence), set(r1.indices_sequence)
    assert s0.isdisjoint(s1), "DP ranks overlap in their assigned indices"
    # Together they cover every example exactly once.
    assert s0 | s1 == set(range(len(gids)))


def test_set_epoch_reshuffles_consistently():
    """set_epoch changes the per-iteration order, but siblings at the SAME epoch stay identical."""
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
    # Same multiset of indices each epoch — only the order changes.
    assert sorted(epoch1_a) == sorted(epoch0)


def test_no_shuffle_is_deterministic_and_seed_independent():
    """shuffle=False (the eval path) ignores the seed and preserves the small-before-large order."""
    gids = _group_ids()
    a = MultiGroupSampler(gids, rank=0, world_size=2, shuffle=False, seed=1)
    b = MultiGroupSampler(gids, rank=0, world_size=2, shuffle=False, seed=999)
    assert list(a) == list(b) == a.indices_sequence


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
