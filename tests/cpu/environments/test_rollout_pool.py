#!/usr/bin/env python
"""How the actor budget and the in-flight cap resolve per rollout rank, on a local Ray (budget per node) and a shared
cluster (budget cluster-wide), from one rank to 512."""

import pytest

from halo.environments.rollouts.rollout_pool import MIN_ACTORS_PER_RANK, resolve_rollout_pool


def _resolve(*, ranks, address, workers=256, per_node=8, rows=24, eval_rows=None, cap=None):
    return resolve_rollout_pool(
        num_workers=workers,
        shared_cluster=address,
        rollout_ranks=ranks,
        rollout_ranks_per_node=min(ranks, per_node),
        rows_per_round=rows,
        eval_rows=eval_rows,
        max_concurrent_rollouts=cap,
    )


@pytest.mark.parametrize(
    ("ranks", "local_actors", "shared_actors"),
    [(1, 256, 256), (8, 32, 32), (64, 32, 4), (512, 32, 1)],
)
def test_the_actor_split_per_rank(ranks, local_actors, shared_actors):
    assert _resolve(ranks=ranks, address=False).actors_per_rank == local_actors
    assert _resolve(ranks=ranks, address=True).actors_per_rank == shared_actors


def test_a_local_pool_does_not_grow_with_the_node_count():
    """The budget is per node, so 64 nodes hold what one does."""
    assert _resolve(ranks=512, address=False).actors_per_rank == _resolve(ranks=8, address=False).actors_per_rank


def test_the_default_cap_is_one_wave_of_the_round_not_a_fixed_multiple_of_the_actors():
    shared_at_512 = _resolve(ranks=512, address=True, rows=64)
    assert shared_at_512.actors_per_rank == 1
    assert shared_at_512.max_concurrent == 64, "1 actor must not cap a 64-row round at 4 episodes"
    assert _resolve(ranks=8, address=False, rows=1024).max_concurrent == 1024
    assert _resolve(ranks=8, address=False, workers=64, rows=24).max_concurrent == 24


def test_the_default_cap_holds_the_larger_eval_round():
    assert _resolve(ranks=8, address=False, workers=64, rows=24, eval_rows=34).max_concurrent == 34
    assert _resolve(ranks=8, address=False, workers=64, rows=24, eval_rows=8).max_concurrent == 24


def test_an_explicit_cap_is_kept_and_a_round_over_it_warns_with_its_waves():
    plan = _resolve(ranks=8, address=False, workers=64, rows=24, cap=10)
    assert plan.max_concurrent == 10
    assert any("3 waves" in w for w in plan.warnings)


def test_an_explicit_cap_below_the_actors_is_raised_to_them():
    plan = _resolve(ranks=8, address=False, rows=4, workers=64, cap=2)
    assert (plan.actors_per_rank, plan.max_concurrent) == (8, 8)
    assert any("raised to 8" in w for w in plan.warnings)


@pytest.mark.parametrize(("ranks", "warns"), [(8, False), (64, False), (65, True), (512, True)])
def test_a_shared_split_that_leaves_a_rank_thin_warns(ranks, warns):
    plan = _resolve(ranks=ranks, address=True, workers=256)
    assert bool(plan.warnings) is warns
    if warns:
        assert str(MIN_ACTORS_PER_RANK * ranks) in plan.warnings[0], "the warning names the budget that would do"


def test_a_local_split_over_many_ranks_per_node_warns():
    assert _resolve(ranks=8, address=False, workers=16).warnings


def test_a_rank_with_fewer_rows_than_the_minimum_is_not_called_thin():
    assert not _resolve(ranks=8, address=False, workers=8, rows=1).warnings


def test_a_default_single_node_run_warns_of_nothing():
    plan = _resolve(ranks=8, address=False)
    assert plan.warnings == ()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
