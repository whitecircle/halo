#!/usr/bin/env python
"""``tests.common.parallel_shape.expert_tp_group_problems`` names an expert-TP layout that does not tile the world.

The EP-modes GPU suite gates its pure-ETP rows on this verdict over every rank's reported group, and runs only on
the GPU tiers; with several DP replicas the loss and gradient comparisons on an identical batch cannot tell which
ranks pair up, or which batch each partner would read. These pins hold the verdict to the layouts that are valid
(contiguous or strided pairs whose partners read one batch) and to each way a layout can be wrong, and the gather
to one view per rank, a rank without a group included.

Run: python tests/cpu/conventions/test_expert_tp_group_layout.py
"""

import types

import pytest
import torch.distributed as dist

from tests.common.parallel_shape import ExpertTPMembership, expert_tp_group_problems, expert_tp_group_views


def _views(
    groups: dict[int, tuple[int, ...] | None], shard_ranks: dict[int, int], dp_ranks: dict[int, int] | None = None
) -> list[ExpertTPMembership]:
    """Per global rank, in order, the group it reports, the expert-TP rank it holds and its DP rank (``rank //
    2``, contiguous pairs' layout, unless given)."""
    dp_ranks = dp_ranks or {rank: rank // 2 for rank in groups}
    return [ExpertTPMembership(groups[rank], shard_ranks[rank], dp_ranks[rank]) for rank in sorted(groups)]


def _pairs(*pairs: tuple[int, int]) -> list[ExpertTPMembership]:
    """Every rank in its pair, holding its position in the pair as its expert-TP rank and the pair's index as its DP
    rank."""
    groups = {rank: pair for pair in pairs for rank in pair}
    shard_ranks = {rank: pair.index(rank) for pair in pairs for rank in pair}
    dp_ranks = {rank: index for index, pair in enumerate(pairs) for rank in pair}
    return _views(groups, shard_ranks, dp_ranks)


@pytest.mark.parametrize(
    "views",
    [
        pytest.param(_pairs((0, 1), (2, 3)), id="contiguous_pairs_over_two_replicas"),
        pytest.param(_pairs((0, 2), (1, 3)), id="strided_pairs"),
        pytest.param(_pairs((0, 1)), id="one_pair_on_two_ranks"),
    ],
)
def test_pairs_that_tile_the_world_pass(views):
    assert expert_tp_group_problems(views, 2) == []


def test_a_member_listing_its_group_out_of_order_still_agrees():
    views = _pairs((0, 1), (2, 3))
    views[1] = views[1]._replace(members=(1, 0))
    assert expert_tp_group_problems(views, 2) == []


@pytest.mark.parametrize(
    ("views", "named"),
    [
        pytest.param(
            _views({rank: (0, 1, 2, 3) for rank in range(4)}, {0: 0, 1: 1, 2: 0, 3: 1}),
            "has 4 ranks, not 2",
            id="a_group_spanning_both_replicas",
        ),
        pytest.param(
            _views({0: (0, 1), 1: (0, 1), 2: (0, 3), 3: (0, 3)}, {0: 0, 1: 1, 2: 0, 3: 1}),
            "rank 2 is not in its own expert-TP group",
            id="a_rank_outside_its_own_group",
        ),
        pytest.param(
            _views({0: (0, 1), 1: (0, 1), 2: (2, 3), 3: (2, 3)}, {0: 0, 1: 0, 2: 0, 3: 1}),
            "holds expert-TP ranks [0, 0]",
            id="a_shard_index_held_twice",
        ),
        pytest.param(
            _views({0: (0, 1), 1: (1, 2), 2: (1, 2), 3: (2, 3)}, {0: 0, 1: 1, 2: 0, 3: 1}),
            "do not report rank 0's expert-TP group",
            id="members_disagreeing_on_the_group",
        ),
        pytest.param(
            _views({0: (0, 1), 1: None, 2: (2, 3), 3: (2, 3)}, {0: 0, 1: 0, 2: 0, 3: 1}),
            "rank 1 reports no expert-TP group",
            id="a_rank_without_a_group",
        ),
        pytest.param(
            _views({0: (0, 4), 1: (1, 2)}, {0: 0, 1: 0}),
            "names a rank outside the world",
            id="a_group_naming_a_rank_past_the_world",
        ),
        pytest.param(
            _views({0: (0, 2), 1: (1, 3), 2: (0, 2), 3: (1, 3)}, {0: 0, 1: 0, 2: 1, 3: 1}),
            "reads the batches of DP ranks [0, 1]",
            id="strided_partners_reading_different_batches",
        ),
    ],
)
def test_a_layout_that_does_not_tile_the_world_is_named(views, named):
    problems = expert_tp_group_problems(views, 2)
    assert any(named in problem for problem in problems), problems


def test_the_gather_reports_every_rank_once_with_or_without_a_group(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path}/pg_init")
    try:
        grouped = types.SimpleNamespace(expert_tp_group=dist.new_group([0]), expert_tp_rank=0)
        ungrouped = types.SimpleNamespace(expert_tp_group=None, expert_tp_rank=0)
        assert expert_tp_group_views(grouped, 3) == [ExpertTPMembership((0,), 0, 3)]
        assert expert_tp_group_views(ungrouped, 0) == [ExpertTPMembership(None, 0, 0)]
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
