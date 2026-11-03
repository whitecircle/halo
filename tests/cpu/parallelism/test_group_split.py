#!/usr/bin/env python
"""Group construction over an NCCL default group bound to its device: one communicator split per call. Every
``new_group`` splits the world communicator, blocking every rank."""

import pytest

from halo.distributed.pipeline_parallel.groups import build_pipeline_group, build_stage_group
from halo.runtime.collectives import build_groups
from tests.common.goldens.rank_layout import LITERAL_SHAPES
from tests.common.parallelism import StubGroup, stubbed_process_groups

EP_GROUP_FIELDS = ("process_group", "dispatch_ep_group", "expert_tp_group", "expert_replica_group", "dp_scope_group")


def _held_groups(config) -> dict[str, tuple | None]:
    ep_config = config.ep_config()
    held = {field: getattr(ep_config, field) for field in EP_GROUP_FIELDS}
    if config.is_cp_mode:
        held["cp"] = config.cp_config().process_group
    held["pp_chain"], held["pp_stage"] = build_pipeline_group(config), build_stage_group(config)
    return {name: None if group is None else tuple(group) for name, group in held.items()}


@pytest.mark.parametrize("shape", LITERAL_SHAPES, ids=lambda shape: shape.key)
def test_one_split_per_call_builds_what_one_group_per_list_builds(shape):
    for rank in range(shape.world_size):
        # It keeps the groups it built, which the second path would read back instead.
        listed: list[StubGroup] = []
        config = shape.config_as_rank(rank)
        with stubbed_process_groups(shape.world_size, rank, listed):
            per_list = _held_groups(config)
        split_calls: list[tuple] = []
        unexpected: list[StubGroup] = []
        config = shape.config_as_rank(rank)
        with stubbed_process_groups(shape.world_size, rank, unexpected, splits=split_calls):
            split = _held_groups(config)
        assert split == per_list, f"rank {rank}"
        assert not unexpected, f"rank {rank} built {len(unexpected)} groups through new_group beside the splits"
        # Exactly the lists the per-list path creates one group each for.
        assert [ranks for _what, lists in split_calls for ranks in lists] == [list(group) for group in listed]
        described = [what for what, _lists in split_calls]
        assert len(described) == len(set(described)), f"rank {rank}: more than one split per call: {described}"


@pytest.mark.parametrize("splits", [None, []], ids=["per-list", "split"])
def test_a_rank_outside_its_list_or_overlapping_lists_raise(splits):
    with stubbed_process_groups(4, 1, [], splits=splits):
        with pytest.raises(RuntimeError, match="was not assigned a test group"):
            build_groups([[0, 1], [2, 3]], mine=1, what="test group")
        with pytest.raises(RuntimeError, match="was not assigned a test group"):
            build_groups([[0, 2], [3]], mine=0, what="test group")
        with pytest.raises(ValueError, match="overlap"):
            build_groups([[0, 1], [1, 2, 3]], mine=0, what="test group")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
