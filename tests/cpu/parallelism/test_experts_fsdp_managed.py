#!/usr/bin/env python3
"""``experts_fsdp_managed`` is one verdict, ``ParallelismConfig``'s, carried by its ``EPConfig``.
Disagreeing, the experts sync twice or not at all, and neither raises."""

import pytest

from halo.distributed.parallelism_config import ParallelismConfig

WORLD = 8

# The only replicated shape (ep_group_size == 1) plus each way ep_group_size grows past it.
EP_SHAPES = [(1, 1), (1, 2), (2, 1), (4, 2)]


def _config(ep_size: int, expert_tp_size: int, shard_ep1: bool) -> ParallelismConfig:
    return ParallelismConfig(
        world_size=WORLD,
        gpus_per_node=WORLD,
        nvlink_domain_size=WORLD,
        ep_size=ep_size,
        expert_tp_size=expert_tp_size,
        fsdp_shard_ep1_experts=shard_ep1,
    )


@pytest.mark.parametrize("shard_ep1", [True, False])
@pytest.mark.parametrize(("ep_size", "expert_tp_size"), EP_SHAPES)
def test_only_truly_replicated_experts_are_fsdp_managed(ep_size, expert_tp_size, shard_ep1):
    config = _config(ep_size, expert_tp_size, shard_ep1)
    expected = shard_ep1 and config.ep_group_size == 1
    assert config.experts_fsdp_managed is expected, (
        f"ep_size={ep_size} expert_tp_size={expert_tp_size} shard_ep1={shard_ep1}: "
        f"ep_group_size={config.ep_group_size} but experts_fsdp_managed={config.experts_fsdp_managed}"
    )
    # Its reduce-scatter would average shards holding different experts.
    if config.ep_group_size > 1:
        assert not config.experts_fsdp_managed


@pytest.mark.parametrize("shard_ep1", [True, False])
@pytest.mark.parametrize(("ep_size", "expert_tp_size"), EP_SHAPES)
def test_parallelism_config_and_ep_config_agree(ep_size, expert_tp_size, shard_ep1):
    config = _config(ep_size, expert_tp_size, shard_ep1)
    ep_config = config.ep_config()

    assert ep_config.experts_fsdp_managed is config.experts_fsdp_managed, (
        f"ep_size={ep_size} expert_tp_size={expert_tp_size} shard_ep1={shard_ep1}: "
        f"EPConfig says {ep_config.experts_fsdp_managed}, ParallelismConfig says "
        f"{config.experts_fsdp_managed} — the experts would be synced twice or not at all."
    )


def test_the_predicate_is_not_vacuous():
    """Both answers must be reachable."""
    answers = {_config(ep, etp, shard).experts_fsdp_managed for ep, etp in EP_SHAPES for shard in (True, False)}
    assert answers == {True, False}, f"experts_fsdp_managed only ever returns {answers}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
