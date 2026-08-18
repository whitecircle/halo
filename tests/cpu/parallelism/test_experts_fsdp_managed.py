#!/usr/bin/env python3
"""``experts_fsdp_managed`` is one predicate, and both config objects must answer it identically.

Whether FSDP2 owns the experts (``ep_group_size == 1`` + ``fsdp_shard_ep1_experts``) or the EP
layer's own grad hooks do decides FSDP-ignored membership, whether the expert/router hooks register
at all, the grad-norm bucket, and whether ``fp32_experts`` does anything. It is one predicate rather
than a condition spelled by hand at each call site, and its call sites read two different config
objects. If ``ParallelismConfig`` and the ``EPConfig`` it builds ever disagree, the experts are
either synced twice (reduce-scatter *and* hooks) or not at all — and neither raises, so only a test
catches it.

Usage:
    python tests/cpu/parallelism/test_experts_fsdp_managed.py
"""

import sys

import pytest

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.parallelism_config import ParallelismConfig

WORLD = 8

# (ep_size, expert_tp_size) on one 8-GPU domain: the only shape with truly replicated experts
# (ep_group_size == 1) plus one representative of each way ep_group_size grows past it — pure ETP,
# pure EP, and both at once.
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
    """FSDP may own the experts only when they are genuinely replicated (``ep_group_size == 1``)."""
    config = _config(ep_size, expert_tp_size, shard_ep1)
    expected = shard_ep1 and config.ep_group_size == 1
    assert config.experts_fsdp_managed is expected, (
        f"ep_size={ep_size} expert_tp_size={expert_tp_size} shard_ep1={shard_ep1}: "
        f"ep_group_size={config.ep_group_size} but experts_fsdp_managed={config.experts_fsdp_managed}"
    )
    # Distributed experts must never be handed to FSDP: its reduce-scatter would average shards that
    # hold *different* experts, not replicas of the same one.
    if config.ep_group_size > 1:
        assert not config.experts_fsdp_managed


@pytest.mark.parametrize("shard_ep1", [True, False])
@pytest.mark.parametrize(("ep_size", "expert_tp_size"), EP_SHAPES)
def test_parallelism_config_and_ep_config_agree(ep_size, expert_tp_size, shard_ep1):
    """The mirror on ``ParallelismConfig`` must match the ``EPConfig`` the layers actually read."""
    config = _config(ep_size, expert_tp_size, shard_ep1)
    # Build the EPConfig the way ParallelismConfig does, minus the process groups (no CPU backend).
    ep_config = EPConfig.__new__(EPConfig)
    ep_config.ep_group_size = config.ep_group_size
    ep_config.fsdp_shard_ep1_experts = config.fsdp_shard_ep1_experts

    assert ep_config.experts_fsdp_managed is config.experts_fsdp_managed, (
        f"ep_size={ep_size} expert_tp_size={expert_tp_size} shard_ep1={shard_ep1}: "
        f"EPConfig says {ep_config.experts_fsdp_managed}, ParallelismConfig says "
        f"{config.experts_fsdp_managed} — the experts would be synced twice or not at all."
    )


def test_the_predicate_is_not_vacuous():
    """Both answers must be reachable, or the tests above pass on a constant."""
    answers = {_config(ep, etp, shard).experts_fsdp_managed for ep, etp in EP_SHAPES for shard in (True, False)}
    assert answers == {True, False}, f"experts_fsdp_managed only ever returns {answers}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
