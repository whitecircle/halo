#!/usr/bin/env python
"""EP+ETP dispatch-rank agreement between ``ParallelismConfig`` and ``EPConfig``, across node-local and cross-node
formulas. A divergence gives ETP partners different batches and desyncs the ``SumAcrossGroup`` all-reduce."""

import pytest

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.group_layout import ep_layout
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common.parallelism import make_parallelism_config, stubbed_process_groups

# (world_size, domain, ep_size, expert_tp_size, node_local) of shapes both classes accept.
SHAPES = [
    (8, 8, 4, 2, True),  # The verified EP+ETP shape.
    (8, 8, 2, 2, True),
    (8, 8, 2, 4, True),  # ETP-major.
    (16, 8, 2, 8, False),  # Cross-node: one EP group spanning 2 domains, ETP = domain block.
]

# The multi-domain multi-group shapes ParallelismConfig rejects.
EP_SHAPES = SHAPES + [
    (16, 8, 2, 4, True),
    (16, 8, 2, 4, False),  # Cross-node: two EP groups of 8, one ETP group per domain half.
]


def _make_parallelism_config(rank: int, world_size: int, gpus_per_node: int, **kwargs) -> ParallelismConfig:
    return make_parallelism_config(world_size=world_size, gpus_per_node=gpus_per_node, rank=rank, **kwargs)


def _make_ep_config(rank: int, world_size: int, gpus_per_node: int, **kwargs) -> EPConfig:
    """new_group returns the rank tuple."""
    with stubbed_process_groups(world_size, rank):
        return EPConfig(
            stage_world_size=world_size, nvlink_domain_size=gpus_per_node, experts_fsdp_managed=False, **kwargs
        )


def test_dispatch_rank_agreement_across_layouts():
    for world_size, gpus_per_node, ep_size, expert_tp_size, node_local in SHAPES:
        for rank in range(world_size):
            ep = _make_ep_config(
                rank, world_size, gpus_per_node, ep_size=ep_size, expert_tp_size=expert_tp_size, node_local=node_local
            )
            pc = _make_parallelism_config(
                rank,
                world_size,
                gpus_per_node,
                ep_size=ep_size,
                expert_tp_size=expert_tp_size,
                ep_scope="node" if node_local else "global",
            )
            coords = pc.ep_layout.rank_and_group(pc.stage_local_rank)
            assert coords == (ep.ep_rank, ep.ep_group_idx), (rank, world_size, ep_size, expert_tp_size, node_local)
            # Partners sharing dispatch_ep_rank share a batch.
            expected_dp = ep.dispatch_ep_rank * ep.num_ep_groups + ep.ep_group_idx
            assert pc.get_data_parallel_rank() == expected_dp, (
                rank,
                world_size,
                ep_size,
                expert_tp_size,
                node_local,
                pc.get_data_parallel_rank(),
                expected_dp,
            )


def test_expert_replica_ranks_agree_with_ep_config():
    """A divergence averages expert grads over the wrong ranks: silently wrong weights."""
    for world_size, gpus_per_node, ep_size, expert_tp_size, node_local in SHAPES:
        for rank in range(world_size):
            ep = _make_ep_config(
                rank, world_size, gpus_per_node, ep_size=ep_size, expert_tp_size=expert_tp_size, node_local=node_local
            )
            pc = _make_parallelism_config(
                rank,
                world_size,
                gpus_per_node,
                ep_size=ep_size,
                expert_tp_size=expert_tp_size,
                ep_scope="node" if node_local else "global",
            )
            replicas = pc.ep_layout.replica_ranks(ep.ep_rank) if pc.num_ep_groups > 1 else [rank]
            assert sorted(replicas) == sorted(ep.expert_replica_ranks), (
                rank,
                world_size,
                ep_size,
                expert_tp_size,
                node_local,
            )


def test_dispatch_rank_partition_is_balanced():
    for world_size, gpus_per_node, ep_size, expert_tp_size, node_local in EP_SHAPES:
        claims = {}
        for rank in range(world_size):
            ep = _make_ep_config(
                rank, world_size, gpus_per_node, ep_size=ep_size, expert_tp_size=expert_tp_size, node_local=node_local
            )
            dp = ep.dispatch_ep_rank * ep.num_ep_groups + ep.ep_group_idx
            claims.setdefault(dp, []).append((rank, ep.expert_tp_rank))
        assert len(claims) == world_size // expert_tp_size, (claims, world_size, expert_tp_size)
        for dp, partners in claims.items():
            assert len(partners) == expert_tp_size, (dp, partners)
            assert sorted(tp for _, tp in partners) == list(range(expert_tp_size)), (dp, partners)


def test_ep_config_dispatch_formulas_pinned():
    ep = _make_ep_config(6, 8, 8, ep_size=4, expert_tp_size=2, node_local=True)
    assert (ep.ep_rank, ep.dispatch_ep_rank, ep.expert_tp_rank) == (6, 2, 1)
    ep = _make_ep_config(11, 16, 8, ep_size=2, expert_tp_size=8, node_local=False)
    assert (ep.ep_rank, ep.dispatch_ep_rank, ep.expert_tp_rank) == (11, 1, 3)


def test_leaf_function_matches_group_construction():
    """EPConfig assigns its own coords through the layout, so the constructed membership is the only independent
    spelling."""
    for world_size, gpus_per_node, ep_size, expert_tp_size, node_local in EP_SHAPES:
        layout = ep_layout(
            world_size=world_size,
            group_size=ep_size * expert_tp_size,
            domain_size=gpus_per_node,
            node_local=node_local,
        )
        for rank in range(world_size):
            ep = _make_ep_config(
                rank, world_size, gpus_per_node, ep_size=ep_size, expert_tp_size=expert_tp_size, node_local=node_local
            )
            group_ranks = list(ep._my_ep_group_ranks)
            expected = sorted(
                group_ranks[k]
                for k in range(len(group_ranks))
                if layout.etp_partition(k, ep_size, expert_tp_size)[0] == ep.dispatch_ep_rank
            )
            shape = (rank, world_size, ep_size, expert_tp_size, node_local)
            assert len(expected) == expert_tp_size, shape
            assert rank in expected, shape
            assert sorted(ep.expert_tp_group) == expected, (shape, ep.expert_tp_group, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
