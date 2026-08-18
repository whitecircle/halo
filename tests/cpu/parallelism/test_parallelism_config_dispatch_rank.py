#!/usr/bin/env python
"""EP+ETP dispatch-rank agreement between ``ParallelismConfig`` and ``EPConfig``.

Under expert TP, an EP group splits into ``expert_tp_size`` dispatch (sub-EP) chunks; ETP partners
share a ``dispatch_ep_rank`` and therefore a DP batch, so ``ParallelismConfig.get_data_parallel_rank``
and ``EPConfig._create_expert_tp_groups`` must derive the SAME dispatch rank from ``ep_rank`` —
node-local (contiguous dispatch chunks of ``ep_size``) and cross-node (one ETP group per NVLink
domain) use different formulas. A divergence would give ETP partners different batches and desync
the token-space ``ReduceFromExpertTP`` all-reduce.

Drives the REAL ``EPConfig`` group construction with ``torch.distributed`` stubbed (``new_group``
returns the rank list) and the REAL ``ParallelismConfig`` rank math, for every rank of several
shapes across both layouts, and asserts the two agree.

    python tests/cpu/parallelism/test_parallelism_config_dispatch_rank.py
"""

import types
from unittest.mock import patch

import pytest

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.parallelism import make_parallelism_config

_EP_MOD = "src.distributed.expert_parallel.config"

# (world_size, domain, ep_size, expert_tp_size, node_local) — shapes BOTH classes accept.
SHAPES = [
    (8, 8, 4, 2, True),  # single node-local group of 8 (the verified EP+ETP shape)
    (8, 8, 2, 2, True),  # two node-local groups of 4, one domain
    (8, 8, 2, 4, True),  # single node-local group of 8, ETP-major
    (16, 8, 2, 8, False),  # cross-node: one EP group spanning 2 domains, ETP = domain block
]

# EPConfig-only superset: the multi-domain multi-group shapes ParallelismConfig rejects.
EP_SHAPES = SHAPES + [
    (16, 8, 2, 4, True),  # two nodes, one node-local group of 8 each
    (16, 8, 2, 4, False),  # cross-node: two EP groups of 8, one ETP group per domain half
]


def _make_parallelism_config(rank: int, world_size: int, gpus_per_node: int, **kwargs) -> ParallelismConfig:
    return make_parallelism_config(world_size=world_size, gpus_per_node=gpus_per_node, rank=rank, **kwargs)


def _make_ep_config(rank: int, world_size: int, gpus_per_node: int, **kwargs) -> EPConfig:
    """Build the REAL EPConfig for ``rank`` with distributed stubbed (new_group → rank tuple)."""
    fake_dist = types.SimpleNamespace(
        is_initialized=lambda: True,
        new_group=lambda ranks, timeout=None: tuple(ranks),
    )
    with (
        patch(f"{_EP_MOD}.dist", fake_dist),
        patch(f"{_EP_MOD}.get_global_rank", return_value=rank),
        patch(f"{_EP_MOD}.get_global_world_size", return_value=world_size),
        patch(f"{_EP_MOD}.get_local_world_size", return_value=gpus_per_node),
        patch(f"{_EP_MOD}.get_nccl_timeout", return_value=None),
        patch(f"{_EP_MOD}.is_global_main_process", return_value=False),
    ):
        return EPConfig(world_size=world_size, gpus_per_node=gpus_per_node, **kwargs)


def test_dispatch_rank_agreement_across_layouts():
    """Both call sites must derive the same dispatch_ep_rank (hence DP rank) for every rank/shape."""
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
            assert pc.get_ep_rank() == ep.ep_rank, (rank, world_size, ep_size, expert_tp_size, node_local)
            assert pc.get_ep_group_idx() == ep.ep_group_idx
            # DP rank keys on EPConfig's dispatch layout: partners sharing dispatch_ep_rank share a batch.
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
    """``ParallelismConfig.get_expert_replica_ranks`` has no production caller — it is the deliberate
    independent oracle for ``EPConfig.expert_replica_ranks``, the membership the deferred
    cross-replica expert-grad sweep reduces over. Nothing pinned the two against each other, so the
    oracle could drift into agreeing with nothing. A divergence averages expert grads over the wrong
    ranks: silently wrong weights, no error."""
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
            assert sorted(pc.get_expert_replica_ranks()) == sorted(ep.expert_replica_ranks), (
                rank,
                world_size,
                ep_size,
                expert_tp_size,
                node_local,
            )


def test_dispatch_rank_partition_is_balanced():
    """Across each shape, every DP rank is claimed by exactly expert_tp_size ranks (the ETP partners)."""
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
    """Pin the two layout formulas at concrete ranks (a silent formula swap must fail here)."""
    # Node-local ep4+etp2, one group of 8: dispatch = ep_rank % 4, expert_tp = ep_rank // 4.
    ep = _make_ep_config(6, 8, 8, ep_size=4, expert_tp_size=2, node_local=True)
    assert (ep.ep_rank, ep.dispatch_ep_rank, ep.expert_tp_rank) == (6, 2, 1)
    # Cross-node ep2+etp8 over 2 domains: expert_tp = ep_rank % 8, dispatch = ep_rank // 8.
    ep = _make_ep_config(11, 16, 8, ep_size=2, expert_tp_size=8, node_local=False)
    assert (ep.ep_rank, ep.dispatch_ep_rank, ep.expert_tp_rank) == (11, 1, 3)


def test_leaf_function_matches_group_construction():
    """``etp_dispatch_coords`` must agree with the INDEPENDENT indexing the real group construction
    uses: the ranks ``_create_expert_tp_groups`` put in this rank's expert-TP group (captured via
    the stubbed ``new_group``) are exactly the EP-group members the leaf assigns this rank's
    dispatch coordinate. EPConfig assigns its own coords THROUGH the leaf, so comparing those would
    certify nothing — the constructed membership is the only independent spelling."""
    from src.distributed.group_layout import etp_dispatch_coords

    for world_size, gpus_per_node, ep_size, expert_tp_size, node_local in EP_SHAPES:
        for rank in range(world_size):
            ep = _make_ep_config(
                rank, world_size, gpus_per_node, ep_size=ep_size, expert_tp_size=expert_tp_size, node_local=node_local
            )
            group_ranks = list(ep._my_ep_group_ranks)
            expected = sorted(
                group_ranks[k]
                for k in range(len(group_ranks))
                if etp_dispatch_coords(k, ep_size, expert_tp_size, node_local)[0] == ep.dispatch_ep_rank
            )
            shape = (rank, world_size, ep_size, expert_tp_size, node_local)
            assert len(expected) == expert_tp_size, shape
            assert rank in expected, shape
            assert sorted(ep.expert_tp_group) == expected, (shape, ep.expert_tp_group, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
