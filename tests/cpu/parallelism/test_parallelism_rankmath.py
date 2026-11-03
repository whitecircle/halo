#!/usr/bin/env python
"""Group rank math and the multi-group EP reject; complements ``test_parallelism_config.py``."""

from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from halo.trainers.parallelism.ep_layers import EPLayersMixin
from tests.common.parallelism import create_config


class _GCModel(nn.Module):
    """The EP path re-points ``_gradient_checkpointing_func`` at the scoped wrapper its dispatch replay hangs off
    and raises if it finds none."""

    def __init__(self):
        super().__init__()
        self.gc_kwargs = None

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs=None, every_n_layers: int = 1, offload: bool = False
    ):
        self.gc_kwargs = gradient_checkpointing_kwargs
        self._gradient_checkpointing_func = checkpoint


def _run_ep_gc_guard(config):
    """Binds the unbound mixin method to a stub with only the attributes it reads; ``args.gradient_checkpointing``
    is True so an accept shape reaches the real GC path."""

    class _Args:
        gradient_checkpointing = True
        gradient_checkpointing_kwargs = None

    class _Stub:
        pass

    stub = _Stub()
    stub.parallelism_config = config
    # Mirrors EPConfig.is_deferred_dp: the post-backward DP sweep engages only for multi-group, multi-node,
    # non-expert-TP EP.
    stub._ep_config = SimpleNamespace(
        is_deferred_dp=config.num_ep_groups > 1 and config.num_nodes > 1 and config.expert_tp_size == 1
    )
    stub.args = _Args()
    stub.model = _GCModel()
    stub._top_level_model = lambda: stub.model
    stub._ep_layers = lambda: [SimpleNamespace(FAMILY=SimpleNamespace(supports_gradient_checkpointing=True))]

    with patch("halo.trainers.parallelism.ep_layers.is_global_main_process", return_value=True):
        EPLayersMixin._install_ep_gradient_checkpointing(stub)


def _assert_config_rejects_multigroup_ep(*, ep_size, world, gpus_per_node):
    try:
        create_config(ep_size=ep_size, world_size=world, gpus_per_node=gpus_per_node, ep_scope="node")
    except ValueError as e:
        msg = str(e)
        # Pins the ep_size, domain size and derived group count, so a guard firing for the wrong reason or with wrong
        # arithmetic fails.
        n_groups = gpus_per_node // ep_size
        assert f"expert_parallel_size={ep_size}" in msg, msg
        assert f"{gpus_per_node}-GPU NVLink domain" in msg, msg
        assert f"{n_groups} " in msg and "dispatch groups" in msg, msg
        return
    raise AssertionError(
        f"ep_size={ep_size} on world={world} (gpus_per_node={gpus_per_node}) "
        f"MUST be rejected at config time as multi-group pure EP"
    )


def test_h1_multigroup_pure_ep_ep4_on_8_rejected():
    """Two 4-rank groups, the canonical run-killer."""
    _assert_config_rejects_multigroup_ep(ep_size=4, world=8, gpus_per_node=8)


def test_h1_multigroup_pure_ep_ep3_on_6_rejected():
    _assert_config_rejects_multigroup_ep(ep_size=3, world=6, gpus_per_node=6)


def test_h1_pure_ep_ep2_on_8_accepted():
    """ep_size is not > 2, so the race guard does not fire."""
    cfg = create_config(ep_size=2, world_size=8, gpus_per_node=8, ep_scope="node")
    _run_ep_gc_guard(cfg)


def test_h1_pure_ep_ep8_on_8_accepted():
    cfg = create_config(ep_size=8, world_size=8, gpus_per_node=8, ep_scope="node")
    _run_ep_gc_guard(cfg)


def test_h1_pure_ep_ep4_on_4_accepted():
    cfg = create_config(ep_size=4, world_size=4, gpus_per_node=4, ep_scope="node")
    _run_ep_gc_guard(cfg)


def test_h1_ep_etp_exempt_from_multigroup_reject():
    """The exemption is arithmetic: ETP raises ``ep_group_size`` to 8. ``test_racy_ep_dispatch_topology.py`` pins
    the two 4-rank dispatch groups underneath."""
    cfg = create_config(ep_size=4, expert_tp_size=2, world_size=8, gpus_per_node=8, ep_scope="node")
    _run_ep_gc_guard(cfg)


def test_h2_ep_etp_dp_rank_partition_and_partners():
    """The two ranks sharing a dp-rank are ETP partners: same dispatch_ep_rank, ep_rank differing by exactly
    ep_size."""
    ep_size, etp = 2, 2
    world = 8
    dp_by_rank = {}
    ep_rank_by_rank = {}
    group_by_rank = {}
    for r in range(world):
        cfg = create_config(
            ep_size=ep_size,
            expert_tp_size=etp,
            world_size=world,
            gpus_per_node=8,
            rank=r,
            ep_scope="node",
        )
        assert cfg.data_parallel_size == 4
        assert cfg.ep_group_size == ep_size * etp == 4
        dp_by_rank[r] = cfg.get_data_parallel_rank()
        ep_rank_by_rank[r], group_by_rank[r] = cfg.ep_layout.rank_and_group(r)

    distinct = set(dp_by_rank.values())
    assert len(distinct) == 4, dp_by_rank
    assert distinct == {0, 1, 2, 3}, dp_by_rank

    counts = Counter(dp_by_rank.values())
    assert all(c == etp for c in counts.values()), dp_by_rank

    for dp in distinct:
        partners = sorted(r for r, d in dp_by_rank.items() if d == dp)
        assert len(partners) == etp, (dp, partners)
        a, b = partners
        assert group_by_rank[a] == group_by_rank[b], (dp, partners, group_by_rank)
        assert ep_rank_by_rank[a] % ep_size == ep_rank_by_rank[b] % ep_size, (dp, partners)
        assert abs(ep_rank_by_rank[a] - ep_rank_by_rank[b]) == ep_size, (
            dp,
            partners,
            ep_rank_by_rank,
        )


def test_h2_ep_etp_dp_rank_concrete_values():
    """dp_rank = dispatch_ep_rank * num_ep_groups + ep_group_idx, with dispatch_ep_rank = ep_rank % ep_size."""
    expected = {0: 0, 1: 2, 2: 0, 3: 2, 4: 1, 5: 3, 6: 1, 7: 3}
    for r, want in expected.items():
        cfg = create_config(
            ep_size=2,
            expert_tp_size=2,
            world_size=8,
            gpus_per_node=8,
            rank=r,
            ep_scope="node",
        )
        assert cfg.get_data_parallel_rank() == want, (r, cfg.get_data_parallel_rank())


def _all_global_groups(ep_size, world_size, gpus_per_node):
    layout = create_config(
        ep_size=ep_size, world_size=world_size, gpus_per_node=gpus_per_node, ep_scope="global"
    ).ep_layout
    groups = {}
    for rank in range(world_size):
        group_idx = layout.rank_and_group(rank)[1]
        groups.setdefault(group_idx, layout.group_ranks(group_idx))
    return groups


def test_h3_cross_node_world32_partition_and_contiguity():
    """m=2, 4 EP groups; the existing tests only do 2 domains."""
    world, dom = 32, 8
    cfg0 = create_config(ep_size=8, world_size=world, gpus_per_node=dom, rank=0, ep_scope="global")
    assert cfg0.num_nvlink_domains == 4
    assert cfg0.num_ep_groups == 4

    groups = _all_global_groups(ep_size=8, world_size=world, gpus_per_node=dom)
    assert len(groups) == 4

    flat = [r for ranks in groups.values() for r in ranks]
    assert sorted(flat) == list(range(world))
    assert len(flat) == world

    m = 2
    for ranks in groups.values():
        assert len(ranks) == 8
        for blk in range(4):
            block = ranks[blk * m : (blk + 1) * m]
            assert block == list(range(block[0], block[0] + m)), ranks
            assert all(r // dom == blk for r in block), ranks

    assert groups[0] == [0, 1, 8, 9, 16, 17, 24, 25]
    assert groups[3] == [6, 7, 14, 15, 22, 23, 30, 31]


def test_h3_cross_node_world32_replica_round_trip():
    """For each rank, replicas are the same group-rank position across the 4 EP groups."""
    world, dom = 32, 8
    cfg = create_config(ep_size=8, world_size=world, gpus_per_node=dom, ep_scope="global")
    layout = cfg.ep_layout
    for src in (0, 5, 17, 30):
        ep_rank = layout.rank_and_group(src)[0]
        replicas = layout.replica_ranks(ep_rank)
        assert len(replicas) == cfg.num_ep_groups == 4, (src, replicas)
        assert src in replicas, (src, replicas)
        for r in replicas:
            assert layout.rank_and_group(r)[0] == ep_rank, (src, r, layout.rank_and_group(r), ep_rank)
        assert sorted(layout.rank_and_group(r)[1] for r in replicas) == [0, 1, 2, 3], src


def test_h4_multinode_dp_rank_is_global_no_parallelism():
    cfg = create_config(world_size=16, gpus_per_node=8, rank=11)
    assert cfg.get_data_parallel_rank() == 11  # Global DP rank, not the domain-local 3.


def test_h4_multinode_dp_rank_tp():
    cfg = create_config(tp_size=4, world_size=16, gpus_per_node=8, rank=13)
    assert cfg.data_parallel_size == 4
    assert cfg.get_data_parallel_rank() == 3  # Global, not domain-local (5 // 4 == 1).


def test_h6_data_parallel_size_derivation_matrix():
    """One place for dp_size = world / max(tp, cp, expert_tp); the same world makes the divisor the only moving
    part."""
    world, dom = 16, 8
    cases = [
        ({"ep_size": 8, "ep_scope": "node"}, world),
        ({"tp_size": 4}, world // 4),
        ({"cp_size": 4}, world // 4),
        ({"ep_size": 1, "expert_tp_size": 2}, world // 2),
        # Multi-domain multi-group EP+TP is rejected (cross-replica average vs (dp,tp) mesh).
        ({"ep_size": 16, "tp_size": 4, "ep_scope": "global"}, world // 4),
        ({"ep_size": 8, "cp_size": 8, "ep_scope": "node"}, world // 8),
        # The one supported EP+ETP cross-node shape: one ETP group per NVLink domain.
        ({"ep_size": 2, "expert_tp_size": 8, "ep_scope": "global"}, world // 8),
    ]
    for kw, expected in cases:
        cfg = create_config(world_size=world, gpus_per_node=dom, **kw)
        assert cfg.data_parallel_size == expected, (kw, cfg.data_parallel_size, expected)

    # A finer ETP split would stride ETP groups across NVLink domains, the rule EPConfig enforces at group
    # construction.
    with pytest.raises(ValueError, match="one ETP group per NVLink domain"):
        create_config(world_size=world, gpus_per_node=dom, ep_size=8, expert_tp_size=2, ep_scope="global")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
