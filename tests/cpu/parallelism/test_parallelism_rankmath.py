#!/usr/bin/env python
"""Group rank-math + multi-group-EP-reject tests for the parallelism subsystem.

Complements ``test_parallelism_config.py`` with the cases it does not cover:

* H1 — the multi-group pure-EP REJECT guard (the documented ep4-on-8 run-killer). It lives on
  ``EpIntrospectionMixin._setup_ep_gradient_checkpointing`` (not in ParallelismConfig), so this
  drives the REAL method via a minimal stub trainer — a regression in the actual guard predicate
  fails the test, which a re-implementation of the predicate could not.
* H2 — EP+ETP ``get_data_parallel_rank`` correctness with ep_size>1 (exercises the ``% ep_size``
  modulo that ep_size=1 makes a no-op): the DP-rank partition + genuine ETP-partner geometry.
* H3 — cross-node column-block layout at a SECOND topology (world=32, 4 domains of 8).
* H4 — multi-node DP rank is GLOBAL, not domain-local (no-parallelism and TP).
* H6 — the ``data_parallel_size`` "largest-divisor-wins" derivation matrix in one place.

Run: python tests/cpu/parallelism/test_parallelism_rankmath.py
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from tests.common.parallelism import create_config


class _GCModel(nn.Module):
    """Stub policy model: records the GC kwargs the EP path forces on it.

    Installing ``_gradient_checkpointing_func`` mirrors what ``PreTrainedModel`` does; the EP path
    re-points it at the scoped wrapper its dispatch replay hangs off and raises if it finds none.
    """

    def __init__(self):
        super().__init__()
        self.gc_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, every_n_layers: int = 1):
        self.gc_kwargs = gradient_checkpointing_kwargs
        self._gradient_checkpointing_func = checkpoint


# H1 — Multi-group pure-EP reject matrix (drives the REAL guard)


def _run_ep_gc_guard(config):
    """Invoke the real ``_setup_ep_gradient_checkpointing`` guard against ``config``.

    Binds the unbound mixin method to a minimal stub that carries exactly the attributes the method
    reads: ``parallelism_config``, ``_has_ep_layers`` (True so the guard runs), ``_ep_config`` (the
    EPConfig — production guarantees it is set whenever ``_has_ep_layers`` is True; the guard reads
    ``_ep_config.is_deferred_dp``), and ``args.gradient_checkpointing`` (True so an ACCEPT shape falls
    through to the real GC-enable path rather than short-circuiting before the guard's downstream
    work). The stub ``model`` is an empty
    nn.Module: it owns no ``EPMoELayerBase`` and no ``gradient_checkpointing_enable``, so the accept
    path completes (logs + sets args.gradient_checkpointing=False) without needing a real model. A
    REJECT shape raises RuntimeError before reaching any of that.
    """
    from src.trainers.mixins.ep_introspection import EpIntrospectionMixin

    class _Args:
        gradient_checkpointing = True

    class _Stub:
        pass

    stub = _Stub()
    stub.parallelism_config = config
    stub._has_ep_layers = True
    # Production captures _ep_config from the first EP module; _has_ep_layers being True guarantees it
    # is set (the property is literally ``_ep_config is not None``). The guard reads
    # ``_ep_config.is_deferred_dp`` — mirror EPConfig.is_deferred_dp (config.py): the deferred
    # post-backward DP sweep engages only for multi-group, multi-node, non-expert-TP EP.
    stub._ep_config = SimpleNamespace(
        is_deferred_dp=config.num_ep_groups > 1 and config.num_nodes > 1 and config.expert_tp_size == 1
    )
    stub.args = _Args()
    stub.model = _GCModel()

    with patch("src.trainers.mixins.ep_introspection.is_global_main_process", return_value=True):
        EpIntrospectionMixin._setup_ep_gradient_checkpointing(stub)


def _assert_config_rejects_multigroup_ep(*, ep_size, world, gpus_per_node):
    """The reject fires at ParallelismConfig construction, ahead of the trainer guard."""
    try:
        create_config(ep_size=ep_size, world_size=world, gpus_per_node=gpus_per_node, ep_scope="node")
    except ValueError as e:
        msg = str(e)
        # Pin the load-bearing facts of the guard message, not a generic token: the offending
        # ep_size, the NVLink-domain size, AND the derived per-domain group count — so a guard that
        # fired for the WRONG reason (or with wrong arithmetic) would not satisfy this.
        n_groups = gpus_per_node // ep_size
        assert f"ep_size={ep_size}" in msg, msg
        assert f"{gpus_per_node}-GPU NVLink domain" in msg, msg
        assert f"{n_groups} " in msg and "dispatch groups" in msg, msg
        return
    raise AssertionError(
        f"ep_size={ep_size} on world={world} (gpus_per_node={gpus_per_node}) "
        f"MUST be rejected at config time as multi-group pure EP"
    )


def test_h1_multigroup_pure_ep_ep4_on_8_rejected():
    """ep_size=4 on 8 GPUs → two 4-rank groups: the canonical ep4-on-8 run-killer. REJECT."""
    _assert_config_rejects_multigroup_ep(ep_size=4, world=8, gpus_per_node=8)


def test_h1_multigroup_pure_ep_ep3_on_6_rejected():
    """ep_size=3 on 6 GPUs → two 3-rank groups (gpus_per_node > ep_size > 2). REJECT."""
    _assert_config_rejects_multigroup_ep(ep_size=3, world=6, gpus_per_node=6)


def test_h1_pure_ep_ep2_on_8_accepted():
    """ep_size=2 on 8 GPUs → four 2-rank groups: NOT > 2, so the race guard does not fire. ACCEPT."""
    cfg = create_config(ep_size=2, world_size=8, gpus_per_node=8, ep_scope="node")
    _run_ep_gc_guard(cfg)  # must not raise


def test_h1_pure_ep_ep8_on_8_accepted():
    """ep_size=8 on 8 GPUs → a SINGLE 8-rank group (gpus_per_node == ep_size). ACCEPT."""
    cfg = create_config(ep_size=8, world_size=8, gpus_per_node=8, ep_scope="node")
    _run_ep_gc_guard(cfg)  # must not raise


def test_h1_pure_ep_ep4_on_4_accepted():
    """ep_size=4 on world=4 → a SINGLE 4-rank group (the online-GRPO shape). ACCEPT.

    ep4-on-4 is fine (one group); only ep4-on-8 (two groups) is the run-killer — the guard keys on
    gpus_per_node > ep_size, which is 4 > 4 == False here.
    """
    cfg = create_config(ep_size=4, world_size=4, gpus_per_node=4, ep_scope="node")
    _run_ep_gc_guard(cfg)  # must not raise


def test_h1_ep_etp_exempt_from_multigroup_reject():
    """ep_size=4 + expert_tp_size=2 on 8 GPUs: ACCEPT, via ``ep_group_size == nvlink_domain_size``.

    The guard has no expert-TP clause — the exemption is purely arithmetic: ETP raises
    ``ep_group_size`` to 4*2 = 8, so ``nvlink_domain_size > ep_group_size`` is False. The dispatch
    topology underneath is still two 4-rank groups, which
    ``test_racy_ep_dispatch_topology.py`` pins against the config-time dispatch-group gate.
    """
    cfg = create_config(ep_size=4, expert_tp_size=2, world_size=8, gpus_per_node=8, ep_scope="node")
    _run_ep_gc_guard(cfg)  # must not raise


def test_h1_guard_skipped_when_no_ep_layers():
    """The trainer guard is gated on _has_ep_layers — a reject shape with no EP layers must NOT raise.

    Config-time validation rejects this shape outright, so build a legal ep2-on-8 config and
    mutate it into the ep4-on-8 reject shape (simulating a config that escaped validation) to prove
    the trainer guard is defense-in-depth: hot when _has_ep_layers is True, skipped when False.
    """
    from src.trainers.mixins.ep_introspection import EpIntrospectionMixin

    cfg = create_config(ep_size=2, world_size=8, gpus_per_node=8, ep_scope="node")
    cfg.ep_size = 4
    cfg.ep_group_size = 4  # stored in __post_init__, must track the mutated ep_size

    class _Args:
        gradient_checkpointing = True

    class _Stub:
        pass

    stub = _Stub()
    stub.parallelism_config = cfg
    stub._has_ep_layers = False  # no EP layers → early return before the guard
    stub.args = _Args()
    stub.model = _GCModel()
    EpIntrospectionMixin._setup_ep_gradient_checkpointing(stub)  # must not raise

    # Prove the mutated shape actually arms the guard (the skip assertion above is not vacuous).
    stub._has_ep_layers = True
    stub._ep_config = SimpleNamespace(is_deferred_dp=False)
    with patch("src.trainers.mixins.ep_introspection.is_global_main_process", return_value=True):
        with pytest.raises(RuntimeError, match="dispatch groups"):
            EpIntrospectionMixin._setup_ep_gradient_checkpointing(stub)


# H2 — EP+ETP get_data_parallel_rank correctness (ep_size > 1 exercises % ep_size)


def test_h2_ep_etp_dp_rank_partition_and_partners():
    """EP+ETP (ep_size=2, expert_tp_size=2, world=8): enumerate every rank's DP-rank.

    ep_group_size = ep_size(2) * expert_tp_size(2) = 4, single NVLink domain of 8 → 2 EP groups,
    num_ep_groups=2, data_parallel_size = world(8) / expert_tp_size(2) = 4.

    Assertions (all derived, none tautological):
      * exactly data_parallel_size (=4) DISTINCT dp-ranks across the 8 global ranks,
      * each dp-rank appears exactly expert_tp_size (=2) times (the two ETP shards share a batch),
      * the two ranks sharing a dp-rank are genuine ETP PARTNERS — same dispatch_ep_rank but
        ep_rank differing by exactly ep_size (the strided expert-TP layout, e.g. ep_rank 0 & 2).
    """
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
        assert cfg.data_parallel_size == 4  # world(8) / expert_tp_size(2)
        assert cfg.ep_group_size == ep_size * etp == 4
        dp_by_rank[r] = cfg.get_data_parallel_rank()
        ep_rank_by_rank[r] = cfg.get_ep_rank()
        group_by_rank[r] = cfg.get_ep_group_idx()

    distinct = set(dp_by_rank.values())
    assert len(distinct) == 4, dp_by_rank  # exactly data_parallel_size distinct dp-ranks
    assert distinct == {0, 1, 2, 3}, dp_by_rank  # contiguous 0..dp-1

    # Each dp-rank shared by exactly expert_tp_size ranks.
    from collections import Counter

    counts = Counter(dp_by_rank.values())
    assert all(c == etp for c in counts.values()), dp_by_rank

    # The ranks sharing a dp-rank are genuine ETP partners: same dispatch_ep_rank (= ep_rank %
    # ep_size) and same EP group, with ep_rank differing by exactly ep_size (strided ETP group).
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
    """Pin the exact dp-rank of each rank for the EP+ETP case so the formula can't silently drift.

    Layout (ep_size=2, etp=2, domain=8): ep_group_size=4 → groups [0,1,2,3] and [4,5,6,7].
    Within a group of 4, ep_rank = global_rank % 4 (node-local). dispatch_ep_rank = ep_rank % 2,
    expert_tp_rank = ep_rank // 2. dp_rank = dispatch_ep_rank * num_ep_groups(2) + ep_group_idx.

      rank 0: ep_rank0 disp0 grp0 → 0*2+0 = 0      rank 4: ep_rank0 disp0 grp1 → 0*2+1 = 1
      rank 1: ep_rank1 disp1 grp0 → 1*2+0 = 2      rank 5: ep_rank1 disp1 grp1 → 1*2+1 = 3
      rank 2: ep_rank2 disp0 grp0 → 0*2+0 = 0      rank 6: ep_rank2 disp0 grp1 → 0*2+1 = 1
      rank 3: ep_rank3 disp1 grp0 → 1*2+0 = 2      rank 7: ep_rank3 disp1 grp1 → 1*2+1 = 3
    """
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


# H3 — Cross-node column-block layout at a SECOND topology (world=32, 4 domains of 8)


def _all_global_groups(ep_size, world_size, gpus_per_node):
    groups = {}
    for rank in range(world_size):
        cfg = create_config(
            ep_size=ep_size,
            world_size=world_size,
            gpus_per_node=gpus_per_node,
            rank=rank,
            ep_scope="global",
        )
        groups.setdefault(cfg.get_ep_group_idx(), cfg.get_ep_group_ranks())
    return groups


def test_h3_cross_node_world32_partition_and_contiguity():
    """world=32, 4 domains of 8, ep_size=8 (ep_group_size=8) → num_domains=4, m=2, 4 EP groups.

    Asserts the column-block layout at a 4-domain topology (existing tests only do 2 domains):
      * the 4 groups PARTITION all 32 ranks (union==all, disjoint),
      * within each domain a group's members are a contiguous device block of m=2,
      * group-rank order is domain-major (domain 0's block, then domain 1's, ...).
    """
    world, dom = 32, 8
    cfg0 = create_config(ep_size=8, world_size=world, gpus_per_node=dom, rank=0, ep_scope="global")
    assert cfg0.num_nvlink_domains == 4
    assert cfg0.num_ep_groups == 4  # nvlink_domain(8) / m(=ep_group_size 8 / num_domains 4 = 2)

    groups = _all_global_groups(ep_size=8, world_size=world, gpus_per_node=dom)
    assert len(groups) == 4

    flat = [r for ranks in groups.values() for r in ranks]
    assert sorted(flat) == list(range(world))  # exact partition: union == world, disjoint (no dup)
    assert len(flat) == world

    m = 2
    for ranks in groups.values():
        assert len(ranks) == 8  # ep_group_size
        # domain-major group-rank order: 4 blocks of m, each block in its own domain, ascending.
        for blk in range(4):
            block = ranks[blk * m : (blk + 1) * m]
            assert block == list(range(block[0], block[0] + m)), ranks  # contiguous device block
            assert all(r // dom == blk for r in block), ranks  # block blk lives in domain blk

    # Concrete first/last group to lock the math (group 0 = block 0 of each domain).
    assert groups[0] == [0, 1, 8, 9, 16, 17, 24, 25]
    assert groups[3] == [6, 7, 14, 15, 22, 23, 30, 31]


def test_h3_cross_node_world32_replica_round_trip():
    """Every expert-replica rank shares the same group-rank (get_ep_rank) — at the 4-domain topology.

    For ep_size=8 on world=32 (4 domains, m=2, 4 EP groups), the replicas of a rank are exactly the
    same-group-rank position across the 4 EP groups. Round-trip: build a config at each replica and
    confirm its get_ep_rank equals the source rank's, and that there are num_ep_groups of them.
    """
    world, dom = 32, 8
    for src in (0, 5, 17, 30):
        cfg = create_config(ep_size=8, world_size=world, gpus_per_node=dom, rank=src, ep_scope="global")
        replicas = cfg.get_expert_replica_ranks()
        assert len(replicas) == cfg.num_ep_groups == 4, (src, replicas)
        assert src in replicas, (src, replicas)
        for r in replicas:
            rcfg = create_config(ep_size=8, world_size=world, gpus_per_node=dom, rank=r, ep_scope="global")
            assert rcfg.get_ep_rank() == cfg.get_ep_rank(), (src, r, rcfg.get_ep_rank(), cfg.get_ep_rank())
        # Replicas are one-per-group: their ep_group_idx values are all distinct and cover 0..3.
        gidx = sorted(
            create_config(ep_size=8, world_size=world, gpus_per_node=dom, rank=r, ep_scope="global").get_ep_group_idx()
            for r in replicas
        )
        assert gidx == [0, 1, 2, 3], (src, gidx)


# H4 — Multi-node DP rank is GLOBAL, not domain-local


def test_h4_multinode_dp_rank_is_global_no_parallelism():
    """world=16, gpus_per_node=8, no parallelism, rank=11 → dp_rank == 11 (global, NOT 11%8=3)."""
    cfg = create_config(world_size=16, gpus_per_node=8, rank=11)
    assert cfg.get_data_parallel_rank() == 11  # global DP rank, not the domain-local 3


def test_h4_multinode_dp_rank_tp():
    """TP multi-node: tp_size=4, world=16, rank=13 → dp_rank == 13 // 4 == 3 (global // tp_size)."""
    cfg = create_config(tp_size=4, world_size=16, gpus_per_node=8, rank=13)
    assert cfg.data_parallel_size == 4  # 16 / 4
    assert cfg.get_data_parallel_rank() == 3  # 13 // 4 — global, not domain-local (5 // 4 == 1)


# H6 — data_parallel_size derivation matrix ("largest divisor wins")


def test_h6_data_parallel_size_derivation_matrix():
    """One place documenting the dp_size = world / max(tp, cp, expert_tp) rule (EP never divides).

    Each row is (kwargs, expected_dp). All on world=16 (gpus_per_node=8, 2 NVLink domains) so the
    same world makes the divisor the only moving part.
    """
    world, dom = 16, 8
    cases = [
        # EP-only: EP is orthogonal to DP → full world.
        ({"ep_size": 8, "ep_scope": "node"}, world),  # 16
        # TP-only: world / tp.
        ({"tp_size": 4}, world // 4),  # 4
        # CP-only: world / cp (cp must divide the domain → 4 <= 8).
        ({"cp_size": 4}, world // 4),  # 4
        # pure ETP (ep_size=1): world / expert_tp.
        ({"ep_size": 1, "expert_tp_size": 2}, world // 2),  # 8
        # EP+TP: only TP reduces DP. Cross-domain EP+TP must be a SINGLE EP group (ep_size==world,
        # global) — multi-domain multi-group EP+TP is rejected (cross-replica average vs (dp,tp) mesh).
        ({"ep_size": 16, "tp_size": 4, "ep_scope": "global"}, world // 4),  # 4
        # EP+CP orthogonal node-local: only CP reduces DP (ep_group_size==domain==cp).
        ({"ep_size": 8, "cp_size": 8, "ep_scope": "node"}, world // 8),  # 2
        # EP+ETP cross-node: only expert_tp reduces DP (EP stays orthogonal). The one supported
        # shape is one ETP group per NVLink domain: expert_tp == members-per-domain (8),
        # ep_size == domains spanned (2).
        ({"ep_size": 2, "expert_tp_size": 8, "ep_scope": "global"}, world // 8),  # 2
    ]
    for kw, expected in cases:
        cfg = create_config(world_size=world, gpus_per_node=dom, **kw)
        assert cfg.data_parallel_size == expected, (kw, cfg.data_parallel_size, expected)

    # An ETP split finer than members-per-domain would stride ETP groups across NVLink domains —
    # rejected at config time (the same rule EPConfig enforces at group construction).
    with pytest.raises(ValueError, match="one ETP group per NVLink domain"):
        create_config(world_size=world, gpus_per_node=dom, ep_size=8, expert_tp_size=2, ep_scope="global")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
