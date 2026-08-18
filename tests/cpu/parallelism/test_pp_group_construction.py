#!/usr/bin/env python
"""Pipeline chain/stage groups, and EP/CP process-group construction under PP (rank_offset threading).

PP splits the world into ``pp_size`` contiguous blocks of ``stage_world_size``; ``EPConfig`` /
``CPConfig`` receive ONE block's width plus its base global rank (``rank_offset``) and must

  * do all coordinate math in block-local rank space (subtract the offset), and
  * emit GLOBAL rank lists to ``dist.new_group`` (add the offset back), and
  * create the groups of EVERY block, in one order, keeping only their own.

The last point is the deadlock invariant: ``dist.new_group`` is a collective over the default process
group, so a rank that only created its own stage's groups would issue a different call sequence than
the other stage and hang the job before the first step.

These tests drive the REAL ``EPConfig`` / ``CPConfig`` / ``ParallelismConfig`` with
``torch.distributed`` stubbed (``new_group`` records and returns the rank tuple), so a regression in
the rank math or the enumeration order fails here rather than on 16 GPUs.

Run: python tests/cpu/parallelism/test_pp_group_construction.py
"""

import types
from unittest.mock import patch

import pytest

from src.distributed.context_parallel.config import CPConfig
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.groups import create_pipeline_group, create_stage_group
from tests.common.parallelism import make_parallelism_config

_EP_MOD = "src.distributed.expert_parallel.config"
_CP_MOD = "src.distributed.context_parallel.config"
_PP_MOD = "src.distributed.pipeline_parallel.groups"


def _recording_dist(calls: list, rank: int = 0):
    """Stub ``torch.distributed``: ``new_group`` records the rank list and returns it as the handle."""

    def _new_group(ranks, timeout=None):
        calls.append(tuple(ranks))
        return tuple(ranks)

    return types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_rank=lambda: rank,
        new_group=_new_group,
    )


def _make_config(cls, module: str, rank: int, *, world_size: int, stage_world_size: int, **kwargs):
    """Build the REAL ``cls`` for ``rank`` of a ``world_size`` job split into stage-wide blocks.

    Returns ``(config, calls)`` where ``calls`` is the ordered ``new_group`` rank-list sequence this
    rank would issue.
    """
    calls: list = []
    rank_offset = (rank // stage_world_size) * stage_world_size
    with (
        patch(f"{module}.dist", _recording_dist(calls)),
        patch(f"{module}.get_global_rank", return_value=rank),
        patch(f"{module}.get_global_world_size", return_value=world_size),
        patch(f"{module}.get_local_world_size", return_value=8),
        patch(f"{module}.get_nccl_timeout", return_value=None),
        patch(f"{module}.is_global_main_process", return_value=False),
    ):
        return cls(world_size=stage_world_size, rank_offset=rank_offset, **kwargs), calls


def _ep_over_world(*, world_size: int, stage_world_size: int, **kwargs):
    """Build the EPConfig of every rank of the job. Returns ``{rank: (config, calls)}``."""
    return {
        rank: _make_config(EPConfig, _EP_MOD, rank, world_size=world_size, stage_world_size=stage_world_size, **kwargs)
        for rank in range(world_size)
    }


def _assert_partitions_world(groups: set, world_size: int, stage_world_size: int):
    """The distinct groups tile the world exactly and none straddles a stage boundary."""
    flat = [r for g in groups for r in g]
    assert sorted(flat) == list(range(world_size)), sorted(flat)  # union == world AND disjoint
    for g in groups:
        assert len({r // stage_world_size for r in g}) == 1, g


# Pipeline chain and stage groups


def _pp_groups(rank: int, *, world_size: int, pp_size: int):
    """``(chain group, stage group, ordered new_group calls, config)`` as ``rank`` of a PP job.

    The stub returns the rank tuple as the handle, so each returned "group" IS the membership the
    real ``dist.new_group`` would have been given.
    """
    config = make_parallelism_config(world_size=world_size, gpus_per_node=8, rank=rank, pp_size=pp_size)
    calls: list = []
    with (
        patch(f"{_PP_MOD}.dist", _recording_dist(calls, rank)),
        patch(f"{_PP_MOD}.get_nccl_timeout", return_value=None),
    ):
        return create_pipeline_group(config), create_stage_group(config), calls, config


@pytest.mark.parametrize(("world_size", "pp_size"), [(16, 2), (32, 4)])
def test_pipeline_groups_match_the_config_rank_math(world_size, pp_size):
    """The chain groups PP builds must be exactly ``ParallelismConfig.get_pp_group_ranks``.

    The two are computed independently — ``groups.py`` from the stage rank blocks, the config from
    ``stage_local_rank + s * stage_world_size`` — and the trainer mixes them: it broadcasts the last
    stage's loss to ``get_pp_group_ranks()[-1]`` OVER the group built here. A drift would name a rank
    outside the group and hang the chain, so this pins them against each other on every rank.
    """
    for rank in range(world_size):
        chain, stage, _, config = _pp_groups(rank, world_size=world_size, pp_size=pp_size)
        assert chain == tuple(config.get_pp_group_ranks()), (rank, chain)
        assert rank in chain, (rank, chain)
        assert len(chain) == pp_size, (rank, chain)
        base, width = config.stage_base_rank, config.stage_world_size
        assert stage == tuple(range(base, base + width)), (rank, stage)
        assert rank in stage, (rank, stage)


def test_pipeline_group_creation_sequence_is_identical_on_every_rank():
    """``dist.new_group`` is collective: every rank must issue the same calls in the same order.

    Both group families are created by every rank (each keeps only its own), so the recorded
    sequence must match rank for rank, and together the chains must tile the world exactly.
    """
    built = {rank: _pp_groups(rank, world_size=16, pp_size=2) for rank in range(16)}
    reference = built[0][2]
    for rank, (_, _, calls, _) in built.items():
        assert calls == reference, (rank, calls, reference)

    chains = {chain for chain, _, _, _ in built.values()}
    assert len(chains) == 8, chains
    assert sorted(r for chain in chains for r in chain) == list(range(16)), chains
    assert {stage for _, stage, _, _ in built.values()} == {tuple(range(8)), tuple(range(8, 16))}


def test_pipeline_groups_are_absent_without_pp():
    config = make_parallelism_config(world_size=8, gpus_per_node=8, rank=0)
    assert create_pipeline_group(config) is None
    assert create_stage_group(config) is None


# (ep_size, groups per stage) on world=16 = 2 stages x 8 ranks, NVLink domain 8.
_NODE_LOCAL_EP_SHAPES = [(8, 1), (4, 2), (2, 4)]


def test_ep_every_rank_selects_a_group_containing_itself():
    """Every rank of every stage lands in an EP group it is a member of.

    Without this, stage-1 ranks match no created group and EPConfig raises
    "EP group selection inconsistency" — the exact 16-GPU crash.
    """
    for ep_size, groups_per_stage in _NODE_LOCAL_EP_SHAPES:
        built = _ep_over_world(world_size=16, stage_world_size=8, ep_size=ep_size, gpus_per_node=8, node_local=True)
        for rank, (cfg, _) in built.items():
            assert cfg.num_ep_groups == groups_per_stage, (ep_size, rank, cfg.num_ep_groups)
            assert cfg.process_group is not None, (ep_size, rank)
            assert rank in cfg._my_ep_group_ranks, (ep_size, rank, cfg._my_ep_group_ranks)
            # The stub returns the rank tuple, so the handle IS this rank's group.
            assert cfg.process_group == tuple(cfg._my_ep_group_ranks), (ep_size, rank)


def test_ep_groups_are_stage_confined_and_tile_the_world():
    """The EP groups partition all 16 ranks and none straddles the stage-0 / stage-1 boundary."""
    for ep_size, groups_per_stage in _NODE_LOCAL_EP_SHAPES:
        built = _ep_over_world(world_size=16, stage_world_size=8, ep_size=ep_size, gpus_per_node=8, node_local=True)
        groups = {tuple(cfg._my_ep_group_ranks) for cfg, _ in built.values()}
        assert len(groups) == 2 * groups_per_stage, (ep_size, groups)
        _assert_partitions_world(groups, 16, 8)


def test_ep_new_group_sequence_identical_on_every_rank():
    """Every rank issues the SAME ``new_group`` calls in the SAME order (the deadlock invariant)."""
    for ep_size, _ in _NODE_LOCAL_EP_SHAPES:
        built = _ep_over_world(world_size=16, stage_world_size=8, ep_size=ep_size, gpus_per_node=8, node_local=True)
        sequences = {rank: calls for rank, (_, calls) in built.items()}
        reference = sequences[0]
        for rank, seq in sequences.items():
            assert seq == reference, (ep_size, rank, seq, reference)
        # Both stages' groups are actually created (not just this rank's own block).
        created = {r for call in reference for r in call}
        assert created == set(range(16)), (ep_size, reference)


def test_ep_pinned_call_sequence_ep2_two_stages():
    """Pin the exact ordered sequence for ep_size=2 on world=16, pp=2 (routing, replica, DP-scope)."""
    _, calls = _make_config(
        EPConfig, _EP_MOD, 12, world_size=16, stage_world_size=8, ep_size=2, gpus_per_node=8, node_local=True
    )
    assert calls == [
        # Routing groups, block-major.
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        # Expert-replica groups: same ep_rank across the stage's groups.
        (0, 2, 4, 6),
        (1, 3, 5, 7),
        (8, 10, 12, 14),
        (9, 11, 13, 15),
        # DP-scope groups: one per rank block (replicated-param averaging domain).
        (0, 1, 2, 3, 4, 5, 6, 7),
        (8, 9, 10, 11, 12, 13, 14, 15),
    ], calls


def test_ep_expert_replica_sets_are_stage_confined():
    """Cross-group expert-grad replica sets never span stages (a different stage owns other layers)."""
    for ep_size, groups_per_stage in _NODE_LOCAL_EP_SHAPES:
        if groups_per_stage == 1:
            continue  # single group per stage → no cross-group sync
        built = _ep_over_world(world_size=16, stage_world_size=8, ep_size=ep_size, gpus_per_node=8, node_local=True)
        for rank, (cfg, _) in built.items():
            replicas = cfg.expert_replica_ranks
            assert cfg.needs_expert_grad_sync, (ep_size, rank)
            assert len(replicas) == groups_per_stage, (ep_size, rank, replicas)
            assert rank in replicas, (ep_size, rank, replicas)
            assert {r // 8 for r in replicas} == {rank // 8}, (ep_size, rank, replicas)


# Cross-node (global-scope) EP under PP


def test_ep_cross_node_scope_under_pp():
    """world=32, 4 NVLink domains of 8, pp=2 → each stage runs the column-block layout on its 16 ranks.

    Stage 0 groups are [0-3, 8-11] / [4-7, 12-15]; stage 1's are the same shifted by 16. A group that
    straddled the boundary would put PP-incompatible ranks in one DeepEP dispatch group.
    """
    built = _ep_over_world(world_size=32, stage_world_size=16, ep_size=8, gpus_per_node=8, node_local=False)
    for rank, (cfg, _) in built.items():
        assert cfg.num_ep_groups == 2, (rank, cfg.num_ep_groups)
        assert rank in cfg._my_ep_group_ranks, (rank, cfg._my_ep_group_ranks)
        assert rank in cfg.expert_replica_ranks, (rank, cfg.expert_replica_ranks)
        assert {r // 16 for r in cfg.expert_replica_ranks} == {rank // 16}, (rank, cfg.expert_replica_ranks)

    groups = {tuple(cfg._my_ep_group_ranks) for cfg, _ in built.values()}
    _assert_partitions_world(groups, 32, 16)
    assert groups == {
        (0, 1, 2, 3, 8, 9, 10, 11),
        (4, 5, 6, 7, 12, 13, 14, 15),
        (16, 17, 18, 19, 24, 25, 26, 27),
        (20, 21, 22, 23, 28, 29, 30, 31),
    }, groups

    reference = built[0][1]
    for rank, (_, calls) in built.items():
        assert calls == reference, (rank, calls, reference)


# EP + expert TP under PP (the second group-creation call site)


def test_ep_etp_groups_stage_confined_under_pp():
    """ep_size=4 + expert_tp_size=2 (one EP group of 8 per stage): ETP and dispatch groups too.

    ``_create_expert_tp_groups`` enumerates the same all-block EP group list, so its groups must be
    stage-local, contain their rank, and be created in an identical order on every rank.
    """
    built = _ep_over_world(
        world_size=16, stage_world_size=8, ep_size=4, expert_tp_size=2, gpus_per_node=8, node_local=True
    )
    etp_groups, dispatch_groups = set(), set()
    for rank, (cfg, _) in built.items():
        assert rank in cfg.expert_tp_group, (rank, cfg.expert_tp_group)
        assert rank in cfg.dispatch_ep_group, (rank, cfg.dispatch_ep_group)
        assert {r // 8 for r in cfg.expert_tp_group} == {rank // 8}, (rank, cfg.expert_tp_group)
        assert {r // 8 for r in cfg.dispatch_ep_group} == {rank // 8}, (rank, cfg.dispatch_ep_group)
        etp_groups.add(tuple(cfg.expert_tp_group))
        dispatch_groups.add(tuple(cfg.dispatch_ep_group))

    _assert_partitions_world(etp_groups, 16, 8)
    _assert_partitions_world(dispatch_groups, 16, 8)
    assert dispatch_groups == {(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)}, dispatch_groups
    assert (8, 12) in etp_groups, etp_groups  # strided ETP pair inside stage 1

    reference = built[0][1]
    for rank, (_, calls) in built.items():
        assert calls == reference, (rank, calls, reference)


def test_pp1_ep_construction_is_unchanged():
    """Without PP (offset defaulted, world == stage) the created groups are exactly the pre-PP ones."""
    calls: list = []
    with (
        patch(f"{_EP_MOD}.dist", _recording_dist(calls)),
        patch(f"{_EP_MOD}.get_global_rank", return_value=3),
        patch(f"{_EP_MOD}.get_global_world_size", return_value=8),
        patch(f"{_EP_MOD}.get_local_world_size", return_value=8),
        patch(f"{_EP_MOD}.get_nccl_timeout", return_value=None),
        patch(f"{_EP_MOD}.is_global_main_process", return_value=False),
    ):
        cfg = EPConfig(ep_size=2, world_size=8, gpus_per_node=8, node_local=True)

    assert cfg.rank_offset == 0
    assert calls == [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2, 4, 6), (1, 3, 5, 7)], calls
    assert cfg._my_ep_group_ranks == [2, 3]
    assert cfg.expert_replica_ranks == [1, 3, 5, 7]


def test_pp1_cp_construction_is_unchanged():
    """Same no-op guarantee for CP: default offset reproduces the pre-PP group list."""
    calls: list = []
    with (
        patch(f"{_CP_MOD}.dist", _recording_dist(calls)),
        patch(f"{_CP_MOD}.get_global_rank", return_value=5),
        patch(f"{_CP_MOD}.get_global_world_size", return_value=8),
        patch(f"{_CP_MOD}.get_local_world_size", return_value=8),
        patch(f"{_CP_MOD}.get_nccl_timeout", return_value=None),
        patch(f"{_CP_MOD}.is_global_main_process", return_value=False),
    ):
        cfg = CPConfig(cp_size=2, world_size=8, gpus_per_node=8)

    assert cfg.rank_offset == 0
    assert calls == [(0, 1), (2, 3), (4, 5), (6, 7)], calls
    assert cfg._cp_group_ranks == [4, 5]
    assert (cfg.cp_rank, cfg.cp_group_idx) == (1, 2)


# CP under PP


def test_cp_groups_under_pp():
    """CP groups are stage-confined, tile the world, and are created in one identical order.

    ``ParallelismConfig`` currently rejects PP + CP (the loss normalizer is unvalidated), so this
    drives ``CPConfig`` directly — the rank math must already be right when that gate lifts.
    """
    for cp_size, groups_per_stage in ((2, 4), (4, 2), (8, 1)):
        built = {
            rank: _make_config(
                CPConfig,
                _CP_MOD,
                rank,
                world_size=16,
                stage_world_size=8,
                cp_size=cp_size,
                gpus_per_node=8,
            )
            for rank in range(16)
        }
        for rank, (cfg, _) in built.items():
            assert cfg.process_group is not None, (cp_size, rank)
            assert rank in cfg._cp_group_ranks, (cp_size, rank, cfg._cp_group_ranks)

        groups = {tuple(cfg._cp_group_ranks) for cfg, _ in built.values()}
        assert len(groups) == 2 * groups_per_stage, (cp_size, groups)
        _assert_partitions_world(groups, 16, 8)

        reference = built[0][1]
        for rank, (_, calls) in built.items():
            assert calls == reference, (cp_size, rank, calls, reference)
        assert len(reference) == 2 * groups_per_stage, (cp_size, reference)


def test_cp_rank_math_agrees_with_parallelism_config():
    """``ParallelismConfig.get_cp_group_ranks``/``get_cp_rank`` have no production caller — they are
    the independent oracle for the membership ``CPConfig`` actually builds its process group from.
    An oracle nothing pins against the implementation certifies only itself; a divergence would send
    the Ulysses all-to-all over a different rank set than the one the config reports."""
    for cp_size in (2, 4, 8):
        for rank in range(8):
            cfg, _ = _make_config(
                CPConfig,
                _CP_MOD,
                rank,
                world_size=8,
                stage_world_size=8,
                cp_size=cp_size,
                gpus_per_node=8,
            )
            pc = make_parallelism_config(world_size=8, gpus_per_node=8, rank=rank, cp_size=cp_size)
            assert pc.get_cp_rank() == cfg.cp_rank, (cp_size, rank)
            assert pc.get_cp_group_ranks() == list(cfg._cp_group_ranks), (cp_size, rank)


def _make_parallelism_config(rank: int, **kwargs) -> ParallelismConfig:
    """Two 8-GPU nodes (one NVLink domain each) — the smallest world a pipeline can span."""
    return make_parallelism_config(world_size=16, gpus_per_node=8, rank=rank, **kwargs)


def test_parallelism_config_passes_stage_base_rank_to_ep_config():
    """``create_ep_config`` must hand EPConfig this stage's base rank, not just its width.

    Without the offset a stage-1 rank builds stage-0's group list and EPConfig raises; with it, the
    EPConfig membership matches ``ParallelismConfig.get_ep_group_ranks`` exactly.
    """
    for rank in (0, 5, 8, 12):
        pc = _make_parallelism_config(rank, pp_size=2, ep_size=8, ep_scope="node")
        assert pc.stage_base_rank == (rank // 8) * 8

        calls: list = []
        with (
            patch(f"{_EP_MOD}.dist", _recording_dist(calls)),
            patch(f"{_EP_MOD}.get_global_rank", return_value=rank),
            patch(f"{_EP_MOD}.get_global_world_size", return_value=16),
            patch(f"{_EP_MOD}.get_nccl_timeout", return_value=None),
            patch(f"{_EP_MOD}.is_global_main_process", return_value=False),
        ):
            ep = pc.create_ep_config()

        assert ep.rank_offset == pc.stage_base_rank, rank
        assert ep.world_size == pc.stage_world_size == 8
        assert ep._my_ep_group_ranks == pc.get_ep_group_ranks(), (rank, ep._my_ep_group_ranks)
        assert ep._my_ep_group_ranks == list(range(pc.stage_base_rank, pc.stage_base_rank + 8)), rank
        # EP routing groups (one per stage), then the per-block DP-scope groups (same rank sets).
        assert calls == [tuple(range(8)), tuple(range(8, 16))] * 2, (rank, calls)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
