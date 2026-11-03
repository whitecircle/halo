#!/usr/bin/env python
"""Pipeline chain/stage groups and EP/CP group construction under PP. ``dist.new_group`` is a collective over the
default group, so every rank creates every stage's groups in one order."""

import pytest

from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.pipeline_parallel.groups import build_pipeline_group, build_stage_group
from tests.common.parallelism import make_parallelism_config, stubbed_process_groups


def _make_ep(rank: int, *, world_size: int, stage_world_size: int, **kwargs):
    """Returns ``(config, calls)``, ``calls`` being the ordered ``new_group`` rank lists this rank issued."""
    calls: list = []
    stage_base_rank = (rank // stage_world_size) * stage_world_size
    with stubbed_process_groups(world_size, rank, calls):
        config = EPConfig(
            stage_world_size=stage_world_size, stage_base_rank=stage_base_rank, experts_fsdp_managed=False, **kwargs
        )
        return config, calls


def _ep_over_world(*, world_size: int, stage_world_size: int, **kwargs):
    return {
        rank: _make_ep(rank, world_size=world_size, stage_world_size=stage_world_size, **kwargs)
        for rank in range(world_size)
    }


def _cp_config(rank: int, *, world_size: int, stage_world_size: int, cp_size: int):
    calls: list = []
    with stubbed_process_groups(world_size, rank, calls):
        config = CPConfig(
            cp_size,
            stage_world_size=stage_world_size,
            stage_base_rank=(rank // stage_world_size) * stage_world_size,
            nvlink_domain_size=8,
        )
    return config, calls


def _assert_partitions_world(groups: set, world_size: int, stage_world_size: int):
    flat = [r for g in groups for r in g]
    assert sorted(flat) == list(range(world_size)), sorted(flat)
    for g in groups:
        assert len({r // stage_world_size for r in g}) == 1, g


def _pp_groups(rank: int, *, world_size: int, pp_size: int):
    """The stub returns the rank tuple as the handle, so each returned group is the membership ``dist.new_group``
    got."""
    config = make_parallelism_config(world_size=world_size, gpus_per_node=8, rank=rank, pp_size=pp_size)
    calls: list = []
    with stubbed_process_groups(world_size, rank, calls):
        return build_pipeline_group(config), build_stage_group(config), calls, config


# By intra-stage position: one rank per stage, ``stage_world_size`` (8) apart.
_EXPECTED_CHAINS = {
    (16, 2): [(0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)],
    (32, 4): [
        (0, 8, 16, 24),
        (1, 9, 17, 25),
        (2, 10, 18, 26),
        (3, 11, 19, 27),
        (4, 12, 20, 28),
        (5, 13, 21, 29),
        (6, 14, 22, 30),
        (7, 15, 23, 31),
    ],
}


@pytest.mark.parametrize(("world_size", "pp_size"), sorted(_EXPECTED_CHAINS))
def test_pipeline_groups_are_the_expected_chains_and_stage_blocks(world_size, pp_size):
    """The trainer broadcasts the last stage's loss from ``pp_chain_ranks()[-1]`` over this group; a drift names a
    rank outside it and hangs the chain."""
    expected = _EXPECTED_CHAINS[(world_size, pp_size)]
    for rank in range(world_size):
        chain, stage, _, config = _pp_groups(rank, world_size=world_size, pp_size=pp_size)
        assert chain == expected[rank % 8], (rank, chain)
        assert chain == tuple(config.pp_chain_ranks()), (rank, chain)
        base = (rank // 8) * 8
        assert stage == tuple(range(base, base + 8)), (rank, stage)


def test_pipeline_group_creation_sequence_is_identical_on_every_rank():
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
    assert build_pipeline_group(config) is None
    assert build_stage_group(config) is None


# On world=16 = 2 stages x 8 ranks, NVLink domain 8.
_NODE_LOCAL_EP_SHAPES = [(8, 1), (4, 2), (2, 4)]


def test_ep_every_rank_selects_a_group_containing_itself():
    """Otherwise stage-1 ranks match no created group and EPConfig raises."""
    for ep_size, groups_per_stage in _NODE_LOCAL_EP_SHAPES:
        built = _ep_over_world(
            world_size=16, stage_world_size=8, ep_size=ep_size, nvlink_domain_size=8, node_local=True
        )
        for rank, (cfg, _) in built.items():
            assert cfg.num_ep_groups == groups_per_stage, (ep_size, rank, cfg.num_ep_groups)
            assert cfg.process_group is not None, (ep_size, rank)
            assert rank in cfg._my_ep_group_ranks, (ep_size, rank, cfg._my_ep_group_ranks)
            assert cfg.process_group == tuple(cfg._my_ep_group_ranks), (ep_size, rank)


def test_ep_groups_are_stage_confined_and_tile_the_world():
    for ep_size, groups_per_stage in _NODE_LOCAL_EP_SHAPES:
        built = _ep_over_world(
            world_size=16, stage_world_size=8, ep_size=ep_size, nvlink_domain_size=8, node_local=True
        )
        groups = {tuple(cfg._my_ep_group_ranks) for cfg, _ in built.values()}
        assert len(groups) == 2 * groups_per_stage, (ep_size, groups)
        _assert_partitions_world(groups, 16, 8)


def test_ep_new_group_sequence_identical_on_every_rank():
    for ep_size, _ in _NODE_LOCAL_EP_SHAPES:
        built = _ep_over_world(
            world_size=16, stage_world_size=8, ep_size=ep_size, nvlink_domain_size=8, node_local=True
        )
        sequences = {rank: calls for rank, (_, calls) in built.items()}
        reference = sequences[0]
        for rank, seq in sequences.items():
            assert seq == reference, (ep_size, rank, seq, reference)
        created = {r for call in reference for r in call}
        assert created == set(range(16)), (ep_size, reference)


def test_ep_pinned_call_sequence_ep2_two_stages():
    """Routing, replica, DP-scope."""
    _, calls = _make_ep(12, world_size=16, stage_world_size=8, ep_size=2, nvlink_domain_size=8, node_local=True)
    assert calls == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        (0, 2, 4, 6),
        (1, 3, 5, 7),
        (8, 10, 12, 14),
        (9, 11, 13, 15),
        # One per rank block (the replicated-param averaging domain).
        (0, 1, 2, 3, 4, 5, 6, 7),
        (8, 9, 10, 11, 12, 13, 14, 15),
    ], calls


def test_ep_expert_replica_sets_are_stage_confined():
    """A different stage owns other layers."""
    for ep_size, groups_per_stage in _NODE_LOCAL_EP_SHAPES:
        if groups_per_stage == 1:
            continue
        built = _ep_over_world(
            world_size=16, stage_world_size=8, ep_size=ep_size, nvlink_domain_size=8, node_local=True
        )
        for rank, (cfg, _) in built.items():
            replicas = cfg.expert_replica_ranks
            assert cfg.needs_expert_grad_sync, (ep_size, rank)
            assert len(replicas) == groups_per_stage, (ep_size, rank, replicas)
            assert rank in replicas, (ep_size, rank, replicas)
            assert {r // 8 for r in replicas} == {rank // 8}, (ep_size, rank, replicas)


def test_ep_cross_node_scope_under_pp():
    """A straddling group would put PP-incompatible ranks in one DeepEP dispatch group."""
    built = _ep_over_world(world_size=32, stage_world_size=16, ep_size=8, nvlink_domain_size=8, node_local=False)
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


def test_ep_etp_groups_stage_confined_under_pp():
    built = _ep_over_world(
        world_size=16, stage_world_size=8, ep_size=4, expert_tp_size=2, nvlink_domain_size=8, node_local=True
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
    assert (8, 12) in etp_groups, etp_groups

    reference = built[0][1]
    for rank, (_, calls) in built.items():
        assert calls == reference, (rank, calls, reference)


def test_pp1_ep_construction_is_unchanged():
    """World == stage reproduces the pre-PP groups."""
    calls: list = []
    with stubbed_process_groups(8, 3, calls):
        cfg = EPConfig(
            ep_size=2, stage_world_size=8, nvlink_domain_size=8, node_local=True, experts_fsdp_managed=False
        )

    assert cfg.stage_base_rank == 0
    assert calls == [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2, 4, 6), (1, 3, 5, 7)], calls
    assert cfg._my_ep_group_ranks == [2, 3]
    assert cfg.expert_replica_ranks == [1, 3, 5, 7]


def test_misaligned_stage_base_rank_raises_before_any_group():
    """No rank is left waiting in a collective its peers skipped."""
    calls: list = []
    with (
        stubbed_process_groups(16, 3, calls),
        pytest.raises(ValueError, match=r"stage_base_rank \(3\) must be a multiple of stage_world_size \(8\)"),
    ):
        EPConfig(ep_size=8, stage_world_size=8, stage_base_rank=3, nvlink_domain_size=8, experts_fsdp_managed=False)
    assert calls == []


@pytest.mark.parametrize("missing", ["stage_world_size", "nvlink_domain_size"])
def test_stage_width_and_domain_are_required(missing):
    """EPConfig never guesses these from the launcher's environment."""
    sizes = {"stage_world_size": 8, "nvlink_domain_size": 8}
    del sizes[missing]
    with pytest.raises(TypeError, match=missing):
        EPConfig(ep_size=1, experts_fsdp_managed=True, **sizes)


def test_pp1_cp_construction_is_unchanged():
    calls: list = []
    with stubbed_process_groups(8, 5, calls):
        cfg = CPConfig(2, stage_world_size=8, stage_base_rank=0, nvlink_domain_size=8)

    assert cfg.stage_base_rank == 0
    assert calls == [(0, 1), (2, 3), (4, 5), (6, 7)], calls
    assert cfg._cp_group_ranks == [4, 5]
    assert (cfg.cp_rank, cfg.cp_group_idx) == (1, 2)


def test_cp_groups_under_pp():
    """``ParallelismConfig`` rejects PP + CP today, so this drives ``CPConfig`` directly for when that gate lifts."""
    for cp_size, groups_per_stage in ((2, 4), (4, 2), (8, 1)):
        built = {rank: _cp_config(rank, world_size=16, stage_world_size=8, cp_size=cp_size) for rank in range(16)}
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


def _make_parallelism_config(rank: int, **kwargs) -> ParallelismConfig:
    """Two 8-GPU nodes, the smallest world a pipeline can span."""
    return make_parallelism_config(world_size=16, gpus_per_node=8, rank=rank, **kwargs)


def test_parallelism_config_passes_stage_base_rank_to_ep_config():
    """Without the offset a stage-1 rank builds stage-0's group list and EPConfig raises."""
    for rank in (0, 5, 8, 12):
        pc = _make_parallelism_config(rank, pp_size=2, ep_size=8, ep_scope="node")
        assert pc.stage_base_rank == (rank // 8) * 8

        calls: list = []
        with stubbed_process_groups(16, rank, calls):
            ep = pc.ep_config()

        assert ep.stage_base_rank == pc.stage_base_rank, rank
        assert ep.stage_world_size == pc.stage_world_size == 8
        assert ep._my_ep_group_ranks == list(range(pc.stage_base_rank, pc.stage_base_rank + 8)), rank
        # EP routing groups (one per stage), then the per-block DP-scope groups (same rank sets).
        assert calls == [tuple(range(8)), tuple(range(8, 16))] * 2, (rank, calls)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
