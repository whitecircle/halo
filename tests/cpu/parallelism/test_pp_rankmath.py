#!/usr/bin/env python
"""Pipeline-parallelism rank math and config-time rejections.

PP is the OUTERMOST parallelism dimension: the world splits into ``pp_size`` contiguous rank blocks
of ``stage_world_size``, and every other mode (EP/CP/TP/ETP/FSDP) runs unchanged *inside* one block.
These tests pin the properties that make that decomposition correct — a regression in any of them is
a silent wrong-gradient or hang bug, not a crash:

* P1 — ``pp_size=1`` is bit-identical to the pre-PP behaviour (no-op guarantee).
* P2 — stage assignment and the pipeline chain (P2P peers) over a whole world.
* P3 — every rank of one chain shares a DP coordinate, so the chain consumes ONE batch (stage 0
  reads input_ids, the last stage reads labels). If this breaks, stages train on different data.
* P4 — EP groups and expert-replica sets never straddle a stage boundary. The same ep_rank in a
  different stage owns DIFFERENT layers, so averaging across stages corrupts gradients.
* P5 — the config-time rejections (indivisible world, bad pp args) and the PP+EP shapes that must
  still be accepted. Node alignment, PP+HSDP and PP+ZeRO-3 live in test_pp_data_parallel_composition.

Run: python tests/cpu/parallelism/test_pp_rankmath.py
"""

import pytest

from tests.common.parallelism import create_config


def _two_node_stage(rank, **kwargs):
    """world=8 as 2 simulated nodes of 4 (NVLink domain = 4), PP across the node boundary."""
    return create_config(world_size=8, gpus_per_node=4, nvlink_domain_size=4, rank=rank, **kwargs)


# P1 — pp_size=1 is a no-op


def test_p1_pp1_is_noop():
    """At pp_size=1 the stage coordinates collapse so every pre-PP formula is unchanged."""
    for rank in range(8):
        c = create_config(rank=rank, ep_size=8)
        assert c.stage_world_size == c.world_size == 8, f"rank{rank}: stage_world must equal world"
        assert c.pp_rank == 0 and c.stage_base_rank == 0
        assert c.stage_local_rank == c.global_rank == rank
        assert c.is_first_pp_stage and c.is_last_pp_stage, "a 1-stage pipeline is both first and last"
        assert not c.is_pp_mode
        assert c.get_pp_group_ranks() == [rank]


def test_p1_pp1_preserves_dp_and_ep_layout():
    """pp_size=1 must leave data_parallel_size and EP group membership exactly as before."""
    c = create_config(rank=3, ep_size=8)
    assert c.data_parallel_size == 8
    assert c.get_ep_group_ranks() == list(range(8))
    assert c.get_expert_replica_ranks() == [3], "single EP group -> replica set is just this rank"

    tp = create_config(rank=3, tp_size=2)
    assert tp.data_parallel_size == 4
    assert tp.get_data_parallel_rank() == 1, "rank 3 with tp_size=2 -> DP rank 3//2"


# P2 — stage assignment and pipeline chains


def test_p2_stage_assignment_partitions_the_world():
    """Stages are contiguous, equal, non-overlapping rank blocks covering the world exactly once."""
    seen = {}
    for rank in range(8):
        c = _two_node_stage(rank, pp_size=2)
        assert c.stage_world_size == 4
        assert c.pp_rank == rank // 4, f"rank{rank} belongs to stage {rank // 4}"
        assert c.stage_base_rank == c.pp_rank * 4
        assert c.stage_local_rank == rank % 4
        seen.setdefault(c.pp_rank, []).append(rank)
    assert seen == {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}


def test_p2_chain_members_are_one_stage_apart():
    """A chain holds the same intra-stage position in every stage — the P2P peers."""
    for rank in range(8):
        c = _two_node_stage(rank, pp_size=2)
        chain = c.get_pp_group_ranks()
        assert chain == [rank % 4, rank % 4 + 4], f"rank{rank} chain"
        assert len(chain) == c.pp_size
        assert rank in chain
        # Consecutive chain members sit in different NVLink domains — PP is the axis that crosses
        # RDMA while everything else stays domain-local.
        assert all(b - a == c.stage_world_size for a, b in zip(chain, chain[1:], strict=False))


def test_p2_first_and_last_stage_flags():
    c0, c1 = _two_node_stage(0, pp_size=2), _two_node_stage(4, pp_size=2)
    assert c0.is_first_pp_stage and not c0.is_last_pp_stage
    assert c1.is_last_pp_stage and not c1.is_first_pp_stage


def test_p2_four_stages():
    """pp_size=4 over 8 ranks with a 2-GPU domain: 4 stages of 2."""
    for rank in range(8):
        c = create_config(world_size=8, gpus_per_node=2, nvlink_domain_size=2, rank=rank, pp_size=4)
        assert c.stage_world_size == 2
        assert c.pp_rank == rank // 2
        assert c.get_pp_group_ranks() == [rank % 2, rank % 2 + 2, rank % 2 + 4, rank % 2 + 6]
    assert c.data_parallel_size == 2, "8 ranks / pp 4 = 2 distinct batches"


# P3 — one chain consumes one batch


def test_p3_chain_shares_a_dp_rank():
    """THE dataloader contract: every rank of a chain returns the same DP rank, and distinct chains
    return distinct DP ranks covering 0..data_parallel_size-1."""
    by_chain = {}
    for rank in range(8):
        c = _two_node_stage(rank, pp_size=2)
        by_chain.setdefault(tuple(c.get_pp_group_ranks()), set()).add(c.get_data_parallel_rank())

    assert len(by_chain) == 4, "4 chains over 8 ranks at pp_size=2"
    for chain, dp_ranks in by_chain.items():
        assert len(dp_ranks) == 1, f"chain {chain} disagrees on its DP rank: {dp_ranks}"
    assert {next(iter(v)) for v in by_chain.values()} == {0, 1, 2, 3}


def test_p3_pp_pairs_with_no_other_sharding_axis():
    """PP composes with the expert axes only (EP, or pure ETP). TP/CP alongside PP — and all three
    expert-sharding axes at once — are refused by the capability matrix, each naming its own
    mechanism rather than a generic 'unsupported'."""
    for kwargs, combo in (
        ({"tp_size": 2}, "PP + TP"),
        ({"cp_size": 2}, "PP + CP"),
        ({"tp_size": 2, "ep_size": 2}, "PP + EP + TP"),
        ({"ep_size": 2, "expert_tp_size": 2}, "PP + EP + ETP"),
    ):
        with pytest.raises(ValueError, match="is not a supported parallelism combination") as excinfo:
            _two_node_stage(0, pp_size=2, **kwargs)
        assert combo in str(excinfo.value), (kwargs, str(excinfo.value))


# P4 — intra-stage groups never straddle a stage boundary


def test_p4_ep_groups_stay_inside_one_stage():
    """Node-local EP inside each stage: every EP group's ranks share one pp_rank."""
    for rank in range(8):
        c = _two_node_stage(rank, pp_size=2, ep_size=4)
        ep_ranks = c.get_ep_group_ranks()
        stages = {r // c.stage_world_size for r in ep_ranks}
        assert stages == {c.pp_rank}, f"rank{rank}: EP group {ep_ranks} straddles stages {stages}"
        assert ep_ranks == [c.stage_base_rank + i for i in range(4)]


def test_p4_expert_replicas_confined_to_the_stage():
    """With 2 EP groups per stage, expert replicas span the stage's groups but never cross stages."""
    for rank in range(8):
        c = _two_node_stage(rank, pp_size=2, ep_size=2)
        assert c.num_ep_groups == 2, "4 ranks per stage / ep_group_size 2"
        replicas = c.get_expert_replica_ranks()
        stages = {r // c.stage_world_size for r in replicas}
        assert stages == {c.pp_rank}, f"rank{rank}: replicas {replicas} cross stages {stages}"
        assert len(replicas) == c.num_ep_groups


def test_p4_ep_group_partition_is_exact():
    """Union of all EP groups = the world, each rank in exactly one group (no overlap, no orphan)."""
    covered = []
    for rank in range(8):
        covered.append(tuple(_two_node_stage(rank, pp_size=2, ep_size=2).get_ep_group_ranks()))
    groups = set(covered)
    flat = [r for g in groups for r in g]
    assert sorted(flat) == list(range(8)), f"EP groups do not tile the world exactly: {sorted(flat)}"


# P5 — config-time rejections


def test_p5_rejects_pp_not_dividing_world():
    with pytest.raises(ValueError, match="divisible by pp_size"):
        _two_node_stage(0, pp_size=3)


def test_p5_rejects_bad_pp_size_and_schedule():
    with pytest.raises(ValueError, match="pp_size must be >= 1"):
        _two_node_stage(0, pp_size=0)
    with pytest.raises(ValueError, match="pp_schedule must be"):
        _two_node_stage(0, pp_size=2, pp_schedule="dualpipe")
    with pytest.raises(ValueError, match="pp_microbatches must be"):
        _two_node_stage(0, pp_size=2, pp_microbatches=-1)


def test_p5_accepts_pp_with_ep_inside_a_stage():
    """The supported shapes must NOT be rejected — a validator that rejects everything also passes
    the rejection tests above."""
    for kwargs in ({"ep_size": 4}, {"ep_size": 2}, {}):
        c = _two_node_stage(0, pp_size=2, **kwargs)
        assert c.is_pp_mode and c.pp_size == 2, f"{kwargs} should be accepted with PP"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
