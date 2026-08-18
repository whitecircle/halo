#!/usr/bin/env python3
"""How pipeline parallelism composes with the data-parallel axes, and where it stops.

PP is the outermost dimension: the world splits into ``pp_size`` contiguous rank blocks and every
other mode runs unchanged inside one block. Three consequences are load-bearing and none of them is
visible from a single-node run — which is the only kind that can be launched here:

* ``data_parallel_size`` divides the STAGE's world, not the job's. A formula that omits ``pp_size``
  gets exactly this wrong, so the per-node-count table below is pinned.
* A stage must own whole NVLink domains, which makes the legal ``pp_size`` values the divisors of
  the domain count — and makes PP unreachable on a single node.
* Therefore a node never straddles a stage. ``is_pp_shard_writer`` relies on that to pick one
  checkpoint writer per NODE on a non-shared filesystem; if it failed, two stages' shards would be
  written by ranks that disagree about which layers they hold.

Usage:
    python tests/cpu/parallelism/test_pp_data_parallel_composition.py
"""

import sys
from unittest.mock import patch

import pytest

from src.distributed import mesh as mesh_mod
from src.distributed.checkpoint.save import is_pp_shard_writer
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.parallelism import make_parallelism_config

GPUS = 8


def _config(rank: int = 0, world: int = 16, gpus_per_node: int = GPUS, **kwargs) -> ParallelismConfig:
    return make_parallelism_config(world_size=world, gpus_per_node=gpus_per_node, rank=rank, **kwargs)


# (nodes, pp_size, extra kwargs, expected data_parallel_size) on 8-GPU nodes with domain == node.
# dp = (world / pp) / max(cp, tp, expert_tp); EP never divides it.
_DP_CASES = [
    (1, 1, {}, 8),
    (2, 1, {}, 16),
    (2, 2, {}, 8),
    (2, 2, {"ep_size": 8}, 8),  # EP is orthogonal: a stage's whole width stays data-parallel
    (2, 2, {"ep_size": 2}, 8),
    (3, 3, {}, 8),
    (4, 2, {}, 16),
    (4, 4, {}, 8),
    (4, 2, {"ep_size": 8}, 16),
    (8, 2, {}, 32),
    (8, 4, {}, 16),
    (8, 8, {}, 8),
]


@pytest.mark.parametrize("nodes,pp,kwargs,expected_dp", _DP_CASES)
def test_data_parallel_size_divides_the_stage_world(nodes, pp, kwargs, expected_dp):
    """dp counts a STAGE's distinct batches. Using the job's world instead would overstate it
    ``pp_size``-fold, and the sampler would shard the dataset into slices no rank ever reads."""
    config = _config(world=nodes * GPUS, pp_size=pp, **kwargs)
    assert config.stage_world_size == nodes * GPUS // pp
    assert config.data_parallel_size == expected_dp, (nodes, pp, kwargs, config.data_parallel_size)


@pytest.mark.parametrize("nodes", [1, 2, 3, 4, 8])
def test_legal_pp_sizes_are_the_divisors_of_the_domain_count(nodes):
    """A stage owns whole NVLink domains, so on N single-domain nodes the legal pp sizes are exactly
    the divisors of N — in particular PP cannot run on one node, which is why no GPU test here
    launches PP without simulating smaller domains."""
    world = nodes * GPUS
    accepted = []
    for pp in range(1, world + 1):
        try:
            _config(world=world, pp_size=pp)
        except ValueError:
            continue
        accepted.append(pp)
    assert accepted == [pp for pp in range(1, nodes + 1) if nodes % pp == 0], (nodes, accepted)


def test_single_node_pp_is_rejected_for_the_domain_reason():
    """The message must name the domain rule, not divisibility: pp2 DOES divide 8 evenly."""
    with pytest.raises(ValueError, match="Stage boundaries must fall on NVLink-domain boundaries"):
        _config(world=GPUS, pp_size=2)


def test_one_rank_per_stage_is_rejected_honestly():
    """Reachable only when a domain is one GPU. The FSDP wrap skips itself at dp <= 1, so without
    this the run would die on a wrap-contract assert that blames a bug elsewhere."""
    with pytest.raises(ValueError, match="single rank"):
        _config(world=4, gpus_per_node=1, nvlink_domain_size=1, pp_size=4)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"use_hsdp": True}, "PP \\+ HSDP is not supported"),
        ({"fsdp_reshard_after_forward": True}, "FULL_SHARD / ZeRO-3"),
        ({"lowp_precision": "fp8"}, "lowp_precision"),
        ({"fsdp_shard_ep1_experts": False}, "fsdp_shard_ep1_experts=False"),
        ({"cp_size": 2}, "not a supported parallelism combination"),
        ({"tp_size": 2}, "not a supported parallelism combination"),
        ({"ep_size": 2, "tp_size": 2}, "not a supported parallelism combination"),
        ({"ep_size": 2, "expert_tp_size": 2}, "not a supported parallelism combination"),
    ],
)
def test_pp_rejects_the_dp_modes_it_cannot_compose_with(kwargs, expected):
    """Each of these is rejected at CONFIG time — before a model loads — with its own mechanism."""
    with pytest.raises(ValueError, match=expected):
        _config(world=2 * GPUS, pp_size=2, **kwargs)


def test_zero2_is_the_only_fsdp_mode_pp_accepts():
    """Anti-vacuity for the ZeRO-3 row above: the same shape with ZeRO-2 must build."""
    config = _config(world=2 * GPUS, pp_size=2, fsdp_reshard_after_forward=False)
    assert config.fsdp_reshard_after_forward is False
    assert config.dp_replicate_size == 1 and config.dp_shard_size == config.stage_world_size


@pytest.mark.parametrize("nodes,pp,domain", [(2, 2, 8), (4, 2, 8), (4, 4, 8), (8, 4, 8), (2, 2, 2), (4, 2, 4)])
def test_a_node_never_straddles_a_stage(nodes, pp, domain):
    """The invariant ``is_pp_shard_writer`` rests on for non-shared filesystems: with one writer per
    NODE, every writer must hold exactly one stage's layers."""
    world = nodes * GPUS
    gpus_per_node = min(domain, GPUS)
    config = _config(world=world, gpus_per_node=gpus_per_node, nvlink_domain_size=domain, pp_size=pp)
    assert config.stage_world_size % gpus_per_node == 0, config.stage_world_size
    stage_of = [
        _config(rank=r, world=world, gpus_per_node=gpus_per_node, nvlink_domain_size=domain, pp_size=pp).pp_rank
        for r in range(world)
    ]
    for node in range(world // gpus_per_node):
        ranks = range(node * gpus_per_node, (node + 1) * gpus_per_node)
        assert len({stage_of[r] for r in ranks}) == 1, (node, [stage_of[r] for r in ranks])


def test_the_non_shared_fs_writer_set_is_one_rank_per_node_per_stage():
    """Shared FS: one writer per stage. Non-shared: one per node — so a stage spanning k nodes
    writes its shard k times, which is duplication, never a missing shard."""
    world, domain, pp = 32, 8, 2
    configs = [_config(rank=r, world=world, nvlink_domain_size=domain, pp_size=pp) for r in range(world)]
    with patch("src.distributed.checkpoint.save.is_local_main_process", side_effect=lambda: False):
        shared = [is_pp_shard_writer(c, shared_fs=True) for c in configs]
    assert [r for r, w in enumerate(shared) if w] == [0, 16], shared  # stage_local_rank == 0
    for rank, c in enumerate(configs):
        with patch(
            "src.distributed.checkpoint.save.is_local_main_process",
            side_effect=lambda rank=rank: rank % GPUS == 0,
        ):
            assert is_pp_shard_writer(c, shared_fs=False) == (rank % GPUS == 0)


@pytest.mark.parametrize(
    "nodes,pp,ep,scope", [(2, 2, 8, "node"), (4, 2, 8, "node"), (4, 2, 16, "global"), (8, 4, 2, "node")]
)
def test_no_intra_stage_group_crosses_a_pipeline_boundary(nodes, pp, ep, scope):
    """Only PP's point-to-point activations may leave a stage. Every EP group (and the expert-replica
    group that averages across them) must live inside one stage, or a collective would join ranks
    holding different layers."""
    world = nodes * GPUS
    for rank in range(world):
        c = _config(rank=rank, world=world, pp_size=pp, ep_size=ep, ep_scope=scope)
        lo = c.stage_base_rank
        hi = lo + c.stage_world_size
        assert all(lo <= r < hi for r in c.get_ep_group_ranks()), (rank, c.get_ep_group_ranks())
        assert all(lo <= r < hi for r in c.get_expert_replica_ranks()), (rank, c.get_expert_replica_ranks())
        # A pipeline chain, by contrast, must span every stage exactly once.
        chain = c.get_pp_group_ranks()
        assert len(chain) == pp and len({r // c.stage_world_size for r in chain}) == pp, chain


def test_a_stage_sized_hsdp_mesh_is_refused_rather_than_silently_misbuilt():
    """Why PP + HSDP cannot simply be un-rejected.

    ``init_device_mesh`` numbers a mesh ``0..prod(shape)-1`` and does NOT check it against the world,
    so a 2-D mesh sized to a pipeline STAGE would hand every rank a mesh made of the FIRST stage's
    ranks — stage 1 would then reduce-scatter over stage 0's ranks, silently. ``create_dp_mesh``
    refuses that shape instead; a real PP+HSDP needs the pipeline as an outer mesh dimension.
    """
    with (
        patch.object(mesh_mod.dist, "is_initialized", return_value=True),
        patch.object(mesh_mod.dist, "get_world_size", return_value=16),
    ):
        with pytest.raises(ValueError, match="must span the whole world"):
            mesh_mod.create_dp_mesh(8, dp_replicate_size=2, device_type="cpu")


def test_nvl72_pipeline_needs_more_than_one_rack():
    """With NVLINK_DOMAIN_SIZE=72 a stage must own whole racks, so one rack cannot be pipelined and
    two racks give pp2 — the case where nvlink_domain_size != gpus_per_node changes the answer."""
    with pytest.raises(ValueError, match="Stage boundaries must fall on NVLink-domain boundaries"):
        _config(world=72, gpus_per_node=8, nvlink_domain_size=72, pp_size=2)
    config = _config(world=144, gpus_per_node=8, nvlink_domain_size=72, pp_size=2)
    assert config.stage_world_size == 72
    assert config.num_nvlink_domains == 1  # one rack per stage
    assert config.data_parallel_size == 72
    # A stage spanning 9 OS nodes still never straddles a stage boundary.
    assert config.stage_world_size % config.gpus_per_node == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
