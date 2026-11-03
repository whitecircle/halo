#!/usr/bin/env python3
"""How pipeline parallelism composes with data-parallel axes: PP is outermost, the world splits into ``pp_size``
blocks. A stage owns whole NVLink domains, which makes PP unreachable on one node."""

from unittest.mock import patch

import pytest

from halo.checkpoint.parallel.pp import is_pp_shard_writer
from halo.distributed import mesh as mesh_mod
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.lowp.spec import LowpSpec
from tests.common.parallelism import built_ep_config, make_parallelism_config

GPUS = 8


def _config(rank: int = 0, world: int = 16, gpus_per_node: int = GPUS, **kwargs) -> ParallelismConfig:
    return make_parallelism_config(world_size=world, gpus_per_node=gpus_per_node, rank=rank, **kwargs)


# dp = (world / pp) / max(cp, tp, expert_tp); EP never divides it. On 8-GPU nodes with domain == node.
_DP_CASES = [
    (1, 1, {}, 8),
    (2, 1, {}, 16),
    (2, 2, {}, 8),
    (2, 2, {"ep_size": 8}, 8),  # EP is orthogonal.
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
    """Overstated ``pp_size``-fold, the sampler would shard into slices no rank ever reads."""
    config = _config(world=nodes * GPUS, pp_size=pp, **kwargs)
    assert config.stage_world_size == nodes * GPUS // pp
    assert config.data_parallel_size == expected_dp, (nodes, pp, kwargs, config.data_parallel_size)


@pytest.mark.parametrize("nodes", [1, 2, 3, 4, 8])
def test_legal_pp_sizes_are_the_divisors_of_the_domain_count(nodes):
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
    """pp2 does divide 8 evenly, so the message must name the domain rule."""
    with pytest.raises(ValueError, match="Stage boundaries must fall on NVLink-domain boundaries"):
        _config(world=GPUS, pp_size=2)


def test_one_rank_per_stage_is_rejected_honestly():
    """The FSDP wrap skips itself at dp <= 1, so the run would die on an assert that blames a bug elsewhere."""
    with pytest.raises(ValueError, match="single rank"):
        _config(world=4, gpus_per_node=1, nvlink_domain_size=1, pp_size=4)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"use_hsdp": True}, "PP \\+ HSDP is not supported"),
        ({"fsdp_reshard_after_forward": True}, "FULL_SHARD / ZeRO-3"),
        ({"lowp": LowpSpec(precision="fp8")}, "lowp_precision"),
        ({"fsdp_shard_ep1_experts": False}, "fsdp_shard_ep1_experts=False"),
        ({"cp_size": 2}, "not a supported parallelism combination"),
        ({"tp_size": 2}, "not a supported parallelism combination"),
        ({"ep_size": 2, "tp_size": 2}, "not a supported parallelism combination"),
        ({"ep_size": 2, "expert_tp_size": 2}, "not a supported parallelism combination"),
    ],
)
def test_pp_rejects_the_dp_modes_it_cannot_compose_with(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        _config(world=2 * GPUS, pp_size=2, **kwargs)


def test_zero2_is_the_only_fsdp_mode_pp_accepts():
    """Anti-vacuity for the ZeRO-3 row."""
    config = _config(world=2 * GPUS, pp_size=2, fsdp_reshard_after_forward=False)
    assert config.fsdp_reshard_after_forward is False
    assert config.dp_replicate_size == 1 and config.dp_shard_size == config.stage_world_size


@pytest.mark.parametrize("nodes,pp,domain", [(2, 2, 8), (4, 2, 8), (4, 4, 8), (8, 4, 8), (2, 2, 2), (4, 2, 4)])
def test_a_node_never_straddles_a_stage(nodes, pp, domain):
    """``is_pp_shard_writer`` rests on this for non-shared filesystems."""
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
    """A stage spanning k nodes writes its shard k times: duplication, never a missing shard."""
    world, domain, pp = 32, 8, 2
    configs = [_config(rank=r, world=world, nvlink_domain_size=domain, pp_size=pp) for r in range(world)]
    with patch("halo.checkpoint.parallel.pp.is_local_main_process", side_effect=lambda: False):
        shared = [is_pp_shard_writer(c, shared_fs=True) for c in configs]
    assert [r for r, w in enumerate(shared) if w] == [0, 16], shared  # stage_local_rank == 0.
    for rank, c in enumerate(configs):
        with patch(
            "halo.checkpoint.parallel.pp.is_local_main_process",
            side_effect=lambda rank=rank: rank % GPUS == 0,
        ):
            assert is_pp_shard_writer(c, shared_fs=False) == (rank % GPUS == 0)


@pytest.mark.parametrize(
    "nodes,pp,ep,scope", [(2, 2, 8, "node"), (4, 2, 8, "node"), (4, 2, 16, "global"), (8, 4, 2, "node")]
)
def test_no_intra_stage_group_crosses_a_pipeline_boundary(nodes, pp, ep, scope):
    """A collective would otherwise join ranks holding different layers."""
    world = nodes * GPUS
    for rank in range(world):
        c = _config(rank=rank, world=world, pp_size=pp, ep_size=ep, ep_scope=scope)
        lo = c.stage_base_rank
        hi = lo + c.stage_world_size
        ep_config = built_ep_config(c)
        assert all(lo <= r < hi for r in ep_config._my_ep_group_ranks), (rank, ep_config._my_ep_group_ranks)
        assert all(lo <= r < hi for r in ep_config.expert_replica_ranks), (rank, ep_config.expert_replica_ranks)
        # A pipeline chain spans every stage exactly once.
        chain = c.pp_chain_ranks()
        assert len(chain) == pp and len({r // c.stage_world_size for r in chain}) == pp, chain


def test_a_stage_sized_hsdp_mesh_is_refused_rather_than_silently_misbuilt():
    """``init_device_mesh`` numbers ``0..prod(shape)-1`` unchecked, so a stage-sized mesh hands stage 1 the first
    stage's ranks; ``build_dp_mesh`` refuses that."""
    with (
        patch.object(mesh_mod.dist, "is_initialized", return_value=True),
        patch.object(mesh_mod.dist, "get_world_size", return_value=16),
    ):
        with pytest.raises(ValueError, match="must span the whole world"):
            mesh_mod.build_dp_mesh(8, dp_replicate_size=2, device_type="cpu")


def test_nvl72_pipeline_needs_more_than_one_rack():
    """The case where nvlink_domain_size != gpus_per_node changes the answer."""
    with pytest.raises(ValueError, match="Stage boundaries must fall on NVLink-domain boundaries"):
        _config(world=72, gpus_per_node=8, nvlink_domain_size=72, pp_size=2)
    config = _config(world=144, gpus_per_node=8, nvlink_domain_size=72, pp_size=2)
    assert config.stage_world_size == 72
    assert config.num_nvlink_domains == 1  # One rack per stage.
    assert config.data_parallel_size == 72
    assert config.stage_world_size % config.gpus_per_node == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
