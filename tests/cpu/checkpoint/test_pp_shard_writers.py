"""PP checkpoint writer selection must adapt to the filesystem.

On a shared filesystem one rank per stage writes into the single output directory. On per-node
storage that leaves every node but one without a checkpoint, so the writer set widens to one rank per
node — each writing its own stage's shard onto its own filesystem.

The widening is only sound because a node never straddles a stage: ``stage_world_size`` is a multiple
of ``nvlink_domain_size``, which is a multiple of ``gpus_per_node``. These tests pin both the writer
selection and that structural guarantee, across one-stage-per-node, multi-node-per-stage, and an
NVL72-shaped topology where one NVLink domain spans nine OS nodes.

    python tests/cpu/checkpoint/test_pp_shard_writers.py
"""

import pytest

from src.distributed.checkpoint import save as save_mod
from src.distributed.checkpoint.save import is_pp_shard_writer
from tests.common.parallelism import make_parallelism_config


def _configs(world_size: int, gpus_per_node: int, pp_size: int, domain: int | None = None):
    """One ParallelismConfig per rank of the job (``domain`` defaults to one domain per node)."""
    return [
        make_parallelism_config(
            world_size=world_size,
            gpus_per_node=gpus_per_node,
            nvlink_domain_size=domain,
            rank=rank,
            pp_size=pp_size,
            max_concurrent_loading=0,
        )
        for rank in range(world_size)
    ]


@pytest.mark.parametrize(
    "world_size,gpus_per_node,pp_size,domain",
    [
        # No single-node case: stage_world_size is a multiple of nvlink_domain_size (>= gpus_per_node),
        # so pp>1 within one node is rejected at config time.
        (16, 8, 2, None),  # 2 nodes, one stage per node
        (32, 8, 4, None),  # 4 nodes, one stage per node
        (32, 8, 2, 16),  # domain spans 2 nodes: each stage is 2 nodes
        (144, 8, 2, 72),  # NVL72: domain = 72 GPUs over 9 nodes; each stage is 9 nodes
    ],
)
def test_shared_fs_writes_exactly_one_shard_per_stage(world_size, gpus_per_node, pp_size, domain):
    configs = _configs(world_size, gpus_per_node, pp_size, domain)
    writers = [c for c in configs if is_pp_shard_writer(c, shared_fs=True)]
    assert len(writers) == pp_size, f"expected one writer per stage, got {len(writers)}"
    assert sorted(c.pp_rank for c in writers) == list(range(pp_size)), "every stage must have a writer"


@pytest.mark.parametrize(
    "world_size,gpus_per_node,pp_size,domain",
    [
        (16, 8, 2, None),
        (32, 8, 4, None),
        (32, 8, 2, 16),
        (144, 8, 2, 72),
    ],
)
def test_non_shared_fs_writes_one_shard_per_node(world_size, gpus_per_node, pp_size, domain, monkeypatch):
    configs = _configs(world_size, gpus_per_node, pp_size, domain)
    n_nodes = world_size // gpus_per_node
    writers = []
    for rank, config in enumerate(configs):
        # is_local_main_process() reads the launcher's LOCAL_RANK; emulate each rank's view.
        monkeypatch.setattr(save_mod, "is_local_main_process", lambda r=rank: r % gpus_per_node == 0)
        if is_pp_shard_writer(config, shared_fs=False):
            writers.append(config)
    assert len(writers) == n_nodes, f"every node needs its own checkpoint, got {len(writers)} of {n_nodes}"
    per_stage = {}
    for c in writers:
        per_stage.setdefault(c.pp_rank, []).append(c)
    assert sorted(per_stage) == list(range(pp_size)), "every stage must be written"
    expected = n_nodes // pp_size
    for pp_rank, group in per_stage.items():
        assert len(group) == expected, f"stage {pp_rank} written by {len(group)} nodes, expected {expected}"


@pytest.mark.parametrize(
    "world_size,gpus_per_node,pp_size,domain",
    [(16, 8, 2, None), (32, 8, 4, None), (32, 8, 2, 16), (144, 8, 2, 72)],
)
def test_a_node_never_straddles_a_stage(world_size, gpus_per_node, pp_size, domain):
    """The premise the per-node writer rests on. If a node held ranks from two stages, its single
    writer would emit one stage's shard and silently drop the other's."""
    configs = _configs(world_size, gpus_per_node, pp_size, domain)
    for node in range(world_size // gpus_per_node):
        stages = {configs[r].pp_rank for r in range(node * gpus_per_node, (node + 1) * gpus_per_node)}
        assert len(stages) == 1, f"node {node} spans stages {sorted(stages)} — a per-node writer would lose one"


def test_non_shared_fs_is_not_the_same_rule_as_shared():
    """Anti-vacuity: at more than one node per stage the two rules must actually differ, or the tests
    above would pass on an unchanged implementation."""
    configs = _configs(32, 8, 2, None)  # 4 nodes, 2 stages ⇒ 2 nodes per stage
    shared = {c.global_rank for c in configs if is_pp_shard_writer(c, shared_fs=True)}
    assert len(shared) == 2
    assert shared != {0, 8, 16, 24}, "shared-FS rule must NOT already be one writer per node"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
