"""PP checkpoint writer selection adapts to the filesystem: one rank per stage on shared, one per stage per node on
per-node storage. A configured node never straddles a stage."""

import pytest

from halo.checkpoint.parallel import pp as pp_mod
from halo.checkpoint.parallel.pp import is_pp_shard_writer
from tests.common.parallelism import make_parallelism_config


def _configs(world_size: int, gpus_per_node: int, pp_size: int, domain: int | None = None):
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
        # pp>1 within one node is rejected at config time, so no single-node case.
        (16, 8, 2, None),
        (32, 8, 4, None),
        (32, 8, 2, 16),
        (144, 8, 2, 72),  # NVL72: domain = 72 GPUs over 9 nodes; each stage is 9 nodes.
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
        monkeypatch.setattr(pp_mod, "is_local_main_process", lambda r=rank: r % gpus_per_node == 0)
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
    configs = _configs(world_size, gpus_per_node, pp_size, domain)
    for node in range(world_size // gpus_per_node):
        stages = {configs[r].pp_rank for r in range(node * gpus_per_node, (node + 1) * gpus_per_node)}
        assert len(stages) == 1, f"node {node} spans stages {sorted(stages)} — a per-node writer would lose one"


@pytest.mark.parametrize(
    "world_size,launch_node,gpus_per_node,pp_size",
    [(8, 8, 2, 2), (8, 8, 2, 4), (24, 8, 4, 2)],
    ids=["one-host-two-stages", "one-host-four-stages", "stage-starts-mid-node"],
)
def test_a_launch_node_running_several_stages_writes_each_of_them(
    world_size, launch_node, gpus_per_node, pp_size, monkeypatch
):
    """Its local rank 0 alone would write only its own stage's shard beside an index naming every stage's."""
    configs = _configs(world_size, gpus_per_node, pp_size, gpus_per_node)
    writers = []
    for rank, config in enumerate(configs):
        monkeypatch.setattr(pp_mod, "is_local_main_process", lambda r=rank: r % launch_node == 0)
        if is_pp_shard_writer(config, shared_fs=False):
            writers.append(rank)
    for node in range(world_size // launch_node):
        ranks = range(node * launch_node, (node + 1) * launch_node)
        runs = sorted({configs[r].pp_rank for r in ranks})
        writes = sorted(configs[r].pp_rank for r in writers if r in ranks)
        assert writes == runs, f"launch node {node} runs stages {runs} but writes {writes}"


def test_non_shared_fs_is_not_the_same_rule_as_shared():
    """Anti-vacuity: at more than one node per stage the rules must differ."""
    configs = _configs(32, 8, 2, None)  # 4 nodes, 2 stages ⇒ 2 nodes per stage.
    shared = {c.global_rank for c in configs if is_pp_shard_writer(c, shared_fs=True)}
    assert len(shared) == 2
    assert shared != {0, 8, 16, 24}, "shared-FS rule must NOT already be one writer per node"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
