#!/usr/bin/env python
"""Every topology a shipped diagram depicts is one ``ParallelismConfig`` accepts, at the DP the figure states."""

from __future__ import annotations

import pytest

from halo.distributed.parallelism_config import ParallelismConfig

# The shapes the PNGs under agent-docs/assets/diagrams draw; redrawing a figure updates its row.
TOPOLOGIES = {
    "ep_group_hierarchy": {"nodes": 1, "gpus": 4, "ep": 2, "tp": 1, "scope": "node", "dp": 4},
    "ep_multi_node_layout": {"nodes": 2, "gpus": 8, "ep": 16, "tp": 8, "scope": "global", "dp": 2},
}


@pytest.mark.parametrize("figure", sorted(TOPOLOGIES))
def test_depicted_topology_is_accepted_by_parallelism_config(figure):
    shape = TOPOLOGIES[figure]
    gpus_per_node = int(shape["gpus"])
    config = ParallelismConfig(
        ep_size=int(shape["ep"]),
        tp_size=int(shape["tp"]),
        ep_scope=str(shape["scope"]),
        world_size=gpus_per_node * int(shape["nodes"]),
        gpus_per_node=gpus_per_node,
        nvlink_domain_size=gpus_per_node,
    )
    assert config.data_parallel_size == int(shape["dp"]), f"{figure} claims a DP the config does not give"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
