#!/usr/bin/env python
"""``node_placement_contradiction``: node-local groups are contiguous rank blocks. A round-robin launcher straddles
machines while ``requires_rdma`` reports False, sending DeepEP's IPC to another host."""

from unittest.mock import patch

import pytest

from halo.distributed.nvlink import NO_FABRIC, node_placement_contradiction, validate_nvlink_domain_against_fabric

_MOD = "halo.distributed.nvlink"


def _block(num_nodes: int, gpus_per_node: int) -> list[str]:
    return [f"host{rank // gpus_per_node}" for rank in range(num_nodes * gpus_per_node)]


def _cyclic(num_nodes: int, gpus_per_node: int) -> list[str]:
    """``srun --distribution=cyclic``."""
    return [f"host{rank % num_nodes}" for rank in range(num_nodes * gpus_per_node)]


def test_block_placement_is_accepted():
    assert node_placement_contradiction(_block(4, 8), gpus_per_node=8) is None


def test_one_process_per_node_is_accepted():
    """Declares gpus_per_node=1, so distinct hostnames are correct."""
    assert node_placement_contradiction([f"host{rank}" for rank in range(8)], gpus_per_node=1) is None


def test_a_node_holding_several_blocks_is_accepted():
    """Only a block spanning machines is this verdict's business."""
    assert node_placement_contradiction(_block(2, 8), gpus_per_node=4) is None


def test_cyclic_placement_is_rejected():
    """A node-local ep8 group would be half remote while requires_rdma reports False."""
    reason = node_placement_contradiction(_cyclic(2, 8), gpus_per_node=8)
    assert reason is not None
    assert "not placed 8 per node" in reason
    # It must name the offending block and its machines.
    assert "host0" in reason and "host1" in reason


def test_a_single_straddling_block_among_correct_ones_is_rejected():
    """A verdict firing only when every block straddled would pass partial misplacement."""
    hosts = _block(4, 8)
    hosts[9] = "host3"  # Rank 9 belongs to block 1 (host1) but sits on host3.
    reason = node_placement_contradiction(hosts, gpus_per_node=8)
    assert reason is not None and "[1]" in reason


def test_an_undeclared_node_width_is_not_judged():
    """``gpus_per_node=0`` is the caller's skip spelling, and would divide by zero."""
    assert node_placement_contradiction(_cyclic(2, 8), gpus_per_node=0) is None
    assert node_placement_contradiction([], gpus_per_node=8) is None


def test_a_trailing_partial_block_is_not_invented():
    """ParallelismConfig owns that rejection; firing here gives the wrong diagnosis."""
    assert node_placement_contradiction(_block(2, 8) + ["host9"], gpus_per_node=8) is None


def _run_fabric_check(hosts: list[str], cliques: list[int | None], gpus_per_node: int):
    def fake_all_gather(out_list, obj):
        out_list[:] = list(zip(cliques, [obj[1]] * len(hosts), hosts, strict=True))

    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=True),
        patch(f"{_MOD}.dist.get_world_size", return_value=len(hosts)),
        patch(f"{_MOD}.dist.all_gather_object", side_effect=fake_all_gather),
        patch(f"{_MOD}.nvlink_fabric_clique_id", return_value=cliques[0]),
        patch(f"{_MOD}.is_global_main_process", return_value=True),
    ):
        validate_nvlink_domain_against_fabric(gpus_per_node, len(hosts), gpus_per_node)


def test_the_placement_verdict_reaches_the_collective_check():
    """The pure verdict is worthless unless the gathered identities drive it."""
    with pytest.raises(ValueError, match="not placed 8 per node"):
        _run_fabric_check(_cyclic(2, 8), [NO_FABRIC] * 16, gpus_per_node=8)


def test_the_placement_verdict_does_not_need_a_readable_clique():
    """The fabric legs no-op on an unreadable clique, and on NVL8 every rank reports one sentinel."""
    with pytest.raises(ValueError, match="not placed 8 per node"):
        _run_fabric_check(_cyclic(2, 8), [None] * 16, gpus_per_node=8)


def test_correct_placement_still_passes_the_collective_check():
    _run_fabric_check(_block(2, 8), [NO_FABRIC] * 16, gpus_per_node=8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
