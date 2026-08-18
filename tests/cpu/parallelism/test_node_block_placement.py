#!/usr/bin/env python
"""``node_placement_contradiction`` — the rank→machine assumption every node-local group rests on.

``src.distributed.group_layout`` cuts every node-local EP/CP/TP/ETP group as a CONTIGUOUS global-rank
block, and ``ParallelismConfig`` derives ``num_nodes`` as ``world_size // gpus_per_node``. Both are a
machine only while rank ``r`` sits on node ``r // gpus_per_node``. One torchrun per node makes that
true by construction; a launcher that numbers ranks round-robin over the allocation does not, and
nothing else in the toolkit would notice — the groups straddle machines while ``requires_rdma`` still
reports False, so DeepEP takes its intranode CUDA-IPC path to peers on another host.

The verdict is pure so it is testable without a job, and it is taken over the GATHERED machine
identities inside ``validate_nvlink_domain_against_fabric``, ahead of that function's fabric legs:
those no-op wherever a clique is unreadable, and this assumption has to hold anyway.

    python tests/cpu/parallelism/test_node_block_placement.py
"""

from unittest.mock import patch

import pytest

from src.distributed.nvlink import NO_FABRIC, node_placement_contradiction, validate_nvlink_domain_against_fabric

_MOD = "src.distributed.nvlink"


def _block(num_nodes: int, gpus_per_node: int) -> list[str]:
    """The ordinary placement: rank ``r`` on node ``r // gpus_per_node``."""
    return [f"host{rank // gpus_per_node}" for rank in range(num_nodes * gpus_per_node)]


def _cyclic(num_nodes: int, gpus_per_node: int) -> list[str]:
    """Round-robin placement (``srun --distribution=cyclic``): rank ``r`` on node ``r % num_nodes``."""
    return [f"host{rank % num_nodes}" for rank in range(num_nodes * gpus_per_node)]


def test_block_placement_is_accepted():
    """Anti-over-rejection: the shape every documented launch recipe produces must pass."""
    assert node_placement_contradiction(_block(4, 8), gpus_per_node=8) is None


def test_one_process_per_node_is_accepted():
    """One rank per node (a 1-GPU-per-pod scheduler) declares gpus_per_node=1, so every block is a
    single rank and distinct hostnames are correct rather than a straddle."""
    assert node_placement_contradiction([f"host{rank}" for rank in range(8)], gpus_per_node=1) is None


def test_a_node_holding_several_blocks_is_accepted():
    """``gpus_per_node`` narrower than the machine keeps every group INSIDE a machine — safe, and not
    this verdict's business. Only a block spanning machines is."""
    assert node_placement_contradiction(_block(2, 8), gpus_per_node=4) is None


def test_cyclic_placement_is_rejected():
    """The failure this exists to catch: ranks 0..7 land on 2 different hosts, so a 'node-local'
    ep8 group is half remote while requires_rdma reports False."""
    reason = node_placement_contradiction(_cyclic(2, 8), gpus_per_node=8)
    assert reason is not None
    assert "not placed 8 per node" in reason
    # The message must name the offending block and its machines, or it diagnoses nothing.
    assert "host0" in reason and "host1" in reason


def test_a_single_straddling_block_among_correct_ones_is_rejected():
    """One mis-placed rank is enough: its group's dispatch would cross the network. A verdict that
    only fired when EVERY block straddled would pass the realistic partial-misplacement case."""
    hosts = _block(4, 8)
    hosts[9] = "host3"  # rank 9 belongs to block 1 (host1) but sits on host3
    reason = node_placement_contradiction(hosts, gpus_per_node=8)
    assert reason is not None and "[1]" in reason


def test_an_undeclared_node_width_is_not_judged():
    """``gpus_per_node=0`` is the caller's 'skip this leg' spelling (it is optional on the fabric
    check), and 0 would make the block arithmetic divide by zero."""
    assert node_placement_contradiction(_cyclic(2, 8), gpus_per_node=0) is None
    assert node_placement_contradiction([], gpus_per_node=8) is None


def test_a_trailing_partial_block_is_not_invented():
    """``world_size % gpus_per_node != 0`` is ParallelismConfig's own rejection; this verdict must not
    also fire on the ragged tail, or the operator gets the wrong diagnosis for a node-count problem."""
    assert node_placement_contradiction(_block(2, 8) + ["host9"], gpus_per_node=8) is None


def _run_fabric_check(hosts: list[str], cliques: list[int | None], gpus_per_node: int):
    """Drive ``validate_nvlink_domain_against_fabric`` over a fake gather of ``(clique, width, host)``."""

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
    """Wiring pin: the pure verdict is worthless unless the gathered identities actually drive it."""
    with pytest.raises(ValueError, match="not placed 8 per node"):
        _run_fabric_check(_cyclic(2, 8), [NO_FABRIC] * 16, gpus_per_node=8)


def test_the_placement_verdict_does_not_need_a_readable_clique():
    """It must be judged BEFORE the fabric legs: those no-op on an unreadable clique, and on ordinary
    NVL8 hardware every rank reports the same sentinel, so no clique comparison can ever see this."""
    with pytest.raises(ValueError, match="not placed 8 per node"):
        _run_fabric_check(_cyclic(2, 8), [None] * 16, gpus_per_node=8)


def test_correct_placement_still_passes_the_collective_check():
    """Anti-vacuity for the two raises above: the same harness must NOT raise on a good placement."""
    _run_fabric_check(_block(2, 8), [NO_FABRIC] * 16, gpus_per_node=8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
