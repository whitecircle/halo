#!/usr/bin/env python
"""``validate_nvlink_domain_against_fabric``: the declared NVLink domain vs the one the GPUs report. A COLLECTIVE
inside ``ParallelismConfig.__post_init__``, so it never raises on a subset of ranks: a partial raise is a hang."""

import subprocess
from unittest.mock import patch

import pytest

from halo.distributed.nvlink import (
    NO_FABRIC,
    _fabric_clique_for_device,
    _parse_fabric_clique,
    nvlink_fabric_clique_id,
    validate_nvlink_domain_against_fabric,
)
from halo.distributed.parallelism_config import ParallelismConfig

_MOD = "halo.distributed.nvlink"
_CONFIG_MOD = "halo.distributed.parallelism_config"


def _run(
    domain: int,
    world: int,
    cliques: list[int | None],
    local: int | None = 0,
    gpus_per_node: int = 0,
    node_widths: list[int] | None = None,
    hosts: list[str] | None = None,
):
    """``gpus_per_node`` comes from the launcher per node, so a heterogeneous job can disagree; ``hosts`` defaults
    to block placement."""

    def fake_all_gather(out_list, obj):
        widths = node_widths if node_widths is not None else [obj[1]] * len(cliques)
        placement = hosts
        if placement is None:
            width = gpus_per_node or len(cliques)
            placement = [f"node{rank // width}" for rank in range(len(cliques))]
        out_list[:] = list(zip(cliques, widths, placement, strict=True))

    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=True),
        patch(f"{_MOD}.dist.get_world_size", return_value=len(cliques)),
        patch(f"{_MOD}.dist.all_gather_object", side_effect=fake_all_gather),
        patch(f"{_MOD}.nvlink_fabric_clique_id", return_value=local),
        patch(f"{_MOD}.is_global_main_process", return_value=True),
    ):
        validate_nvlink_domain_against_fabric(domain, world, gpus_per_node)


def test_a_domain_wider_than_a_node_is_rejected_when_no_rank_has_a_fabric():
    """The NVL8 hole: with no fabric there are no cliques to straddle, yet groups cross the network while
    ``requires_rdma`` reports False."""
    with pytest.raises(ValueError, match="every rank reports NO NVLink fabric"):
        _run(domain=16, world=32, cliques=[NO_FABRIC] * 32, local=NO_FABRIC, gpus_per_node=8)


def test_a_domain_within_a_node_is_fine_without_a_fabric():
    _run(domain=8, world=32, cliques=[NO_FABRIC] * 32, local=NO_FABRIC, gpus_per_node=8)


def test_an_unreadable_fabric_still_no_ops_even_when_wider_than_a_node():
    """An NVL72 rack mid-registration would otherwise be rejected for declaring the domain it has."""
    _run(domain=16, world=32, cliques=[None] * 32, local=None, gpus_per_node=8)


def test_a_heterogeneous_node_width_cannot_raise_on_a_subset():
    """Deciding from this rank's own width would raise on the narrow node while the wide node walks into the next
    collective."""
    widths = [8] * 16 + [4] * 16
    verdicts = []
    for rank_width in (8, 4):
        try:
            _run(
                domain=8,
                world=32,
                cliques=[NO_FABRIC] * 32,
                local=NO_FABRIC,
                gpus_per_node=rank_width,
                node_widths=widths,
            )
            verdicts.append("pass")
        except ValueError:
            verdicts.append("raise")
    assert len(set(verdicts)) == 1, f"ranks disagreed on the width verdict: {verdicts}"


def test_domain_straddling_two_cliques_is_rejected():
    with pytest.raises(ValueError, match="span more than one fabric clique"):
        _run(domain=8, world=8, cliques=[1, 1, 1, 1, 2, 2, 2, 2])


def test_domain_matching_the_clique_passes():
    _run(domain=4, world=8, cliques=[1, 1, 1, 1, 2, 2, 2, 2])
    _run(domain=8, world=8, cliques=[7] * 8)


def test_an_unreadable_clique_on_any_rank_disables_the_check():
    """A partial view is not evidence, and a raise on a subset of ranks would hang the job."""
    _run(domain=8, world=8, cliques=[1, 1, 1, None, 2, 2, 2, 2])


def test_a_declared_world_that_disagrees_with_the_group_is_not_judged():
    """Sizing the buffer from the declared world would read the pre-fill as cliques or raise ``IndexError``."""
    _run(domain=8, world=16, cliques=[1, 1, 1, 1, 2, 2, 2, 2])
    _run(domain=4, world=4, cliques=[1, 1, 1, 1, 2, 2, 2, 2])


def test_no_process_group_is_a_no_op():
    with patch(f"{_MOD}.dist.is_initialized", return_value=False):
        validate_nvlink_domain_against_fabric(8, 8)
    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=True),
        patch(f"{_MOD}.dist.get_world_size", return_value=1),
        patch(f"{_MOD}.dist.all_gather_object", side_effect=AssertionError("must not gather")),
    ):
        validate_nvlink_domain_against_fabric(1, 1)  # A one-rank group returns before the collective.


def test_a_wider_clique_than_declared_warns_but_runs():
    """Under-claiming only leaves NVLink width unused."""
    with patch(f"{_MOD}.logger") as logger:
        _run(domain=4, world=8, cliques=[9] * 8)
    assert logger.warning.called, "leaving half the fabric unused should say so"


def test_a_per_node_domain_drift_is_rejected_before_this_collective():
    """The verdict belongs to ``__post_init__`` before the topology collectives, since this check would see the
    drift only after the MNNVL gate split the world on it."""
    order: list[str] = []
    with (
        patch(f"{_CONFIG_MOD}.get_global_world_size", return_value=8),
        patch(f"{_CONFIG_MOD}.get_local_world_size", return_value=8),
        patch(f"{_CONFIG_MOD}.get_global_rank", return_value=0),
        patch(f"{_CONFIG_MOD}.is_global_main_process", return_value=False),
        patch(f"{_CONFIG_MOD}.verify_mnnvl_prerequisites", side_effect=lambda *a: order.append("mnnvl")),
        patch(f"{_CONFIG_MOD}.validate_nvlink_domain_against_fabric", side_effect=lambda *a: order.append("fabric")),
        patch(f"{_CONFIG_MOD}.reject_divergent_settings") as divergent,
    ):
        divergent.side_effect = lambda values, *a: order.append(f"divergent:{sorted(values)}")
        ParallelismConfig(world_size=8, gpus_per_node=8, nvlink_domain_size=8)

    assert order[0] == "divergent:['gpus_per_node', 'nvlink_domain_size']", (
        f"the per-node drift must be gathered before either topology collective: {order}"
    )
    assert order[1:] == ["mnnvl", "fabric"]


def test_every_rank_reaches_the_same_verdict():
    """Drive it once per rank with that rank's own clique: agreement keeps a misconfiguration from becoming a hang."""
    cliques = [1, 1, 1, 1, 2, 2, 2, 2]
    verdicts = []
    for rank_local in cliques:
        try:
            _run(domain=8, world=8, cliques=cliques, local=rank_local)
            verdicts.append("pass")
        except ValueError:
            verdicts.append("raise")
    assert len(set(verdicts)) == 1, f"ranks disagreed: {verdicts}"
    assert verdicts[0] == "raise"


class _Props:
    """``torch.cuda.get_device_properties`` exposes the PCI address as three ints, never a string."""

    pci_domain_id = 0
    pci_bus_id = 4
    pci_device_id = 0


_FABRIC_QUERY = """GPU 00000000:04:00.0
    Product Name                          : NVIDIA B300
    GPU Fabric GUID                       : 0x0000000000000000
    Inforom Version
        State                             : NotAFabricField
        CliqueId                          : 999
    Fabric
        State                             : {state}
        Status                            : Success
        CliqueId                          : {clique}
        ClusterUUID                       : 00000000-0000-0000-0000-000000000000
    Clocks
        State                             : AlsoNotAFabricField
"""


def _read_clique(state: str, clique: str) -> tuple[int | None, list[list[str]]]:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=_FABRIC_QUERY.format(state=state, clique=clique))

    # The read is memoized per device, so simulating different hardware drops the memo first.
    _fabric_clique_for_device.cache_clear()
    with (
        patch(f"{_MOD}.subprocess.run", side_effect=fake_run),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_properties", return_value=_Props()),
    ):
        return nvlink_fabric_clique_id(), calls


def test_the_clique_read_actually_reports_a_fabric():
    """NVML's Python binding raises on the int PCI address and segfaults on the fabric getter; out of process, a
    wrong driver pair is a non-zero exit."""
    clique, calls = _read_clique(state="Completed", clique="7")

    assert clique == 7, "a COMPLETED fabric with clique 7 must read back as 7, not None"
    assert calls and calls[0][0] == "nvidia-smi", "the clique must be read out of process"
    assert "00000000:04:00.0" in calls[0], "the GPU is selected by its composed PCI address, not an index"


def test_the_clique_is_read_from_the_fabric_block_only():
    """The sample has decoys around the block, including a top-level ``GPU Fabric GUID``, so a flat scan picks the
    wrong value."""
    assert _read_clique(state="Completed", clique="7")[0] == 7


def test_absent_and_unregistered_fabrics_are_told_apart():
    """Only a definite negative lets the cross-check reject a too-wide domain; ``None`` must keep meaning a partial
    view is not evidence."""
    assert _read_clique(state="Completed", clique="0")[0] == NO_FABRIC
    assert _read_clique(state="In Progress", clique="7")[0] is None
    assert _read_clique(state="N/A", clique="N/A")[0] is None


def test_a_dump_without_a_fabric_block_is_a_definite_negative():
    """A plain HGX/NVL8 part emits no ``Fabric`` block at all."""
    assert _parse_fabric_clique("GPU 00000000:01:00.0\n    Product Name : NVIDIA B200\n") == NO_FABRIC


def test_a_failing_nvidia_smi_is_unreadable_not_a_crash():
    for boom in (FileNotFoundError, subprocess.TimeoutExpired("nvidia-smi", 30)):
        _fabric_clique_for_device.cache_clear()
        with (
            patch(f"{_MOD}.subprocess.run", side_effect=boom),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties", return_value=_Props()),
        ):
            assert nvlink_fabric_clique_id() is None


def test_the_clique_read_is_memoized_per_device():
    """The read costs ~60 ms and the CPU config grid builds tens of thousands of configs. Correct because fabric
    membership is fixed once the driver is up."""
    _fabric_clique_for_device.cache_clear()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=_FABRIC_QUERY.format(state="Completed", clique="7"))

    with (
        patch(f"{_MOD}.subprocess.run", side_effect=fake_run),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_properties", return_value=_Props()),
    ):
        first = nvlink_fabric_clique_id()
        repeats = [nvlink_fabric_clique_id() for _ in range(5)]

    assert first == 7
    assert repeats == [7] * 5, "the memo must return the clique, not None"
    assert len(calls) == 1, f"nvidia-smi ran {len(calls)} times for one device; the read is not memoized"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
