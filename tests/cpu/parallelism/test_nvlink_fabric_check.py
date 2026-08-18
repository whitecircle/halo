#!/usr/bin/env python
"""``validate_nvlink_domain_against_fabric`` — the declared NVLink domain vs the one the GPUs report.

``nvlink_domain_size`` is an unchecked assertion about physics, and both ways of getting it wrong are
silent: a domain that straddles two racks makes every "node-local" group half RDMA while
``requires_rdma`` still reports False, and leaving it at ``gpus_per_node`` on an NVL72 rack caps
expert sharding far below the fabric.

The check is a COLLECTIVE inside ``ParallelismConfig.__post_init__``, so the property that matters
most is that it can never raise on a subset of ranks — a partial raise is a hang, which is worse than
the bug it catches. Every rank decides from the same gathered list, and a single unreadable clique
disables the whole check.

The declared domain is already rank-uniform by the time it gets here: ``__post_init__`` gathers that
verdict through ``reject_divergent_settings`` before either topology collective, which is where the
last test pins it.

    python tests/cpu/parallelism/test_nvlink_fabric_check.py
"""

import subprocess
import sys
from unittest.mock import patch

import pytest

from src.distributed.nvlink import (
    NO_FABRIC,
    _fabric_clique_for_device,
    _parse_fabric_clique,
    nvlink_fabric_clique_id,
    validate_nvlink_domain_against_fabric,
)
from src.distributed.parallelism_config import ParallelismConfig

_MOD = "src.distributed.nvlink"
_CONFIG_MOD = "src.distributed.parallelism_config"


def _run(
    domain: int,
    world: int,
    cliques: list[int | None],
    local: int | None = 0,
    gpus_per_node: int = 0,
    node_widths: list[int] | None = None,
    hosts: list[str] | None = None,
):
    """Drive the check with a fabric whose per-rank clique ids are ``cliques``.

    ``node_widths`` supplies a PER-RANK ``gpus_per_node`` (it comes from the launcher per node, so a
    heterogeneous job really can disagree); ``None`` means every rank reports ``gpus_per_node``.
    ``hosts`` supplies the per-rank machine identity; ``None`` means the ordinary block placement
    (rank ``r`` on node ``r // gpus_per_node``), which is what the fabric legs below assume.
    """

    def fake_all_gather(out_list, obj):
        # Every rank contributes (clique, node width, machine identity).
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
    """The NVL8 hole: with no MNNVL fabric there are no cliques to straddle, so the check below
    cannot see this — yet a 16-wide "node-local" domain on 8-GPU nodes builds EP/TP/CP groups that
    cross the network while ``requires_rdma`` still reports False."""
    with pytest.raises(ValueError, match="every rank reports NO NVLink fabric"):
        _run(domain=16, world=32, cliques=[NO_FABRIC] * 32, local=NO_FABRIC, gpus_per_node=8)


def test_a_domain_within_a_node_is_fine_without_a_fabric():
    """Anti-over-rejection: the ordinary NVL8 job declares nothing wider than its node."""
    _run(domain=8, world=32, cliques=[NO_FABRIC] * 32, local=NO_FABRIC, gpus_per_node=8)


def test_an_unreadable_fabric_still_no_ops_even_when_wider_than_a_node():
    """A rank that could not tell must not be turned into a negative: an NVL72 rack mid-registration
    would otherwise be rejected for declaring the domain it genuinely has."""
    _run(domain=16, world=32, cliques=[None] * 32, local=None, gpus_per_node=8)


def test_a_heterogeneous_node_width_cannot_raise_on_a_subset():
    """``gpus_per_node`` is per-NODE, so the width verdict must come from the GATHERED values.

    Deciding from this rank's own width would raise on the narrow node's ranks while the wide node's
    walked into the next collective — the subset raise this check exists to avoid. Every rank must
    reach the same answer, so drive it once per rank and compare.
    """
    widths = [8] * 16 + [4] * 16  # node A has 8 GPUs, node B has 4
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
    """8 ranks declared as one domain, but ranks 0-3 and 4-7 sit on different fabrics."""
    with pytest.raises(ValueError, match="span more than one fabric clique"):
        _run(domain=8, world=8, cliques=[1, 1, 1, 1, 2, 2, 2, 2])


def test_domain_matching_the_clique_passes():
    """Anti-over-rejection: the shape the NVL72 recipe asks for must not raise."""
    _run(domain=4, world=8, cliques=[1, 1, 1, 1, 2, 2, 2, 2])
    _run(domain=8, world=8, cliques=[7] * 8)


def test_an_unreadable_clique_on_any_rank_disables_the_check():
    """A partial view is not evidence, and a raise on a subset of ranks would hang the job.

    ``None`` is what a rank contributes when NVML has no answer; one of them must silence the whole
    check even though the remaining ids would look like a straddle.
    """
    _run(domain=8, world=8, cliques=[1, 1, 1, None, 2, 2, 2, 2])


def test_a_declared_world_that_disagrees_with_the_group_is_not_judged():
    """``all_gather_object`` fills exactly ``get_world_size()`` slots.

    Sizing the buffer from the DECLARED world instead would leave the tail at its pre-fill on an
    oversized list — read as real cliques by the straddle math — and raise ``IndexError`` on an
    undersized one. A config built for a different topology than the live group cannot be judged here.
    """
    _run(domain=8, world=16, cliques=[1, 1, 1, 1, 2, 2, 2, 2])  # group of 8, config declares 16
    _run(domain=4, world=4, cliques=[1, 1, 1, 1, 2, 2, 2, 2])  # group of 8, config declares 4


def test_no_process_group_is_a_no_op():
    """Constructed in a test or a single-process run: nothing to gather, nothing to judge."""
    with patch(f"{_MOD}.dist.is_initialized", return_value=False):
        validate_nvlink_domain_against_fabric(8, 8)
    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=True),
        patch(f"{_MOD}.dist.get_world_size", return_value=1),
        patch(f"{_MOD}.dist.all_gather_object", side_effect=AssertionError("must not gather")),
    ):
        validate_nvlink_domain_against_fabric(1, 1)  # a one-rank group returns before the collective


def test_a_wider_clique_than_declared_warns_but_runs():
    """Under-claiming the domain is always SAFE — it only leaves NVLink width unused, so it warns."""
    with patch(f"{_MOD}.logger") as logger:
        _run(domain=4, world=8, cliques=[9] * 8)
    assert logger.warning.called, "leaving half the fabric unused should say so"


def test_a_per_node_domain_drift_is_rejected_before_this_collective():
    """``NVLINK_DOMAIN_SIZE`` set differently per node builds different groups on different ranks.

    The verdict belongs to ``ParallelismConfig.__post_init__``, which gathers it through
    ``reject_divergent_settings`` BEFORE any of the topology collectives — not to this check, which
    would see the drift only after the MNNVL gate had already split the world on it. Pinned on the
    live gate, and on the ordering that makes it reachable.
    """
    order: list[str] = []
    with (
        patch(f"{_CONFIG_MOD}.get_global_world_size", return_value=8),
        patch(f"{_CONFIG_MOD}.get_local_world_size", return_value=8),
        patch(f"{_CONFIG_MOD}.get_global_rank", return_value=0),
        patch(f"{_CONFIG_MOD}.is_global_main_process", return_value=False),
        patch(f"{_CONFIG_MOD}.check_mnnvl_prerequisites", side_effect=lambda *a: order.append("mnnvl")),
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
    """The decision is a pure function of the gathered list, so it cannot differ by rank.

    This is the property that keeps the check from converting a misconfiguration into a hang: drive
    it once per rank with that rank's own local clique and assert all eight agree.
    """
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
    """``torch.cuda.get_device_properties`` exposes the PCI address as three INTS, never a string."""

    pci_domain_id = 0
    pci_bus_id = 4
    pci_device_id = 0


_FABRIC_QUERY = """GPU 00000000:04:00.0
    Product Name                          : NVIDIA B300
    GPU Fabric GUID                       : 0x6d9419000347bb4c
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

    # The read is memoized per device, so simulating different hardware must drop the memo first.
    _fabric_clique_for_device.cache_clear()
    with (
        patch(f"{_MOD}.subprocess.run", side_effect=fake_run),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_properties", return_value=_Props()),
    ):
        return nvlink_fabric_clique_id(), calls


def test_the_clique_read_actually_reports_a_fabric():
    """The whole fabric check is worth nothing if this read cannot succeed.

    NVML's Python binding cannot perform it, and fails silently: torch's PCI address is three ints
    (``.encode()`` on one raises) and the fabric getter needs a versioned out-param struct — one
    raise and one **segfault**, the first swallowed by the advisory ``except`` into "no fabric" (the
    same answer a healthy single node gives, so nothing looks wrong), the second uncatchable and
    fatal to every rank inside ``ParallelismConfig.__post_init__``. Reading out of process is what
    makes a wrong driver/binding pair a non-zero exit instead of the job.
    """
    clique, calls = _read_clique(state="Completed", clique="7")

    assert clique == 7, "a COMPLETED fabric with clique 7 must read back as 7, not None"
    assert calls and calls[0][0] == "nvidia-smi", "the clique must be read out of process"
    assert "00000000:04:00.0" in calls[0], "the GPU is selected by its composed PCI address, not an index"


def test_the_clique_is_read_from_the_fabric_block_only():
    """``State``/``CliqueId`` are common key names; only the ones under ``Fabric`` count.

    The sample carries decoys above and below the block — including a top-level ``GPU Fabric GUID``
    line, which must not open it — so a flat scan for either key picks the wrong value.
    """
    assert _read_clique(state="Completed", clique="7")[0] == 7


def test_absent_and_unregistered_fabrics_are_told_apart():
    """ "No fabric" and "cannot tell" are DIFFERENT answers, and the difference is load-bearing.

    Only a definite negative lets the cross-check reject a domain declared wider than a node on
    hardware with no fabric to span; ``None`` has to keep meaning "a partial view is not evidence",
    or that rejection would fire on a healthy NVL72 rack whose registration was merely mid-flight.
    """
    assert _read_clique(state="Completed", clique="0")[0] == NO_FABRIC  # COMPLETED, no island
    # Genuinely unknown: a fabric exists but is not yet trustworthy.
    assert _read_clique(state="In Progress", clique="7")[0] is None
    assert _read_clique(state="N/A", clique="N/A")[0] is None


def test_a_dump_without_a_fabric_block_is_a_definite_negative():
    """A plain HGX/NVL8 part emits no ``Fabric`` block at all — the commonest topology in practice."""
    assert _parse_fabric_clique("GPU 00000000:01:00.0\n    Product Name : NVIDIA B200\n") == NO_FABRIC


def test_a_failing_nvidia_smi_is_unreadable_not_a_crash():
    """No nvidia-smi, a timeout, or a non-zero exit must read as "could not tell", never propagate."""
    for boom in (FileNotFoundError, subprocess.TimeoutExpired("nvidia-smi", 30)):
        _fabric_clique_for_device.cache_clear()  # each failure mode is a fresh read
        with (
            patch(f"{_MOD}.subprocess.run", side_effect=boom),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_properties", return_value=_Props()),
        ):
            assert nvlink_fabric_clique_id() is None


def test_the_clique_read_is_memoized_per_device():
    """One read per device, then never again — and it must be the answer, not just a fast path.

    ``ParallelismConfig`` construction reaches this through the MNNVL advisory, and the read costs
    ~60 ms (an NVML property lookup plus an ``nvidia-smi`` subprocess). A training process builds one
    config, but the CPU config grid builds tens of thousands, which is over an hour of subprocess
    spawning. Memoizing is only correct because a GPU's fabric membership is fixed once the driver is
    up — so this pins BOTH halves: the second call must not re-run nvidia-smi, and it must still
    return the same clique.
    """
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
    sys.exit(pytest.main([__file__, "-v"]))
