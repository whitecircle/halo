#!/usr/bin/env python
"""Multi-Node NVLink prerequisites are judged on EVERY rank, and the verdict is enforced.

``check_mnnvl_prerequisites`` fires only where the topology REQUIRES MNNVL — a declared
``NVLINK_DOMAIN_SIZE`` wider than the OS node, i.e. NVL72, where "node-local" EP/TP/CP groups span
OS nodes over NVLink. Two properties matter there:

* it must run on every rank. Rank 0 is the one node an operator does check, so a rank-0-only look
  inspects the node least likely to be broken and passes the job through;
* an incomplete fabric must FAIL. ``nvlink_fabric_clique_id()`` returns ``None`` precisely when
  ``Fabric.State != COMPLETED`` — Fabric Manager still bringing the node in — which is the
  prerequisite failure this check exists to catch, not a benign "could not tell".

The reason string is a pure per-rank verdict, so the aggregation is the existing
``reject_across_ranks`` and the interesting logic is testable with synthetic values.

    python tests/cpu/parallelism/test_mnnvl_prereq.py
"""

import sys
from unittest.mock import patch

import pytest

from src.distributed.nvlink import NO_FABRIC, check_mnnvl_prerequisites, mnnvl_prerequisite_reason

_MOD = "src.distributed.nvlink"


def test_a_fully_provisioned_rank_reports_nothing():
    """IMEX channels present and a real fabric clique — the passing NVL72 node."""
    assert mnnvl_prerequisite_reason(imex_ok=True, fabric_clique=7) is None


def test_missing_imex_channels_is_a_reason():
    assert "IMEX" in (mnnvl_prerequisite_reason(imex_ok=False, fabric_clique=7) or "")


def test_an_incomplete_fabric_registration_is_a_reason():
    """The disarm this fix closes: ``None`` is exactly what a non-COMPLETED Fabric State reports."""
    reason = mnnvl_prerequisite_reason(imex_ok=True, fabric_clique=None)
    assert reason is not None, "an unregistered fabric must not read as 'nothing to report'"
    assert "COMPLETED" in reason, f"the nvidia-smi field to look at must be named: {reason}"


def test_a_node_with_no_fabric_at_all_is_a_reason():
    reason = mnnvl_prerequisite_reason(imex_ok=True, fabric_clique=NO_FABRIC)
    assert reason is not None and "no NVLink fabric" in reason


def test_imex_is_reported_before_the_fabric():
    """Both broken is one node with no MNNVL provisioning; the kernel-side fact is the actionable one."""
    assert "IMEX" in (mnnvl_prerequisite_reason(imex_ok=False, fabric_clique=None) or "")


def _run_required(clique, imex_ok: bool, *, world: int = 16, all_reasons=None):
    """Drive the check on a topology that REQUIRES MNNVL (domain 72 > 8 GPUs per node)."""
    gathered = all_reasons

    def fake_all_gather(out_list, obj):
        out_list[:] = gathered if gathered is not None else [obj] * world

    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=True),
        patch(f"{_MOD}.dist.get_world_size", return_value=world),
        patch(f"{_MOD}.dist.get_rank", return_value=0),
        patch(f"{_MOD}.dist.all_gather_object", side_effect=fake_all_gather),
        patch(f"{_MOD}.imex_channels_present", return_value=imex_ok),
        patch(f"{_MOD}.nvlink_fabric_clique_id", return_value=clique),
    ):
        check_mnnvl_prerequisites(nvlink_domain_size=72, gpus_per_node=8)


def test_a_provisioned_job_passes():
    _run_required(clique=3, imex_ok=True)


def test_one_bad_node_fails_the_job_by_name():
    """Rank 8 (node 1) has no IMEX channels while rank 0's node is perfect — the rank-0-only check
    saw nothing and the job went on to fault inside a dispatch."""
    reasons = [None] * 8 + ["no IMEX channels at /dev/nvidia-caps-imex-channels"] * 8
    with pytest.raises(ValueError) as excinfo:
        _run_required(clique=3, imex_ok=True, all_reasons=reasons)
    message = str(excinfo.value)
    assert "rank 8" in message, f"the failing rank must be named: {message}"
    assert "IMEX" in message, f"the missing prerequisite must be named: {message}"
    assert "NVLINK_DOMAIN_SIZE=72" in message, f"the topology that requires it must be named: {message}"


def test_a_topology_that_does_not_need_mnnvl_is_never_enforced():
    """A domain within one node needs no fabric at all — an NVL8 box with no IMEX must run."""
    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=True),
        patch(f"{_MOD}.dist.get_world_size", return_value=16),
        patch(f"{_MOD}.dist.get_rank", return_value=0),
        patch(f"{_MOD}.imex_channels_present", return_value=False),
        patch(f"{_MOD}.nvlink_fabric_clique_id", return_value=NO_FABRIC),
        patch(f"{_MOD}.detect_nvlink_fabric_present", return_value=False),
    ):
        check_mnnvl_prerequisites(nvlink_domain_size=8, gpus_per_node=8)


def test_outside_a_live_job_it_only_advises():
    """``ParallelismConfig`` is built tens of thousands of times by the CPU config sweeps, with no
    hardware to judge and no peers to join — enforcing there would fail every one of them."""
    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=False),
        patch(f"{_MOD}.imex_channels_present", return_value=False),
        patch(f"{_MOD}.nvlink_fabric_clique_id", return_value=None),
    ):
        check_mnnvl_prerequisites(nvlink_domain_size=72, gpus_per_node=8)  # must not raise


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
