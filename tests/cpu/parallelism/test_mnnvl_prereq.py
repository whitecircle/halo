#!/usr/bin/env python
"""Multi-Node NVLink prerequisites are judged on every rank where the topology requires them (NVL72). Rank 0 is the
node an operator checks; a ``None`` clique means an incomplete fabric, a failure."""

from unittest.mock import patch

import pytest
import torch

from halo.distributed.nvlink import NO_FABRIC, mnnvl_prerequisite_reason, verify_mnnvl_prerequisites

_MOD = "halo.distributed.nvlink"


def test_a_fully_provisioned_rank_reports_nothing():
    assert mnnvl_prerequisite_reason(imex_ok=True, fabric_clique=7) is None


def test_missing_imex_channels_is_a_reason():
    assert "IMEX" in (mnnvl_prerequisite_reason(imex_ok=False, fabric_clique=7) or "")


def test_an_incomplete_fabric_registration_is_a_reason():
    """``None`` is what a non-COMPLETED Fabric State reports."""
    reason = mnnvl_prerequisite_reason(imex_ok=True, fabric_clique=None)
    assert reason is not None, "an unregistered fabric must not read as 'nothing to report'"
    assert "COMPLETED" in reason, f"the nvidia-smi field to look at must be named: {reason}"


def test_a_node_with_no_fabric_at_all_is_a_reason():
    reason = mnnvl_prerequisite_reason(imex_ok=True, fabric_clique=NO_FABRIC)
    assert reason is not None and "no NVLink fabric" in reason


def test_imex_is_reported_before_the_fabric():
    """One node with no MNNVL provisioning; the kernel-side fact is the actionable one."""
    assert "IMEX" in (mnnvl_prerequisite_reason(imex_ok=False, fabric_clique=None) or "")


def _run_required(clique, imex_ok: bool, *, world: int = 16, all_reasons=None):
    """Domain 72 > 8 GPUs per node."""
    gathered = all_reasons

    def fake_all_gather(out_list, obj):
        out_list[:] = gathered if gathered is not None else [obj] * world

    def fake_has_reason_flag(flag, op=None, group=None):
        # The rejection's fast path: the MAX of every rank's has-reason flag.
        flag.fill_(int(any(gathered or ())) or int(flag.item()))

    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=True),
        patch(f"{_MOD}.dist.get_world_size", return_value=world),
        patch(f"{_MOD}.dist.get_rank", return_value=0),
        patch("halo.runtime.consensus.collective_device", return_value=torch.device("cpu")),
        patch(f"{_MOD}.dist.all_reduce", side_effect=fake_has_reason_flag),
        patch(f"{_MOD}.dist.all_gather_object", side_effect=fake_all_gather),
        patch(f"{_MOD}.imex_channels_present", return_value=imex_ok),
        patch(f"{_MOD}.nvlink_fabric_clique_id", return_value=clique),
    ):
        verify_mnnvl_prerequisites(nvlink_domain_size=72, gpus_per_node=8)


def test_a_provisioned_job_passes():
    _run_required(clique=3, imex_ok=True)


def test_one_bad_node_fails_the_job_by_name():
    """A rank-0-only check sees nothing, and the job faults inside a dispatch."""
    reasons = [None] * 8 + ["no IMEX channels at /dev/nvidia-caps-imex-channels"] * 8
    with pytest.raises(ValueError) as excinfo:
        _run_required(clique=3, imex_ok=True, all_reasons=reasons)
    message = str(excinfo.value)
    assert "rank 8" in message, f"the failing rank must be named: {message}"
    assert "IMEX" in message, f"the missing prerequisite must be named: {message}"
    assert "NVLINK_DOMAIN_SIZE=72" in message, f"the topology that requires it must be named: {message}"


def test_a_topology_that_does_not_need_mnnvl_is_never_enforced():
    """An NVL8 box with no IMEX must run."""
    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=True),
        patch(f"{_MOD}.dist.get_world_size", return_value=16),
        patch(f"{_MOD}.dist.get_rank", return_value=0),
        patch(f"{_MOD}.imex_channels_present", return_value=False),
        patch(f"{_MOD}.nvlink_fabric_clique_id", return_value=NO_FABRIC),
    ):
        verify_mnnvl_prerequisites(nvlink_domain_size=8, gpus_per_node=8)


def test_outside_a_live_job_it_only_advises():
    """CPU config sweeps build it tens of thousands of times with no hardware or peers."""
    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=False),
        patch(f"{_MOD}.imex_channels_present", return_value=False),
        patch(f"{_MOD}.nvlink_fabric_clique_id", return_value=None),
    ):
        verify_mnnvl_prerequisites(nvlink_domain_size=72, gpus_per_node=8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
