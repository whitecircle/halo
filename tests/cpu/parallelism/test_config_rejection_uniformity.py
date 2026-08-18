#!/usr/bin/env python3
"""A rejected parallelism config must not leave the other ranks inside a collective.

``ParallelismConfig.__post_init__`` ends in two collectives (the MNNVL prerequisite check and the
fabric ``all_gather_object``), and every rule before them is rank-LOCAL. Two of those rules read
per-NODE inputs — ``gpus_per_node`` from the launcher, ``nvlink_domain_size`` from
``NVLINK_DOMAIN_SIZE`` — so an env drift on one node can fail a divisibility rule *there only*.

Unordered, that is a 512-rank hang: 8 ranks raise a clean ValueError and 504 sit in the fabric
gather until the 30-minute watchdog, with the real diagnostic buried in the 8 logs nobody is
reading. The verdict therefore has to be gathered BEFORE either collective runs.

Usage:
    python tests/cpu/parallelism/test_config_rejection_uniformity.py
"""

import sys
from unittest.mock import patch

import pytest

from tests.common.parallelism import make_parallelism_config

_MOD = "src.distributed.parallelism_config"


def _build(**kwargs):
    """Construct on a single 8-GPU node, leaving the real validation in place.

    The two topology collectives are stubbed so the test can assert WHETHER they ran — the ordering
    this file exists to pin — not merely that the config was rejected.
    """
    with (
        patch(f"{_MOD}.check_mnnvl_prerequisites") as mnnvl,
        patch(f"{_MOD}.validate_nvlink_domain_against_fabric") as fabric,
    ):
        try:
            config = make_parallelism_config(world_size=8, gpus_per_node=8, **kwargs)
        except ValueError as exc:
            return None, exc, mnnvl, fabric
        return config, None, mnnvl, fabric


def test_a_rejected_config_never_reaches_the_fabric_collective():
    """The ordering itself. If the collectives ran first, a one-node drift would hang the job."""
    _, error, mnnvl, fabric = _build(pp_size=3)  # 8 % 3 != 0 — stages are equal rank blocks
    assert error is not None, "an indivisible pp_size must be rejected"
    assert not fabric.called, "the fabric all_gather ran even though the config was already invalid"
    assert not mnnvl.called, "the MNNVL prerequisite check ran even though the config was invalid"


def test_a_valid_config_still_runs_both_collectives():
    """Anti-over-fitting: the gather must not swallow the checks on the happy path.

    The fabric check's ARGUMENTS are asserted too: it needs ``gpus_per_node`` to reject a domain
    declared wider than a node on hardware with no fabric, and dropping that argument disables the
    rejection silently — the call still happens, so ``fabric.called`` alone stays green.
    """
    config, error, mnnvl, fabric = _build(ep_size=8)
    assert error is None and config is not None
    assert mnnvl.called
    fabric.assert_called_once_with(config.nvlink_domain_size, config.world_size, config.gpus_per_node)


def test_the_rejection_stays_a_ValueError_carrying_the_real_reason():
    """The uniformity seam raises RuntimeError by default; this gate documents ValueError.

    Losing either the type or the message would turn a precise config error into an unrelated-looking
    failure — the opposite of the point.
    """
    _, error, _, _ = _build(ep_size=0)
    assert isinstance(error, ValueError)
    assert "ep_size must be >= 1" in str(error)


def test_every_rule_is_still_enforced_through_the_seam():
    """A sample across the validators, so the try/except cannot silently swallow a rejection."""
    for kwargs, expected in (
        ({"pp_schedule": "nonsense"}, "pp_schedule must be one of"),
        ({"ep_buffer_backend": "nonsense"}, "ep_buffer_backend must be one of"),
        ({"max_concurrent_loading": -1}, "max_concurrent_loading must be >= 0"),
        ({"ep_scope": "nonsense"}, "ep_scope must be one of"),
    ):
        _, error, _, _ = _build(**kwargs)
        assert error is not None and expected in str(error), f"{kwargs} should have been rejected"


def test_a_legacy_backend_this_topology_cannot_drive_is_rejected_at_config_time():
    """DeepEP V1 is intranode CUDA-IPC with a fixed table of tuned dispatch widths.

    The dispatcher enforces the same rules, but only inside ``EPMoELayerBase.__init__`` — after the
    checkpoint download and the model load on every rank. These are pure arithmetic over the declared
    topology, so a rejection that costs a full load is a rejection in the wrong place.
    """
    # Wider than one node's NVLink peers.
    _, error, _, _ = _build(ep_size=8, ep_buffer_backend="legacy")
    assert error is None, "ep8 on an 8-GPU node is exactly what legacy is for"

    config, error, _, _ = _build(ep_size=2, ep_buffer_backend="legacy")
    assert error is None and config is not None

    # A width DeepEP V1 ships no Config for: it asserts at the first dispatch otherwise.
    config.ep_size = 6  # past __post_init__, as the racy-topology tests do
    with pytest.raises(ValueError, match="no tuned Config"):
        config._validate_ep_buffer_backend()

    config.ep_size = 16  # a legal V1 table width, but more NVLink peers than V1 drives
    with pytest.raises(ValueError, match="intranode CUDA-IPC"):
        config._validate_ep_buffer_backend()


def _build_multinode(**kwargs):
    """Two 8-GPU nodes — the smallest world where a cross-node EP group is expressible."""
    with (
        patch(f"{_MOD}.check_mnnvl_prerequisites"),
        patch(f"{_MOD}.validate_nvlink_domain_against_fabric"),
    ):
        try:
            return make_parallelism_config(world_size=16, gpus_per_node=8, **kwargs), None
        except ValueError as exc:
            return None, exc


def test_the_default_elastic_backend_is_never_judged_by_the_v1_table():
    """Anti-over-rejection, on a shape where the two backends genuinely differ.

    On one 8-GPU node every legal ep_size is also a legal V1 width, so no value there can tell the
    table apart from no table at all. A 16-wide cross-node group can: elastic drives it, V1 cannot.
    """
    config, error = _build_multinode(ep_size=16, ep_scope="global")  # elastic default
    assert error is None and config is not None, f"elastic ep16 must be accepted: {error}"

    _, error = _build_multinode(ep_size=16, ep_scope="global", ep_buffer_backend="legacy")
    assert error is not None and "CROSS-NODE EP group" in str(error), (
        "legacy ep16 cross-node must be rejected at CONFIG time, through __post_init__ — not by the "
        "dispatcher after the whole model has loaded"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
