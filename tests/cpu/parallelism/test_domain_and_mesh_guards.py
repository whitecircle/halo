#!/usr/bin/env python
"""Two fail-loud guards on the way into the rank math; without them both failures are LATE and
per-rank.

1. ``requested_nvlink_domain_size`` must not hand ``NVLINK_DOMAIN_SIZE`` back verbatim. ``0`` is also
   ``ParallelismConfig``'s own "unset" sentinel, so it slips past every domain validator (``0 %
   gpus_per_node == 0``) and dies as a bare ``ZeroDivisionError`` inside ``world_size %
   nvlink_domain_size``; a negative value silently builds garbage groups.

2. ``build_dp_mesh`` must guard its 1-D branch against a sub-world ``dp_size``, not only its 2-D
   (HSDP) one. On the 1-D branch ``init_device_mesh`` numbers the mesh ``0..dp_size-1``, so the
   ranks OUTSIDE it raise a bare ``IndexError: list index out of range`` while the ranks inside
   proceed and then block forever on collectives their peers never reach — a rank-dependent failure
   with no usable message. The sub-world caller (a pipeline stage) is supposed to pass its process
   group as ``dp_group``.

    python tests/cpu/parallelism/test_domain_and_mesh_guards.py
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch.distributed as dist

import src.distributed.mesh as mesh_mod
import src.distributed.nvlink as nvlink_mod
from src.distributed.nvlink import requested_nvlink_domain_size


@pytest.mark.parametrize("bad", ["0", "-4", "seventy-two"])
def test_unusable_nvlink_domain_size_reads_as_unset(monkeypatch, bad):
    """A non-positive or malformed domain must not reach the rank math, where it divides by zero."""
    monkeypatch.setenv("NVLINK_DOMAIN_SIZE", bad)
    assert requested_nvlink_domain_size() is None


def test_a_non_positive_domain_warns_once(monkeypatch):
    """The reader runs on every ParallelismConfig construction (tens of thousands in a config sweep)."""
    monkeypatch.setenv("NVLINK_DOMAIN_SIZE", "0")
    monkeypatch.setattr(nvlink_mod, "_warned_domain_sizes", set())
    with patch.object(nvlink_mod, "logger") as logger:
        requested_nvlink_domain_size()
        requested_nvlink_domain_size()
    assert logger.warning.call_count == 1


def test_valid_nvlink_domain_size_is_still_honored(monkeypatch):
    """Anti-vacuity: the guard must not swallow a real rack-wide domain."""
    monkeypatch.setenv("NVLINK_DOMAIN_SIZE", "72")
    assert requested_nvlink_domain_size() == 72


def test_unset_nvlink_domain_size_reads_as_unset(monkeypatch):
    monkeypatch.delenv("NVLINK_DOMAIN_SIZE", raising=False)
    assert requested_nvlink_domain_size() is None


def test_1d_dp_mesh_rejects_a_sub_world_dp_size(monkeypatch):
    """Without dp_group, a 1-D mesh narrower than the world is built from ranks 0..dp_size-1."""
    monkeypatch.setattr(mesh_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mesh_mod.dist, "get_world_size", lambda: 8)
    with pytest.raises(ValueError, match="must span the whole world"):
        mesh_mod.build_dp_mesh(4, device_type="cpu")


def test_2d_dp_mesh_rejects_a_sub_world_dp_size(monkeypatch):
    """The sibling branch's guard, pinned alongside so the two cannot drift apart."""
    monkeypatch.setattr(mesh_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mesh_mod.dist, "get_world_size", lambda: 8)
    with pytest.raises(ValueError, match="must span the whole world"):
        mesh_mod.build_dp_mesh(4, dp_replicate_size=2, device_type="cpu")


def test_full_world_1d_dp_mesh_is_accepted(tmp_path):
    """Anti-vacuity: the ordinary single-node case must still build."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        assert mesh_mod.build_dp_mesh(1, device_type="cpu").size() == 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
