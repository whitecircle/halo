#!/usr/bin/env python
"""Two fail-loud guards ahead of the rank math. ``NVLINK_DOMAIN_SIZE=0`` passes every validator and divides by zero; a
sub-world 1-D ``build_dp_mesh`` raises on some ranks while others block forever."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch.distributed as dist

import halo.distributed.mesh as mesh_mod
import halo.distributed.nvlink as nvlink_mod
from halo.distributed.nvlink import requested_nvlink_domain_size


@pytest.mark.parametrize("bad", ["0", "-4", "seventy-two"])
def test_unusable_nvlink_domain_size_reads_as_unset(monkeypatch, bad):
    """It must not reach the rank math."""
    monkeypatch.setenv("NVLINK_DOMAIN_SIZE", bad)
    assert requested_nvlink_domain_size() is None


def test_a_non_positive_domain_warns_once(monkeypatch):
    """The reader runs on every ParallelismConfig construction, tens of thousands in a sweep."""
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
    """Without dp_group, a narrower mesh is built from ranks 0..dp_size-1."""
    monkeypatch.setattr(mesh_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mesh_mod.dist, "get_world_size", lambda: 8)
    with pytest.raises(ValueError, match="must span the whole world"):
        mesh_mod.build_dp_mesh(4, device_type="cpu")


def test_2d_dp_mesh_rejects_a_sub_world_dp_size(monkeypatch):
    """Pinned alongside so the two branches cannot drift apart."""
    monkeypatch.setattr(mesh_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mesh_mod.dist, "get_world_size", lambda: 8)
    with pytest.raises(ValueError, match="must span the whole world"):
        mesh_mod.build_dp_mesh(4, dp_replicate_size=2, device_type="cpu")


def test_full_world_1d_dp_mesh_is_accepted(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        assert mesh_mod.build_dp_mesh(1, device_type="cpu").size() == 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
