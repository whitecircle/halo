#!/usr/bin/env python
"""Two fail-loud guards on the way into the rank math; without them both failures are LATE and
per-rank.

1. ``get_nvlink_domain_size`` must not hand ``NVLINK_DOMAIN_SIZE`` back verbatim. ``0`` is also
   ``ParallelismConfig``'s own "unset" sentinel, so it slips past every domain validator (``0 %
   gpus_per_node == 0``) and dies as a bare ``ZeroDivisionError`` inside ``world_size %
   nvlink_domain_size``; a negative value silently builds garbage groups.

2. ``create_dp_mesh`` must guard its 1-D branch against a sub-world ``dp_size``, not only its 2-D
   (HSDP) one. On the 1-D branch ``init_device_mesh`` numbers the mesh ``0..dp_size-1``, so the
   ranks OUTSIDE it raise a bare ``IndexError: list index out of range`` while the ranks inside
   proceed and then block forever on collectives their peers never reach — a rank-dependent failure
   with no usable message. The sub-world caller (a pipeline stage) is supposed to pass its process
   group as ``dp_group``.

    python tests/cpu/parallelism/test_domain_and_mesh_guards.py
"""

from __future__ import annotations

import pytest
import torch.distributed as dist

import src.distributed.mesh as mesh_mod
from src.distributed.nvlink import get_nvlink_domain_size


@pytest.mark.parametrize("bad", ["0", "-4"])
def test_non_positive_nvlink_domain_size_falls_back(monkeypatch, bad):
    """A non-positive domain must not reach the rank math, where it divides by zero."""
    monkeypatch.setenv("NVLINK_DOMAIN_SIZE", bad)
    assert get_nvlink_domain_size(default=8) == 8


def test_valid_nvlink_domain_size_is_still_honored(monkeypatch):
    """Anti-vacuity: the guard must not swallow a real rack-wide domain."""
    monkeypatch.setenv("NVLINK_DOMAIN_SIZE", "72")
    assert get_nvlink_domain_size(default=8) == 72


def test_unset_nvlink_domain_size_uses_the_default(monkeypatch):
    monkeypatch.delenv("NVLINK_DOMAIN_SIZE", raising=False)
    assert get_nvlink_domain_size(default=8) == 8


def test_1d_dp_mesh_rejects_a_sub_world_dp_size(monkeypatch):
    """Without dp_group, a 1-D mesh narrower than the world is built from ranks 0..dp_size-1."""
    monkeypatch.setattr(mesh_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mesh_mod.dist, "get_world_size", lambda: 8)
    with pytest.raises(ValueError, match="must span the whole world"):
        mesh_mod.create_dp_mesh(4, device_type="cpu")


def test_2d_dp_mesh_rejects_a_sub_world_dp_size(monkeypatch):
    """The sibling branch's guard, pinned alongside so the two cannot drift apart again."""
    monkeypatch.setattr(mesh_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mesh_mod.dist, "get_world_size", lambda: 8)
    with pytest.raises(ValueError, match="must span the whole world"):
        mesh_mod.create_dp_mesh(4, dp_replicate_size=2, device_type="cpu")


def test_full_world_1d_dp_mesh_is_accepted(tmp_path):
    """Anti-vacuity: the ordinary single-node case must still build."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        assert mesh_mod.create_dp_mesh(1, device_type="cpu").size() == 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
