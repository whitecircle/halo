#!/usr/bin/env python
"""Pin for torch's private ``default_pg_nccl_timeout``, rebound for subgroups that take no timeout kwarg. Assigning
a removed name would silently leave them on 10 minutes, tripping the watchdog."""

import datetime

import pytest
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from halo.env import nccl_timeout
from halo.runtime.process_group import _apply_default_pg_timeout
from tests.common.distributed import init_distributed as harness_init_distributed


def test_torch_still_exposes_the_new_group_timeout_default():
    assert hasattr(c10d, "default_pg_nccl_timeout"), (
        "torch.distributed.distributed_c10d.default_pg_nccl_timeout is gone on this PyTorch build — "
        "_apply_default_pg_timeout can no longer pin the subgroup default, and every DeviceMesh "
        "(DP/HSDP/TP) subgroup falls back to the 10-minute NCCL watchdog. Point "
        "_apply_default_pg_timeout at the renamed symbol."
    )


def test_apply_default_pg_timeout_rebinds_the_module_default(monkeypatch):
    monkeypatch.setenv("DIST_NCCL_TIMEOUT_MINUTES", "123")
    original = c10d.default_pg_nccl_timeout
    try:
        _apply_default_pg_timeout(nccl_timeout())
        assert c10d.default_pg_nccl_timeout == datetime.timedelta(minutes=123)
    finally:
        c10d.default_pg_nccl_timeout = original


def test_a_renamed_symbol_raises_instead_of_pinning_nothing(monkeypatch):
    """``setattr`` on a missing name succeeds, so only a read-before-write tells a rename from a pin."""
    monkeypatch.delattr(c10d, "default_pg_nccl_timeout")
    with pytest.raises(RuntimeError, match="default_pg_nccl_timeout is absent"):
        _apply_default_pg_timeout(datetime.timedelta(minutes=45))


def test_a_gpu_row_brings_its_group_up_with_the_production_watchdog(monkeypatch):
    """A large gathered save or cross-node row would otherwise abort where production waits."""
    built = []
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: built.append(kwargs))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(c10d, "default_pg_nccl_timeout", c10d.default_pg_nccl_timeout)
    for name, value in {"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "2"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setenv("DIST_NCCL_TIMEOUT_MINUTES", "60")

    harness_init_distributed()

    assert [call["timeout"] for call in built] == [datetime.timedelta(minutes=60)]
    assert c10d.default_pg_nccl_timeout == datetime.timedelta(minutes=60), "mesh subgroups keep the 10 min default"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
