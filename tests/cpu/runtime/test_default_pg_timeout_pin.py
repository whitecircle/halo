#!/usr/bin/env python
"""CI pin for torch's private ``default_pg_nccl_timeout`` seam.

``_apply_default_pg_timeout`` rebinds ``torch.distributed.distributed_c10d.default_pg_nccl_timeout``
so every subsequent ``new_group`` / ``init_device_mesh`` subgroup (which takes no timeout kwarg)
inherits ``DIST_NCCL_TIMEOUT_MINUTES`` instead of PyTorch's 10-minute default. The symbol is a
private torch internal, and assigning a name torch no longer defines would merely create an unused
module global — every EP/CP/TP mesh subgroup back on 10 minutes, tripping the watchdog on long
cross-node collectives (e.g. a gathered save), with nothing said. Hence both pins below: the symbol
must exist on this torch, and its absence must RAISE rather than pass.

    python tests/cpu/runtime/test_default_pg_timeout_pin.py
"""

import datetime

import pytest
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from halo.env import nccl_timeout
from halo.runtime.process_group import _apply_default_pg_timeout
from tests.common.distributed import init_distributed as harness_init_distributed


def test_torch_still_exposes_the_new_group_timeout_default():
    """The name ``new_group(timeout=None)`` resolves its default from must exist on this torch."""
    assert hasattr(c10d, "default_pg_nccl_timeout"), (
        "torch.distributed.distributed_c10d.default_pg_nccl_timeout is gone on this PyTorch build — "
        "_apply_default_pg_timeout can no longer pin the subgroup default, and every DeviceMesh "
        "(DP/HSDP/TP) subgroup falls back to the 10-minute NCCL watchdog. Point "
        "_apply_default_pg_timeout at the renamed symbol."
    )


def test_apply_default_pg_timeout_rebinds_the_module_default(monkeypatch):
    """The rebind must actually land: DIST_NCCL_TIMEOUT_MINUTES → the module-level default."""
    monkeypatch.setenv("DIST_NCCL_TIMEOUT_MINUTES", "123")
    original = c10d.default_pg_nccl_timeout
    try:
        _apply_default_pg_timeout(nccl_timeout())
        assert c10d.default_pg_nccl_timeout == datetime.timedelta(minutes=123)
    finally:
        c10d.default_pg_nccl_timeout = original


def test_a_renamed_symbol_raises_instead_of_pinning_nothing(monkeypatch):
    """The failure mode this file exists for: a silent no-op on a future torch.

    ``setattr`` on a missing name succeeds, so only a read-before-write can tell a rename from a
    successful pin. Deleting the symbol must therefore end the run, naming the consequence.
    """
    monkeypatch.delattr(c10d, "default_pg_nccl_timeout")
    with pytest.raises(RuntimeError, match="default_pg_nccl_timeout is absent"):
        _apply_default_pg_timeout(datetime.timedelta(minutes=45))


def test_a_gpu_row_brings_its_group_up_with_the_production_watchdog(monkeypatch):
    """The GPU harness inits through production, so a row waits ``DIST_NCCL_TIMEOUT_MINUTES`` on its world and
    mesh groups, not PyTorch's 10 minutes: a large gathered save or a cross-node row would otherwise abort where
    production waits. Driven as rank 1 of 2 with the group construction recorded, on a CPU host."""
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
