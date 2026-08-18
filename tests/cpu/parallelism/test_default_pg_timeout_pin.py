#!/usr/bin/env python
"""CI pin for torch's private ``default_pg_nccl_timeout`` seam.

``apply_default_pg_timeout`` rebinds ``torch.distributed.distributed_c10d.default_pg_nccl_timeout``
so every subsequent ``new_group`` / ``init_device_mesh`` subgroup (which takes no timeout kwarg)
inherits ``DIST_NCCL_TIMEOUT_MINUTES`` instead of PyTorch's 10-minute default. The symbol is a
private torch internal, and assigning a name torch no longer defines would merely create an unused
module global — every EP/CP/TP mesh subgroup back on 10 minutes, tripping the watchdog on long
cross-node collectives (e.g. a gathered save), with nothing said. Hence both pins below: the symbol
must exist on this torch, and its absence must RAISE rather than pass.

    python tests/cpu/parallelism/test_default_pg_timeout_pin.py
"""

import datetime
import sys

import pytest
import torch.distributed.distributed_c10d as c10d

from src.distributed.runtime import apply_default_pg_timeout, get_nccl_timeout


def test_torch_still_exposes_the_new_group_timeout_default():
    """The name ``new_group(timeout=None)`` resolves its default from must exist on this torch."""
    assert hasattr(c10d, "default_pg_nccl_timeout"), (
        "torch.distributed.distributed_c10d.default_pg_nccl_timeout is gone on this PyTorch build — "
        "apply_default_pg_timeout can no longer pin the subgroup default, and every DeviceMesh "
        "(DP/HSDP/TP) subgroup falls back to the 10-minute NCCL watchdog. Update the pin in "
        "src/distributed/runtime.py to the renamed symbol."
    )


def test_apply_default_pg_timeout_rebinds_the_module_default(monkeypatch):
    """The rebind must actually land: DIST_NCCL_TIMEOUT_MINUTES → the module-level default."""
    monkeypatch.setenv("DIST_NCCL_TIMEOUT_MINUTES", "123")
    original = c10d.default_pg_nccl_timeout
    try:
        apply_default_pg_timeout(get_nccl_timeout())
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
        apply_default_pg_timeout(datetime.timedelta(minutes=45))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
