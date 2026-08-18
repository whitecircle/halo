#!/usr/bin/env python
"""The cross-node GIN dispatch ceiling must fail loud, at the measured boundary, and stay overridable.

Above ~8k tokens/rank an EFA proxy-GIN dispatch wedges instead of erroring (receive counts never
arrive; CPU-wait timeout or Xid 109→43), measured on 2× 8×B300 at ep8 and ep16 on every node pair.
The guard turns that wedge into a config-time raise with the levers; these tests pin the boundary,
the disable path, and that the message names both the mechanism and the escape hatch.

Run: python tests/cpu/parallelism/test_gin_dispatch_ceiling.py  (or pytest -m cpu)
"""

from __future__ import annotations

import importlib

import pytest

from src.distributed.expert_parallel import config as ep_config_mod


def test_default_ceiling_is_the_validated_boundary():
    """8,192 trains, 16,384 wedges (measured) — the default must sit exactly on the passing side."""
    assert ep_config_mod.GIN_MAX_TOKENS_PER_RANK == 8192
    ep_config_mod.reject_oversized_gin_dispatch(8192)  # at the ceiling: allowed
    with pytest.raises(ValueError, match="proxy-GIN ceiling"):
        ep_config_mod.reject_oversized_gin_dispatch(8193)


def test_message_names_the_mechanism_and_the_levers():
    with pytest.raises(ValueError) as err:
        ep_config_mod.reject_oversized_gin_dispatch(16384)
    text = str(err.value)
    assert "received count 0" in text, "the wedge symptom must be searchable from the raise"
    assert "ep_scope=node" in text
    assert "HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK" in text


def test_env_override_and_disable(monkeypatch):
    """The env var raises the ceiling or (0) disables the guard entirely, via module reload."""
    monkeypatch.setenv("HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK", "0")
    importlib.reload(ep_config_mod)
    try:
        ep_config_mod.reject_oversized_gin_dispatch(1_000_000)  # disabled: no raise
        monkeypatch.setenv("HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK", "32768")
        importlib.reload(ep_config_mod)
        ep_config_mod.reject_oversized_gin_dispatch(32768)
        with pytest.raises(ValueError, match="proxy-GIN ceiling"):
            ep_config_mod.reject_oversized_gin_dispatch(32769)
    finally:
        monkeypatch.delenv("HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK", raising=False)
        importlib.reload(ep_config_mod)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
