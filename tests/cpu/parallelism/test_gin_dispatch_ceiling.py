#!/usr/bin/env python
"""The cross-node GIN dispatch ceiling must fail loud, at the measured boundary, and stay overridable.

Above ~8k tokens/rank an EFA proxy-GIN dispatch wedges instead of erroring (receive counts never
arrive; CPU-wait timeout or Xid 109→43), measured on cross-node EFA proxy GIN at ep8 and ep16.
The guard turns that wedge into a config-time raise with the levers; these tests pin the boundary,
the disable path, and that the message names both the mechanism and the escape hatch.

Run: python tests/cpu/parallelism/test_gin_dispatch_ceiling.py  (or pytest -m cpu)
"""

from __future__ import annotations

import importlib

import pytest

from halo.distributed.expert_parallel.deepep import capacity as capacity_mod

# The process capacity scope the backends hold; reloading the module for its ceiling must not replace it.
_CAPACITY_SCOPE = capacity_mod.capacity_scope


def _reject_cross_node(capacity: int) -> None:
    """The ceiling alone: a wire-index extent far below DeepEP's 32-bit limit, on a group spanning domains."""
    capacity_mod.reject_oversized_dispatch(capacity, num_topk=1, padded_hidden=1, spans_domains=True)


def _reload() -> None:
    importlib.reload(capacity_mod).capacity_scope = _CAPACITY_SCOPE


def test_default_ceiling_is_the_validated_boundary():
    """8,192 trains, 16,384 wedges (measured) — the default must sit exactly on the passing side."""
    assert capacity_mod.GIN_MAX_TOKENS_PER_RANK == 8192
    _reject_cross_node(8192)  # at the ceiling: allowed
    with pytest.raises(ValueError, match="proxy-GIN ceiling"):
        _reject_cross_node(8193)


def test_the_ceiling_binds_only_a_group_spanning_domains():
    """The NVLink path has no such limit, so a node-local dispatch past the ceiling passes."""
    capacity_mod.reject_oversized_dispatch(16384, num_topk=1, padded_hidden=1, spans_domains=False)


def test_message_names_the_mechanism_and_the_levers():
    with pytest.raises(ValueError) as err:
        _reject_cross_node(16384)
    text = str(err.value)
    assert "received count 0" in text, "the wedge symptom must be searchable from the raise"
    assert "ep_scope=node" in text
    assert "HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK" in text


def test_env_override_and_disable(monkeypatch):
    """The env var raises the ceiling or (0) disables the guard entirely, via module reload."""
    monkeypatch.setenv("HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK", "0")
    _reload()
    try:
        _reject_cross_node(1_000_000)  # disabled: no raise
        monkeypatch.setenv("HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK", "32768")
        _reload()
        _reject_cross_node(32768)
        with pytest.raises(ValueError, match="proxy-GIN ceiling"):
            _reject_cross_node(32769)
    finally:
        monkeypatch.delenv("HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK", raising=False)
        _reload()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
