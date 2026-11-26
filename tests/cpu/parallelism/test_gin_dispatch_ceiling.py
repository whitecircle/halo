#!/usr/bin/env python
"""The cross-node GIN dispatch ceiling fails loud at the measured boundary and stays overridable.
Above ~8k tokens/rank an EFA proxy-GIN dispatch wedges instead of erroring (measured at ep8 and ep16)."""

from __future__ import annotations

import importlib

import pytest

from halo.distributed.expert_parallel.deepep import capacity as capacity_mod

# Reloading the module for its ceiling must not replace the scope the backends hold.
_CAPACITY_SCOPE = capacity_mod.capacity_scope


def _reject_cross_node(capacity: int) -> None:
    capacity_mod.reject_oversized_dispatch(capacity, spans_domains=True)


def _reload() -> None:
    importlib.reload(capacity_mod).capacity_scope = _CAPACITY_SCOPE


def test_default_ceiling_is_the_validated_boundary():
    """8,192 trains, 16,384 wedges (measured)."""
    assert capacity_mod.GIN_MAX_TOKENS_PER_RANK == 8192
    _reject_cross_node(8192)  # At the ceiling: allowed.
    with pytest.raises(ValueError, match="proxy-GIN ceiling"):
        _reject_cross_node(8193)


def test_the_ceiling_binds_only_a_group_spanning_domains():
    """The NVLink path has no ceiling at all, however large the dispatch."""
    for capacity in (16384, 262_144, 2**31):
        capacity_mod.reject_oversized_dispatch(capacity, spans_domains=False)
    assert capacity_mod.max_dispatch_capacity(spans_domains=False) is None


def test_message_names_the_mechanism_and_the_levers():
    with pytest.raises(ValueError) as err:
        _reject_cross_node(16384)
    text = str(err.value)
    assert "received count 0" in text, "the wedge symptom must be searchable from the raise"
    assert "ep_scope=node" in text
    assert "HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK" in text


def test_a_negative_ceiling_keeps_the_default(monkeypatch):
    """A negative is truthy, so every cross-domain dispatch would raise against it."""
    monkeypatch.setenv("HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK", "-1")
    _reload()
    try:
        assert capacity_mod.GIN_MAX_TOKENS_PER_RANK == 8192
        _reject_cross_node(8192)
    finally:
        monkeypatch.delenv("HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK", raising=False)
        _reload()


def test_env_override_and_disable(monkeypatch):
    """Via module reload."""
    monkeypatch.setenv("HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK", "0")
    _reload()
    try:
        _reject_cross_node(1_000_000)  # Disabled: no raise.
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
