#!/usr/bin/env python
"""The cross-rank DeepEP env check compares resolved settings, since an absent knob and its default are one behaviour.
Comparing raw strings would kill a job whose launcher exports a default on the head node only."""

from __future__ import annotations

import importlib

import pytest

from halo.distributed import grad_reduce as grad_reduce_mod
from halo.distributed import rank_uniform_env
from halo.distributed.expert_parallel.deepep import capacity as capacity_mod
from halo.distributed.expert_parallel.deepep import settings as settings_mod

# Knob -> the string spelling of its own default, per halo/env.py resolution.
_KNOBS_AT_DEFAULT = {
    "HALO_EP_CAPACITY_DEDUP": "1",
    "HALO_DEEPEP_GPU_TIMEOUT_SECONDS": "100",
    "HALO_DEEPEP_NUM_SMS": "0",
    "HALO_DEEPEP_NUM_QPS": "0",
    "HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK": "8192",
    "HALO_GRAD_BUCKET_MB": "256",
    "DIST_NCCL_TIMEOUT_MINUTES": "30",
    "DIST_STORE_TIMEOUT_HOURS": "4",
    "DIST_FS_PROBE_TIMEOUT_SECONDS": "60",
}


# Reloading ``capacity`` for its Gin ceiling must not replace the scope the backends hold.
_CAPACITY_SCOPE = capacity_mod.capacity_scope


def _reload_modules() -> None:
    """Suppliers first, then the check that binds their values."""
    importlib.reload(grad_reduce_mod)
    importlib.reload(settings_mod)
    importlib.reload(capacity_mod).capacity_scope = _CAPACITY_SCOPE
    importlib.reload(rank_uniform_env)


def _reload_settings() -> dict:
    _reload_modules()
    return rank_uniform_env.rank_uniform_settings()


def _settings_with(monkeypatch, env: dict[str, str]) -> dict:
    for name in _KNOBS_AT_DEFAULT:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return _reload_settings()


@pytest.fixture(autouse=True)
def _restore_module():
    """So a reloaded module never leaks into other tests."""
    yield
    _reload_modules()


def test_default_valued_env_is_not_a_divergence(monkeypatch):
    unset = _settings_with(monkeypatch, {})
    explicit = _settings_with(monkeypatch, _KNOBS_AT_DEFAULT)

    assert explicit == unset, (
        "a knob set to its own default must resolve identically to unset; comparing raw env "
        f"strings would reject this aligned pair (unset={unset}, explicit={explicit})"
    )


def test_truthy_spellings_agree(monkeypatch):
    """One boolean to env_flag."""
    spellings = [
        _settings_with(monkeypatch, {"HALO_EP_CAPACITY_DEDUP": v})["HALO_EP_CAPACITY_DEDUP"]
        for v in ("1", "true", "TRUE", "yes", "on")
    ]
    assert spellings == [True] * len(spellings), f"truthy spellings resolved inconsistently: {spellings}"


def test_a_real_difference_still_differs(monkeypatch):
    default = _settings_with(monkeypatch, {})
    disabled = _settings_with(monkeypatch, {"HALO_EP_CAPACITY_DEDUP": "0"})
    assert disabled != default, "HALO_EP_CAPACITY_DEDUP=0 changes collective counts and must be caught"

    retimed = _settings_with(monkeypatch, {"HALO_DEEPEP_GPU_TIMEOUT_SECONDS": "250"})
    assert retimed != default, "a non-default GPU timeout is a real divergence"

    ceiling = _settings_with(monkeypatch, {"HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK": "16384"})
    assert ceiling != default, "a divergent Gin ceiling makes one rank raise while its peers enter dispatch"

    bucket = _settings_with(monkeypatch, {"HALO_GRAD_BUCKET_MB": "128"})
    assert bucket != default, "divergent bucket sizes desync the deferred sweep's chunk boundaries"


def test_coordination_timeouts_are_compared(monkeypatch):
    """The node giving up first dies in a healthy collective, and the survivors blame the next one."""
    default = _settings_with(monkeypatch, {})
    assert default["DIST_NCCL_TIMEOUT_MINUTES"] == 30, "the NCCL watchdog bound must be in the compared set"
    assert default["DIST_STORE_TIMEOUT_HOURS"] == 4, "the store-join bound must be in the compared set"

    assert _settings_with(monkeypatch, {"DIST_NCCL_TIMEOUT_MINUTES": "60"}) != default
    assert _settings_with(monkeypatch, {"DIST_STORE_TIMEOUT_HOURS": "8"}) != default


def test_the_filesystem_probe_budget_is_compared(monkeypatch):
    """A shorter budget reports a slow shared mount as unshared while peers saw it."""
    default = _settings_with(monkeypatch, {})
    assert default["DIST_FS_PROBE_TIMEOUT_SECONDS"] == 60
    assert _settings_with(monkeypatch, {"DIST_FS_PROBE_TIMEOUT_SECONDS": "180"}) != default


def test_grad_bucket_is_reported_in_operator_units(monkeypatch):
    """MB, the unit the operator can grep their ``--env-file`` for."""
    assert _settings_with(monkeypatch, {})["HALO_GRAD_BUCKET_MB"] == 256


def test_every_resolved_knob_is_a_plain_value(monkeypatch):
    settings = _settings_with(monkeypatch, _KNOBS_AT_DEFAULT)
    for name in _KNOBS_AT_DEFAULT:
        assert isinstance(settings[name], (bool, int)), (
            f"{name} resolved to {settings[name]!r} ({type(settings[name]).__name__}); a raw string "
            f"here means unset-vs-default would compare unequal across ranks"
        )
    # EP_DISABLE_GIN is DeepEP-owned and read as a raw string, so it stays a string/None.
    assert "EP_DISABLE_GIN" in settings, "the DeepEP-owned Gin switch must still be compared"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
