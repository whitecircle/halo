#!/usr/bin/env python
"""The cross-rank DeepEP env check compares resolved settings, since an absent knob and its default are one behaviour.
Comparing raw strings would kill a job whose launcher exports a default on the head node only."""

from __future__ import annotations

import importlib
import logging
import os

import pytest

from halo.distributed import grad_reduce as grad_reduce_mod
from halo.distributed import rank_uniform_env
from halo.distributed.expert_parallel.deepep import backends as backends_mod
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
    "HALO_FLEX_SLIDING": "1",
    "HALO_FP32_MATMUL_PRECISION": "highest",
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


@pytest.mark.parametrize("knob", ["HALO_DEEPEP_NUM_SMS", "HALO_DEEPEP_NUM_QPS"])
def test_a_negative_deepep_count_means_auto(monkeypatch, caplog, knob):
    """A negative is truthy, so it would reach DeepEP as the count to use."""
    with caplog.at_level(logging.WARNING, logger="halo.env"):
        assert _settings_with(monkeypatch, {knob: "-4"})[knob] == 0
    assert f"{knob}=-4" in caplog.text
    assert _settings_with(monkeypatch, {knob: "8"})[knob] == 8


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


@pytest.mark.parametrize("raw", ["0", "-64"])
def test_a_non_positive_grad_bucket_keeps_the_default(monkeypatch, raw):
    """A cap of zero or less means one collective per tensor."""
    assert _settings_with(monkeypatch, {"HALO_GRAD_BUCKET_MB": raw})["HALO_GRAD_BUCKET_MB"] == 256
    assert grad_reduce_mod.BUCKET_MAX_BYTES == 256 * 1024 * 1024


def test_the_attention_backend_and_matmul_precision_knobs_are_compared(monkeypatch):
    """One rank on another backend compiles and waits where its peers do not; TF32 on one rank drifts its replica."""
    default = _settings_with(monkeypatch, {})
    assert default["HALO_FLEX_SLIDING"] is True
    assert default["HALO_FP32_MATMUL_PRECISION"] == "highest"
    assert _settings_with(monkeypatch, {"HALO_FLEX_SLIDING": "0"}) != default
    assert _settings_with(monkeypatch, {"HALO_FP32_MATMUL_PRECISION": "high"}) != default


def test_an_invalid_matmul_precision_compares_as_its_fallback(monkeypatch):
    default = _settings_with(monkeypatch, {})
    with pytest.warns(UserWarning, match="HALO_FP32_MATMUL_PRECISION"):
        assert _settings_with(monkeypatch, {"HALO_FP32_MATMUL_PRECISION": "not-a-precision"}) == default


def test_grad_bucket_is_reported_in_operator_units(monkeypatch):
    """MB, the unit the operator can grep their ``--env-file`` for."""
    assert _settings_with(monkeypatch, {})["HALO_GRAD_BUCKET_MB"] == 256


@pytest.mark.parametrize(
    ("raw", "spans_domains", "expected"), [("", True, "0"), ("  ", False, "1"), ("1", True, "1"), ("0", False, "0")]
)
def test_the_gin_switch_derives_from_the_topology_unless_given_a_value(monkeypatch, raw, spans_domains, expected):
    """A blank export must not reach DeepEP's own read of the variable."""
    monkeypatch.setattr(backends_mod, "_deepep_env_configured", False)
    monkeypatch.setenv("EP_DISABLE_GIN", raw)
    backends_mod._configure_deepep_env(spans_domains)
    assert os.environ["EP_DISABLE_GIN"] == expected


def test_a_blank_gin_switch_compares_as_unset(monkeypatch):
    monkeypatch.delenv("EP_DISABLE_GIN", raising=False)
    unset = _settings_with(monkeypatch, {})
    assert _settings_with(monkeypatch, {"EP_DISABLE_GIN": " "}) == unset
    assert _settings_with(monkeypatch, {"EP_DISABLE_GIN": "1"}) != unset


def test_every_resolved_knob_is_a_plain_value(monkeypatch):
    settings = _settings_with(monkeypatch, _KNOBS_AT_DEFAULT)
    for name in _KNOBS_AT_DEFAULT.keys() - {"HALO_FP32_MATMUL_PRECISION"}:  # a precision is a name by nature
        assert isinstance(settings[name], (bool, int)), (
            f"{name} resolved to {settings[name]!r} ({type(settings[name]).__name__}); a raw string "
            f"here means unset-vs-default would compare unequal across ranks"
        )
    # EP_DISABLE_GIN is DeepEP-owned and read as a raw string, so it stays a string/None.
    assert "EP_DISABLE_GIN" in settings, "the DeepEP-owned Gin switch must still be compared"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
