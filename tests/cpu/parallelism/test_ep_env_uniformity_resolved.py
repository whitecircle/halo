#!/usr/bin/env python
"""The cross-rank DeepEP env check compares RESOLVED settings, not raw strings.

``verify_rank_uniform_env`` refuses a job whose ranks disagree on a knob that changes how many
collectives a rank runs. Every ``HALO_*`` knob in that set is consumed through :mod:`src.env`, where
an ABSENT variable and the variable set to its default are the same behaviour. Comparing the raw
strings instead would reject the ordinary multi-node shape — a launcher or per-node ``--env-file``
exporting a default on the head node only — killing a correctly configured job and naming variables
that are in fact aligned.

These tests pin both directions: default-valued equals unset, and a genuinely different value still
differs. Run: ``pytest -m cpu tests/cpu/parallelism/test_ep_env_uniformity_resolved.py``
"""

from __future__ import annotations

import importlib
import sys

import pytest

from src.distributed import grad_reduce as grad_reduce_mod
from src.distributed.expert_parallel import config as ep_config_mod
from src.distributed.expert_parallel import dispatcher as dispatcher_mod

# Knob -> the string spelling of its own default, per src/env.py resolution.
_KNOBS_AT_DEFAULT = {
    "HALO_EP_CAPACITY_DEDUP": "1",
    "HALO_DEEPEP_GPU_TIMEOUT_SECONDS": "100",
    "HALO_DEEPEP_NUM_SMS": "0",
    "HALO_DEEPEP_NUM_QPS": "0",
    "HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK": "8192",
    "HALO_GRAD_BUCKET_MB": "256",
    "DIST_NCCL_TIMEOUT_MINUTES": "30",
    "DIST_STORE_TIMEOUT_HOURS": "4",
}


def _reload_settings() -> dict:
    """Re-resolve the settings from the current environment.

    The dispatcher's suppliers first — it imports its bucket constant from ``grad_reduce`` and the
    Gin dispatch ceiling from the EP config leaf, so reloading the dispatcher alone would re-bind
    their stale values.
    """
    importlib.reload(grad_reduce_mod)
    importlib.reload(ep_config_mod)
    return importlib.reload(dispatcher_mod).rank_uniform_ep_settings()


def _settings_with(monkeypatch, env: dict[str, str]) -> dict:
    """Resolved rank-uniform settings with exactly ``env`` set (others cleared)."""
    for name in _KNOBS_AT_DEFAULT:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return _reload_settings()


@pytest.fixture(autouse=True)
def _restore_module():
    """Reload with the ambient environment so a reloaded module never leaks into other tests."""
    yield
    importlib.reload(grad_reduce_mod)
    importlib.reload(ep_config_mod)
    importlib.reload(dispatcher_mod)


def test_default_valued_env_is_not_a_divergence(monkeypatch):
    """A node exporting each knob's own default must compare EQUAL to a node leaving it unset."""
    unset = _settings_with(monkeypatch, {})
    explicit = _settings_with(monkeypatch, _KNOBS_AT_DEFAULT)

    assert explicit == unset, (
        "a knob set to its own default must resolve identically to unset; comparing raw env "
        f"strings would reject this aligned pair (unset={unset}, explicit={explicit})"
    )


def test_truthy_spellings_agree(monkeypatch):
    """``1``/``true``/``on`` are one boolean to env_flag, so they must not read as three settings."""
    spellings = [
        _settings_with(monkeypatch, {"HALO_EP_CAPACITY_DEDUP": v})["HALO_EP_CAPACITY_DEDUP"]
        for v in ("1", "true", "TRUE", "yes", "on")
    ]
    assert spellings == [True] * len(spellings), f"truthy spellings resolved inconsistently: {spellings}"


def test_a_real_difference_still_differs(monkeypatch):
    """Anti-vacuity: the check must still catch a knob genuinely set differently."""
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
    """A per-rank timeout divergence is one rank aborting the job alone, ahead of its peers.

    Both are set exactly the way a divergence happens — a per-node ``--env-file`` or a launcher task
    that pins one of them — and the symptom is the worst kind: the node that gives up first dies
    inside a healthy collective, and the survivors then fail on the NEXT one and blame it.
    """
    default = _settings_with(monkeypatch, {})
    assert default["DIST_NCCL_TIMEOUT_MINUTES"] == 30, "the NCCL watchdog bound must be in the compared set"
    assert default["DIST_STORE_TIMEOUT_HOURS"] == 4, "the store-join bound must be in the compared set"

    assert _settings_with(monkeypatch, {"DIST_NCCL_TIMEOUT_MINUTES": "60"}) != default
    assert _settings_with(monkeypatch, {"DIST_STORE_TIMEOUT_HOURS": "8"}) != default


def test_grad_bucket_is_reported_in_operator_units(monkeypatch):
    """The bucket entry carries MB — the unit the operator sets — so a divergence message names a
    value they can grep their ``--env-file`` for, not the derived byte count."""
    assert _settings_with(monkeypatch, {})["HALO_GRAD_BUCKET_MB"] == 256


def test_every_resolved_knob_is_a_plain_value(monkeypatch):
    """Values must be resolved (bool/int), never the raw string — that is the whole point."""
    settings = _settings_with(monkeypatch, _KNOBS_AT_DEFAULT)
    for name in _KNOBS_AT_DEFAULT:
        assert isinstance(settings[name], (bool, int)), (
            f"{name} resolved to {settings[name]!r} ({type(settings[name]).__name__}); a raw string "
            f"here means unset-vs-default would compare unequal across ranks"
        )
    # EP_DISABLE_GIN is DeepEP-owned and read by DeepEP as a raw string, so it stays a string/None.
    assert "EP_DISABLE_GIN" in settings, "the DeepEP-owned Gin switch must still be compared"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
