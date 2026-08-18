#!/usr/bin/env python
"""The rank-uniform environment check is ONE seam, run EARLY, and it covers the non-EP consumers.

Two things break if it is pinned to ``create_ep_buffers``:

* it runs only on the DeepEP path, while ``HALO_GRAD_BUCKET_MB`` also sets the chunk boundaries of
  the TP replicated sweep and the QLoRA sweep — both bucketed reductions over groups that span OS
  nodes, both of which desync silently when one node's buckets are a different size;
* it runs AFTER the FS-heavy weight load. A world ``all_gather_object`` placed there is the first
  collective a straggler node can miss, so the job dies at ``DIST_NCCL_TIMEOUT_MINUTES`` with a
  traceback naming the env check rather than the load that ran long. Every value it compares is
  known at import, so there is nothing to wait for.

These tests pin the pure divergence verdict, the latch that keeps the post-load call free, and the
fact that the bucket knob is part of the compared set.

    python tests/cpu/parallelism/test_env_uniformity.py
"""

import importlib
import sys
from unittest.mock import patch

import pytest

from src.distributed.expert_parallel import dispatcher as dispatcher_mod
from src.distributed.runtime import divergent_settings, reject_divergent_settings

_MOD = "src.distributed.runtime"


def test_an_aligned_world_reports_nothing():
    values = {"HALO_GRAD_BUCKET_MB": 256, "HALO_EP_CAPACITY_DEDUP": True}
    assert divergent_settings([values] * 8) == {}


def test_one_drifting_node_is_reported_with_both_spellings():
    aligned = {"HALO_GRAD_BUCKET_MB": 256, "HALO_EP_CAPACITY_DEDUP": True}
    drifted = {"HALO_GRAD_BUCKET_MB": 128, "HALO_EP_CAPACITY_DEDUP": True}
    differing = divergent_settings([aligned] * 8 + [drifted] * 8)
    assert list(differing) == ["HALO_GRAD_BUCKET_MB"], f"only the drifting knob may be named: {differing}"
    assert differing["HALO_GRAD_BUCKET_MB"] == ["128", "256"]


def test_a_name_missing_on_one_rank_is_a_divergence():
    """A rank whose module resolved a different key set has a different behaviour, not a smaller one."""
    assert "EP_DISABLE_GIN" in divergent_settings([{"EP_DISABLE_GIN": "1"}, {}])


def test_an_empty_gather_is_not_a_divergence():
    assert divergent_settings([]) == {}


def test_the_collective_raises_with_the_caller_s_guidance():
    """The message must carry the operator's own knob names and rank 0's reference values."""

    def fake_all_gather(out_list, obj):
        del obj
        out_list[:] = [{"HALO_GRAD_BUCKET_MB": 256}] * 4 + [{"HALO_GRAD_BUCKET_MB": 64}] * 4

    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=True),
        patch(f"{_MOD}.dist.get_world_size", return_value=8),
        patch(f"{_MOD}.dist.all_gather_object", side_effect=fake_all_gather),
        pytest.raises(ValueError) as excinfo,
    ):
        reject_divergent_settings({"HALO_GRAD_BUCKET_MB": 256}, "Rank-uniform toolkit environment", "GUIDANCE HERE.")
    message = str(excinfo.value)
    assert "HALO_GRAD_BUCKET_MB" in message and "GUIDANCE HERE." in message
    assert "rank 0 has {'HALO_GRAD_BUCKET_MB': 256}" in message, f"the reference must be shown: {message}"


def test_the_grad_bucket_knob_is_in_the_compared_set():
    """It is consumed by the TP and QLoRA sweeps too, which never touch DeepEP — dropping it from
    the set would leave those two paths with no uniformity guard at all."""
    assert "HALO_GRAD_BUCKET_MB" in dispatcher_mod.rank_uniform_ep_settings()


def _count_joins(monkeypatch) -> list:
    """Arm a fresh latch and record every ``reject_divergent_settings`` the seam issues."""
    calls: list = []
    monkeypatch.setattr(dispatcher_mod, "_ENV_UNIFORMITY_VERIFIED", False)
    monkeypatch.setattr(dispatcher_mod, "reject_divergent_settings", lambda *a, **k: calls.append(a))
    return calls


def test_the_check_runs_at_most_once_per_process(monkeypatch):
    """The entry scripts run it in distributed setup; ``create_ep_buffers`` must then cost nothing,
    or the post-load world join is simply back."""
    calls = _count_joins(monkeypatch)
    monkeypatch.setattr(dispatcher_mod.dist, "is_initialized", lambda: True)

    dispatcher_mod.verify_rank_uniform_env()
    dispatcher_mod.verify_rank_uniform_env()
    dispatcher_mod.verify_rank_uniform_env()
    assert len(calls) == 1, f"the world join ran {len(calls)} times; it is latched to one"

    dispatcher_mod.verify_rank_uniform_env(force=True)
    assert len(calls) == 2, "force= must still re-run it for a caller without the script scaffold"


def test_a_pre_init_call_does_not_disarm_the_backstop(monkeypatch):
    """``reject_divergent_settings`` no-ops before ``init_process_group``, so a call that lands
    there compares nothing.

    Latching on it would leave ``create_ep_buffers``' backstop permanently satisfied by a check that
    never ran — and the backstop exists exactly for the caller that reached DeepEP without the entry
    script's ordered setup, which is the same caller likeliest to have called this early.
    """
    calls = _count_joins(monkeypatch)
    monkeypatch.setattr(dispatcher_mod.dist, "is_initialized", lambda: False)
    dispatcher_mod.verify_rank_uniform_env()
    assert not dispatcher_mod._ENV_UNIFORMITY_VERIFIED, "a no-op call must not latch"

    monkeypatch.setattr(dispatcher_mod.dist, "is_initialized", lambda: True)
    dispatcher_mod.verify_rank_uniform_env()
    assert len(calls) == 2, "the real join must still run once the group exists"
    assert dispatcher_mod._ENV_UNIFORMITY_VERIFIED

    dispatcher_mod.verify_rank_uniform_env()
    assert len(calls) == 2, "and it latches from there, so the backstop stays free"


def test_the_guidance_does_not_scope_the_rule_to_an_ep_group(monkeypatch):
    """The seam is world-wide — the grad-bucket knob is read by the TP and QLoRA sweeps too — so a
    message telling the operator to align "every rank of an EP group" points a non-EP job at a
    group it does not have."""
    guidance: list = []
    monkeypatch.setattr(dispatcher_mod, "_ENV_UNIFORMITY_VERIFIED", False)
    monkeypatch.setattr(dispatcher_mod, "reject_divergent_settings", lambda *a: guidance.append(a[2]))
    dispatcher_mod.verify_rank_uniform_env()
    assert "EP group" not in guidance[0], guidance[0]
    assert "every rank of the job" in guidance[0].lower(), guidance[0]


def test_the_legacy_backend_timeout_warning_gates_on_the_parsed_value(monkeypatch):
    """A docker-compose pass-through (``- HALO_DEEPEP_GPU_TIMEOUT_SECONDS``) exports the name with an
    EMPTY value when it is absent. A presence test warns there about a knob nobody set; the parsed
    value resolves to the default and correctly says nothing."""
    monkeypatch.setenv("HALO_DEEPEP_GPU_TIMEOUT_SECONDS", "")
    reloaded = importlib.reload(dispatcher_mod)
    try:
        assert reloaded._GPU_TIMEOUT_SECONDS == reloaded._DEEPEP_DEFAULT_GPU_TIMEOUT_SECONDS
        monkeypatch.setenv("HALO_DEEPEP_GPU_TIMEOUT_SECONDS", "250")
        reloaded = importlib.reload(dispatcher_mod)
        assert reloaded._GPU_TIMEOUT_SECONDS != reloaded._DEEPEP_DEFAULT_GPU_TIMEOUT_SECONDS, (
            "a value the operator really set must still reach the legacy-backend warning"
        )
    finally:
        monkeypatch.delenv("HALO_DEEPEP_GPU_TIMEOUT_SECONDS", raising=False)
        importlib.reload(dispatcher_mod)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
