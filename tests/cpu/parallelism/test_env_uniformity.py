#!/usr/bin/env python
"""The rank-uniform environment check is ONE seam, run EARLY, and it covers the non-EP consumers.

Two things break if it runs only on the DeepEP path:

* it runs only on the DeepEP path, while ``HALO_GRAD_BUCKET_MB`` also sets the chunk boundaries of
  the TP replicated sweep and the QLoRA sweep — both bucketed reductions over groups that span OS
  nodes, both of which desync silently when one node's buckets are a different size;
* it runs AFTER the FS-heavy weight load. A world collective placed there is the first collective
  a straggler node can miss, so the job dies at ``DIST_NCCL_TIMEOUT_MINUTES`` with a
  traceback naming the env check rather than the load that ran long. Every value it compares is
  known at import, so there is nothing to wait for.

These tests pin the pure divergence verdict, the latch that keeps the loader's backstop free, that every
distributed load runs that backstop first, and the fact that the bucket knob is part of the compared set.

    python tests/cpu/parallelism/test_env_uniformity.py
"""

import importlib
from unittest.mock import patch

import pytest
import torch

from src.distributed import rank_uniform_env
from src.distributed.expert_parallel.deepep import settings as settings_mod
from src.model_loading.parallel import loader as model_loading
from src.runtime.consensus import divergent_settings, reject_divergent_settings
from tests.common.parallelism import make_parallelism_config

_MOD = "src.runtime.consensus"


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

    def disagreeing_digests(slots, op=None, group=None):
        slots.copy_(torch.tensor([7, -3, 0, 0]))  # max(d) != -max(-d): two digests met

    with (
        patch(f"{_MOD}.dist.is_available", return_value=True),
        patch(f"{_MOD}.dist.is_initialized", return_value=True),
        patch(f"{_MOD}.dist.get_world_size", return_value=8),
        patch(f"{_MOD}.dist.get_rank", return_value=0),
        patch(f"{_MOD}.collective_device", return_value=torch.device("cpu")),
        patch(f"{_MOD}.dist.all_reduce", side_effect=disagreeing_digests),
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
    assert "HALO_GRAD_BUCKET_MB" in rank_uniform_env.rank_uniform_settings()


def _count_joins(monkeypatch) -> list:
    """Arm a fresh latch and record every ``reject_divergent_settings`` the seam issues."""
    calls: list = []
    monkeypatch.setattr(rank_uniform_env, "_env_uniformity_verified", False)
    monkeypatch.setattr(rank_uniform_env, "reject_divergent_settings", lambda *a, **k: calls.append(a))
    return calls


def test_the_check_runs_at_most_once_per_process(monkeypatch):
    """The entry scripts run it in distributed setup; the loader's backstop must then cost nothing,
    or the post-load world join is simply back."""
    calls = _count_joins(monkeypatch)
    monkeypatch.setattr(rank_uniform_env.dist, "is_initialized", lambda: True)

    rank_uniform_env.agree_rank_uniform_env()
    rank_uniform_env.agree_rank_uniform_env()
    rank_uniform_env.agree_rank_uniform_env()
    assert len(calls) == 1, f"the world join ran {len(calls)} times; it is latched to one"


def test_a_pre_init_call_does_not_disarm_the_backstop(monkeypatch):
    """``reject_divergent_settings`` no-ops before ``init_process_group``, so a call that lands
    there compares nothing.

    Latching on it would leave ``load_distributed_model``'s backstop permanently satisfied by a check that
    never ran — and the backstop exists exactly for the caller that reached DeepEP without the entry
    script's ordered setup, which is the same caller likeliest to have called this early.
    """
    calls = _count_joins(monkeypatch)
    monkeypatch.setattr(rank_uniform_env.dist, "is_initialized", lambda: False)
    rank_uniform_env.agree_rank_uniform_env()
    assert not rank_uniform_env._env_uniformity_verified, "a no-op call must not latch"

    monkeypatch.setattr(rank_uniform_env.dist, "is_initialized", lambda: True)
    rank_uniform_env.agree_rank_uniform_env()
    assert len(calls) == 2, "the real join must still run once the group exists"
    assert rank_uniform_env._env_uniformity_verified

    rank_uniform_env.agree_rank_uniform_env()
    assert len(calls) == 2, "and it latches from there, so the backstop stays free"


class _LoadStarted(Exception):
    """Raised by the loader's first step after the check, so the test reads the order and loads nothing."""


@pytest.mark.parametrize("ep_size", [1, 8], ids=["dense", "ep"])
def test_every_distributed_load_agrees_the_env_before_it_loads(monkeypatch, ep_size):
    """The loader's backstop runs before anything else, for a dense load as for an EP one: a caller that
    skipped the entry script's setup still gets the join before its first rank-asymmetric work."""
    events: list = []
    monkeypatch.setattr(model_loading, "agree_rank_uniform_env", lambda: events.append("agree"))

    def first_load_step(*args, **kwargs):
        events.append("load")
        raise _LoadStarted

    monkeypatch.setattr(model_loading, "reject_unsupported_run", first_load_step)
    with pytest.raises(_LoadStarted):
        model_loading.load_distributed_model(
            "unused", make_parallelism_config(ep_size=ep_size), trust_remote_code=True
        )
    assert events == ["agree", "load"]


def test_the_guidance_does_not_scope_the_rule_to_an_ep_group(monkeypatch):
    """The seam is world-wide — the grad-bucket knob is read by the TP and QLoRA sweeps too — so a
    message telling the operator to align "every rank of an EP group" points a non-EP job at a
    group it does not have."""
    guidance: list = []
    monkeypatch.setattr(rank_uniform_env, "_env_uniformity_verified", False)
    monkeypatch.setattr(rank_uniform_env, "reject_divergent_settings", lambda *a: guidance.append(a[2]))
    rank_uniform_env.agree_rank_uniform_env()
    assert "EP group" not in guidance[0], guidance[0]
    assert "every rank of the job" in guidance[0].lower(), guidance[0]


def test_the_legacy_backend_timeout_warning_gates_on_the_parsed_value(monkeypatch):
    """A docker-compose pass-through (``- HALO_DEEPEP_GPU_TIMEOUT_SECONDS``) exports the name with an
    EMPTY value when it is absent. A presence test warns there about a knob nobody set; the parsed
    value resolves to the default and correctly says nothing."""
    monkeypatch.setenv("HALO_DEEPEP_GPU_TIMEOUT_SECONDS", "")
    reloaded = importlib.reload(settings_mod)
    try:
        assert reloaded.DEEPEP_GPU_TIMEOUT_S == reloaded.DEEPEP_DEFAULT_GPU_TIMEOUT_S
        monkeypatch.setenv("HALO_DEEPEP_GPU_TIMEOUT_SECONDS", "250")
        reloaded = importlib.reload(settings_mod)
        assert reloaded.DEEPEP_GPU_TIMEOUT_S != reloaded.DEEPEP_DEFAULT_GPU_TIMEOUT_S, (
            "a value the operator really set must still reach the legacy-backend warning"
        )
    finally:
        monkeypatch.delenv("HALO_DEEPEP_GPU_TIMEOUT_SECONDS", raising=False)
        importlib.reload(settings_mod)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
