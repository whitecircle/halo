#!/usr/bin/env python
"""The rank-uniform environment check is one seam, run early, covering the TP and QLoRA sweeps too. A world
collective after the load is the first a straggler can miss, blaming the env check."""

import datetime
import importlib
import json
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from halo.checkpoint.writing import StreamingShardWriter
from halo.distributed import rank_uniform_env
from halo.distributed.expert_parallel.deepep import settings as settings_mod
from halo.model_loading.parallel import loader as model_loading
from halo.models.attention.selection import FLASH_ATTENTION_DETERMINISTIC_ENV
from halo.runtime.consensus import divergent_settings, reject_divergent_settings
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import make_parallelism_config

_MOD = "halo.runtime.consensus"

PG_TIMEOUT = datetime.timedelta(seconds=90)
# A rank reaching the stand-in gather alone waits this long before the case fails.
WRITE_THREADS_PG_TIMEOUT = datetime.timedelta(seconds=20)
_WRITE_THREAD_COUNTS = {"one_rank_negative": ("-1", "0"), "both_negative": ("-2", "-2"), "differing": ("2", "0")}
# None: unset.
_DETERMINISM_SPELLINGS = {
    "one_rank_deterministic": (None, "1"),
    "zero_and_unset": ("0", None),
    "both_deterministic": ("1", "1"),
}


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
    """A different key set is a different behaviour."""
    assert "EP_DISABLE_GIN" in divergent_settings([{"EP_DISABLE_GIN": "1"}, {}])


def test_an_empty_gather_is_not_a_divergence():
    assert divergent_settings([]) == {}


def test_the_collective_raises_with_the_caller_s_guidance():
    def fake_all_gather(out_list, obj):
        del obj
        out_list[:] = [{"HALO_GRAD_BUCKET_MB": 256}] * 4 + [{"HALO_GRAD_BUCKET_MB": 64}] * 4

    def disagreeing_digests(slots, op=None, group=None):
        slots.copy_(torch.tensor([7, -3]))  # max(d) != -max(-d): two digests met.

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
    """Dropping it would leave the TP and QLoRA sweeps with no uniformity guard."""
    assert "HALO_GRAD_BUCKET_MB" in rank_uniform_env.rank_uniform_settings()


@pytest.mark.parametrize(("value", "expected"), [(None, 1), ("0", 0), ("3", 3)])
def test_the_weight_sync_gather_order_is_compared_across_ranks(value, expected):
    """Ranks issuing their expert gathers in different orders run different collective sequences."""
    env = {} if value is None else {"HALO_WEIGHT_SYNC_GATHER_AHEAD": value}
    with patch.dict(os.environ, env):
        settings = rank_uniform_env.rank_uniform_settings()
    assert settings["HALO_WEIGHT_SYNC_GATHER_AHEAD"] == expected


def test_the_parity_probe_mode_is_in_the_compared_set(monkeypatch):
    """'off' skips a collective forward, so a rank holding it alone would hang its peers."""
    monkeypatch.setenv("HALO_WEIGHT_SYNC_PARITY_PROBE", "off")
    assert rank_uniform_env.rank_uniform_settings()["HALO_WEIGHT_SYNC_PARITY_PROBE"] == "off"


def test_ranks_holding_different_parity_probe_modes_are_refused(monkeypatch):
    """The probe forward is collective: 'off' on one rank and 'warn' on another must be named, not passed."""
    per_rank = []
    for mode in ("off", "warn"):
        monkeypatch.setenv("HALO_WEIGHT_SYNC_PARITY_PROBE", mode)
        per_rank.append(rank_uniform_env.rank_uniform_settings())
    assert divergent_settings(per_rank) == {"HALO_WEIGHT_SYNC_PARITY_PROBE": ["off", "warn"]}


def _count_joins(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(rank_uniform_env, "_env_uniformity_verified", False)
    monkeypatch.setattr(rank_uniform_env, "reject_divergent_settings", lambda *a, **k: calls.append(a))
    return calls


def test_the_check_runs_at_most_once_per_process(monkeypatch):
    """Otherwise the post-load world join is back."""
    calls = _count_joins(monkeypatch)
    monkeypatch.setattr(rank_uniform_env, "has_process_group", lambda: True)

    rank_uniform_env.agree_rank_uniform_env()
    rank_uniform_env.agree_rank_uniform_env()
    rank_uniform_env.agree_rank_uniform_env()
    assert len(calls) == 1, f"the world join ran {len(calls)} times; it is latched to one"


def test_a_pre_init_call_does_not_disarm_the_backstop(monkeypatch):
    """``reject_divergent_settings`` no-ops before ``init_process_group``; latching there would permanently satisfy
    the backstop with a check that never ran."""
    calls = _count_joins(monkeypatch)
    monkeypatch.setattr(rank_uniform_env, "has_process_group", lambda: False)
    rank_uniform_env.agree_rank_uniform_env()
    assert not rank_uniform_env._env_uniformity_verified, "a no-op call must not latch"

    monkeypatch.setattr(rank_uniform_env, "has_process_group", lambda: True)
    rank_uniform_env.agree_rank_uniform_env()
    assert len(calls) == 2, "the real join must still run once the group exists"
    assert rank_uniform_env._env_uniformity_verified

    rank_uniform_env.agree_rank_uniform_env()
    assert len(calls) == 2, "and it latches from there, so the backstop stays free"


class _LoadStarted(Exception):
    """Raised by the loader's first step after the check, so the test reads the order and loads nothing."""


@pytest.mark.parametrize("ep_size", [1, 8], ids=["dense", "ep"])
def test_every_distributed_load_agrees_the_env_before_it_loads(monkeypatch, ep_size):
    """A caller that skipped the entry script's setup still gets the join before its first rank-asymmetric work."""
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
    """The grad-bucket knob is world-wide, so naming an EP group points a non-EP job at a group it lacks."""
    guidance: list = []
    monkeypatch.setattr(rank_uniform_env, "_env_uniformity_verified", False)
    monkeypatch.setattr(rank_uniform_env, "reject_divergent_settings", lambda *a: guidance.append(a[2]))
    rank_uniform_env.agree_rank_uniform_env()
    assert "EP group" not in guidance[0], guidance[0]
    assert "every rank of the job" in guidance[0].lower(), guidance[0]


def test_the_legacy_backend_timeout_warning_gates_on_the_parsed_value(monkeypatch):
    """Compose's ``${VAR:-}`` and a blank ``--env-file`` line export the name empty, so a presence test warns about a
    knob nobody set."""
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


def _determinism_worker(rank: int, tmp_dir: str) -> None:
    outcomes = {}
    for case, spellings in _DETERMINISM_SPELLINGS.items():
        if spellings[rank] is None:
            os.environ.pop(FLASH_ATTENTION_DETERMINISTIC_ENV, None)
        else:
            os.environ[FLASH_ATTENTION_DETERMINISTIC_ENV] = spellings[rank]
        rank_uniform_env._env_uniformity_verified = False
        try:
            rank_uniform_env.agree_rank_uniform_env()
            outcomes[case] = None
        except ValueError as error:
            outcomes[case] = str(error)
    with open(os.path.join(tmp_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(outcomes, handle)


def test_ranks_disagreeing_on_flash_attention_determinism_are_refused(tmp_path):
    """Set on one rank only, it loads FA4 there and its fallback on the peer for a head_dim-256 model. ``0`` and
    unset are one setting to transformers."""
    run_gloo_ranks(_determinism_worker, 2, str(tmp_path), pg_timeout=PG_TIMEOUT)
    for rank in range(2):
        outcomes = json.loads((tmp_path / f"rank{rank}.json").read_text())
        divergence = f"'{FLASH_ATTENTION_DETERMINISTIC_ENV}': ['False', 'True']"
        assert divergence in (outcomes["one_rank_deterministic"] or ""), f"rank {rank}: {outcomes}"
        assert outcomes["zero_and_unset"] is None, f"rank {rank}: {outcomes}"
        assert outcomes["both_deterministic"] is None, f"rank {rank}: {outcomes}"


def _write_threads_worker(rank: int, tmp_dir: str, case: str) -> None:
    os.environ["HALO_CHECKPOINT_WRITE_THREADS"] = _WRITE_THREAD_COUNTS[case][rank]
    rank_uniform_env._env_uniformity_verified = False
    outcome = {"refused": None, "refused_at": None, "reached_gather": False}
    phase = "startup"
    try:
        rank_uniform_env.agree_rank_uniform_env()
        phase = "save"
        with StreamingShardWriter(os.path.join(tmp_dir, f"save{rank}"), "1MB", enabled=rank == 0):
            outcome["reached_gather"] = True
            dist.barrier()
    except ValueError as error:
        outcome["refused"], outcome["refused_at"] = str(error), phase
    with open(os.path.join(tmp_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(outcome, handle)


@pytest.mark.parametrize("case", sorted(_WRITE_THREAD_COUNTS))
def test_a_write_thread_count_refused_anywhere_is_refused_at_startup_on_every_rank(tmp_path, case):
    """A negative count on one rank would raise at the save while its peers wait in the first gather."""
    run_gloo_ranks(_write_threads_worker, 2, str(tmp_path), case, pg_timeout=WRITE_THREADS_PG_TIMEOUT)
    expected = "-2: expected a thread count >= 0" if case == "both_negative" else "'HALO_CHECKPOINT_WRITE_THREADS': ["
    for rank in range(2):
        outcome = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert not outcome["reached_gather"], f"rank {rank} reached the save: {outcome}"
        assert outcome["refused_at"] == "startup" and expected in outcome["refused"], f"rank {rank}: {outcome}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
