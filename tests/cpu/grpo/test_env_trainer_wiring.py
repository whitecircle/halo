#!/usr/bin/env python
"""The environmental GRPO trainer's construction phases and its prefetch thread, run for real on a trainer
built without ``__init__``.

* ``_configure_rollout_state`` takes the rollout, row and objective state from the async config and reaches
  every refusal of a knob that would go unread;
* ``_configure_importance_sampling`` scores at the sampling temperature, turns the correction on only where the
  sampled tokens carry their log-probs, and refuses every stage it would leave inert, the engine re-score
  against the configured backend's client included;
* ``_resolve_environment_spec`` takes a registry config or a custom environment class, and nothing else;
* the prefetch worker thread delivers exactly one output per submitted round, a failure marker included, and
  stops within its join timeout, also while it waits to deliver.

Run: python tests/cpu/grpo/test_env_trainer_wiring.py
"""

import asyncio
import logging
import threading
from types import SimpleNamespace

import pytest
import torch
from transformers.trainer_callback import CallbackHandler
from trl import GRPOConfig

from halo.configs.grpo_environmental import AsyncRolloutConfig, EnvironmentConfig, ISMaskConfig
from halo.environments.environment_base import EnvironmentBase
from halo.trainers.grpo.early_stop import GRPOEarlyStopCallback
from halo.trainers.grpo.environmental import async_rollouts, importance_sampling
from halo.trainers.grpo.environmental.importance_sampling import _BreakerOptimizerSkipCallback
from halo.trainers.grpo.environmental.step_batch import BatchBuildFence
from halo.trainers.grpo.environmental.trainer import (
    DistributedAsyncEnvironmentalGRPOTrainer,
    _resolve_environment_spec,
)
from halo.trainers.grpo.objective.logratio import LOGRATIO_MEAN_KEY
from tests.common.tiny_models import word_level_tokenizer

_STOP = "<stop>"
_TWO_SERVERS = [{"url": "http://127.0.0.1:1"}, {"url": "http://127.0.0.1:2"}]
# Seconds a prefetch test waits on the worker thread before failing instead of hanging.
_WORKER_WAIT_S = 10.0


def _trainer(tmp_path, *, grpo: dict | None = None, **config) -> DistributedAsyncEnvironmentalGRPOTrainer:
    """The trainer with the state its ``__init__`` holds when the configure phases run: TRL's args and
    the attributes TRL's constructor copies off them, the resolved tokenizer, an unbudgeted environment."""
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.async_config = AsyncRolloutConfig(**config)
    trainer.args = GRPOConfig(output_dir=str(tmp_path), bf16=False, use_cpu=True, report_to=[], **(grpo or {}))
    trainer.temperature = trainer.args.temperature
    trainer.vllm_importance_sampling_correction = trainer.args.vllm_importance_sampling_correction
    trainer.use_vllm = False
    trainer._rollout_env = SimpleNamespace(
        reasoning_effort=None,
        thinking_budget_for_effort=lambda level: None,
        thinking_budget_enforced=lambda level: True,
    )
    trainer._tokenizer = word_level_tokenizer([_STOP])
    trainer.model = torch.nn.Module()
    trainer.callback_handler = CallbackHandler([], trainer.model, trainer._tokenizer, None, None)
    return trainer


def _callbacks(trainer, cls) -> list:
    return [callback for callback in trainer.callback_handler.callbacks if isinstance(callback, cls)]


# ── _configure_rollout_state ────────────────────────────────────────────────────


def test_the_rollout_state_follows_the_async_config(tmp_path):
    trainer = _trainer(
        tmp_path,
        rollout_backend="sglang",
        rollout_stop_tokens=[_STOP],
        drop_degenerate_groups=True,
        scale_rewards_std_floor=0.25,
        balance_token_mass=True,
        isr_veto_min=0.1,
        skip_update_masked_frac=0.5,
    )
    trainer._configure_rollout_state()

    assert trainer._rollout_backend == "sglang"
    assert trainer._rollout_stop_token_ids == [trainer._tokenizer.convert_tokens_to_ids(_STOP)]
    assert trainer._reasoning_end_ids is None and trainer._forced_close_ids is None, (
        "an unbudgeted run forces no close"
    )
    assert trainer._is_mask_config == ISMaskConfig(veto_min=0.1)
    assert len(_callbacks(trainer, _BreakerOptimizerSkipCallback)) == 1, "the breaker's optimizer-skip hook"
    assert trainer._breaker_tripped_this_step is False


def test_the_default_config_arms_no_breaker_and_no_stage(tmp_path):
    trainer = _trainer(tmp_path)
    trainer._configure_rollout_state()

    assert trainer._rollout_backend == "vllm"
    assert trainer._rollout_stop_token_ids is None
    assert not trainer._is_mask_config.any_stage_active
    assert _callbacks(trainer, _BreakerOptimizerSkipCallback) == []


@pytest.mark.parametrize(
    ("grpo", "config", "match"),
    [
        ({"off_policy_mask_threshold": 0.5}, {}, r"off_policy_mask_threshold=0.5 is a silent no-op"),
        ({"loss_type": "vespo"}, {}, r"loss_type='vespo' is not supported by environmental GRPO"),
        (
            {"scale_rewards": "none"},
            {"scale_rewards_std_floor": 0.1},
            r"scale_rewards_std_floor=0.1 with scale_rewards='none'",
        ),
        ({"loss_type": "grpo"}, {"balance_token_mass": True}, r"balance_token_mass needs a loss whose loss tokens"),
        ({}, {"rollout_stop_tokens": ["<absent>"]}, r"rollout_stop_tokens \['<absent>'\] are not tokens"),
    ],
    ids=["off-policy-mask", "vespo", "std-floor-without-scaling", "unbalanceable-loss", "unknown-stop-token"],
)
def test_the_rollout_state_reaches_each_refusal(tmp_path, grpo, config, match):
    with pytest.raises(ValueError, match=match):
        _trainer(tmp_path, grpo=grpo, **config)._configure_rollout_state()


# ── _configure_importance_sampling ──────────────────────────────────────────────


def _configured(tmp_path, *, grpo: dict | None = None, **config):
    trainer = _trainer(tmp_path, grpo=grpo, **config)
    trainer._configure_rollout_state()
    return trainer


def test_log_probs_are_scored_at_the_sampling_temperature(tmp_path):
    trainer = _configured(tmp_path, grpo={"temperature": 1.0}, rollout_temperature=0.6)
    trainer._configure_importance_sampling()
    assert trainer.temperature == trainer.args.temperature == 0.6


@pytest.mark.parametrize(
    ("sampled_tokens", "correction", "expected"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_the_correction_runs_only_where_the_sampled_tokens_carry_their_log_probs(
    tmp_path, sampled_tokens, correction, expected
):
    trainer = _configured(
        tmp_path, grpo={"vllm_importance_sampling_correction": correction}, train_on_sampled_tokens=sampled_tokens
    )
    trainer._configure_importance_sampling()
    assert trainer._is_correction is expected
    assert trainer.use_vllm is expected, "TRL reads the sampling log-probs only with use_vllm on"


def test_a_correction_without_sampled_tokens_is_warned_as_disabled(tmp_path, caplog):
    trainer = _configured(tmp_path, train_on_sampled_tokens=False)
    with caplog.at_level(logging.WARNING, logger=importance_sampling.__name__):
        trainer._configure_importance_sampling()
    assert "the correction is DISABLED" in caplog.text


def test_an_early_stop_on_the_gap_attaches_on_the_key_the_correction_logs(tmp_path):
    trainer = _configured(tmp_path, early_stop_logratio_gap=0.01)
    trainer._configure_importance_sampling()
    (stop,) = _callbacks(trainer, GRPOEarlyStopCallback)
    assert stop.gap_key == LOGRATIO_MEAN_KEY


class _NoRescoreClient:
    """A weight-sync client class whose engine declares no re-score route."""

    BACKEND_NAME = "NoRescore"
    SUPPORTS_ENGINE_RESCORE = False


def test_the_engine_reference_is_checked_against_the_configured_backends_client(tmp_path, monkeypatch):
    resolved = []

    def resolve(backend):
        resolved.append(backend)
        return _NoRescoreClient

    monkeypatch.setattr(importance_sampling, "resolve_weight_sync_client", resolve)
    trainer = _configured(tmp_path, rollout_backend="sglang", isr_engine_reference=True)
    with pytest.raises(ValueError, match="isr_engine_reference is not available on NoRescore"):
        trainer._configure_importance_sampling()
    assert resolved == ["sglang"]


@pytest.mark.parametrize(
    ("config", "forced_close_ids", "match"),
    [
        ({"train_on_sampled_tokens": False}, (7,), r"importance-sampling correction is off"),
        (
            {"train_on_sampled_tokens": False, "isr_veto_min": 0.1},
            None,
            r"isr_geo_band/isr_veto/isr_opsm knobs require the importance-sampling correction",
        ),
        (
            {"train_on_sampled_tokens": False, "isr_engine_reference": True},
            None,
            r"isr_engine_reference requires the importance-sampling correction",
        ),
        (
            {"train_on_sampled_tokens": False, "early_stop_logratio_gap": 0.01},
            None,
            r"early_stop_logratio_gap",
        ),
    ],
    ids=["forced-close", "is-stage", "engine-reference", "gap-early-stop"],
)
def test_the_importance_sampling_phase_reaches_each_refusal(tmp_path, config, forced_close_ids, match):
    trainer = _configured(tmp_path, **config)
    trainer._forced_close_ids = forced_close_ids
    with pytest.raises(ValueError, match=match):
        trainer._configure_importance_sampling()


# ── _resolve_environment_spec ───────────────────────────────────────────────────


class _CustomEnvironment(EnvironmentBase):
    """Never instantiated: the spec carries the class for the Ray actors to build."""


def test_a_registry_config_resolves_to_its_type_and_env_config():
    config = EnvironmentConfig(environment_type="EXAM_QA", max_turns=3, environment_kwargs={"k": 1})
    spec, env_config = _resolve_environment_spec(config, _CustomEnvironment, {"ignored": True})
    assert spec == "exam_qa"
    assert env_config == config.to_env_config()


def test_a_custom_class_resolves_to_itself_and_its_kwargs():
    assert _resolve_environment_spec(None, _CustomEnvironment, {"k": 1}) == ((_CustomEnvironment, {"k": 1}), {})
    assert _resolve_environment_spec(None, _CustomEnvironment, None) == ((_CustomEnvironment, {}), {})


def test_anything_else_is_refused():
    with pytest.raises(TypeError, match="environment_cls must be a subclass of EnvironmentBase"):
        _resolve_environment_spec(None, dict, None)
    with pytest.raises(ValueError, match="Either environment_config .* or environment_cls"):
        _resolve_environment_spec(None, None, None)


# ── the prefetch worker thread ──────────────────────────────────────────────────


class _RolloutManager:
    """Collects a round on the worker's own event loop: each prompt's rollout, or a raise for ``"boom"``."""

    def __init__(self):
        self.collected = threading.Event()
        self.rounds: list[list[str]] = []

    async def collect_rollouts(self, prompts, contexts):
        await asyncio.sleep(0)
        self.rounds.append(list(prompts))
        self.collected.set()
        if "boom" in prompts:
            raise RuntimeError("engine unreachable")
        return [f"rollout:{prompt}" for prompt in prompts]


@pytest.fixture
def prefetching(tmp_path, monkeypatch):
    """A trainer whose real async state prefetches over two servers, collecting through :class:`_RolloutManager`,
    its worker's waits shortened; the worker is stopped at teardown."""
    monkeypatch.setattr(async_rollouts, "_PREFETCH_POLL_TIMEOUT_S", 0.01)
    monkeypatch.setattr(async_rollouts, "_PREFETCH_DELIVER_TIMEOUT_S", 0.01)
    # Bounds the wait for an in-flight round, so a lost delivery fails in seconds rather than a minute.
    monkeypatch.setattr(async_rollouts, "_PREFETCH_SUBMIT_TIMEOUT_S", 2.0)
    trainer = _trainer(tmp_path, enable_prefetch=True, rollout_server_configs=_TWO_SERVERS, episode_timeout=1.0)
    trainer._init_async_state()
    trainer._rollout_manager = _RolloutManager()
    trainer._batch_errors = BatchBuildFence()
    yield trainer
    trainer._stop_prefetch_thread()


def test_the_worker_starts_once_and_delivers_each_round_once(prefetching):
    assert prefetching._prefetch_enabled
    prefetching._start_prefetch_thread()
    thread = prefetching._prefetch_thread
    prefetching._start_prefetch_thread()
    assert prefetching._prefetch_thread is thread and thread.is_alive() and thread.name == "PrefetchWorker"

    prefetching._submit_for_prefetch(["a", "b"], [None, None])
    assert prefetching._wait_for_inflight_prefetch() == ["rollout:a", "rollout:b"]
    assert prefetching._prefetch_inflight == 0
    assert prefetching._rollout_manager.rounds == [["a", "b"]]
    assert prefetching._prefetch_queue.empty(), "one output per round, no more"
    assert prefetching._batch_errors.reason is None


def test_a_failed_round_is_delivered_as_a_marker_so_the_inflight_count_stays_exact(prefetching, caplog):
    prefetching._start_prefetch_thread()
    with caplog.at_level(logging.ERROR, logger=async_rollouts.__name__):
        prefetching._submit_for_prefetch(["boom"], [None])
        assert prefetching._wait_for_inflight_prefetch() is None, "a failed round falls back to a sync collection"
    assert prefetching._prefetch_inflight == 0
    assert "Prefetch worker error (1 prompts): engine unreachable" in caplog.text

    prefetching._submit_for_prefetch(["c"], [None])
    assert prefetching._wait_for_inflight_prefetch() == ["rollout:c"], "the worker survives a failed round"


def test_stopping_joins_the_worker_and_drops_what_no_step_consumes(prefetching):
    prefetching._start_prefetch_thread()
    thread = prefetching._prefetch_thread
    prefetching._submit_for_prefetch(["a"], [None])
    assert prefetching._rollout_manager.collected.wait(_WORKER_WAIT_S)

    prefetching._stop_prefetch_thread()

    assert prefetching._prefetch_thread is None and not thread.is_alive()
    assert prefetching._prefetch_inflight == 0
    assert prefetching._prefetch_queue.empty()


def test_a_worker_waiting_to_deliver_abandons_the_round_at_stop(prefetching, caplog):
    """The output queue holds one unconsumed round, so the next one waits in the delivery slices; a stop must
    end that wait instead of leaving the join to time out."""
    prefetching._prefetch_queue.put((1, ["rollout:unconsumed"]))
    prefetching._start_prefetch_thread()
    thread = prefetching._prefetch_thread
    prefetching._submit_for_prefetch(["a"], [None])
    assert prefetching._rollout_manager.collected.wait(_WORKER_WAIT_S)

    with caplog.at_level(logging.WARNING, logger=async_rollouts.__name__):
        prefetching._stop_prefetch_thread()

    assert not thread.is_alive()
    assert "abandoning delivery of a 1-prompt rollout batch" in caplog.text
    assert "did not stop cleanly" not in caplog.text
    assert prefetching._prefetch_queue.empty()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
