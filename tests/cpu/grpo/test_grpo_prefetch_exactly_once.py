#!/usr/bin/env python
"""Env-GRPO prefetch pipeline: exactly-once training on misses + loud drop accounting.

A prefetch MISS that collects the round's rollouts synchronously AND re-submits the same prompts
makes the next hit train a second fresh rollout of a batch already trained, and every mid-stream
miss grows the pipeline lag; a full input queue silently skips a dataset batch, and a full output
queue silently discards completed rollouts. The contract:

- a miss with submissions in flight BLOCKS for the in-flight batch (``_wait_for_inflight_prefetch``)
  instead of duplicating it — only the documented cold-start round primes via a sync collection;
- the first weight push and the prefetch thread start at TRAIN-BEGIN, after the resume restore —
  started at component-init they would serve (and roll out) the pre-restore weights;
- ``_submit_for_prefetch`` counts and warns on a queue-full skip instead of suppressing it;
- the worker delivers exactly one output item per submission (results or a failure marker), so the
  trainer's in-flight accounting never strands a blocking consumer;
- a WEDGED pipeline is recorded in ``_batch_build_error`` and fenced by
  ``_raise_batch_error_uniformly``, never raised on the one rank that hit it — prefetch state is
  per-rank, so a lone raise between two collectives parks every peer until the NCCL watchdog.

    python tests/cpu/grpo/test_grpo_prefetch_exactly_once.py
"""

import queue
import sys
import threading
import types

import pytest
import torch

import src.trainers.grpo.rollout.async_rollouts as async_mod
from src.configs.async_training_config import AsyncTrainingConfig
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer as _T


class _Host:
    """Minimal stand-in running the REAL prefetch decision flow with mocked rollout collection."""

    _generate_and_score_completions_base = _T._generate_and_score_completions_base
    _try_get_prefetched_results = _T._try_get_prefetched_results
    _wait_for_inflight_prefetch = _T._wait_for_inflight_prefetch
    _submit_for_prefetch = _T._submit_for_prefetch
    _extract_prompts_and_contexts = _T._extract_prompts_and_contexts
    _raise_batch_error_uniformly = _T._raise_batch_error_uniformly
    _rollout_engine_name = _T._rollout_engine_name

    def __init__(self, buffer_size: int = 1):
        self.model = types.SimpleNamespace(training=True)
        self.accelerator = types.SimpleNamespace(device=torch.device("cpu"), is_main_process=False)
        self.async_config = types.SimpleNamespace(episode_timeout=5.0, rollout_backend="vllm")
        # profiling_context reads these on the acquire path; report_to=[] makes it a no-op.
        self.state = types.SimpleNamespace(global_step=0)
        self.args = types.SimpleNamespace(report_to=[])
        self._group_random_effort = False
        self._batch_build_error = None
        self._prefetch_enabled = True
        self._prefetch_queue = queue.Queue(maxsize=buffer_size)
        self._prefetch_input_queue = queue.Queue(maxsize=buffer_size + 1)
        self._prefetch_hits = 0
        self._prefetch_misses = 0
        self._prefetch_inflight = 0
        self._prefetch_input_skips = 0
        # Every sync collection is recorded here — the duplication fingerprint.
        self.sync_calls: list[list[str]] = []
        self.trained: list[list[str]] = []
        self._loop = types.SimpleNamespace(run_until_complete=lambda batch: batch)
        self._rollout_manager = types.SimpleNamespace(collect_rollouts=self._collect)

    def _collect(self, prompts, contexts):
        self.sync_calls.append(list(prompts))
        return [f"sync:{p}" for p in prompts]

    def _broadcast_rollouts_for_tp(self, rollout_results):
        return rollout_results

    def _build_training_tensors(self, rollout_results, device, mode):
        self.trained.append(list(rollout_results))
        return {"rollouts": rollout_results}

    def _log_rollout_metrics(self, results, mode):
        pass

    def round(self, prompts: list[str]):
        return self._generate_and_score_completions_base([{"prompt": p} for p in prompts])

    def worker_step(self):
        """One synchronous stand-in for the prefetch worker: input batch → completed rollouts."""
        prompts, _contexts = self._prefetch_input_queue.get_nowait()
        self._prefetch_queue.put((len(prompts), [f"prefetch:{p}" for p in prompts]))


def test_miss_with_inflight_waits_instead_of_duplicating():
    host = _Host()

    # The cold round is the one documented case that both sync-collects and primes on the same prompts.
    host.round(["p0"])
    assert host.sync_calls == [["p0"]]
    assert host.trained == [["sync:p0"]]
    assert host._prefetch_inflight == 1

    host.worker_step()
    host.round(["p1"])
    assert host.trained[-1] == ["prefetch:p0"]
    assert host.sync_calls == [["p0"]]
    assert host._prefetch_inflight == 1

    # p1 is still in flight, so this round must block for it — sync-collecting p2 here would train
    # p1's batch again on the next hit.
    delivery = threading.Timer(0.2, host.worker_step)
    delivery.start()
    try:
        host.round(["p2"])
    finally:
        delivery.join()
    assert host.trained[-1] == ["prefetch:p1"]
    assert host.sync_calls == [["p0"]], "a miss with rollouts in flight must not collect a duplicate"
    assert host._prefetch_inflight == 1  # p2 submitted, pipeline lag still exactly one round

    # Nothing past the cold-start priming may be trained twice.
    flat = [p for batch in host.trained for p in batch]
    assert flat.count("prefetch:p1") == 1
    assert "sync:p2" not in flat


def test_failure_marker_decrements_inflight_and_falls_back():
    host = _Host(buffer_size=2)
    host._prefetch_inflight = 2
    host._prefetch_queue.put((1, None))  # worker failure marker
    host._prefetch_queue.put((1, ["prefetch:pX"]))

    assert host._try_get_prefetched_results() is None
    assert host._prefetch_inflight == 1
    assert host._wait_for_inflight_prefetch() == ["prefetch:pX"]
    assert host._prefetch_inflight == 0


def test_all_failed_inflight_returns_none_for_sync_fallback():
    host = _Host()
    host._prefetch_inflight = 1
    host._prefetch_queue.put((1, None))
    assert host._wait_for_inflight_prefetch() is None
    assert host._prefetch_inflight == 0


def test_wedged_pipeline_is_recorded_for_the_uniform_fence_not_raised():
    """A wedge must RECORD and fall back, so every rank raises together at the next fence.

    Raising here — on the one rank whose prefetch wedged — leaves its peers (which took the hit
    path) inside the collectives that follow, waiting out the whole NCCL watchdog while the only
    informative traceback belongs to a process that already exited.
    """
    host = _Host()
    host.async_config.episode_timeout = 0.01
    host._prefetch_inflight = 1
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(async_mod, "_PREFETCH_SUBMIT_TIMEOUT_S", 0.01)
        assert host._wait_for_inflight_prefetch() is None, "the caller must fall back to sync collection"

    assert "wedged" in (host._batch_build_error or ""), "the wedge must be recorded for the fence to raise"
    # The fence is what fails the job — and it fails on every rank, not just this one.
    with pytest.raises(ValueError, match="wedged"):
        host._raise_batch_error_uniformly(torch.device("cpu"))


def test_input_queue_full_skip_is_counted_not_silent():
    host = _Host()
    while True:
        try:
            host._prefetch_input_queue.put_nowait((["filler"], [None]))
        except queue.Full:
            break
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(async_mod, "_PREFETCH_SUBMIT_TIMEOUT_S", 0.01)
        host._submit_for_prefetch(["p"], [None])
    assert host._prefetch_input_skips == 1
    assert host._prefetch_inflight == 0


class _StubManager:
    max_concurrent = 4
    num_workers = 2

    async def start(self):
        return None

    def warn_if_servers_unreachable_from_actors(self, multinode: bool):
        return None


class _LifecycleHost:
    """Runs the REAL component-init / generation-start flow with Ray, the manager and the push stubbed."""

    _init_async_components = _T._init_async_components
    _start_rollout_generation = _T._start_rollout_generation

    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.accelerator = types.SimpleNamespace(is_main_process=True, num_processes=1)
        self.async_config = types.SimpleNamespace(
            ray_address=None,
            num_rollout_workers=2,
            max_concurrent_rollouts=2,
            get_server_urls=lambda: ["http://10.0.0.1:8000"],
            get_rollout_config=lambda stop_token_ids=None: {},
        )
        self.state = types.SimpleNamespace(global_step=7)
        self._environment_spec = "env"
        self._env_config_dict = {}
        self._rollout_manager = None
        self._weight_sync_client = None
        self._loop = None
        self._num_rollout_servers = 1
        self._prefetch_enabled = True
        self._rollout_start_callback = None
        self._rollout_generation_started = False
        self._last_sync_attempt_step = -1
        self.callbacks: list = []
        # What the engines were handed, in order — the fingerprint of WHICH weights they serve.
        self.pushed: list[float] = []
        self.prefetch_starts = 0

    def _resolve_rollout_stop_token_ids(self):
        return None

    def _init_weight_sync_client(self):
        self._weight_sync_client = object()

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def _sync_weights_to_engine_fenced(self, force: bool = False) -> bool:
        self.pushed.append(float(self.weight.sum()))
        return True

    def _start_prefetch_thread(self):
        self.prefetch_starts += 1


def test_generation_starts_only_after_the_resume_restore(monkeypatch):
    """The first push and the prefetch thread must follow the checkpoint restore.

    ``_init_async_components`` runs before ``super().train()``, i.e. before the trainer restores
    ``resume_from_checkpoint``. A push issued there ships the PRE-restore weights — on a resumed LoRA
    run, the zero-init adapter merged into the base — and every rollout until the next scheduled sync
    is drawn from a policy the trainer is not training.
    """
    weight = torch.zeros(4)  # a freshly initialized adapter
    host = _LifecycleHost(weight)
    monkeypatch.setattr(async_mod, "ray", types.SimpleNamespace(is_initialized=lambda: True, init=lambda **kw: None))
    monkeypatch.setattr(async_mod, "RolloutManager", lambda **kwargs: _StubManager())
    monkeypatch.setattr(async_mod, "broadcast_from_rank0", lambda value: value)

    try:
        host._init_async_components()

        assert host.pushed == [], "the engines were fed pre-restore weights"
        assert host.prefetch_starts == 0, "the prefetch thread rolled out pre-restore weights"
        assert len(host.callbacks) == 1, "nothing will start generation after the restore"

        # The restore lands here, between component init and the training loop's train-begin.
        with torch.no_grad():
            weight.copy_(torch.full_like(weight, 0.5))
        host.callbacks[0].on_train_begin(None, None, None)

        assert host.pushed == [2.0], f"the engines must serve the RESTORED weights: {host.pushed}"
        assert host.prefetch_starts == 1
        assert host._last_sync_attempt_step == 7, (
            "the resumed step must count as attempted, else the first training step re-pushes the "
            "weights this push just sent"
        )

        # Idempotent: a second train-begin (a re-entered loop) must not push a second time.
        host.callbacks[0].on_train_begin(None, None, None)
        assert host.pushed == [2.0]
    finally:
        host._loop.close()


def _async_state(server_configs):
    """Run the REAL ``_init_async_state`` over a config carrying ``server_configs``."""
    host = types.SimpleNamespace(
        async_config=AsyncTrainingConfig(rollout_server_configs=server_configs, enable_prefetch=True)
    )
    _T._init_async_state(host)
    return host


@pytest.mark.parametrize("server_configs", [None, [{"url": "http://s0:8000"}]])
def test_prefetch_is_disabled_whenever_a_single_engine_serves(server_configs):
    """One engine — however it is spelled — must turn prefetch off.

    A one-entry ``rollout_server_configs`` is a single server dressed as a list: it stops serving for
    the whole weight sync, so there is nothing to overlap against, and the rolling sync that prefetch
    selects would leave ZERO servers live instead of N-1 while the prefetch thread keeps posting
    rollouts into the paused engine.
    """
    host = _async_state(server_configs)
    assert host._num_rollout_servers == 1
    assert host._prefetch_enabled is False


def test_prefetch_stays_on_for_two_engines_and_the_client_shape_follows_the_list():
    host = _async_state([{"url": "http://s0:8000"}, {"url": "http://s1:8000"}])
    assert host._prefetch_enabled is True
    # The manager (not the single-client branch) owns any configs list, one entry included: only it
    # reads the per-entry url/group_port that list carries.
    assert host._multi_server_mode is True
    assert _async_state([{"url": "http://s0:8000"}])._multi_server_mode is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
