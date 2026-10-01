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
  per-rank, so a lone raise between two collectives parks every peer until the NCCL watchdog;
- a checkpoint carries the rounds submitted but not trained, and a resume submits them first, so a
  resumed run trains the same batch sequence as an uninterrupted one.

    python tests/cpu/grpo/test_grpo_prefetch_exactly_once.py
"""

import queue
import threading
import types
from collections import deque

import pytest
import torch

import src.trainers.grpo.rollout.async_rollouts as async_mod
from src.configs.async_training_config import AsyncTrainingConfig
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer as _T
from src.trainers.grpo.rollout.async_rollouts import AsyncRolloutMixin
from src.trainers.mixins.checkpointing import CheckpointingMixin


class _Host(AsyncRolloutMixin, CheckpointingMixin):
    """Minimal stand-in running the REAL prefetch decision flow with mocked rollout collection; the
    checkpointing base gives the sidecar hooks the mixin's ``super()`` calls end in."""

    _generate_and_score_completions_base = _T._generate_and_score_completions_base
    _extract_prompts_and_contexts = _T._extract_prompts_and_contexts
    _raise_batch_error_uniformly = _T._raise_batch_error_uniformly

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
        self._prefetch_pending = deque()
        self._resumed_prefetch_rounds = []
        self._prefetch_input_skips = 0
        self._rollout_generation_started = False
        self._last_sync_attempt_step = -1
        self.num_generations = 1
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

    def _sync_weights_to_engine_fenced(self, force: bool = False) -> bool:
        return True

    def _start_prefetch_thread(self):
        pass

    def _train_loader_batch_size(self) -> int:
        return 1

    def get_data_parallel_rank(self) -> int:
        return 0

    def get_data_parallel_size(self) -> int:
        return 1


def _pending(*prompts: str) -> deque:
    return deque(([p], [None]) for p in prompts)


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
    host._prefetch_pending = _pending("pW", "pX")
    host._prefetch_queue.put((1, None))  # worker failure marker
    host._prefetch_queue.put((1, ["prefetch:pX"]))

    assert host._try_get_prefetched_results() is None
    assert host._prefetch_inflight == 1
    assert host._wait_for_inflight_prefetch() == ["prefetch:pX"]
    assert host._prefetch_inflight == 0


def test_all_failed_inflight_returns_none_for_sync_fallback():
    host = _Host()
    host._prefetch_pending = _pending("pX")
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
    host._prefetch_pending = _pending("pX")
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
    _check_eval_round_fits_cap = _T._check_eval_round_fits_cap
    _start_rollout_generation = _T._start_rollout_generation

    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.accelerator = types.SimpleNamespace(is_main_process=True, num_processes=1)
        self.async_config = types.SimpleNamespace(
            ray_address=None,
            num_rollout_workers=2,
            max_concurrent_rollouts=2,
            eval_rollout_batch_size=None,
            get_server_urls=lambda: ["http://10.0.0.1:8000"],
            get_rollout_config=lambda stop_token_ids=None, reasoning_end_token_id=None: {},
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
        self._resumed_prefetch_rounds = []
        self.callbacks: list = []
        # What the engines were handed, in order — the fingerprint of WHICH weights they serve.
        self.pushed: list[float] = []
        self.prefetch_starts = 0

    def _resolve_rollout_stop_token_ids(self):
        return None

    def _resolve_reasoning_end_token_id(self):
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


def _trained_batches(host: _Host) -> list[str]:
    """The prompt each trained round's rollouts were drawn for, whichever path collected them."""
    return [batch[0].split(":", 1)[1] for batch in host.trained]


def _run(host: _Host, batches: list[str]) -> None:
    """One training round per batch, the worker finishing each submission before the next round."""
    for prompt in batches:
        if not host._prefetch_input_queue.empty():
            host.worker_step()
        host.round([prompt])


def test_a_resumed_run_trains_the_batches_an_uninterrupted_one_does(tmp_path):
    """Each round trains the batch submitted one round earlier, so at a save on a generation boundary the
    last submitted batch has left the dataloader untrained. A resume that does not carry it skips that batch and trains
    the next one twice (its cold round collects and submits the same prompts)."""
    batches = [f"b{i}" for i in range(6)]
    uninterrupted = _Host()
    _run(uninterrupted, batches)

    before = _Host()
    _run(before, batches[:3])
    before._persist_trainer_sidecars(str(tmp_path))
    resumed = _Host()
    resumed._restore_trainer_sidecars(str(tmp_path))
    resumed._start_rollout_generation()
    _run(resumed, batches[3:])

    assert _trained_batches(before) + _trained_batches(resumed) == _trained_batches(uninterrupted)
    assert resumed.sync_calls == [], "the resumed run must not open with a cold round"


def test_stopping_the_worker_drops_the_rounds_no_later_step_consumes():
    """A second ``train()`` restarts the worker over the same queues: an output delivered before the stop
    would be popped against that run's own pending rounds."""
    host = _Host()
    host.round(["b0"])
    host.worker_step()  # b0's rollouts are delivered, unconsumed
    host._prefetch_stop_event = threading.Event()
    host._prefetch_thread = threading.Thread(target=lambda: None)
    host._prefetch_thread.start()
    host._stop_prefetch_thread()
    assert host._prefetch_inflight == 0 and host._prefetch_queue.empty()
    host.round(["c0"])
    assert host.trained[-1] == ["sync:c0"], "the next run opens with a cold round of its own batch"


def test_rounds_drawn_under_another_layout_are_not_resubmitted(tmp_path):
    """A stage change from 8 to 12 generations regroups a round's rows across prompts: replayed, its
    baseline would mix problems for a step. The resume drops them and opens cold instead."""
    before = _Host()
    _run(before, ["b0", "b1", "b2"])
    before._persist_trainer_sidecars(str(tmp_path))
    resumed = _Host()
    resumed.num_generations = 2
    resumed._restore_trainer_sidecars(str(tmp_path))
    assert resumed._resumed_prefetch_rounds == []


def test_a_context_only_ray_can_pickle_survives_the_save(tmp_path):
    """A row's context may carry a callable (an answer ``validator``), which Ray ships with cloudpickle and
    the stdlib pickler refuses: a stdlib save would fail every checkpoint of such a run."""
    host = _Host()
    host._prefetch_pending = deque([(["b2"], [{"validator": lambda answer: answer == "42"}])])
    host._persist_trainer_sidecars(str(tmp_path))
    resumed = _Host()
    resumed._restore_trainer_sidecars(str(tmp_path))
    ((prompts, (context,)),) = resumed._resumed_prefetch_rounds
    assert prompts == ["b2"] and context["validator"]("42")


def test_a_failed_save_leaves_no_file_a_resume_would_read(tmp_path):
    """A torn file in a checkpoint that already holds its trainer state fails every later resume from it."""
    host = _Host()
    host._prefetch_pending = deque([(["b2"], [{"lock": threading.Lock()}])])
    with pytest.raises(RuntimeError, match="cannot pickle"):
        host._persist_trainer_sidecars(str(tmp_path))
    assert list(tmp_path.iterdir()) == []


class _LoadHost(CheckpointingMixin):
    """The mixin's resume entry with the weight loader and the bias restore stubbed out."""

    def __init__(self):
        self.restored: list[str] = []

    def _checkpoint_loader(self):
        return types.SimpleNamespace(load_model=lambda checkpoint, model, for_best_model: None)

    def _restore_router_balancing_biases(self, checkpoint):
        pass

    def _restore_trainer_sidecars(self, checkpoint):
        self.restored.append(checkpoint)


def test_a_resume_restores_the_trainer_sidecars_and_a_best_model_load_does_not():
    host = _LoadHost()
    host._load_from_checkpoint("checkpoint-50")
    host._load_from_checkpoint("checkpoint-25", for_best_model=True)
    assert host.restored == ["checkpoint-50"], "a best-model load continues no training"


def test_the_environmental_trainer_takes_the_rollout_mixins_sidecar_hooks():
    """Listed after ``DistributedTrainerMixin``, the mixin's hooks would be shadowed by the empty defaults
    and no checkpoint would carry the pending rounds."""
    for hook in ("_persist_trainer_sidecars", "_restore_trainer_sidecars"):
        assert getattr(_T, hook) is getattr(AsyncRolloutMixin, hook)


def test_a_checkpoint_without_pending_rounds_resumes_cold(tmp_path):
    host = _Host()
    host._restore_trainer_sidecars(str(tmp_path))
    host._start_rollout_generation()
    _run(host, ["b3", "b4"])
    assert _trained_batches(host) == ["b3", "b3"], "the cold round primes the lag with its own batch"


def test_a_run_without_prefetch_writes_and_takes_no_pending_rounds(tmp_path):
    host = _Host()
    host._prefetch_enabled = False
    host._prefetch_pending = _pending("b2")
    host._persist_trainer_sidecars(str(tmp_path))
    assert list(tmp_path.iterdir()) == []

    saved = _Host()
    saved._prefetch_pending = _pending("b2")
    saved._persist_trainer_sidecars(str(tmp_path))
    host._restore_trainer_sidecars(str(tmp_path))
    assert host._resumed_prefetch_rounds == []


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
    raise SystemExit(pytest.main([__file__, "-v"]))
