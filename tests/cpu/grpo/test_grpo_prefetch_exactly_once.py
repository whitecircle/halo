#!/usr/bin/env python
"""Env-GRPO prefetch pipeline: exactly-once training on misses and loud drop accounting. A miss that sync-collects and
re-submits the same prompts would train a batch twice; a full queue must not silently drop a batch or its rollouts."""

import asyncio
import logging
import os
import queue
import stat
import threading
import time
import types
from collections import deque

import pytest
import torch

import halo.runtime.consensus as consensus_mod
import halo.trainers.checkpointing as checkpointing_mod
import halo.trainers.grpo.environmental.async_rollouts as async_mod
import halo.trainers.parallelism.tp_leader as tp_leader_mod
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.trainers.checkpointing import CheckpointingMixin
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.environmental.async_rollouts import AsyncRolloutMixin
from halo.trainers.grpo.environmental.step_batch import BatchBuildFence
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer as _T
from halo.trainers.grpo.phase_timing import PhaseTimedMixin
from tests.common.repo import load_script_module


class _Rollout(str):
    trajectory = None


env_grpo_script = load_script_module("scripts/training/grpo/environmental.py")

_TWO_ENGINES = [{"url": "http://s0:8000"}, {"url": "http://s1:8000"}]


class _Host(AsyncRolloutMixin, CheckpointingMixin):
    """The real prefetch decision flow with mocked rollout collection, over the real ``_init_async_state`` for a
    two-server run at ``depth``."""

    _collect_and_score_rollouts = _T._collect_and_score_rollouts
    _phase_timers = PhaseTimedMixin._phase_timers
    _tool_call_watch = None
    _tokenizer = None  # read only for thinking text, which the stub rollouts carry none of
    _extract_prompts_and_contexts = _T._extract_prompts_and_contexts
    eval_split_rows = _T.eval_split_rows

    def __init__(self, depth: int = 1, wire: dict | None = None, follower: bool = False):
        self.follower = follower
        # The leader's broadcast payload, shared with its follower; ``None`` is a single-rank host.
        self.wire = wire
        self.reject_calls = 0
        self.thread_starts = 0
        self.async_config = AsyncRolloutConfig(
            rollout_server_configs=_TWO_ENGINES, prefetch_depth=depth, episode_timeout=5.0
        )
        self._init_async_state()
        self.model = types.SimpleNamespace(training=True)
        self.accelerator = types.SimpleNamespace(device=torch.device("cpu"), is_main_process=False)
        self.parallelism_config = types.SimpleNamespace(data_parallel_size=1, get_data_parallel_rank=lambda: 0)
        # profiling_context reads these on the acquire path; report_to=[] makes it a no-op.
        self.state = types.SimpleNamespace(global_step=0)
        self.args = types.SimpleNamespace(report_to=[])
        self._group_random_effort = False
        self._batch_errors = BatchBuildFence()
        self._batch_errors.reject = self._counted_reject
        self.num_generations = 1
        # Every sync collection, the duplication fingerprint.
        self.sync_calls: list[list[str]] = []
        self.trained: list[list[str]] = []
        self._loop = types.SimpleNamespace(run_until_complete=lambda batch: batch)
        self._rollout_manager = types.SimpleNamespace(collect_rollouts=self._collect)

    def _collect(self, prompts, contexts):
        self.sync_calls.append(list(prompts))
        return [_Rollout(f"sync:{p}") for p in prompts]

    def _counted_reject(self):
        self.reject_calls += 1
        BatchBuildFence.reject(self._batch_errors)

    def _is_tp_follower(self) -> bool:
        return self.follower

    def _broadcast_object_from_tp_leader(self, rollout_results):
        if self.wire is None:
            return rollout_results
        if self.follower:
            assert rollout_results is None, "a follower has no rollouts of its own to offer"
            return self.wire["rollouts"]
        self.wire["rollouts"] = rollout_results
        return rollout_results

    def _build_training_tensors(self, rollout_results, device, mode, num_padding, reasoning_tokens):
        assert num_padding == 0, "a train round is never padded"
        self.trained.append(list(rollout_results))
        return {"rollouts": rollout_results}

    def _log_rollout_metrics(self, results, mode, reasoning_tokens):
        pass

    def round(self, prompts: list[str]):
        return self._collect_and_score_rollouts([{"prompt": p} for p in prompts])

    def worker_step(self):
        """A synchronous stand-in for the prefetch worker."""
        prompts, _contexts = self._prefetch_input_queue.get_nowait()
        self._prefetch_queue.put_nowait((len(prompts), [_Rollout(f"prefetch:{p}") for p in prompts]))

    def _sync_weights_to_engine_fenced(self, force: bool = False) -> bool:
        return True

    def _start_prefetch_thread(self):
        self.thread_starts += 1

    def _train_loader_batch_size(self) -> int:
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

    # p1 is still in flight, so this round must block for it; sync-collecting p2 here would train p1's batch again on
    # the next hit.
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
    host = _Host(depth=2)
    host._prefetch_pending = _pending("pW", "pX")
    host._prefetch_queue.put_nowait((1, None))  # worker failure marker
    host._prefetch_queue.put_nowait((1, [_Rollout("prefetch:pX")]))

    assert host._try_get_prefetched_results() is None
    assert host._prefetch_inflight == 1
    assert host._wait_for_inflight_prefetch() == ["prefetch:pX"]
    assert host._prefetch_inflight == 0


def test_all_failed_inflight_returns_none_for_sync_fallback():
    host = _Host()
    host._prefetch_pending = _pending("pX")
    host._prefetch_queue.put_nowait((1, None))
    assert host._wait_for_inflight_prefetch() is None
    assert host._prefetch_inflight == 0


def test_wedged_pipeline_is_recorded_for_the_uniform_fence_not_raised():
    """A lone raise on the wedged rank leaves its peers, which took the hit path, in the collectives that follow until
    the NCCL watchdog."""
    host = _Host()
    host.async_config.episode_timeout = 0.01
    host._prefetch_pending = _pending("pX")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(async_mod, "_PREFETCH_SUBMIT_TIMEOUT_S", 0.01)
        assert host._wait_for_inflight_prefetch() is None, "the caller must fall back to sync collection"

    assert "wedged" in (host._batch_errors.reason or ""), "the wedge must be recorded for the fence to raise"
    # The fence fails the job, on every rank.
    with pytest.raises(ValueError, match="wedged"):
        host._batch_errors.reject()


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

    def reject_loopback_servers(self, num_nodes: int):
        return None


class _LifecycleHost:
    """The real component-init and generation-start flow with Ray, the manager and the push stubbed."""

    _init_async_components = _T._init_async_components
    _form_weight_sync_group = _T._form_weight_sync_group
    _weight_sync_group_formed = False
    _start_rank_rollout_components = _T._start_rank_rollout_components
    _start_rollout_pool = _T._start_rollout_pool
    _resolve_rollout_pool = _T._resolve_rollout_pool
    _get_tp_or_etp_group_size = _T._get_tp_or_etp_group_size
    _reject_eval_round_over_cap = _T._reject_eval_round_over_cap
    _rollout_node_count = _T._rollout_node_count
    _start_rollout_generation = _T._start_rollout_generation

    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.accelerator = types.SimpleNamespace(is_main_process=True, num_processes=1)
        self.parallelism_config = types.SimpleNamespace(get_data_parallel_rank=lambda: 0)
        self.async_config = types.SimpleNamespace(
            ray_address=None,
            num_rollout_workers=2,
            max_concurrent_rollouts=2,
            eval_rollout_batch_size=None,
            isr_engine_reference=False,
            get_server_urls=lambda: ["http://10.0.0.1:8000"],
            get_rollout_config=lambda stop_token_ids=None, reasoning_end_token_id=None: {},
        )
        self.state = types.SimpleNamespace(global_step=7)
        self._environment_spec = "env"
        self._env_config_dict = {}
        self._rollout_manager = None
        self._weight_sync_client = None
        self._loop = None
        self._prefetch_enabled = True
        self._rollout_start_callback = None
        self._rollout_generation_started = False
        self._last_sync_attempt_step = -1
        self._resumed_prefetch_rounds = []
        self._rollout_stop_token_ids = None
        self._reasoning_end_ids = None
        self.callbacks: list = []
        # The TP/ETP group this rank sits in: its size, and whether it leads.
        self.tp_group_size = 1
        self.follower = False
        # What the engines were handed, in order: the fingerprint of which weights they serve.
        self.pushed: list[float] = []
        self.prefetch_starts = 0

    def _init_weight_sync_client(self):
        self._weight_sync_client = object()

    def _train_loader_batch_size(self) -> int:
        return 4

    def _is_tp_follower(self) -> bool:
        return self.follower

    def _get_tp_or_etp_process_group(self):
        return types.SimpleNamespace(size=self.tp_group_size) if self.tp_group_size > 1 else None

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def _sync_weights_to_engine_fenced(self, force: bool = False) -> bool:
        self.pushed.append(float(self.weight.sum()))
        return True

    def _start_prefetch_thread(self):
        self.prefetch_starts += 1


def test_a_group_formation_failure_is_joined_over_the_store_not_an_nccl_collective(monkeypatch):
    """Rank 0 forms the groups server by server while the peers wait, for longer than the NCCL watchdog allows."""
    host = _LifecycleHost(torch.zeros(1))
    joined: list[tuple[str, str | None]] = []

    def failing_client() -> None:
        raise RuntimeError("server 3 never loaded")

    def record_join(tag, reason, what, exc_type=RuntimeError, timeout=None):
        joined.append((tag, reason))
        raise exc_type(reason)

    host._init_weight_sync_client = failing_client
    monkeypatch.setattr(consensus_mod, "store_reject_across_ranks", record_join)
    with pytest.raises(RuntimeError, match="server 3 never loaded"):
        host._form_weight_sync_group()
    assert joined == [("rollout/weight_sync_group", "RuntimeError: server 3 never loaded")]


def test_generation_starts_only_after_the_resume_restore(monkeypatch):
    """``_init_async_components`` runs before the trainer restores ``resume_from_checkpoint``. A push there ships the
    pre-restore weights, on a resumed LoRA run the zero-init adapter merged into the base."""
    weight = torch.zeros(4)  # a freshly initialized adapter
    host = _LifecycleHost(weight)
    monkeypatch.setattr(async_mod, "ray", types.SimpleNamespace(is_initialized=lambda: True, init=lambda **kw: None))
    monkeypatch.setattr(async_mod, "RolloutManager", lambda **kwargs: _StubManager())

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
    return [batch[0].split(":", 1)[1] for batch in host.trained]


def _run(host: _Host, batches: list[str]) -> None:
    for prompt in batches:
        if not host._prefetch_input_queue.empty():
            host.worker_step()
        host.round([prompt])


def _tp_pair(depth: int = 1) -> tuple[_Host, _Host]:
    wire: dict = {}
    return _Host(depth, wire=wire), _Host(depth, wire=wire, follower=True)


def _run_pair(leader: _Host, follower: _Host, batches: list[str]) -> None:
    for prompt in batches:
        if not leader._prefetch_input_queue.empty():
            leader.worker_step()
        leader.round([prompt])
        follower.round([prompt])


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_a_tp_follower_issues_no_rollout_request_and_trains_the_leaders_rows(depth):
    """Every follower used to collect a full batch the leader's replaced, so engine and Ray load scaled with the world
    size, not the data-parallel size."""
    leader, follower = _tp_pair(depth)
    _run_pair(leader, follower, [f"b{i}" for i in range(6)])

    assert follower.trained == leader.trained and len(leader.trained) == 6
    assert follower.sync_calls == [], "a follower sent a rollout request"
    assert follower._prefetch_inflight == 0 and follower._prefetch_input_queue.empty(), "a follower submitted a round"
    assert (follower._prefetch_hits, follower._prefetch_misses) == (0, 0)
    assert leader.sync_calls, "the leader must still roll out"
    # The fences are collective: a follower that skipped one would leave the leader waiting in it.
    assert follower.reject_calls == leader.reject_calls == 12


@pytest.mark.parametrize("depth", [2, 3])
def test_a_resume_under_tp_replays_the_leaders_pending_rounds_and_the_follower_none(tmp_path, monkeypatch, depth):
    """Each rank has its sidecar file, or the resume's presence check raises; a follower's holds no rounds, so it
    submits nothing, and the leader's replay trains what an uninterrupted run does."""
    batches = [f"b{i}" for i in range(depth + 4)]
    uninterrupted, _ = _tp_pair(depth)
    _run(uninterrupted, batches)

    saved_at = depth + 1
    before, before_follower = _tp_pair(depth)
    _run_pair(before, before_follower, batches[:saved_at])
    for rank, host in enumerate((before, before_follower)):
        monkeypatch.setattr(async_mod, "get_global_rank", lambda rank=rank: rank)
        monkeypatch.setattr(async_mod, "get_global_world_size", lambda: 2)
        host._persist_trainer_sidecars(str(tmp_path))
    saved = {path.name: torch.load(path, weights_only=False)["rounds"] for path in tmp_path.glob("*prefetch*")}
    assert sorted(len(rounds) for rounds in saved.values()) == [0, depth], (
        "every rank writes, only the leader's holds rounds"
    )

    resumed, resumed_follower = _tp_pair(depth)
    for rank, host in enumerate((resumed, resumed_follower)):
        monkeypatch.setattr(async_mod, "get_global_rank", lambda rank=rank: rank)
        host._restore_trainer_sidecars(str(tmp_path))
    assert [prompts for prompts, _ in resumed._resumed_prefetch_rounds] == [
        [f"b{k}"] for k in range(saved_at - depth, saved_at)
    ]
    assert resumed_follower._resumed_prefetch_rounds == []

    resumed._start_rollout_generation()
    resumed_follower._start_rollout_generation()
    assert (resumed.thread_starts, resumed_follower.thread_starts) == (1, 0)
    assert resumed_follower._prefetch_input_queue.empty() and resumed_follower._prefetch_inflight == 0

    _run_pair(resumed, resumed_follower, batches[saved_at:])
    assert _trained_batches(before) + _trained_batches(resumed) == _trained_batches(uninterrupted)
    assert resumed_follower.trained == resumed.trained
    assert resumed.sync_calls == [] and resumed_follower.sync_calls == []


def test_a_follower_restores_no_rounds_from_a_checkpoint_that_gave_it_some(tmp_path):
    donor = _Host(2)
    _run(donor, ["b0", "b1", "b2"])
    donor._persist_trainer_sidecars(str(tmp_path))
    follower = _Host(2, follower=True)
    follower._restore_trainer_sidecars(str(tmp_path))
    assert follower._resumed_prefetch_rounds == []


def test_a_tp_follower_starts_no_ray_and_no_pool(monkeypatch):
    host = _LifecycleHost(torch.zeros(1))
    host.follower = True
    host.tp_group_size = 4
    ray_calls = []
    monkeypatch.setattr(
        async_mod, "ray", types.SimpleNamespace(is_initialized=lambda: False, init=lambda **kw: ray_calls.append(kw))
    )

    def refuse(**kwargs):
        raise AssertionError("a follower built a rollout manager")

    monkeypatch.setattr(async_mod, "RolloutManager", refuse)
    host._init_async_components()
    host.callbacks[0].on_train_begin(None, None, None)

    assert ray_calls == [] and host._rollout_manager is None and host._loop is None
    assert host.prefetch_starts == 0, "a follower started a prefetch worker"
    assert host.pushed == [0.0], "a follower must still join the weight push"


def test_a_tp_follower_still_builds_the_score_only_clients(monkeypatch):
    host = _LifecycleHost(torch.zeros(1))
    host.follower = True
    host.async_config.isr_engine_reference = True
    host._build_engine_rescore_clients = lambda: ["client"]
    monkeypatch.setattr(async_mod, "ray", types.SimpleNamespace(is_initialized=lambda: True))
    host._start_rank_rollout_components()
    assert host._engine_rescore_clients == ["client"]


@pytest.mark.parametrize(("group_size", "actors"), [(1, 8), (2, 16), (8, 64)])
def test_a_shared_ray_cluster_splits_its_actor_budget_among_the_leaders(monkeypatch, group_size, actors):
    """Followers run no pool, so the budget a shared cluster divides is over the groups, else the cluster-wide pool
    would fall to ``group_size`` times less than ``num_rollout_workers``."""
    host = _LifecycleHost(torch.zeros(1))
    host.tp_group_size = group_size
    host.async_config.ray_address = "auto"
    host.async_config.num_rollout_workers = 64
    built = {}

    def manager(**kwargs):
        built.update(kwargs)
        return _StubManager()

    monkeypatch.setattr(async_mod, "ray", types.SimpleNamespace(is_initialized=lambda: True, nodes=lambda: []))
    monkeypatch.setattr(async_mod, "RolloutManager", manager)
    monkeypatch.setattr(async_mod, "get_global_world_size", lambda: 8)
    monkeypatch.setattr(async_mod, "get_local_world_size", lambda: 8)
    monkeypatch.setattr(tp_leader_mod.dist, "get_world_size", lambda group: group.size)
    try:
        host._init_async_components()
    finally:
        host._loop.close()
    assert built["num_workers"] == actors


def _expected_order(depth: int, steps: int) -> list[str]:
    return [f"b{k}" if k < depth else f"b{k - depth}" for k in range(steps)]


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_each_round_trains_the_batch_submitted_depth_rounds_earlier(depth):
    host = _Host(depth)
    inflight = []
    for prompt in [f"b{i}" for i in range(8)]:
        if not host._prefetch_input_queue.empty():
            host.worker_step()
        host.round([prompt])
        inflight.append(host._prefetch_inflight)

    assert _trained_batches(host) == _expected_order(depth, 8)
    assert host.sync_calls == [[f"b{k}"] for k in range(depth)], "only the opening rounds collect synchronously"
    assert inflight == [min(k + 1, depth) for k in range(8)], "the pipeline must fill to its depth and stay there"
    assert (host._prefetch_hits, host._prefetch_misses) == (8 - depth, depth)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_a_resumed_run_trains_the_batches_an_uninterrupted_one_does(tmp_path, depth):
    """At a save on a generation boundary the last ``depth`` submitted batches have left the dataloader untrained. A
    resume that does not carry them all, oldest first, skips batches and trains others twice."""
    batches = [f"b{i}" for i in range(depth + 4)]
    uninterrupted = _Host(depth)
    _run(uninterrupted, batches)

    saved_at = depth + 1
    before = _Host(depth)
    _run(before, batches[:saved_at])
    before._persist_trainer_sidecars(str(tmp_path))
    resumed = _Host(depth)
    resumed._restore_trainer_sidecars(str(tmp_path))
    assert [prompts for prompts, _ in resumed._resumed_prefetch_rounds] == [
        [f"b{k}"] for k in range(saved_at - depth, saved_at)
    ], "the resume must carry every pending round, oldest first"
    resumed._start_rollout_generation()
    _run(resumed, batches[saved_at:])

    assert _trained_batches(uninterrupted) == _expected_order(depth, len(batches))
    assert _trained_batches(before) + _trained_batches(resumed) == _trained_batches(uninterrupted)
    assert resumed.sync_calls == [], "the resumed run must not open with a cold round"


def test_a_resume_under_a_smaller_depth_resubmits_the_oldest_rounds_and_says_so(tmp_path, monkeypatch, caplog):
    """Each step trains one round and draws one, so a pipeline handed more rounds than its depth would hold
    them all for the rest of the run, past the queues sized for it."""
    before = _Host(3)
    _run(before, ["b0", "b1", "b2", "b3"])
    before._persist_trainer_sidecars(str(tmp_path))
    monkeypatch.setattr(async_mod, "is_global_main_process", lambda: True)
    resumed = _Host(2)
    with caplog.at_level(logging.WARNING, logger=async_mod.logger.name):
        resumed._restore_trainer_sidecars(str(tmp_path))
    assert [prompts for prompts, _ in resumed._resumed_prefetch_rounds] == [["b1"], ["b2"]]
    assert "the newest 1 are not trained" in caplog.text


def test_a_resume_under_a_larger_depth_opens_with_cold_rounds_until_full(tmp_path):
    """Resumed at depth 2, a depth-1 checkpoint is one round short, so the first step collects its own batch and
    submits it too: that batch trains twice and none is skipped."""
    before = _Host(1)
    _run(before, ["b0", "b1", "b2"])
    before._persist_trainer_sidecars(str(tmp_path))
    resumed = _Host(2)
    resumed._restore_trainer_sidecars(str(tmp_path))
    assert [prompts for prompts, _ in resumed._resumed_prefetch_rounds] == [["b2"]]
    resumed._start_rollout_generation()
    _run(resumed, ["b3", "b4", "b5", "b6"])

    assert _trained_batches(before) == ["b0", "b0", "b1"]
    assert _trained_batches(resumed) == ["b3", "b2", "b3", "b4"]
    assert resumed.sync_calls == [["b3"]], "only the one step the deeper pipeline lacks collects synchronously"


def test_a_failed_oldest_round_leaves_the_step_its_own_prompts_and_the_pipeline_at_depth():
    """Waiting on the next round instead would train it a step early and leave the pipeline a round short,
    which the following step refills with a whole synchronous round: two blocking waits for one failure."""
    host = _Host(2)
    host.round(["b0"])
    host.round(["b1"])
    host._prefetch_input_queue.get_nowait()
    host._prefetch_queue.put_nowait((1, None))  # b0's collection raised
    host.worker_step()  # b1 is delivered
    host.round(["b2"])
    assert host.trained[-1] == ["sync:b2"] and host._prefetch_inflight == 2
    host.worker_step()  # b2 is delivered
    host.round(["b3"])
    assert _trained_batches(host) == ["b0", "b1", "b2", "b1"]
    assert host.sync_calls == [["b0"], ["b1"], ["b2"]]


@pytest.mark.usefixtures("real_worker_waits")
def test_a_failure_delivered_while_the_step_waits_leaves_the_younger_round_queued():
    """The failure marker lands during the step's wait, which returns it and not the younger round right behind it."""
    host = _Host(2)
    host.round(["b0"])
    host.round(["b1"])
    host._prefetch_input_queue.get_nowait()
    host._prefetch_input_queue.get_nowait()

    def deliver():
        host._prefetch_queue.put_nowait((1, None))  # b0's collection raised
        host._prefetch_queue.put_nowait((1, [_Rollout("prefetch:b1")]))

    timer = threading.Timer(0.2, deliver)
    timer.start()
    try:
        host.round(["b2"])
    finally:
        timer.join()
    assert host.trained[-1] == ["sync:b2"] and host._prefetch_inflight == 2


class _ConcurrentManager:
    """Rounds finish only when the test releases them, the ``failing`` ones by raising, ``cancelled_inside`` ones by a
    cancel from inside; records which rounds collected at once and which the stop cancelled."""

    def __init__(self, rounds: list[str], failing: frozenset[str] = frozenset(), cancelled_inside=frozenset()):
        self.release = {name: threading.Event() for name in rounds}
        self.failing = failing
        self.cancelled_inside = cancelled_inside
        self.collecting: set[str] = set()
        self.most_at_once = 0
        self.cancelled: set[str] = set()

    async def collect_rollouts(self, prompts, contexts):
        (name,) = prompts
        self.collecting.add(name)
        self.most_at_once = max(self.most_at_once, len(self.collecting))
        try:
            while not self.release[name].is_set():
                await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            self.cancelled.add(name)
            raise
        finally:
            self.collecting.discard(name)
        if name in self.failing:
            raise RuntimeError(f"{name}: every server refused the round")
        if name in self.cancelled_inside:
            raise asyncio.CancelledError
        return [_Rollout(f"prefetch:{name}")]


def _wait_until(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "the prefetch worker never reached the expected state"
        time.sleep(0.005)


@pytest.fixture
def real_worker_waits(monkeypatch):
    """A lost delivery then fails its wait in seconds rather than a minute."""
    monkeypatch.setattr(async_mod, "_PREFETCH_POLL_TIMEOUT_S", 0.01)
    monkeypatch.setattr(async_mod, "_PREFETCH_SUBMIT_TIMEOUT_S", 2.0)


def _start_real_worker(host: _Host, manager: _ConcurrentManager) -> None:
    host._rollout_manager = manager
    AsyncRolloutMixin._start_prefetch_thread(host)  # _Host stubs it out for the lifecycle tests


def _submit_slow_then_fast(host: _Host, manager: _ConcurrentManager) -> None:
    host._submit_for_prefetch(["r0"], [None])
    host._submit_for_prefetch(["r1"], [None])
    _wait_until(lambda: manager.collecting == {"r0", "r1"})
    manager.release["r1"].set()
    _wait_until(lambda: "r1" not in manager.collecting)
    time.sleep(0.05)
    assert host._prefetch_queue.empty(), "the later round was delivered ahead of the earlier one"


@pytest.mark.usefixtures("real_worker_waits")
def test_the_worker_collects_pending_rounds_together_and_delivers_them_in_submission_order():
    """One round at a time, the engines would drain at r0's tail with r1 waiting. r1 is still delivered after r0, since
    the trainer pairs each output with its oldest pending round."""
    host = _Host(2)
    manager = _ConcurrentManager(["r0", "r1"])
    _start_real_worker(host, manager)
    try:
        _submit_slow_then_fast(host, manager)
        manager.release["r0"].set()
        assert host._wait_for_inflight_prefetch() == ["prefetch:r0"]
        assert host._wait_for_inflight_prefetch() == ["prefetch:r1"]
        assert host._prefetch_inflight == 0
    finally:
        for gate in manager.release.values():
            gate.set()
        host._stop_prefetch_thread()


@pytest.mark.usefixtures("real_worker_waits")
@pytest.mark.parametrize(
    ("failure", "logged"),
    [
        ({"failing": frozenset({"r1"})}, "Prefetch worker error (1 prompts): r1: every server refused the round"),
        ({"cancelled_inside": frozenset({"r1"})}, "cancelled while the worker runs; marked failed"),
    ],
    ids=["raised", "cancelled"],
)
def test_a_failed_round_still_yields_one_output_in_its_place(caplog, failure, logged):
    """Delivered ahead of the slower round before it, the marker would pair with that round; not delivered at all,
    every later output would pair one round off."""
    host = _Host(2)
    manager = _ConcurrentManager(["r0", "r1"], **failure)
    _start_real_worker(host, manager)
    try:
        with caplog.at_level(logging.ERROR, logger=async_mod.logger.name):
            _submit_slow_then_fast(host, manager)
        manager.release["r0"].set()
        assert host._wait_for_inflight_prefetch() == ["prefetch:r0"]
        assert host._prefetch_queue.get(timeout=5.0) == (1, None)
        assert logged in caplog.text
    finally:
        for gate in manager.release.values():
            gate.set()
        host._stop_prefetch_thread()


@pytest.mark.usefixtures("real_worker_waits")
@pytest.mark.parametrize("depth", [1, 2])
def test_a_run_through_the_real_worker_trains_in_submission_order_at_most_depth_rounds_in_flight(depth):
    """Each round is released just before the step that trains it, once every younger pending round is collecting."""
    batches = [f"b{i}" for i in range(6)]
    host = _Host(depth)
    manager = _ConcurrentManager(batches)
    host._loop = asyncio.new_event_loop()

    async def collect(prompts, contexts):
        if threading.current_thread() is not host._prefetch_thread:
            host.sync_calls.append(list(prompts))
            return [_Rollout(f"sync:{p}") for p in prompts]
        return await manager.collect_rollouts(prompts, contexts)

    host._rollout_manager = types.SimpleNamespace(collect_rollouts=collect)
    AsyncRolloutMixin._start_prefetch_thread(host)
    try:
        for k, prompt in enumerate(batches):
            if k >= depth:
                _wait_until(lambda k=k: {f"b{j}" for j in range(k - depth, k)} <= manager.collecting)
                manager.release[f"b{k - depth}"].set()
            host.round([prompt])
        assert _trained_batches(host) == _expected_order(depth, len(batches))
        assert manager.most_at_once == depth
    finally:
        for gate in manager.release.values():
            gate.set()
        host._stop_prefetch_thread()
        host._loop.close()


@pytest.mark.usefixtures("real_worker_waits")
def test_stopping_the_worker_cancels_the_rounds_still_collecting(caplog):
    """Every pending round is dropped at the stop, so the worker exits within the join instead of waiting them out."""
    host = _Host(2)
    manager = _ConcurrentManager(["r0", "r1"])
    _start_real_worker(host, manager)
    host._submit_for_prefetch(["r0"], [None])
    host._submit_for_prefetch(["r1"], [None])
    _wait_until(lambda: manager.collecting == {"r0", "r1"})
    thread = host._prefetch_thread
    with caplog.at_level(logging.WARNING, logger=async_mod.logger.name):
        host._stop_prefetch_thread()
    assert not thread.is_alive() and "did not stop cleanly" not in caplog.text
    assert "marked failed" not in caplog.text, "a round cancelled by the stop was reported as a failure"
    assert manager.cancelled == {"r0", "r1"}
    assert host._prefetch_inflight == 0 and host._prefetch_queue.empty()


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
    """A stage change from 8 to 12 generations regroups a round's rows across prompts; replayed, its baseline would mix
    problems for a step."""
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


def test_the_pending_rounds_are_durable_before_the_checkpoint_counts_as_written(tmp_path, monkeypatch):
    """Synced before the rename and the directory after, like every other sidecar: unsynced, a crash
    can leave the checkpoint a renamed but empty file after older checkpoints have rotated away."""
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def recording_fsync(descriptor):
        events.append("directory_sync" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file_sync")
        real_fsync(descriptor)

    def recording_replace(source, destination):
        real_replace(source, destination)
        events.append("rename")

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "replace", recording_replace)
    host = _Host()
    host._prefetch_pending = _pending("b2")
    host._persist_trainer_sidecars(str(tmp_path))

    assert events == ["file_sync", "rename", "directory_sync"]


class _LoadHost(CheckpointingMixin):
    def __init__(self):
        self.restored: list[str] = []
        self.parallelism_config = types.SimpleNamespace(is_pp_mode=False)

    def _checkpoint_load_context(self):
        return None

    def _top_level_model(self):
        return None

    def _restore_trainer_sidecars(self, checkpoint):
        self.restored.append(checkpoint)


def test_a_resume_restores_the_trainer_sidecars_and_a_best_model_load_does_not(monkeypatch):
    monkeypatch.setattr(checkpointing_mod, "load_resume_weights", lambda *args, **kwargs: True)
    monkeypatch.setattr(checkpointing_mod, "restore_balancing_biases", lambda *args, **kwargs: None)
    host = _LoadHost()
    host._load_from_checkpoint("checkpoint-50")
    host._load_from_checkpoint("checkpoint-25", for_best_model=True)
    assert host.restored == ["checkpoint-50"], "a best-model load continues no training"


def test_the_environmental_trainer_takes_the_rollout_mixins_sidecar_hooks():
    """Listed after ``DistributedTrainerMixin``, the mixin's hooks would be shadowed by the empty defaults
    and no checkpoint would carry the pending rounds."""
    for hook in ("_persist_trainer_sidecars", "_restore_trainer_sidecars"):
        assert getattr(_T, hook) is getattr(AsyncRolloutMixin, hook)


def test_resumed_rounds_are_submitted_after_the_train_begin_push():
    """Submitted ahead of the push, the worker rolls the resumed round out of the engines' pre-restore
    weights (on a LoRA resume, the base model)."""
    host = _Host()
    events = []
    host._sync_weights_to_engine_fenced = lambda force=False: events.append("push") or True
    host._submit_for_prefetch = lambda prompts, contexts: events.append(("submit", tuple(prompts)))
    host._resumed_prefetch_rounds = [(["b2"], [None])]
    host._start_rollout_generation()
    assert events == ["push", ("submit", ("b2",))]


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


def _async_state(server_configs, enable_prefetch=None):
    host = types.SimpleNamespace(
        async_config=AsyncRolloutConfig(rollout_server_configs=server_configs, enable_prefetch=enable_prefetch)
    )
    _T._init_async_state(host)
    return host


_SERVERS = {"url": None, "one-entry-list": _TWO_ENGINES[:1], "two-servers": _TWO_ENGINES}


@pytest.mark.parametrize(
    ("enable_prefetch", "servers", "prefetches"),
    [
        (None, "url", False),
        (None, "one-entry-list", False),
        (None, "two-servers", True),
        (True, "url", True),
        (True, "one-entry-list", True),
        (True, "two-servers", True),
        (False, "url", False),
        (False, "two-servers", False),
    ],
)
def test_the_prefetch_gate(enable_prefetch, servers, prefetches):
    """Unset follows the server count, where a one-entry list is one server; an explicit value is taken as given."""
    host = _async_state(_SERVERS[servers], enable_prefetch)
    assert host.async_config.prefetch_active() is prefetches
    assert host._prefetch_enabled is prefetches


@pytest.mark.parametrize("enable_prefetch", [None, True, False], ids=["unset", "true", "false"])
@pytest.mark.parametrize("servers", list(_SERVERS))
def test_the_start_log_states_the_prefetch_the_trainer_runs(servers, enable_prefetch):
    """Read off the knob, it says ``prefetch: disabled`` on an unset run, which prefetches on two servers."""
    host = _async_state(_SERVERS[servers], enable_prefetch)
    lines = env_grpo_script.rollout_start_log({}, host.async_config)
    assert f"prefetch: {'depth 1' if host._prefetch_enabled else 'disabled'}" in lines


def test_the_start_log_states_the_configured_depth():
    config = AsyncRolloutConfig(rollout_server_configs=_TWO_ENGINES, prefetch_depth=2)
    assert "prefetch: depth 2" in env_grpo_script.rollout_start_log({}, config)


@pytest.mark.parametrize(
    ("enable_prefetch", "servers", "hinted"),
    [(None, "url", True), (True, "url", False), (False, "url", False), (None, "two-servers", False)],
)
@pytest.mark.parametrize("main_process", [True, False])
def test_the_single_server_hint_is_said_once_and_only_when_unset(
    monkeypatch, caplog, main_process, enable_prefetch, servers, hinted
):
    monkeypatch.setattr(async_mod, "is_global_main_process", lambda: main_process)
    with caplog.at_level(logging.INFO, logger=async_mod.logger.name):
        _async_state(_SERVERS[servers], enable_prefetch)
    assert ("enable_prefetch: true" in caplog.text) is (hinted and main_process)


class _SubmitLog(queue.Queue):
    def __init__(self, maxsize: int, events: list):
        super().__init__(maxsize)
        self.events = events

    def put(self, item, block=True, timeout=None):
        super().put(item, block, timeout)
        self.events.append(("submit", *item[0]))


class _PopLog(queue.Queue):
    def __init__(self, maxsize: int, events: list):
        super().__init__(maxsize)
        self.events = events

    def get(self, block=True, timeout=None):
        item = super().get(block, timeout)
        self.events.append(("pop", *item[1]))
        return item


class _SteppingHost(_T):
    """The real ``training_step`` and ``_collect_and_score_rollouts`` with the engines, the worker thread and the batch
    build stubbed. A short ``episode_timeout``, so a step that waits on a round never submitted fails in seconds."""

    def __init__(self, server_configs, enable_prefetch, depth=1):
        self.async_config = AsyncRolloutConfig(
            rollout_server_configs=server_configs,
            enable_prefetch=enable_prefetch,
            prefetch_depth=depth,
            episode_timeout=0.01,
        )
        self._init_async_state()
        self.events: list[tuple] = []
        self._prefetch_queue = _PopLog(self._prefetch_queue.maxsize, self.events)
        self._prefetch_input_queue = _SubmitLog(self._prefetch_input_queue.maxsize, self.events)
        self.model = types.SimpleNamespace(training=True)
        self.accelerator = types.SimpleNamespace(device=torch.device("cpu"), is_main_process=False)
        self.parallelism_config = types.SimpleNamespace(
            data_parallel_size=1, get_data_parallel_rank=lambda: 0, is_tp_mode=False, is_expert_tp_mode=False
        )
        self.num_generations = 1
        self._train_batch_size = 1
        self.state = types.SimpleNamespace(global_step=0)
        self.args = types.SimpleNamespace(report_to=[], steps_per_generation=1)
        self._tokenizer = None
        self._group_random_effort = False
        self._batch_errors = BatchBuildFence()
        self._routing_injector = None
        self._loop = types.SimpleNamespace(run_until_complete=lambda batch: batch)
        self._rollout_manager = types.SimpleNamespace(collect_rollouts=self._collect)
        # What train-begin's forced push stamps.
        self._last_sync_attempt_step = 0

    def _collect(self, prompts, contexts):
        self.events.append(("collect", *prompts))
        return [_Rollout(f"sync:{p}") for p in prompts]

    def _sync_weights_to_engine_fenced(self, force: bool = False) -> bool:
        self.events.append(("sync", self.state.global_step))
        return True

    def _broadcast_object_from_tp_leader(self, rollout_results):
        return rollout_results

    def _build_training_tensors(self, rollout_results, device, mode, num_padding, reasoning_tokens):
        self.events.append(("train", *rollout_results))
        return {}

    def _log_rollout_metrics(self, results, mode, reasoning_tokens):
        pass

    def deliver(self):
        """Delivered as the worker delivers: a full output queue raises rather than waits."""
        prompts, _contexts = self._prefetch_input_queue.get_nowait()
        self._prefetch_queue.put_nowait((len(prompts), [_Rollout(f"prefetch:{p}") for p in prompts]))


def _stepped_host(monkeypatch, server_configs, enable_prefetch, depth=1) -> _SteppingHost:
    """One optimizer step per batch: ``depth`` cold steps, then two that pop a prefetched round."""
    monkeypatch.setattr(
        DistributedTrainerMixin,
        "training_step",
        lambda self, model, inputs, n=None: self._collect_and_score_rollouts(inputs),
    )
    monkeypatch.setattr(async_mod, "_PREFETCH_SUBMIT_TIMEOUT_S", 0.01)
    host = _SteppingHost(server_configs, enable_prefetch, depth)
    for step in range(depth + 2):
        host.state.global_step = step
        if not host._prefetch_input_queue.empty():
            host.deliver()
        host.training_step(None, [{"prompt": f"b{step}"}])
    return host


def _step_events(monkeypatch, server_configs, enable_prefetch, depth=1) -> list[tuple]:
    return _stepped_host(monkeypatch, server_configs, enable_prefetch, depth).events


_PREFETCHED_STEPS = {
    1: [
        ("collect", "b0"),
        ("submit", "b0"),
        ("train", "sync:b0"),
        ("sync", 1),
        ("pop", "prefetch:b0"),
        ("submit", "b1"),
        ("train", "prefetch:b0"),
        ("sync", 2),
        ("pop", "prefetch:b1"),
        ("submit", "b2"),
        ("train", "prefetch:b1"),
    ],
    2: [
        ("collect", "b0"),
        ("submit", "b0"),
        ("train", "sync:b0"),
        ("sync", 1),
        ("collect", "b1"),
        ("submit", "b1"),
        ("train", "sync:b1"),
        ("sync", 2),
        ("pop", "prefetch:b0"),
        ("submit", "b2"),
        ("train", "prefetch:b0"),
        ("sync", 3),
        ("pop", "prefetch:b1"),
        ("submit", "b3"),
        ("train", "prefetch:b1"),
    ],
}


@pytest.mark.parametrize("depth", sorted(_PREFETCHED_STEPS))
@pytest.mark.parametrize("servers", ["url", "one-entry-list"])
def test_one_server_that_opts_in_steps_exactly_as_two_servers(monkeypatch, servers, depth):
    """The lone engine's prefetched rounds span the push the way N engines' do, and train ``depth`` steps stale."""
    one_server = _step_events(monkeypatch, _SERVERS[servers], enable_prefetch=True, depth=depth)
    assert one_server == _step_events(monkeypatch, _SERVERS["two-servers"], enable_prefetch=None, depth=depth)
    assert one_server == _PREFETCHED_STEPS[depth]


def test_one_server_left_unset_trains_each_round_fresh(monkeypatch):
    assert _step_events(monkeypatch, _SERVERS["url"], enable_prefetch=None) == [
        ("collect", "b0"),
        ("train", "sync:b0"),
        ("sync", 1),
        ("collect", "b1"),
        ("train", "sync:b1"),
        ("sync", 2),
        ("collect", "b2"),
        ("train", "sync:b2"),
    ]


@pytest.mark.parametrize(("depth", "pending"), [(1, [["b2"]]), (2, [["b2"], ["b3"]])])
@pytest.mark.parametrize("servers", ["url", "one-entry-list"])
def test_one_server_that_opts_in_resumes_the_rounds_it_had_in_flight(monkeypatch, tmp_path, servers, depth, pending):
    """Without them back, oldest first, the one-server run skips those batches and trains the next ones twice."""
    before = _stepped_host(monkeypatch, _SERVERS[servers], enable_prefetch=True, depth=depth)
    before._persist_trainer_sidecars(str(tmp_path))
    resumed = _SteppingHost(_SERVERS[servers], enable_prefetch=True, depth=depth)
    resumed._restore_trainer_sidecars(str(tmp_path))
    assert [prompts for prompts, _contexts in resumed._resumed_prefetch_rounds] == pending


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
