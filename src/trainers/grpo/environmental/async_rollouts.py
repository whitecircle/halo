"""Async rollout plumbing for environmental GRPO: Ray actors, the prefetch thread and engine weight sync. Under
prefetch each round trains one round late, so a checkpoint carries the submitted rounds a resume submits again."""

import asyncio
import contextlib
import logging
import math
import os
import queue
import threading
import time
import weakref
from collections import deque
from functools import partial

import ray
import torch
from transformers import TrainerCallback
from trl.extras.profiling import profiling_context

from src.checkpoint.atomic import atomic_torch_save
from src.checkpoint.format import PREFETCH_PENDING_PREFIX, prefetch_pending_filename
from src.checkpoint.parallel.reading import consensus_read
from src.environments.episode import RolloutResult
from src.environments.ray_actors import RolloutManager, ray_init_kwargs
from src.runtime.collectives import broadcast_from_rank0
from src.runtime.consensus import DeferredRankFailure, rank_consensus
from src.runtime.ranks import get_global_rank, get_global_world_size, get_num_nodes, is_global_main_process
from src.trainers.grpo.rollout.weight_push import sync_trainer_weights
from src.trainers.grpo.world_metrics import world_sums
from src.weight_sync.clients.base import WeightSyncClientBase
from src.weight_sync.pool import WeightSyncPool
from src.weight_sync.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)


# The worker frees an input slot within one collection, so an expired submit means it is wedged.
_PREFETCH_SUBMIT_TIMEOUT_S = 60.0
# Input-queue poll slice, so a quiet round does not hold up shutdown.
_PREFETCH_POLL_TIMEOUT_S = 0.5
# Output-queue delivery slice, retried until the stop event fires.
_PREFETCH_DELIVER_TIMEOUT_S = 1.0
# Grace for the worker to finish its current poll slice and exit at shutdown.
_PREFETCH_JOIN_TIMEOUT_S = 5.0
# Rounds the prefetch pipeline holds: a round pops the previous round's result before it submits its own.
_PREFETCH_DEPTH = 1


class _RolloutStartCallback(TrainerCallback):
    """Starts generation at train-begin, the first hook past both resume restores and still ahead of
    ``eval_on_start``, on every rank: a push from ``_init_async_components`` would ship pre-restore weights.
    Weak, since the push needs the trainer only while it trains."""

    def __init__(self, trainer):
        self._trainer = weakref.ref(trainer)

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self._trainer()
        if trainer is None:
            # Without this raise, every rollout of the run comes from the engines' launch weights.
            raise RuntimeError(
                "Rollout generation could not start: the trainer that registered this callback is "
                "already collected, so the engines were never handed the trained weights."
            )
        trainer._start_rollout_generation()


class AsyncRolloutMixin:
    """Ray-actor rollout collection, prefetch overlap and engine weight sync for the environmental GRPO
    trainer, its state set by :meth:`_init_async_state`."""

    # Whether the weight-sync group is formed, held alike on every rank (the client itself is the main
    # process's alone): set by the fenced formation, cleared first thing by the cleanup.
    _weight_sync_group_formed: bool = False

    def _init_async_state(self):
        """Rollout-manager, weight-sync-client and prefetch state, before any of them is built."""
        self._rollout_manager = None
        self._weight_sync_client = None  # one engine client, or a WeightSyncPool over several
        # Score-only clients, one per rollout server on EVERY rank (isr_engine_reference).
        self._engine_rescore_clients: list[WeightSyncClientBase] | None = None
        self._loop = None
        # Which client shape to build, not how many engines serve (a one-entry list is one server).
        self._multi_server_mode = bool(self.async_config.rollout_server_configs)

        self._prefetch_enabled = self.async_config.prefetch_active()
        if self.async_config.enable_prefetch and not self._prefetch_enabled and is_global_main_process():
            logger.warning(
                "Prefetch auto-disabled: 1 rollout server configured. Every weight sync pauses that one "
                "engine for its whole push, so a prefetched round would only sit frozen (vLLM) or be "
                "aborted (SGLang) across it; prefetch is enabled with two or more servers in "
                "rollout_server_configs."
            )

        self._prefetch_thread = None
        self._prefetch_queue = queue.Queue(maxsize=_PREFETCH_DEPTH)
        self._prefetch_input_queue = queue.Queue(maxsize=_PREFETCH_DEPTH)
        self._prefetch_stop_event = threading.Event()

        self._prefetch_hits = 0
        self._prefetch_misses = 0
        # The (prompts, contexts) of each submitted, unconsumed round, oldest first; trainer thread only.
        self._prefetch_pending: deque[tuple[list[str], list[dict | None]]] = deque()
        # The pending rounds a resumed checkpoint carried, submitted once generation starts.
        self._resumed_prefetch_rounds: list[tuple[list[str], list[dict | None]]] = []
        self._prefetch_input_skips = 0
        # The step the sync path was last entered at, not the step the engines hold.
        self._last_sync_attempt_step = -1
        self._rollout_start_callback = None
        self._rollout_generation_started = False

    def _init_async_components(self):
        """COLLECTIVE. Start every rank's rollout components, then form the weight-sync group, each failure
        deferred to a verdict all ranks join so none is raised on one rank while its peers wait."""
        if self._rollout_manager is not None:
            return

        guard = DeferredRankFailure("Rollout component start")
        guard.run(self._start_rank_rollout_components)
        guard.reject()
        self._form_weight_sync_group()

        # Registered once; a second train() re-arms it through _cleanup_async_components.
        if self._rollout_start_callback is None:
            self._rollout_start_callback = _RolloutStartCallback(self)
            self.add_callback(self._rollout_start_callback)

    def _start_rank_rollout_components(self) -> None:
        """This rank's half of :meth:`_init_async_components`: Ray, the rollout manager and its actors,
        and under ``isr_engine_reference`` the score-only clients the engine re-score fans out over."""
        if not ray.is_initialized():
            ray.init(**ray_init_kwargs(address=self.async_config.ray_address))

        # One manager per rank over its DP shard: on a shared Ray cluster the actor budget is split across ranks.
        self._rollout_manager = RolloutManager(
            num_workers=self.async_config.num_rollout_workers,
            env_type=self._environment_spec,
            env_config=self._env_config_dict,
            server_urls=self.async_config.get_server_urls(),
            rollout_config=self.async_config.get_rollout_config(
                stop_token_ids=self._rollout_stop_token_ids,
                reasoning_end_token_id=self._resolve_reasoning_end_token_id(),
            ),
            max_concurrent_rollouts=self.async_config.max_concurrent_rollouts,
            pool_replicas=self.accelerator.num_processes if self.async_config.ray_address is not None else 1,
        )
        if self.accelerator.is_main_process:
            multinode = self.async_config.ray_address is not None or get_num_nodes() > 1
            self._rollout_manager.warn_if_servers_unreachable_from_actors(multinode)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._reject_eval_round_over_cap(self._rollout_manager.max_concurrent)
        self._loop.run_until_complete(self._rollout_manager.start())
        if self.async_config.isr_engine_reference:
            self._engine_rescore_clients = self._build_engine_rescore_clients()

    def _reject_eval_round_over_cap(self, max_concurrent: int) -> None:
        """Refuse an eval round above the in-flight cap: its serial waves hold the peers at the post-round
        collective for several ``episode_timeout``s."""
        rows = self.async_config.eval_rollout_batch_size
        if rows is not None and rows > max_concurrent:
            raise ValueError(
                f"eval_rollout_batch_size ({rows}) exceeds max_concurrent_rollouts ({max_concurrent}): the eval "
                f"round would run in {math.ceil(rows / max_concurrent)} waves. Raise the cap to the round."
            )

    def _build_engine_rescore_clients(self) -> list[WeightSyncClientBase]:
        """This rank's score-only clients for the engine re-score, one per server a round's trajectories reach
        (trajectory ``t`` on server ``t mod N``); HTTP only, no communicator."""
        client_cls = resolve_weight_sync_client(self.async_config.rollout_backend)
        return [
            client_cls(base_url=url, connection_timeout=self.async_config.rollout_connection_timeout)
            for url in self.async_config.get_server_urls()[: self._train_loader_batch_size()]
        ]

    def _form_weight_sync_group(self) -> None:
        """COLLECTIVE while it forms. Form the engine weight-sync group unless formed; the main process's client
        failure raises on every rank."""
        if self._weight_sync_group_formed:
            return
        guard = DeferredRankFailure("Weight-sync client formation")
        guard.run(self._init_weight_sync_client)
        guard.reject()
        self._weight_sync_group_formed = True

    def _init_weight_sync_client(self):
        """The main process's engine weight-sync client(s), built once under :meth:`_form_weight_sync_group`."""
        if self._weight_sync_client is not None:
            return

        if not self.accelerator.is_main_process:
            return

        device = self.accelerator.device
        client_cls = resolve_weight_sync_client(self.async_config.rollout_backend)

        if self._multi_server_mode:
            self._weight_sync_client = WeightSyncPool(
                server_configs=self.async_config.rollout_server_configs,
                connection_timeout=self.async_config.rollout_connection_timeout,
                client_cls=client_cls,
                base_group_port=self.args.vllm_group_port,
            )
            self._weight_sync_client.init_communicator(device=device)
        else:
            logger.info(f"Initializing {client_cls.BACKEND_NAME} weight-sync client on device: {device}")

            self._weight_sync_client = client_cls(
                base_url=self.async_config.rollout_server_url,
                group_port=self.args.vllm_group_port,
                connection_timeout=self.async_config.rollout_connection_timeout,
            )

            self._weight_sync_client.init_communicator(device=device)

            logger.info(
                f"{client_cls.BACKEND_NAME} weight-sync client initialized: {self.async_config.rollout_server_url}, "
                f"device={self._weight_sync_client.sync_device}, group_port={self.args.vllm_group_port}"
            )

    def _start_rollout_generation(self):
        """COLLECTIVE. Push the trainer's weights to the engines, then start the prefetch thread; idempotent."""
        if self._rollout_generation_started:
            return
        self._rollout_generation_started = True
        # Stamped though forced: a resumed run's first step at this global_step must not push again.
        self._last_sync_attempt_step = self.state.global_step
        self._sync_weights_to_engine_fenced(force=True)
        if self._prefetch_enabled:
            self._start_prefetch_thread()
            # After the push, so the resumed rounds are drawn from the restored policy.
            for prompts, contexts in self._resumed_prefetch_rounds:
                self._submit_for_prefetch(prompts, contexts)
        self._resumed_prefetch_rounds = []

    def _sync_weights_to_engine_fenced(self, force: bool = False) -> bool:
        """COLLECTIVE. :meth:`_sync_weights_to_engine` with whatever a rank raises out of the push raised on every
        rank. The push duration is broadcast first, so every rank credits the servers' pause to its in-flight
        episodes' deadlines whether or not the push failed."""
        manager = self._rollout_manager
        if manager is not None:
            manager.begin_engine_pause()
        guard = DeferredRankFailure(f"{self._rollout_engine_name} weight sync at step {self.state.global_step}")
        started = time.monotonic()
        synced = bool(guard.run(partial(self._sync_weights_to_engine, force=force)))
        paused_seconds = broadcast_from_rank0(time.monotonic() - started if synced else 0.0)
        if manager is not None:
            manager.end_engine_pause(paused_seconds)
        guard.reject()
        return synced

    @property
    def _rollout_engine_name(self) -> str:
        """The configured rollout engine's display name, which every user-facing string here reads."""
        return resolve_weight_sync_client(self.async_config.rollout_backend).BACKEND_NAME

    def _sync_weights_to_engine(self, force: bool = False) -> bool:
        """COLLECTIVE. Push the trainer's weights to every rollout engine unless the cadence gate declines (``force``
        overrides it), forming the group first where none is; returns whether weights went out."""
        # Evaluated alike on every rank: the push it gates is a collective gather.
        if not force and self.state.global_step % self.async_config.sync_weights_every_n_steps != 0:
            return False
        self._form_weight_sync_group()
        with profiling_context(self, "weight_sync"):
            sync_trainer_weights(self, self._weight_sync_client)
        return True

    def _start_prefetch_thread(self):
        """Start the background prefetch thread."""
        if self._prefetch_thread is not None:
            return

        self._prefetch_stop_event.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            name="PrefetchWorker",
            daemon=True,
        )
        self._prefetch_thread.start()
        logger.info("Prefetch thread started for async overlap")

    def _prefetch_worker(self):
        """The prefetch thread: every consumed submission yields exactly one output item, its rollouts or a
        ``(prompt_count, None)`` failure marker, so the in-flight accounting stays exact."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while not self._prefetch_stop_event.is_set():
                try:
                    item = self._prefetch_input_queue.get(timeout=_PREFETCH_POLL_TIMEOUT_S)
                except queue.Empty:
                    continue

                if item is None:  # sentinel
                    break

                prompts, contexts = item
                results = None
                start_time = time.time()
                try:
                    results = loop.run_until_complete(self._rollout_manager.collect_rollouts(prompts, contexts))
                except Exception as e:  # a worker-thread raise is invisible; report and mark
                    logger.error(f"Prefetch worker error ({len(prompts)} prompts): {e}")
                self._put_prefetch_output((len(prompts), results))
                if results is not None:
                    logger.debug(f"Prefetched {len(results)} rollouts in {time.time() - start_time:.2f}s")

        finally:
            loop.close()

    def _put_prefetch_output(self, item: tuple[int, list[RolloutResult] | None]):
        """Deliver a prefetch item, blocking in stop-aware slices; only shutdown abandons one (logged)."""
        while not self._prefetch_stop_event.is_set():
            try:
                self._prefetch_queue.put(item, timeout=_PREFETCH_DELIVER_TIMEOUT_S)
                return
            except queue.Full:
                continue
        logger.warning(f"Prefetch stopping; abandoning delivery of a {item[0]}-prompt rollout batch (shutdown)")

    def _try_get_prefetched_results(self) -> list[RolloutResult] | None:
        """The next prefetched round if one is ready; a popped failure marker counts as a miss."""
        try:
            _count, results = self._prefetch_queue.get_nowait()
        except queue.Empty:
            self._prefetch_misses += 1
            return None
        self._prefetch_pending.popleft()
        if results is None:
            self._prefetch_misses += 1
            return None
        self._prefetch_hits += 1
        logger.debug(f"Prefetch hit: using {len(results)} prefetched rollouts")
        return results

    def _wait_for_inflight_prefetch(self) -> list[RolloutResult] | None:
        """Wait for an in-flight round rather than re-collect it, so each dataset batch trains once; ``None`` when
        every in-flight round failed or the worker is wedged past the episode deadline. A wedge is recorded on
        the ``_batch_errors`` fence, not raised: prefetch state is per rank."""
        timeout = self.async_config.episode_timeout + _PREFETCH_SUBMIT_TIMEOUT_S
        while self._prefetch_inflight > 0:
            try:
                _count, results = self._prefetch_queue.get(timeout=timeout)
            except queue.Empty:
                self._batch_errors.record(
                    f"Prefetch worker produced no rollouts within {timeout:.0f}s with "
                    f"{self._prefetch_inflight} batch(es) in flight — the prefetch pipeline is wedged "
                    f"({self._rollout_engine_name} unreachable or the worker thread died)."
                )
                return None
            self._prefetch_pending.popleft()
            if results is not None:
                logger.info(f"Prefetch miss: waited for {len(results)} in-flight rollouts")
                return results
        return None

    @property
    def _prefetch_inflight(self) -> int:
        """Submitted prefetch rounds whose output no step has consumed (one output item each)."""
        return len(self._prefetch_pending)

    def _submit_for_prefetch(
        self,
        prompts: list[str],
        contexts: list[dict | None],
    ):
        """Submit a round for prefetching; a skipped one is warned, since its prompts then never train."""
        try:
            self._prefetch_input_queue.put((prompts, contexts), timeout=_PREFETCH_SUBMIT_TIMEOUT_S)
        except queue.Full:
            self._prefetch_input_skips += 1
            logger.warning(
                f"Prefetch input queue full after {_PREFETCH_SUBMIT_TIMEOUT_S:.0f}s; skipped prefetch of "
                f"{len(prompts)} prompts — these prompts will NOT train this cycle "
                f"(total skipped submissions: {self._prefetch_input_skips})"
            )
            return
        self._prefetch_pending.append((prompts, contexts))

    def prefetch_metrics(self) -> dict[str, float]:
        """COLLECTIVE. The ``async/*`` prefetch counters since train start, summed over the world."""
        hits, misses, skips = world_sums([self._prefetch_hits, self._prefetch_misses, self._prefetch_input_skips])
        metrics = {}
        if hits + misses > 0:
            metrics = {"async/prefetch_hit_rate": hits / (hits + misses), "async/prefetch_hits": hits}
            metrics["async/prefetch_misses"] = misses
        if skips > 0:
            metrics["async/prefetch_input_skips"] = skips
        return metrics

    def _persist_trainer_sidecars(self, checkpoint_dir: str) -> None:
        """Write this rank's pending prefetch rounds, which a resume submits again first: they left the
        dataloader untrained, and the resumed one starts past them."""
        super()._persist_trainer_sidecars(checkpoint_dir)
        if not self._prefetch_enabled:
            return
        path = os.path.join(checkpoint_dir, prefetch_pending_filename(get_global_rank(), get_global_world_size()))
        payload = {"layout": self._prefetch_round_layout(), "rounds": list(self._prefetch_pending)}
        guard = DeferredRankFailure(f"pending prefetch rounds write to {checkpoint_dir}")
        # Ray's pickler, which ships the same contexts: a row's may carry a callable (an answer
        # ``validator``) that the stdlib pickler refuses.
        guard.run(partial(atomic_torch_save, path, lambda: payload, pickle_module=ray.cloudpickle))
        guard.reject()

    def _prefetch_round_layout(self) -> dict[str, int]:
        """What a round's prompts were drawn and grouped under; a resume under another layout drops them."""
        return {
            "num_generations": self.num_generations,
            "rows_per_round": self._train_loader_batch_size(),
            "data_parallel_rank": self.parallelism_config.get_data_parallel_rank(),
            "data_parallel_size": self.parallelism_config.data_parallel_size,
        }

    def _restore_trainer_sidecars(self, checkpoint: str) -> None:
        """Load this rank's pending prefetch rounds for :meth:`_start_rollout_generation` to submit; rounds drawn
        under another layout are dropped on every rank together. ``weights_only=False``: they carry context values."""
        super()._restore_trainer_sidecars(checkpoint)
        payload, path = consensus_read(
            os.path.join(checkpoint, prefetch_pending_filename(get_global_rank(), get_global_world_size())),
            partial(torch.load, map_location="cpu", weights_only=False),
            what="pending prefetch rounds",
            checkpoint=checkpoint,
            remedy=" Resume with the rank-to-node placement it was saved with, or delete its "
            f"{PREFETCH_PENDING_PREFIX}-* files to open with a cold round.",
        )
        if path is None:
            if self._prefetch_enabled and is_global_main_process():
                logger.warning(
                    f"{checkpoint} holds no pending prefetch rounds for this world size, so the run opens with "
                    "a cold round: if the saving run prefetched, the batch it had drawn but not trained is "
                    "skipped and the next one trained twice."
                )
            return
        if not self._prefetch_enabled:
            if payload["rounds"] and is_global_main_process():
                logger.warning(
                    f"{path} holds pending prefetch rounds this run does not prefetch; they are not trained."
                )
            return
        if not rank_consensus(payload["layout"] == self._prefetch_round_layout()).all_ok:
            if is_global_main_process():
                logger.warning(
                    f"{checkpoint}'s pending prefetch rounds were drawn under {payload['layout']}, this run "
                    f"draws under {self._prefetch_round_layout()}: they are not resubmitted, and the run opens "
                    "with a cold round."
                )
            return
        self._resumed_prefetch_rounds = list(payload["rounds"])

    def _cleanup_async_components(self):
        """Cleanup Ray actors and the engine weight-sync client(s). Run by every rank."""
        # First, on every rank: the client below is the main process's alone, and a teardown step that
        # raises must not leave the ranks disagreeing on whether the next push forms the group.
        self._weight_sync_group_formed = False
        self._stop_prefetch_thread()

        if self._rollout_manager and self._loop:
            self._loop.run_until_complete(self._rollout_manager.shutdown())
            self._rollout_manager = None

        if self._weight_sync_client is not None:
            try:
                self._weight_sync_client.close_communicator()
            except Exception as e:  # teardown must not mask the run's own outcome
                logger.warning(f"Error closing the {self._rollout_engine_name} weight-sync client: {e}")
            self._weight_sync_client = None

        # Score-only clients formed no communicator; their HTTP sessions are what they hold.
        if self._engine_rescore_clients is not None:
            for client in self._engine_rescore_clients:
                client.close_session()
            self._engine_rescore_clients = None

        self._loop = None
        # Re-arm: a second train() rebuilds the components above and needs a fresh push to the
        # engines, at whatever step it starts from.
        self._rollout_generation_started = False
        self._last_sync_attempt_step = -1

    def _stop_prefetch_thread(self):
        """Stop the background prefetch thread."""
        if self._prefetch_thread is None:
            return

        self._prefetch_stop_event.set()

        with contextlib.suppress(queue.Empty):
            while not self._prefetch_input_queue.empty():
                self._prefetch_input_queue.get_nowait()

        with contextlib.suppress(queue.Full):
            self._prefetch_input_queue.put_nowait(None)

        self._prefetch_thread.join(timeout=_PREFETCH_JOIN_TIMEOUT_S)
        if self._prefetch_thread.is_alive():
            logger.warning("Prefetch thread did not stop cleanly")

        self._prefetch_thread = None
        # Drained submissions can no longer produce output items, and delivered ones belong to rounds no
        # later step consumes (a second train() would pop them against its own pending rounds).
        self._prefetch_pending.clear()
        with contextlib.suppress(queue.Empty):
            while True:
                self._prefetch_queue.get_nowait()

        total = self._prefetch_hits + self._prefetch_misses
        if total > 0:
            hit_rate = self._prefetch_hits / total * 100
            logger.info(
                f"Prefetch stats: {self._prefetch_hits} hits, {self._prefetch_misses} misses "
                f"({hit_rate:.1f}% hit rate)"
            )
