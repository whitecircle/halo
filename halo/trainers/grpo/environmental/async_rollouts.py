"""Async rollout plumbing for environmental GRPO. Under prefetch each round trains ``prefetch_depth`` rounds late, so a
checkpoint carries the submitted rounds a resume submits again."""

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
from typing import Any

import ray
import torch
from transformers import TrainerCallback
from trl.extras.profiling import profiling_context

from halo.checkpoint.atomic import atomic_torch_save
from halo.checkpoint.format import PREFETCH_PENDING_PREFIX, prefetch_pending_filename
from halo.checkpoint.parallel.reading import consensus_read
from halo.diagnostics.phase_timers import Phase
from halo.environments.episode import RolloutResult
from halo.environments.ray_actors import RolloutManager, ray_init_kwargs
from halo.environments.rollout_config import resolve_reasoning_end_token_id
from halo.environments.rollout_pool import RolloutPoolPlan, resolve_rollout_pool
from halo.runtime.collectives import broadcast_from_rank0
from halo.runtime.consensus import DeferredRankFailure, rank_consensus
from halo.runtime.ranks import (
    get_global_rank,
    get_global_world_size,
    get_local_world_size,
    get_num_nodes,
    is_global_main_process,
)
from halo.trainers.grpo.early_stop import GRPOEarlyStopCallback
from halo.trainers.grpo.rollout.adapter_sync import (
    AdapterSync,
    build_adapter_requirements,
    reject_unadaptable_config,
    reject_unadaptable_model,
    resolve_adapter_serving,
    resolve_launch_base,
)
from halo.trainers.grpo.rollout.engine_preflight import verify_adapter_servers
from halo.trainers.grpo.rollout.push_parity import verify_first_push_parity
from halo.trainers.grpo.rollout.weight_push import sync_trainer_weights
from halo.trainers.world_metrics import world_sums
from halo.weight_sync.adapter_routes import AdapterRoutes
from halo.weight_sync.adapter_servers import AdapterServers
from halo.weight_sync.adapter_versions import AdapterVersions
from halo.weight_sync.clients.base import WeightSyncClientBase
from halo.weight_sync.pool import WeightSyncPool
from halo.weight_sync.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)


# The worker takes each submission as it arrives, so an expired submit means it is wedged.
_PREFETCH_SUBMIT_TIMEOUT_S = 60.0
# Input-queue poll slice, so a quiet round does not hold up shutdown.
_PREFETCH_POLL_TIMEOUT_S = 0.5
# Grace for the worker to finish its current poll slice, cancel its rounds and exit at shutdown.
_PREFETCH_JOIN_TIMEOUT_S = 5.0


class _RolloutStartCallback(TrainerCallback):
    """Starts generation at train-begin, the first hook past both resume restores and ahead of ``eval_on_start``, on
    every rank: a push from ``_init_async_components`` would ship pre-restore weights."""

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
    # Whether the weight-sync group is formed, held alike on every rank (the client itself is the main
    # process's alone): set by the fenced formation, cleared first thing by the cleanup.
    _weight_sync_group_formed: bool = False
    # Whether the rollout components are up, held alike on every rank (a TP/ETP follower builds no manager, so the
    # manager cannot say): set by the init, cleared by the cleanup.
    _async_components_started: bool = False
    # Whether the first weight push has been checked against the trainer's own log-probs (``verify_first_push_parity``).
    first_push_verified: bool = False
    # Set by _install_adapter_sync when the policy is served as an adapter beside the frozen base.
    _adapter_versions: AdapterVersions | None = None
    _adapter_routes: AdapterRoutes | None = None
    _adapter_sync: AdapterSync | None = None

    def _init_async_state(self):
        self._rollout_manager = None
        self._weight_sync_client: WeightSyncPool | None = None
        # Score-only clients, one per rollout server on EVERY rank (isr_engine_reference).
        self._engine_rescore_clients: list[WeightSyncClientBase] | None = None
        self._loop = None
        self._prefetch_enabled = self.async_config.prefetch_active()
        if self.async_config.enable_prefetch is None and not self._prefetch_enabled and is_global_main_process():
            logger.info(
                "Prefetch off: 1 rollout server, so each round rolls out and then trains on-policy. "
                "enable_prefetch: true rolls the next round out while this one trains, one step stale "
                "(IS-corrected), as two or more servers do by default."
            )

        self._prefetch_thread = None
        # At most prefetch_depth rounds are submitted and unconsumed, so no put on either queue waits while the
        # worker lives.
        depth = self.async_config.prefetch_depth
        self._prefetch_queue = queue.Queue(maxsize=depth)
        self._prefetch_input_queue = queue.Queue(maxsize=depth)
        self._prefetch_stop_event: threading.Event | None = None

        self._prefetch_hits = 0
        self._prefetch_misses = 0
        # The (prompts, contexts) of each submitted, unconsumed round, oldest first; trainer thread only.
        self._prefetch_pending: deque[tuple[list[str], list[dict | None]]] = deque()
        self._resumed_prefetch_rounds: list[tuple[list[str], list[dict | None]]] = []
        self._prefetch_input_skips = 0
        # The step the sync path was last entered at, not the step the engines hold.
        self._last_sync_attempt_step = -1
        # The step whose weights the engines hold.
        self._last_pushed_step = -1
        self._rollout_start_callback = None
        self._rollout_generation_started = False

    def _install_adapter_sync(self) -> None:
        """Refuses on every rank a config or model whose adapters the engine cannot serve, else arms the adapter sync."""
        config = self.async_config
        if not config.rollout_adapter_sync:
            return
        reject_unadaptable_config(config)
        reject_unadaptable_model(
            self.model, self.parallelism_config, resolve_weight_sync_client(config.rollout_backend)
        )
        serving = resolve_adapter_serving(self.model)
        verify_adapter_servers(
            config.get_server_urls(),
            build_adapter_requirements(serving, config),
            backend=config.rollout_backend,
            connection_timeout=config.rollout_connection_timeout,
        )
        self._adapter_versions = AdapterVersions()
        self._adapter_routes = AdapterRoutes(resolve_weight_sync_client(config.rollout_backend).is_adapter_missing)
        self._adapter_sync = AdapterSync(
            serving, self._adapter_versions, self._adapter_routes, resolve_launch_base(self.model, config)
        )

    def _init_async_components(self):
        """COLLECTIVE. Each failure is deferred to a verdict all ranks join, so none is raised on one rank while its
        peers wait."""
        if self._async_components_started:
            return

        guard = DeferredRankFailure("Rollout component start")
        guard.run(self._start_rank_rollout_components)
        guard.reject()
        self._form_weight_sync_group()
        self._async_components_started = True

        # Registered once; a second train() re-arms it through _cleanup_async_components.
        if self._rollout_start_callback is None:
            self._rollout_start_callback = _RolloutStartCallback(self)
            self.add_callback(self._rollout_start_callback)

    def _start_rank_rollout_components(self) -> None:
        """A TP/ETP follower starts no pool: it takes its group leader's rollouts, and only the score-only clients are
        built on every rank."""
        if not self._is_tp_follower():
            self._start_rollout_pool()
        if self.async_config.isr_engine_reference:
            self._engine_rescore_clients = self._build_engine_rescore_clients()

    def _start_rollout_pool(self) -> None:
        if not ray.is_initialized():
            ray.init(**ray_init_kwargs(address=self.async_config.ray_address))

        plan = self._resolve_rollout_pool()
        if is_global_main_process():
            logger.info(
                f"Rollout pool: {plan.actors_per_rank} actors per rank from rollout_actor_budget "
                f"{self.async_config.rollout_actor_budget}, {plan.max_concurrent} rollouts in flight per rank."
            )
            for warning in plan.warnings:
                logger.warning(warning)
        adapter_model_name = None
        if self._adapter_versions is not None:
            client_cls = resolve_weight_sync_client(self.async_config.rollout_backend)
            adapter_model_name = partial(client_cls.adapter_request_model, self.async_config.model_name)
        self._rollout_manager = RolloutManager(
            num_workers=plan.actors_per_rank,
            env_type=self._environment_spec,
            env_config=self._env_config_dict,
            server_urls=self.async_config.get_server_urls(),
            rollout_config=self.async_config.get_rollout_config(
                stop_token_ids=self._rollout_stop_token_ids,
                reasoning_end_token_id=resolve_reasoning_end_token_id(self._reasoning_end_ids),
            ),
            max_concurrent_rollouts=plan.max_concurrent,
            first_server=self.parallelism_config.get_data_parallel_rank(),
            adapter_versions=self._adapter_versions,
            adapter_model_name=adapter_model_name,
            adapter_routes=self._adapter_routes,
        )
        self._rollout_manager.reject_loopback_servers(self._rollout_node_count())

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._reject_eval_round_over_cap(self._rollout_manager.max_concurrent)
        self._loop.run_until_complete(self._rollout_manager.start())

    def _resolve_rollout_pool(self) -> RolloutPoolPlan:
        """Rollout ranks are one per TP/ETP group, its leader."""
        group_size = self._get_tp_or_etp_group_size()
        return resolve_rollout_pool(
            num_workers=self.async_config.rollout_actor_budget,
            shared_cluster=self.async_config.ray_address is not None,
            rollout_ranks=get_global_world_size() // group_size,
            rollout_ranks_per_node=max(1, get_local_world_size() // group_size),
            rows_per_round=self._train_loader_batch_size(),
            eval_rows=self.async_config.eval_rollout_batch_size,
            max_concurrent_rollouts=self.async_config.max_concurrent_rollouts,
        )

    def _rollout_node_count(self) -> int:
        """Actors follow the launch across its nodes and spill over a shared Ray cluster's, which a head on this host
        alone does not have."""
        nodes = get_num_nodes()
        if self.async_config.ray_address is not None:
            nodes = max(nodes, sum(node["Alive"] for node in ray.nodes()))
        return nodes

    def _reject_eval_round_over_cap(self, max_concurrent: int) -> None:
        """An eval round above the in-flight cap runs in serial waves that hold the peers at the post-round collective
        for several ``episode_timeout``s."""
        rows = self.async_config.eval_rollout_batch_size
        if rows is not None and rows > max_concurrent:
            raise ValueError(
                f"eval_rollout_batch_size ({rows}) exceeds max_concurrent_rollouts ({max_concurrent}): the eval "
                f"round would run in {math.ceil(rows / max_concurrent)} waves. Raise the cap to the round."
            )

    def _build_engine_rescore_clients(self) -> list[WeightSyncClientBase]:
        """One per server a round's trajectories reach (trajectory ``t`` on server ``t mod N``); HTTP only, no
        communicator."""
        client_cls = resolve_weight_sync_client(self.async_config.rollout_backend)
        clients = [
            client_cls(base_url=url, connection_timeout=self.async_config.rollout_connection_timeout)
            for url in self.async_config.get_server_urls()[: self._train_loader_batch_size()]
        ]
        if self._adapter_versions is not None:
            for client in clients:
                client.adapter_source = lambda versions=self._adapter_versions: versions.current
        return clients

    def _form_weight_sync_group(self) -> None:
        """COLLECTIVE while it forms, over the c10d store: rank 0 waits on each server in turn for longer than the
        NCCL watchdog allows its peers. The main process's client failure raises on every rank."""
        if self._weight_sync_group_formed:
            return
        guard = DeferredRankFailure("Weight-sync client formation")
        guard.run(self._init_weight_sync_client)
        guard.store_reject("rollout/weight_sync_group")
        self._weight_sync_group_formed = True

    def _init_weight_sync_client(self):
        if self._weight_sync_client is not None:
            return

        if not self.accelerator.is_main_process:
            return

        # A pool at any server count, so a server restarted between syncs is recovered on one URL too.
        config = self.async_config
        self._weight_sync_client = WeightSyncPool(
            server_configs=config.rollout_server_configs or [{"url": config.rollout_server_url}],
            connection_timeout=config.rollout_connection_timeout,
            client_cls=resolve_weight_sync_client(config.rollout_backend),
            base_group_port=config.rollout_group_port,
            topology=config.rollout_sync_topology,
        )
        self._weight_sync_client.init_communicator(device=self.accelerator.device)
        if self._adapter_sync is not None:
            self._adapter_sync.servers = AdapterServers(
                config.get_server_urls(),
                client_cls=resolve_weight_sync_client(config.rollout_backend),
                connection_timeout=config.rollout_connection_timeout,
            )

    def _start_rollout_generation(self):
        """COLLECTIVE. Push the trainer's weights to the engines, then start the prefetch thread; idempotent."""
        if self._rollout_generation_started:
            return
        self._rollout_generation_started = True
        # Stamped though forced: a resumed run's first step at this global_step must not push again.
        self._last_sync_attempt_step = self.state.global_step
        self._sync_weights_to_engine_fenced(force=True)
        if self._prefetch_enabled and not self._is_tp_follower():
            self._start_prefetch_thread()
            # After the push, so the resumed rounds are drawn from the restored policy.
            for prompts, contexts in self._resumed_prefetch_rounds:
                self._submit_for_prefetch(prompts, contexts)
        self._resumed_prefetch_rounds = []

    def _sync_weights_to_engine_fenced(self, force: bool = False) -> bool:
        """COLLECTIVE. A push failure raises on every rank. The duration is broadcast first, so every rank credits the
        servers' pause to its in-flight episodes' deadlines even if the push failed."""
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
        return resolve_weight_sync_client(self.async_config.rollout_backend).BACKEND_NAME

    def _sync_weights_to_engine(self, force: bool = False) -> bool:
        """COLLECTIVE. Push to every engine unless the cadence gate declines (``force`` overrides it); returns whether
        weights went out."""
        # Evaluated alike on every rank: the push it gates is a collective gather.
        if not force and self.state.global_step % self.async_config.sync_weights_every_n_steps != 0:
            if self._adapter_sync is not None:
                self._adapter_sync.recover(self._weight_sync_client)
            return False
        self._form_weight_sync_group()
        with profiling_context(self, "weight_sync"), self._phase_timers.span(Phase.SYNC, "host"):
            if self._adapter_sync is None:
                sync_trainer_weights(self, self._weight_sync_client)
            else:
                self._adapter_sync.sync(self, self._weight_sync_client)
        if self._adapter_sync is None:
            verify_first_push_parity(self, self._weight_sync_client)
        self._last_pushed_step = self.state.global_step
        return True

    def evaluate(self, *args, **kwargs):
        """COLLECTIVE. An eval generates on what the engines hold, which trails the weights being evaluated and saved:
        by this step's update at any cadence, and by up to ``sync_weights_every_n_steps - 1`` steps more. Stamped, so the
        next step's first microbatch does not push the same weights again."""
        if self._rollout_generation_started and self._last_pushed_step != self.state.global_step:
            self._last_sync_attempt_step = self.state.global_step
            self._sync_weights_to_engine_fenced(force=True)
        return super().evaluate(*args, **kwargs)

    def _start_prefetch_thread(self):
        """A stop event of its own: a previous worker still in its round past the stop's bounded join keeps its set
        one, so this start cannot revive it."""
        if self._prefetch_thread is not None:
            return

        self._prefetch_stop_event = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(self._prefetch_stop_event,),
            name="PrefetchWorker",
            daemon=True,
        )
        self._prefetch_thread.start()
        logger.info("Prefetch thread started for async overlap")

    def _prefetch_worker(self, stop: threading.Event):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._prefetch_rounds(stop))
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()

    async def _prefetch_rounds(self, stop: threading.Event):
        """Start each round's collection as it arrives, so a later round's episodes keep the engines busy while an
        earlier round's longest ones finish. Rounds still collecting at ``stop`` are cancelled."""
        loop = asyncio.get_running_loop()
        collecting: set[asyncio.Task] = set()
        previous: asyncio.Task | None = None
        try:
            while not stop.is_set():
                try:
                    prompts, contexts = await loop.run_in_executor(
                        None, partial(self._prefetch_input_queue.get, timeout=_PREFETCH_POLL_TIMEOUT_S)
                    )
                except queue.Empty:
                    continue
                previous = loop.create_task(self._collect_prefetch_round(prompts, contexts, previous, stop))
                collecting.add(previous)
                previous.add_done_callback(collecting.discard)
        finally:
            for task in list(collecting):
                task.cancel()
            await asyncio.gather(*collecting, return_exceptions=True)

    async def _collect_prefetch_round(
        self,
        prompts: list[str],
        contexts: list[dict | None],
        previous: asyncio.Task | None,
        stop: threading.Event,
    ) -> None:
        """Deliver once ``previous`` has delivered, since the trainer pairs each output with its oldest pending round.
        Every round yields exactly one output, its rollouts or a ``(prompt_count, None)`` failure marker."""
        results = None
        start_time = time.time()
        try:
            results = await self._rollout_manager.collect_rollouts(prompts, contexts)
        except asyncio.CancelledError:
            if stop.is_set():
                raise
            # Ended without an output, the round would pair every later output one round off.
            logger.error(f"Prefetch round of {len(prompts)} prompts cancelled while the worker runs; marked failed")
        except Exception as e:  # a worker-thread raise is invisible; report and mark
            logger.error(f"Prefetch worker error ({len(prompts)} prompts): {e}")
        if previous is not None:
            await asyncio.wait([previous])
        self._put_prefetch_output((len(prompts), results), stop)
        if results is not None:
            logger.debug(f"Prefetched {len(results)} rollouts in {time.time() - start_time:.2f}s")

    def _put_prefetch_output(self, item: tuple[int, list[RolloutResult] | None], stop: threading.Event):
        """Only shutdown abandons an item. At most ``prefetch_depth`` are unconsumed, so a full queue raises instead of
        waiting, since a wait on the worker's event loop would freeze every other round's collection."""
        if stop.is_set():
            logger.warning(f"Prefetch stopping; abandoning delivery of a {item[0]}-prompt rollout batch (shutdown)")
            return
        try:
            self._prefetch_queue.put_nowait(item)
        except queue.Full:
            logger.error(
                f"Prefetch output queue full delivering a {item[0]}-prompt round: more rounds in flight than "
                f"prefetch_depth={self.async_config.prefetch_depth}"
            )
            raise

    def _acquire_prefetched_round(self) -> list[RolloutResult] | None:
        """The oldest round once ``prefetch_depth`` are in flight, else ``None``: a cold step collects its own prompts
        and submits them too, deepening the pipeline. A round delivered as a failure is popped and the step collects
        its own."""
        depth = self.async_config.prefetch_depth
        if self._prefetch_inflight < depth:
            self._prefetch_misses += 1
            return None
        results = self._try_get_prefetched_results()
        if results is None and self._prefetch_inflight == depth:
            results = self._wait_for_inflight_prefetch()
        return results

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
        """Wait rather than re-collect, so each dataset batch trains once; ``None`` on failure or a wedge past the
        episode deadline. A wedge is recorded on the ``_batch_errors`` fence, since prefetch state is per rank."""
        timeout = self.async_config.episode_timeout + _PREFETCH_SUBMIT_TIMEOUT_S
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

    @property
    def _prefetch_inflight(self) -> int:
        return len(self._prefetch_pending)

    def _submit_for_prefetch(
        self,
        prompts: list[str],
        contexts: list[dict | None],
    ):
        """A skipped round is warned, since its prompts then never train."""
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
        """Write this rank's pending rounds, oldest first, which a resume submits again first: they left the dataloader
        untrained, and the resumed one starts past them. A TP/ETP follower holds none and writes an empty list, so
        every rank has its file. The halting counters ride beside them, so a resumed run keeps the patience it had
        left."""
        super()._persist_trainer_sidecars(checkpoint_dir)
        path = os.path.join(checkpoint_dir, prefetch_pending_filename(get_global_rank(), get_global_world_size()))
        payload = {
            "layout": self._prefetch_round_layout(),
            "rounds": list(self._prefetch_pending) if self._prefetch_enabled else [],
            "halting": self._halting_counters(),
        }
        guard = DeferredRankFailure(f"pending prefetch rounds write to {checkpoint_dir}")
        # Ray's pickler, which ships the same contexts: a row's may carry a callable (an answer
        # ``validator``) that the stdlib pickler refuses.
        guard.run(partial(atomic_torch_save, path, lambda: payload, pickle_module=ray.cloudpickle))
        guard.reject()

    def _halting_counters(self) -> dict[str, Any]:
        """The streaks the run's halting rules count, which a restart would otherwise reset to a full patience budget."""
        stop = next((cb for cb in self.callback_handler.callbacks if isinstance(cb, GRPOEarlyStopCallback)), None)
        return {
            "zero_gradient_rounds": self._zero_gradient_rounds,
            "policy_gradient_seen": self._policy_gradient_seen,
            "empty_rollout_steps": self._empty_rollout_steps,
            "episode_error_rounds": self._episode_error_rounds,
            "early_stop_streaks": {} if stop is None else dict(stop.streaks),
        }

    def _restore_halting_counters(self, counters: dict[str, Any]) -> None:
        self._zero_gradient_rounds = counters["zero_gradient_rounds"]
        self._policy_gradient_seen = counters["policy_gradient_seen"]
        self._empty_rollout_steps = counters["empty_rollout_steps"]
        self._episode_error_rounds = counters["episode_error_rounds"]
        stop = next((cb for cb in self.callback_handler.callbacks if isinstance(cb, GRPOEarlyStopCallback)), None)
        if stop is not None:
            stop.streaks = dict(counters["early_stop_streaks"])

    def _prefetch_round_layout(self) -> dict[str, int]:
        """What a round's prompts were drawn and grouped under; a resume under another layout drops them."""
        return {
            "num_generations": self.num_generations,
            "rows_per_round": self._train_loader_batch_size(),
            "data_parallel_rank": self.parallelism_config.get_data_parallel_rank(),
            "data_parallel_size": self.parallelism_config.data_parallel_size,
        }

    def _restore_trainer_sidecars(self, checkpoint: str) -> None:
        """Rounds of another layout are dropped on every rank together. ``weights_only=False``: they carry context
        values. A smaller ``prefetch_depth`` keeps the oldest rounds, since the pipeline would never shed the
        surplus."""
        super()._restore_trainer_sidecars(checkpoint)
        payload, path = consensus_read(
            os.path.join(checkpoint, prefetch_pending_filename(get_global_rank(), get_global_world_size())),
            partial(torch.load, map_location="cpu", weights_only=False),
            what="pending prefetch rounds",
            checkpoint=checkpoint,
            remedy=" Resume with the rank-to-node placement it was saved with, or delete its "
            f"{PREFETCH_PENDING_PREFIX}-* files to open with cold rounds.",
        )
        if path is None:
            if is_global_main_process():
                prefetched = (
                    " and opens with cold rounds: if the saving run prefetched, the batches it had drawn but not "
                    "trained are skipped and the next ones trained twice"
                    if self._prefetch_enabled
                    else ""
                )
                logger.warning(
                    f"{checkpoint} holds no resume sidecar for this world size, so the halting counters "
                    f"(zero-gradient, empty-rollout and episode-error rounds, early-stop streaks) restart at zero"
                    f"{prefetched}."
                )
            return
        if "halting" in payload:
            self._restore_halting_counters(payload["halting"])
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
                    "with cold rounds."
                )
            return
        rounds, depth = list(payload["rounds"]), self.async_config.prefetch_depth
        if len(rounds) > depth and is_global_main_process():
            logger.warning(
                f"{path} holds {len(rounds)} pending prefetch rounds, more than prefetch_depth={depth}: the "
                f"oldest {depth} are resubmitted, and the newest {len(rounds) - depth} are not trained."
            )
        self._resumed_prefetch_rounds = [] if self._is_tp_follower() else rounds[:depth]

    def _cleanup_async_components(self):
        """Run by every rank."""
        # First, on every rank: the client below is the main process's alone, and a teardown step that
        # raises must not leave the ranks disagreeing on whether the next push forms the group.
        self._weight_sync_group_formed = False
        self._async_components_started = False
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

        if self._engine_rescore_clients is not None:
            for client in self._engine_rescore_clients:
                client.close_http_session()
            self._engine_rescore_clients = None

        if self._adapter_sync is not None:
            self._adapter_sync.reset()

        self._loop = None
        # Re-arm: a second train() rebuilds the components above and needs a fresh push to the
        # engines, at whatever step it starts from.
        self._rollout_generation_started = False
        self._last_sync_attempt_step = -1
        self._last_pushed_step = -1

    def _stop_prefetch_thread(self):
        """It sees its stop event within a poll slice and cancels the rounds still collecting, so a later start begins
        on an empty input queue."""
        if self._prefetch_thread is None:
            return

        self._prefetch_stop_event.set()

        with contextlib.suppress(queue.Empty):
            while not self._prefetch_input_queue.empty():
                self._prefetch_input_queue.get_nowait()

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
