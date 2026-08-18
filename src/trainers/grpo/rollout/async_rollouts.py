"""Async rollout plumbing for environmental GRPO: Ray actors, the prefetch thread, and weight sync.

Rollouts are collected by Ray actors against the rollout servers while the trainer steps; a
background prefetch thread overlaps the next round's collection with this round's update, and the
trained weights are pushed back to the engines over NCCL between steps.
"""

import asyncio
import contextlib
import logging
import queue
import threading
import time
import weakref

import ray
import torch
from accelerate.utils import is_peft_model
from transformers import TrainerCallback
from trl.extras.profiling import profiling_context

from src.distributed.nccl.registry import resolve_weight_sync_client
from src.distributed.runtime import broadcast_from_rank0, get_num_nodes
from src.environments.episode import RolloutResult
from src.environments.ray_actors import RolloutManager, ray_init_kwargs
from src.trainers.grpo.rollout.weight_sync import (
    expert_layout_for,
    sync_trainer_weights,
    sync_weights_to_client,
)
from src.trainers.grpo.rollout.weight_sync_clients import InferenceClientManager
from src.trainers.mixins.ep_introspection import named_ep_layers

logger = logging.getLogger(__name__)


# The worker frees an input slot within one collection, so an expired submit means it is wedged.
_PREFETCH_SUBMIT_TIMEOUT_S = 60.0
# Input-queue poll slice: bounds how long the worker sits on an empty queue before re-reading the
# stop event, so shutdown is not held up by a quiet round.
_PREFETCH_POLL_TIMEOUT_S = 0.5
# Output-queue delivery slice, retried until the stop event fires, so a completed batch is not dropped.
_PREFETCH_DELIVER_TIMEOUT_S = 1.0
# Grace for the worker to finish its current poll slice and exit at shutdown.
_PREFETCH_JOIN_TIMEOUT_S = 5.0


class _RolloutStartCallback(TrainerCallback):
    """Starts generation at train-begin, the first hook past the resume restore.

    ``_init_async_components`` runs before ``Trainer.train`` and therefore before the checkpoint
    restore, so a first push issued there would ship the pre-restore weights (on a resumed LoRA run,
    the zero-init adapter merged into the base). ``on_train_begin`` fires after both restore paths
    (``train``'s own and ``_prepare_for_training``'s FSDP/DeepSpeed one), still ahead of
    ``eval_on_start`` and the first generation, and on every rank, which the push's collective gather
    requires.

    Held by weak reference: the push needs the trainer only while it is training.
    """

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
    """Ray-actor rollout collection, prefetch overlap and engine weight sync for the environmental
    GRPO trainer.

    Holds ``self._rollout_manager``, ``self._weight_sync_client`` and the prefetch queues and threads,
    initialized by :meth:`_init_async_state` from the trainer's ``__init__``.
    """

    @property
    def _rollout_engine_name(self) -> str:
        """Display name of the configured rollout engine (``vLLM`` / ``SGLang``).

        This path is engine-generic, so every user-facing string in the mixin reads the name from
        here rather than hardcoding one engine.
        """
        return resolve_weight_sync_client(self.async_config.rollout_backend).BACKEND_NAME

    def _init_async_state(self):
        """Rollout-manager, weight-sync-client and prefetch state, before any of them is built."""
        self._rollout_manager = None
        self._weight_sync_client = None  # one engine client, or an InferenceClientManager over several
        self._loop = None
        # Two separate questions: which client shape to build (a configs list of any length carries
        # its own per-server url/ports) and how many engines serve (a one-entry list is one server).
        self._multi_server_mode = bool(self.async_config.rollout_server_configs)
        self._num_rollout_servers = len(self.async_config.get_server_urls())

        self._prefetch_enabled = self.async_config.enable_prefetch
        if self._prefetch_enabled and self._num_rollout_servers < 2:
            logger.warning(
                "Prefetch auto-disabled: %d rollout server configured. The one engine stops serving "
                "during weight sync, so there is nothing to overlap against — and the rolling sync "
                "prefetch selects would leave ZERO servers live instead of N-1. For prefetch "
                "benefits, configure two or more servers in rollout_server_configs.",
                self._num_rollout_servers,
            )
            self._prefetch_enabled = False

        self._prefetch_thread = None
        self._prefetch_queue = queue.Queue(maxsize=self.async_config.num_prefetch_batches)
        self._prefetch_input_queue = queue.Queue(maxsize=self.async_config.num_prefetch_batches + 1)
        self._prefetch_stop_event = threading.Event()

        self._prefetch_hits = 0
        self._prefetch_misses = 0
        # Submitted-but-unconsumed prefetch batches; lock-free — only the trainer thread mutates it.
        self._prefetch_inflight = 0
        self._prefetch_input_skips = 0
        # The step at which the sync path was last entered, which keeps it out of every microbatch of
        # one optimizer step. Not the step the engines hold: the cadence gate
        # (sync_weights_every_n_steps) declines most steps, and only the push's return value says
        # whether anything went out.
        self._last_sync_attempt_step = -1
        self._rollout_start_callback = None
        self._rollout_generation_started = False

    def _init_async_components(self):
        """Initialize Ray actors and the engine weight-sync client (main process only for the client)."""
        if self._rollout_manager is not None:
            return

        if not ray.is_initialized():
            ray.init(**ray_init_kwargs(address=self.async_config.ray_address))

        # Rollout manager on every rank; each collects rollouts for its own DP shard (per-rank pools),
        # so on a shared Ray cluster the configured actor budget is one pool split across the DP ranks.
        self._rollout_manager = RolloutManager(
            num_workers=self.async_config.num_rollout_workers,
            env_type=self._environment_spec,
            env_config=self._env_config_dict,
            server_urls=self.async_config.get_server_urls(),
            rollout_config=self.async_config.get_rollout_config(stop_token_ids=self._resolve_rollout_stop_token_ids()),
            max_concurrent_rollouts=self.async_config.max_concurrent_rollouts,
            pool_replicas=self.accelerator.num_processes if self.async_config.ray_address is not None else 1,
        )
        if self.accelerator.is_main_process:
            multinode = self.async_config.ray_address is not None or get_num_nodes() > 1
            self._rollout_manager.warn_if_servers_unreachable_from_actors(multinode)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._rollout_manager.start())

        # Main-process only; a bare rank-0 raise would leave peers blocked in the first collective gather.
        error: str | None = None
        if self.accelerator.is_main_process:
            try:
                self._init_weight_sync_client()
            except Exception as e:  # re-raised on all ranks below
                error = f"{self._rollout_engine_name} weight-sync client init failed on the main process: {e!r}"
        error = broadcast_from_rank0(error)
        if error is not None:
            raise RuntimeError(error)

        # The first push and the prefetch thread wait for train-begin (see _RolloutStartCallback):
        # this runs before the resume restore, and both would otherwise generate from pre-restore
        # weights. Registered once; a second train() re-arms it through _cleanup_async_components.
        if self._rollout_start_callback is None:
            self._rollout_start_callback = _RolloutStartCallback(self)
            self.add_callback(self._rollout_start_callback)

        logger.info(
            f"Async components initialized: "
            f"{self._rollout_manager.num_workers} Ray actors (this rank), "
            f"{self._num_rollout_servers} rollout servers, "
            f"max_concurrent={self._rollout_manager.max_concurrent}, "
            f"prefetch={'enabled' if self._prefetch_enabled else 'disabled'}, "
            f"main_process={self.accelerator.is_main_process}"
        )

    def _start_rollout_generation(self):
        """Push the trainer's weights to the engines and start the prefetch thread. Collective.

        Before any rollout the engines must serve this trainer's weights, restore included, which is
        why this is called from :class:`_RolloutStartCallback` rather than ``_init_async_components``.
        Idempotent, so a re-entry cannot push twice.
        """
        if self._rollout_generation_started:
            return
        self._rollout_generation_started = True
        # Stamped even though this push is forced: on a resumed run the first training step is at the
        # same global_step, and re-entering the path there would re-push the weights just sent.
        self._last_sync_attempt_step = self.state.global_step
        self._sync_weights_to_engine_fenced(force=True)
        if self._prefetch_enabled:
            self._start_prefetch_thread()

    def _sync_weights_to_engine_fenced(self, force: bool = False) -> bool:
        """``_sync_weights_to_engine`` with the rank-0 failure joined to every rank. Collective.

        The client lives on the main process, so a sync failure there (duplicate group port, dead
        server, trainer and engine on one GPU) would be a rank-0-only raise leaving every peer blocked
        in the next collective until the watchdog fires. Both the train-begin push and the per-step
        sync need this fence.
        """
        local_error: str | None = None
        synced = False
        try:
            synced = self._sync_weights_to_engine(force=force)
        except Exception as e:  # re-raised on all ranks below
            local_error = f"{self._rollout_engine_name} weight sync failed at step {self.state.global_step}: {e!r}"
        main_error = broadcast_from_rank0(local_error if self.accelerator.is_main_process else None)
        if main_error is not None:
            raise RuntimeError(main_error)
        if local_error is not None:
            raise RuntimeError(local_error)
        return synced

    def _init_weight_sync_client(self):
        """Initialize the engine weight-sync client(s) (main process only, idempotent).

        Called before training so the NCCL group is established while the engine is idle.
        """
        if self._weight_sync_client is not None:
            return

        if not self.accelerator.is_main_process:
            return

        # PyNcclCommunicator requires an explicit cuda:N (torch.device("cuda") != cuda:0 in comparisons).
        device = self.accelerator.device
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        client_cls = resolve_weight_sync_client(self.async_config.rollout_backend)

        if self._multi_server_mode:
            server_configs = self.async_config.rollout_server_configs
            logger.info(f"Initializing InferenceClientManager for {len(server_configs)} servers on device: {device}")

            self._weight_sync_client = InferenceClientManager(
                server_configs=server_configs,
                connection_timeout=self.async_config.rollout_connection_timeout,
                client_cls=client_cls,
                base_group_port=self.args.vllm_group_port,
            )

            self._weight_sync_client.init_communicators(device=device)

            logger.info(f"InferenceClientManager initialized: {len(server_configs)} servers, device={device}")
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
                f"device={device}, group_port={self.args.vllm_group_port}"
            )

    def _cleanup_async_components(self):
        """Cleanup Ray actors and the engine weight-sync client(s)."""
        self._stop_prefetch_thread()

        if self._rollout_manager and self._loop:
            self._loop.run_until_complete(self._rollout_manager.shutdown())
            self._rollout_manager = None

        if self._weight_sync_client is not None:
            try:
                if self._multi_server_mode:
                    self._weight_sync_client.close_communicators()
                else:
                    self._weight_sync_client.close_communicator()
            except Exception as e:  # teardown must not mask the run's own outcome
                logger.warning(f"Error closing the {self._rollout_engine_name} weight-sync client: {e}")
            self._weight_sync_client = None

        self._loop = None
        # Re-arm: a second train() rebuilds the components above and needs a fresh push to the
        # engines, at whatever step it starts from.
        self._rollout_generation_started = False
        self._last_sync_attempt_step = -1

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
        # Drained submissions can no longer produce output items; reset the in-flight view.
        self._prefetch_inflight = 0

        total = self._prefetch_hits + self._prefetch_misses
        if total > 0:
            hit_rate = self._prefetch_hits / total * 100
            logger.info(
                f"Prefetch stats: {self._prefetch_hits} hits, {self._prefetch_misses} misses "
                f"({hit_rate:.1f}% hit rate)"
            )

    def _prefetch_worker(self):
        """Background worker that prefetches rollouts.

        Every consumed submission yields exactly one output item: the collected rollouts, or a
        ``(prompt_count, None)`` failure marker when the collection raised. That keeps the
        trainer-side in-flight accounting exact and stops a blocking consumer waiting on a batch that
        will never arrive.
        """
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
        """Deliver a completed (or failed) prefetch to the output queue without discarding it: a
        dropped item would leave the trainer's in-flight accounting wrong and lose collected rollouts.
        Blocks in stop-aware slices; only shutdown abandons delivery (logged)."""
        while not self._prefetch_stop_event.is_set():
            try:
                self._prefetch_queue.put(item, timeout=_PREFETCH_DELIVER_TIMEOUT_S)
                return
            except queue.Full:
                continue
        logger.warning(f"Prefetch stopping; abandoning delivery of a {item[0]}-prompt rollout batch (shutdown)")

    def _try_get_prefetched_results(self) -> list[RolloutResult] | None:
        """Try to get prefetched results from the queue.

        A popped worker failure marker (``results is None``) counts as a miss; the in-flight counter
        drops either way (exactly one output item exists per submission — see ``_prefetch_worker``).
        """
        try:
            _count, results = self._prefetch_queue.get_nowait()
        except queue.Empty:
            self._prefetch_misses += 1
            return None
        self._prefetch_inflight -= 1
        if results is None:
            self._prefetch_misses += 1
            return None
        self._prefetch_hits += 1
        logger.debug(f"Prefetch hit: using {len(results)} prefetched rollouts")
        return results

    def _wait_for_inflight_prefetch(self) -> list[RolloutResult] | None:
        """Block for an in-flight prefetched batch (counted as a miss — the latency was not hidden).

        Called on a miss with submissions in flight: waiting trains each dataset batch once, where a
        synchronous re-collection would duplicate it. Skips failure markers and returns None when
        every in-flight submission failed or the worker produces nothing within the episode deadline;
        either way the caller falls back to synchronous collection.

        A wedged pipeline is recorded in ``_batch_build_error`` rather than raised: the prefetch
        counters and worker thread are per-rank, so a rank whose engine route is wedged would raise
        alone while its peers entered the collectives below. ``_raise_batch_error_uniformly`` then
        fails every rank together and names this one.
        """
        timeout = self.async_config.episode_timeout + _PREFETCH_SUBMIT_TIMEOUT_S
        while self._prefetch_inflight > 0:
            try:
                _count, results = self._prefetch_queue.get(timeout=timeout)
            except queue.Empty:
                if self._batch_build_error is None:
                    self._batch_build_error = (
                        f"Prefetch worker produced no rollouts within {timeout:.0f}s with "
                        f"{self._prefetch_inflight} batch(es) in flight — the prefetch pipeline is wedged "
                        f"({self._rollout_engine_name} unreachable or the worker thread died)."
                    )
                return None
            self._prefetch_inflight -= 1
            if results is not None:
                logger.info(f"Prefetch miss: waited for {len(results)} in-flight rollouts")
                return results
        return None

    def _submit_for_prefetch(
        self,
        prompts: list[str],
        contexts: list[dict | None],
    ):
        """Submit prompts for prefetching in the background.

        A skipped submission is logged as a warning: on the steady-state hit path this round's prompts
        train only via the prefetched set, so dropping them here drops them from training. The bounded
        blocking put cannot deadlock (see ``_PREFETCH_SUBMIT_TIMEOUT_S``).
        """
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
        self._prefetch_inflight += 1

    def _sync_cadence_declines(self, force: bool) -> bool:
        """Whether ``sync_weights_every_n_steps`` declines this step (``force`` overrides it).

        Both push paths gate on this, and the collective one must evaluate it identically on every
        rank or the ranks split over a gather.
        """
        return not force and self.state.global_step % self.async_config.sync_weights_every_n_steps != 0

    def _sync_weights_to_engine_single(self, force: bool = False) -> bool:
        """Sync model weights to the rollout engine via NCCL (main process only); ``force`` ignores the step gate.

        Multi-server mode syncs one server at a time, keeping (N-1) available for generation during
        the sync. Returns whether weights were pushed, so the caller does not record a sync the
        cadence gate declined.
        """
        if not self.accelerator.is_main_process:
            return False

        if self._sync_cadence_declines(force):
            return False

        self._init_weight_sync_client()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Rolling sync uses the raw-model client API, which forwards ``named_parameters()``
        # untouched, so it is reachable for plain adapter-free dense models only.
        if self._multi_server_mode and not named_ep_layers(unwrapped_model) and not is_peft_model(unwrapped_model):
            self._weight_sync_client.update_model_params(unwrapped_model)
            num_servers = self._weight_sync_client.num_servers
            logger.debug(f"Synced weights to {num_servers} rollout servers (rolling) at step {self.state.global_step}")
        else:
            # One gather for every remaining case: the EP-wrapped layouts the engine cannot map (EP
            # wrappers are present even at ep_size==1), PEFT (the gather merges the adapters and the
            # engine serves the merged base), and plain dense, where ``update_model_params`` would
            # forward raw ``named_parameters()`` — which FSDP2 hands out as DTensors describing a
            # local shard under the global shape, so the broadcast reads past the shard.
            sync_weights_to_client(
                unwrapped_model,
                self._weight_sync_client,
                is_main=True,
                is_tp_main=True,
                expert_layout=expert_layout_for(self),
            )
            logger.debug(f"Synced weights to the {self._rollout_engine_name} engine at step {self.state.global_step}")
        return True

    def _sync_weights_to_engine(self, force: bool = False) -> bool:
        """Sync weights to the rollout engine with EP/TP/FSDP/PEFT awareness; ``force`` ignores the step gate.

        Single-process runs use the main-process-only single path; EP/TP/ETP and any multi-rank run
        route through the all-ranks collective gather. Returns whether weights were pushed, which is
        what the caller records: the cadence gate declines most steps, so a stamp taken on the attempt
        would claim the engines hold weights that were never sent.
        """
        config = self.parallelism_config

        # full_tensor() on an FSDP2 param is a DP-mesh collective: every rank must enter, or main-only deadlocks.
        needs_collective_gather = (
            config.is_ep_mode or config.is_tp_mode or config.is_expert_tp_mode or self.accelerator.num_processes > 1
        )

        if not needs_collective_gather:
            with profiling_context(self, "weight_sync"):
                return self._sync_weights_to_engine_single(force=force)

        # The step gate must evaluate identically on all ranks: the collectives need all of them.
        if self._sync_cadence_declines(force):
            return False

        # Called on every rank of the collective path: the method is main-process-gated and
        # idempotent, so a caller-side rank gate here would put the fence on the wrong side of a push
        # that every rank has to enter.
        self._init_weight_sync_client()

        with profiling_context(self, "weight_sync"):
            sync_trainer_weights(self, self._weight_sync_client)
        return True
