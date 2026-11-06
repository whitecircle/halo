"""Async environmental GRPO trainer: multi-turn rollouts from Ray actors over vLLM or SGLang, weights synced
back over NCCL. No CP (``logits_to_keep`` and global log-prob sums)."""

from typing import Any

import torch
from trl import GRPOTrainer
from trl.extras.profiling import profiling_context

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.diagnostics.phase_timers import Phase
from halo.environments.environment_base import EnvironmentBase
from halo.environments.messages import task_prompt
from halo.environments.reasoning_effort import has_thinking_budget, resolve_reasoning_effort, stable_reasoning_effort
from halo.environments.rollout_config import RANDOM_REASONING_EFFORT, resolve_rollout_stop_token_ids
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.models.unwrapping import unwrap_framework_wrappers
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.environmental.async_rollouts import AsyncRolloutMixin
from halo.trainers.grpo.environmental.environmental_rules import (
    reject_answerless_datasets,
    reject_inert_reasoning_terms,
    reject_off_policy_mask_threshold,
    reject_open_ended_tool_turns,
    reject_prefetch_past_retry_budget,
    reject_straddling_eval_groups,
    reject_straddling_train_groups,
    reject_unread_grpo_knobs,
    reject_unresolved_forced_closes,
    reject_unverified_carried_reasoning,
    reject_vespo_loss,
)
from halo.trainers.grpo.environmental.importance_sampling import ImportanceSamplingMixin
from halo.trainers.grpo.environmental.loss_forward_weights import LossForwardImportanceMixin
from halo.trainers.grpo.environmental.packed_rounds import PackedMicrobatchMixin
from halo.trainers.grpo.environmental.reasoning import ReasoningBudgetMixin, reasoning_token_counts
from halo.trainers.grpo.environmental.rollout_metrics import RolloutMetricsMixin
from halo.trainers.grpo.environmental.routing_replay import RoutingReplayMixin
from halo.trainers.grpo.environmental.row_order import LengthOrderedRowsMixin
from halo.trainers.grpo.environmental.step_batch import BatchBuildFence, StepBatchMixin
from halo.trainers.grpo.environmental.tool_call_watch import ToolCallParserWatch, build_tool_call_watch
from halo.trainers.grpo.environmental.trajectory_tokenize import TrajectoryTokenizeMixin
from halo.trainers.grpo.grpo_rules import reject_inert_std_floor, reject_unbalanceable_token_mass
from halo.trainers.grpo.logprob_scoring import ChunkedGRPOLogprobsMixin, LogitsWidth, ProtectedTokenEntropyMixin
from halo.trainers.grpo.on_policy import GenerationRoundMixin, OnPolicyGRPOInitMixin
from halo.trainers.grpo.phase_timing import PhaseTimedMixin
from halo.trainers.grpo.rollout.completions_logging import DecoupledCompletionsLogMixin, unbounded_completion_logs
from halo.trainers.grpo.row_span_stash import RowSpanStashMixin
from halo.trainers.training_args_overrides import keep_all_dataset_columns
from halo.trainers.world_metrics import WorldMetrics
from halo.weight_sync.registry import resolve_weight_sync_client


def _resolve_environment_spec(environment_config, environment_cls, environment_kwargs) -> tuple[Any, dict]:
    if environment_config is not None:
        return environment_config.environment_type, environment_config.to_env_config()
    if environment_cls is not None:
        if not (isinstance(environment_cls, type) and issubclass(environment_cls, EnvironmentBase)):
            raise TypeError(
                f"environment_cls must be a subclass of EnvironmentBase, got {environment_cls}. "
                "Ensure your environment class inherits from halo.environments.environment_base.EnvironmentBase."
            )
        return (environment_cls, environment_kwargs or {}), {}
    raise ValueError(
        "Either environment_config (EnvironmentConfig from YAML) or "
        "environment_cls (custom EnvironmentBase subclass) is required."
    )


def placeholder_reward_func(prompts, _completions, **_kwargs) -> list[float]:
    """``GRPOTrainer``'s required reward function; the environment's rewards go straight into the batch."""
    return [0.0] * len(prompts)


class DistributedAsyncEnvironmentalGRPOTrainer(
    OnPolicyGRPOInitMixin,
    StepBatchMixin,
    PackedMicrobatchMixin,
    LengthOrderedRowsMixin,
    ImportanceSamplingMixin,
    ReasoningBudgetMixin,
    RoutingReplayMixin,
    AsyncRolloutMixin,
    RolloutMetricsMixin,
    TrajectoryTokenizeMixin,
    GenerationRoundMixin,
    LossForwardImportanceMixin,
    PhaseTimedMixin,
    ProtectedTokenEntropyMixin,
    ChunkedGRPOLogprobsMixin,
    RowSpanStashMixin,
    DecoupledCompletionsLogMixin,
    DistributedTrainerMixin,
    GRPOTrainer,
):
    """Its environment is given as ``environment_config`` or ``environment_cls`` + ``environment_kwargs``."""

    # The batches are built text-only: the build assembles a fixed set of text tensors.
    _SCORED_BATCHES_CARRY_IMAGES = False

    PP_REFUSAL = (
        "inherits online GRPO's cross-stage engine weight-sync and rollout-phase forward blockers, and "
        "adds Ray-driven multi-turn async rollouts whose generation lengths vary per turn — the "
        "pipeline freezes its boundary activation shape on the first step"
    )
    # Declared so a trainer built without __init__ still reads a value on the tokenize and arm paths.
    _rollout_backend: str | None = None
    _rollout_routing_replay = False
    _replay_row_spans = False
    _tool_call_watch: ToolCallParserWatch | None = None

    def __init__(self, *args, **kwargs):
        self.async_config = kwargs.pop("async_config", None) or AsyncRolloutConfig()
        self._use_chunked_grpo_logprobs = self.async_config.use_chunked_grpo_logprobs
        self._log_phase_times = self.async_config.log_phase_times
        _, kwargs = self._begin_on_policy_init(args, kwargs)
        self._environment_spec, self._env_config_dict = _resolve_environment_spec(
            kwargs.pop("environment_config", None),
            kwargs.pop("environment_cls", None),
            kwargs.pop("environment_kwargs", None),
        )
        # Read off the env instance, not the config dict: the class owns its defaults and validation.
        self._group_random_effort = self._rollout_env.reasoning_effort == RANDOM_REASONING_EFFORT
        reject_unverified_carried_reasoning(self._rollout_env, self.async_config.rollout_backend)
        kwargs.setdefault("reward_funcs", placeholder_reward_func)
        with self._supplying_kl_reference():
            super().__init__(*args, **kwargs)

        # TRL exposes no self.{pad,eos}_token_id, and a VLM processing_class nests the real tokenizer;
        # TRL's __init__ has already set a missing pad token to eos on it.
        self._tokenizer = resolve_tokenizer(self.processing_class)
        self.pad_token_id = self._tokenizer.pad_token_id
        self.eos_token_id = self._tokenizer.eos_token_id
        self._batch_errors = BatchBuildFence()
        self._empty_rollout_steps: int = 0
        # Fail at construction, not mid-training: the tokenize-time overflow check needs a real limit.
        self._context_limit()
        reject_straddling_train_groups(
            self.args.per_device_train_batch_size, self.args.steps_per_generation, self.num_generations
        )
        reject_straddling_eval_groups(self.args, self.async_config, self.num_generations_eval)
        # The rollout context IS the row minus ``prompt``: pruning would drop ``answer`` and the context fields.
        keep_all_dataset_columns(
            self.args,
            "Environmental GRPO requires remove_unused_columns=False (the rollout context is the row's "
            "non-prompt columns, which pruning would strip); forcing it off. Set it explicitly in your GRPOConfig.",
        )
        reject_answerless_datasets(self._rollout_env, self.train_dataset, self.eval_dataset)
        reject_inert_reasoning_terms(
            self.async_config,
            self._rollout_env,
            budgeted=has_thinking_budget(self._efforts_by_level()),
            every_budget_soft=self._every_budget_soft,
        )
        self._configure_rollout_state()
        self._configure_importance_sampling()
        reject_unread_grpo_knobs(self.args)
        self._init_async_state()
        self._finish_on_policy_init()
        self._configure_phase_timers()
        self._configure_row_span_stash()
        # After _configure_distributed_modes (needs the EP layers); eager, so an unsupported setup fails now.
        self._routing_injector = self._build_routing_replay_injector(self.async_config.routing_replay)
        # After the model is placed: the deal reads its attention implementation.
        self._resolve_length_ordered_rows()
        self._configure_packed_microbatches()
        self._configure_loss_forward_weights()

    def _configure_rollout_state(self) -> None:
        cfg = self.async_config
        self._rollout_backend = cfg.rollout_backend
        self._rollout_stop_token_ids = resolve_rollout_stop_token_ids(self._tokenizer, cfg.rollout_stop_tokens)
        reject_open_ended_tool_turns(
            self._rollout_env, unwrap_framework_wrappers(self.model), self._tokenizer, self._rollout_stop_token_ids
        )
        self._reasoning_end_ids = self._resolve_reasoning_end_ids()
        self._forced_close_ids = self._resolve_forced_close_ids(self._reasoning_end_ids)
        reject_unresolved_forced_closes(
            self._any_episode_capped, self._reasoning_end_ids, cfg.rollout_backend, cfg.rollout_reasoning_end_token
        )
        self._rows_over_cap = 0
        self._logs = unbounded_completion_logs()
        self._world_metrics = WorldMetrics()
        reject_inert_std_floor(self.args.scale_rewards, cfg.scale_rewards_std_floor)
        reject_prefetch_past_retry_budget(resolve_weight_sync_client(cfg.rollout_backend), cfg, self.args)
        # Ahead of the balance gate, which refuses both beside it: here each is refused on its own.
        reject_off_policy_mask_threshold(self.args)
        reject_vespo_loss(self.args)
        if cfg.balance_token_mass:
            reject_unbalanceable_token_mass(self.args)
        self._arm_update_breaker()
        self._warned_once: set[str] = set()
        self._isr_mask_config = cfg.build_isr_mask_config()
        self._tool_call_watch = build_tool_call_watch(
            self._rollout_env, self._tokenizer, cfg.rollout_chat_template_kwargs or {}, cfg.tool_parser_watch_rounds
        )

    def train(self, *args, **kwargs):
        try:
            self._init_async_components()
            return super().train(*args, **kwargs)
        finally:
            self._cleanup_async_components()

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Gated on the attempt stamp, not the synced one: the cadence gate declines most steps, and
        # re-entering the path on every microbatch of a declined step costs a collective and a barrier.
        if self.state.global_step > 0 and self.state.global_step != self._last_sync_attempt_step:
            self._last_sync_attempt_step = self.state.global_step
            self._sync_weights_to_engine_fenced()
        if self._routing_injector is None:
            return super().training_step(model, inputs, num_items_in_batch)
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        finally:
            # Disarm only after backward (the reentrant-GC recompute reads the armed mask during it).
            self._routing_injector.disarm()

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """The TP/ETP leader's batch on every group rank: ranks sample apart, and compute_loss expects
        DTensor-consistent inputs."""
        return self._with_row_spans(self._broadcast_tensors_from_tp_leader(self._collect_and_score_rollouts(inputs)))

    def _collect_and_score_rollouts(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "eval" if not self.model.training else "train"
        # Rows past the split's own are never rolled out: they enter the batch as inert padding.
        episodes = self.eval_split_rows(len(inputs))
        # One rollout per row: TRL's RepeatSampler already expanded each prompt num_generations times.
        expanded_prompts, expanded_contexts = self._extract_prompts_and_contexts(inputs[:episodes])
        # Before the rollout: an unusable prompt would cost a full round and fail first inside a Ray actor.
        self._batch_errors.reject()

        with profiling_context(self, "rollout_acquire"), self._phase_timers.span(Phase.ROLLOUT, "host"):
            rollout_results = None
            if self._prefetch_enabled and mode == "train":
                rollout_results = self._acquire_prefetched_round()

            # A wedged prefetch is fenced before the synchronous fallback: peers that hit are already
            # waiting here, and a full fallback round can outlast DIST_NCCL_TIMEOUT_MINUTES.
            self._batch_errors.reject()
            if rollout_results is None:
                rollout_results = self._loop.run_until_complete(
                    self._rollout_manager.collect_rollouts(expanded_prompts, expanded_contexts)
                )

        # Popped prefetch_depth steps later, that many steps stale (the IS ratio corrects it); a cold round
        # rolls out twice.
        if self._prefetch_enabled and mode == "train":
            self._submit_for_prefetch(expanded_prompts, expanded_contexts)

        # Every TP/ETP group rank tokenizes the leader's rollouts, or the in-forward collectives deadlock.
        rollout_results = self._broadcast_object_from_tp_leader(rollout_results)
        if self._tool_call_watch is not None and self._tool_call_watch.armed:
            self._tool_call_watch.observe(rollout_results)
        # Counted once: the reasoning terms price these and the rollout metrics slice them.
        reasoning_tokens = [reasoning_token_counts(self._tokenizer, r.trajectory) for r in rollout_results]

        with profiling_context(self, "build_training_tensors"), self._phase_timers.span(Phase.BUILD, "host"):
            result = self._build_training_tensors(
                rollout_results, device, mode, len(inputs) - episodes, reasoning_tokens
            )
        self._log_rollout_metrics(rollout_results, mode, reasoning_tokens)
        return result

    def _extract_prompts_and_contexts(self, inputs: list[dict]) -> tuple[list[str], list[dict | None]]:
        """The task text for the environment, which builds the messages, and the other columns as rollout context; an
        unusable prompt is recorded for the caller's fence."""
        prompts = []
        contexts = []
        for inp in inputs:
            try:
                prompt_text = task_prompt(inp["prompt"])
            except ValueError as exc:
                self._batch_errors.record(f"Environmental GRPO row: {exc}")
                prompt_text = ""
            prompts.append(prompt_text)
            ctx = {k: v for k, v in inp.items() if k != "prompt"}
            contexts.append(ctx if ctx else None)
        if self._group_random_effort:
            self._stamp_group_efforts(prompts, contexts)
        return prompts, contexts

    def _stamp_group_efforts(self, prompts: list[str], contexts: list[dict | None]) -> None:
        """One draw per generation group, since a per-episode draw would enter the advantage, which compares siblings
        under one conditioning. Evaluation draws a group's level from its problem."""
        group = (self.num_generations if self.model.training else self.num_generations_eval) or 1
        if len(contexts) % group != 0:
            self._batch_errors.record(
                f"rollout context count ({len(contexts)}) is not a multiple of the group size "
                f"({group}); group-level reasoning-effort draws require whole groups per rank. "
                f"In eval this means the eval round (eval_rollout_batch_size, else the eval batch) "
                f"does not divide by num_generations_eval."
            )
            return
        for start in range(0, len(contexts), group):
            level = (
                resolve_reasoning_effort(RANDOM_REASONING_EFFORT)
                if self.model.training
                else stable_reasoning_effort(prompts[start])
            )
            for i in range(start, start + group):
                ctx = contexts[i] or {}
                ctx["reasoning_effort"] = level
                contexts[i] = ctx

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """The routing mask stays armed through the backward, whose reentrant-GC recompute reads it;
        :meth:`training_step` disarms. A no-op slot of a packed window returns before arming, as the mask it would arm
        has no forward to consume it."""
        if self._is_noop_slot(model, inputs):
            return self._noop_loss()
        if self._routing_injector is not None and model.training and isinstance(inputs, dict):
            self._arm_loss_forward_replay(inputs)
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "eval" if not self.model.training else "train"
        # World reductions, gated on the config and the mode alone.
        if self._routing_injector is not None and mode == "train":
            flip = self._routing_injector.drain_flip_rate()
            if flip is not None:
                logs["routing/replay_flip_rate"] = flip
        if self._rollout_manager:
            logs.update(self.cumulative_rollout_metrics())
        if self._prefetch_enabled:
            logs.update(self.prefetch_metrics())
        super().log(logs, start_time)

    def _loss_logits_width(self) -> LogitsWidth:
        """A per-turn row completes within one turn's cap, a whole-trajectory row runs to the row cap; one logit past
        the completion is kept for the shift. Per turn this is no bound, as a trajectory that lost its ids falls back
        to one whole row."""
        cfg = self.async_config
        if cfg.train_on_sampled_tokens:
            return LogitsWidth(cfg.rollout_max_tokens + 1, "rollout_max_tokens")
        if cfg.max_train_row_tokens is not None:
            return LogitsWidth(cfg.max_train_row_tokens + 1, "max_train_row_tokens")
        context = self._context_limit()
        return LogitsWidth(context + 1, f"max_train_row_tokens (unset: the {context}-token context window)")

    def _eval_loader_batch_size(self) -> int:
        """``eval_rollout_batch_size`` sizes an eval round, one synchronous rollout as slow as its slowest
        episode; the loss forward still chunks by ``per_device_eval_batch_size``."""
        return self.async_config.eval_rollout_batch_size or super()._eval_loader_batch_size()
