"""Distributed async environmental GRPO trainer (EP/TP-aware).

Multi-turn env RL: Ray actors collect rollouts over HTTP from the rollout engine (vLLM or SGLang); the
trainer computes the loss and syncs weights back to the engine over NCCL. CP unsupported
(``logits_to_keep`` + global log-prob sums).
"""

import contextlib
import dataclasses
import math
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any, get_args

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.logging import get_logger
from accelerate.utils import gather
from transformers import TrainerCallback
from trl import GRPOTrainer
from trl.extras.profiling import profiling_context
from trl.trainer.utils import pad

from src.configs.async_training_config import AsyncTrainingConfig
from src.configs.rollout_config import IDENTITY_SAMPLER
from src.distributed.nccl.registry import resolve_weight_sync_client
from src.distributed.runtime import (
    DeferredRankFailure,
    get_global_rank,
    is_multi_rank_run,
    reject_across_ranks,
)
from src.environments.base import (
    ANSWER_KEY,
    EPISODE_INVALID_REASON_KEY,
    RANDOM_REASONING_EFFORT,
    VALID_REASONING_EFFORTS,
    BaseEnvironment,
    resolve_reasoning_effort,
    solve_verdict,
    stable_reasoning_effort,
    task_prompt,
)
from src.environments.engine_wire import SGLANG_BACKEND
from src.environments.episode import (
    RolloutResult,
    resolve_reasoning_end_ids,
    resolve_reasoning_end_token_id,
    resolve_rollout_stop_token_ids,
    thinking_caps_by_level,
)
from src.models.structure import resolve_tokenizer
from src.trainers.grpo.early_stop import build_early_stop_callback
from src.trainers.grpo.mixins.chunked_logprobs import (
    ChunkedGRPOLogprobsMixin,
    LogitsWidth,
    dense_row_spans,
    rows_forward_densely,
)
from src.trainers.grpo.mixins.dataloader import GRPOTrainDataLoaderMixin
from src.trainers.grpo.mixins.entropy_mask import ProtectedTokenEntropyMixin
from src.trainers.grpo.mixins.generation_buffer import GRPOGenerationBufferMixin
from src.trainers.grpo.mixins.on_policy_init import OnPolicyGRPOInitMixin
from src.trainers.grpo.objective.advantages import (
    degenerate_group_mask,
    group_relative_advantages,
    reject_inert_std_floor,
    valid_group_stats,
)
from src.trainers.grpo.objective.application import (
    DEGENERATE_GROUP_FRAC_KEY,
    expand_traj_to_rows,
    gathered_num_items,
    narrow_loss_masks,
    record_token_mass,
    validate_token_mass_balance,
)
from src.trainers.grpo.objective.logratio import (
    KL_CLAMP_FRAC_KEY,
    LOGRATIO_MEAN_KEY,
    UPDATE_SKIPPED_KEY,
    apply_is_masks,
    apply_opsm,
    clamp_ref_logps,
    compute_is_ratio,
    kl_clamp_counts,
    sampled_token_mask,
    sampler_certain_mask,
    select_mask_logratio,
    zero_engine_forced_closes,
)
from src.trainers.grpo.reasoning_terms import (
    reasoning_floor_term,
    reasoning_price_term,
    reasoning_token_counts,
    turn_overlong_term,
)
from src.trainers.grpo.reference_policy import reference_policy
from src.trainers.grpo.rollout.async_rollouts import AsyncRolloutMixin
from src.trainers.grpo.rollout.completions_logging import DecoupledCompletionsLogMixin, unbounded_completion_logs
from src.trainers.grpo.rollout.rollout_metrics import RolloutMetricsMixin, group_solve_counts
from src.trainers.grpo.rollout.routing_replay import (
    ROLLOUT_COVERAGE_SHAPES,
    ROUTING_MASKS_KEY,
    RoutingReplayInjector,
    assemble_rollout_masks,
    build_routing_replay_injector,
)
from src.trainers.grpo.rollout.trajectory_tokenize import TrajectoryTokenizeMixin, TurnRouting, single_trajectory_row
from src.trainers.grpo.world_metrics import WorldMetrics, gathered_fractions
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.loss_masks import effective_loss_mask
from src.trainers.mixins.validation import evaluation_runs, keep_all_dataset_columns

logger = get_logger(__name__, log_level="INFO")

# Derived from the config's own Literal annotation, so the trainer's gate and the parse-time one can
# never drift apart.
ROUTING_REPLAY_MODES: tuple[str, ...] = get_args(AsyncTrainingConfig.__annotations__["routing_replay"])

# TRL sampling knobs the rollout servers control here: generation runs in the environment actors
# from RolloutConfig, so a value set on GRPOConfig never reaches a sampler. ``temperature`` is the
# one exception, reconciled the other way round: the trainer scores at the sampling temperature.
_ROLLOUT_OWNED_SAMPLING_KNOBS = ("top_p", "top_k", "min_p", "repetition_penalty", "generation_kwargs")
# Concurrent re-score prefills per rank under isr_engine_reference (each is one HTTP request in flight).
_ENGINE_RESCORE_CONCURRENCY = 16

# Consecutive steps with no valid episode anywhere before the run is halted. One such step can be a
# blip (a grader outage, a restarting engine); a second means the rollout path is down and every
# further step would train on an all-masked batch.
EMPTY_ROLLOUT_STEP_LIMIT = 2


@dataclasses.dataclass(frozen=True)
class BatchRows:
    """This step's rollouts and the training rows they became.

    Everything a per-trajectory tensor needs to be laid out over rows travels together: how many turn
    rows each trajectory contributed, how many masked padding rows this rank added to match its
    peers, and whether the per-turn split happened at all. One value, rather than the same three
    parameters repeated down every phase helper. ``turns_per_traj`` runs past ``rollout_results`` by
    the padding trajectories of an eval split's final round, one inert row each.
    """

    rollout_results: list[RolloutResult]
    turns_per_traj: list[int]
    num_dummy_rows: int
    per_turn: bool

    def to_rows(self, values: torch.Tensor, dummy_fill: float = 0) -> torch.Tensor:
        """Per-trajectory ``values`` laid out over this step's rows (see :func:`expand_traj_to_rows`)."""
        return expand_traj_to_rows(values, self.turns_per_traj, self.num_dummy_rows, self.per_turn, dummy_fill)

    def traj_row_ids(self, device: torch.device) -> torch.Tensor:
        """Each row's index into ``rollout_results``; -1 on the padding and dummy rows, which are no rollout's."""
        ids = torch.arange(len(self.rollout_results), device=device)
        return self.to_rows(F.pad(ids, (0, len(self.turns_per_traj) - len(ids)), value=-1), dummy_fill=-1)


def rollout_valid_mask(rollout_results: list[RolloutResult], device: torch.device) -> torch.Tensor:
    """Per-rollout bool mask of :attr:`RolloutResult.counts_toward_baseline`."""
    return torch.tensor([r.counts_toward_baseline for r in rollout_results], device=device, dtype=torch.bool)


def env_reward_func(prompts, _completions, **_kwargs) -> list[float]:
    """Stub reward function satisfying ``GRPOTrainer``'s required ``reward_funcs``: the real rewards
    come from the environment and are written straight into the scored batch."""
    return [0.0] * len(prompts)


def batch_reward_std(rewards: torch.Tensor) -> float:
    """Sample std of a gathered reward batch; 0.0 when there is a single reward.

    ``Tensor.std()`` is ``correction=1``, so a 1-element batch (one prompt on one rank, or a
    heavily dropped step) returns NaN and poisons the mean of the whole logging window.
    """
    return rewards.std().item() if rewards.numel() > 1 else 0.0


def within_group_reward_std(rewards: torch.Tensor, valid: torch.Tensor, num_generations: int) -> float:
    """Mean over the groups of the reward std the group-scaled advantage divides by
    (:func:`valid_group_stats`): over each group's valid members, 0 where fewer than two. Read on the
    host, once per generation round."""
    return valid_group_stats(rewards, num_generations, valid)[1].mean().item()


def reject_off_policy_mask_threshold(args) -> None:
    """Refuse TRL's ``off_policy_mask_threshold``: its mask reads ``sampling_per_token_logps``, a batch
    key this trainer never emits, so TRL falls back to ``old_per_token_logps`` — ``None`` at
    ``num_iterations=1``, where the KL it thresholds is identically 0 and every token passes. The
    sequence mask against the sampling policy here is ``isr_opsm_delta``."""
    if args.off_policy_mask_threshold is not None:
        raise ValueError(
            f"off_policy_mask_threshold={args.off_policy_mask_threshold} is a silent no-op in environmental "
            "GRPO: TRL's mask reads sampling_per_token_logps, which this trainer's batch never carries, so it "
            "thresholds a KL of exactly 0. Use isr_opsm_delta (DeepSeek-V3.2 off-policy sequence masking "
            "against the engine's sampling log-probs) instead."
        )


class BatchBuildFence(DeferredRankFailure):
    """The rank-uniform fence of the environmental trainer's batch construction, reused every round.

    A writer records a fatal failure rather than raising it: it need not know whether the failure is
    symmetric across ranks, and a rank raising ahead of the fence leaves its peers in the next collective
    until the NCCL watchdog. :meth:`reject` raises ``ValueError`` on every rank when any recorded one,
    naming the first failing rank and its cause. The trainer fences in the phase that can still explain
    each failure: before the rollout (prompt shapes), after the prefetch wait and before any synchronous
    fallback (a wedged prefetch), after tokenization (row shapes) and in the routing-replay assembly.
    """

    def __init__(self) -> None:
        super().__init__("Environmental GRPO batch construction", exc_type=ValueError)

    def record(self, message: str) -> None:
        """Keep the first failure: a later one on the same batch would replace the root cause."""
        if self.reason is None:
            self.reason = message

    def reject(self) -> None:
        """COLLECTIVE. Raise on every rank if any recorded a failure; this rank's starts the next round clear."""
        reason, self.reason = self.reason, None
        reject_across_ranks(reason, self.what, exc_type=self.exc_type)


class _BreakerOptimizerSkipCallback(TrainerCallback):
    """Turns a tripped trust-region breaker into a skipped optimizer step
    (:meth:`DistributedAsyncEnvironmentalGRPOTrainer._skip_optimizer_step_if_breaker_tripped`).
    Pre-optimizer-step is the one hook between gradient clipping and ``optimizer.step``. Weak
    reference: the trainer owns the callback handler that owns this."""

    def __init__(self, trainer):
        self._trainer = weakref.ref(trainer)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        trainer = self._trainer()
        if trainer is not None:
            trainer._skip_optimizer_step_if_breaker_tripped()


class DistributedAsyncEnvironmentalGRPOTrainer(
    OnPolicyGRPOInitMixin,
    AsyncRolloutMixin,
    RolloutMetricsMixin,
    TrajectoryTokenizeMixin,
    GRPOTrainDataLoaderMixin,
    GRPOGenerationBufferMixin,
    ProtectedTokenEntropyMixin,
    ChunkedGRPOLogprobsMixin,
    DecoupledCompletionsLogMixin,
    DistributedTrainerMixin,
    GRPOTrainer,
):
    """Async environmental GRPO with Expert/Tensor Parallelism.

    Environment specified via kwargs: ``environment_config`` (from the registry) or
    ``environment_cls`` + ``environment_kwargs`` (a custom ``BaseEnvironment`` subclass). Distributed
    setup via ``parallelism_config`` consumed by ``DistributedTrainerMixin``; rest flow to ``GRPOTrainer``.
    """

    _supports_pp = False
    _pp_unsupported_reason = (
        "inherits online GRPO's cross-stage engine weight-sync and rollout-phase forward blockers, and "
        "adds Ray-driven multi-turn async rollouts whose generation lengths vary per turn — the "
        "pipeline freezes its boundary activation shape on the first step"
    )
    # Set from async_config in __init__; declared here so the tokenize path's engine-specific remedy
    # reads a value on any trainer, constructed or not.
    _rollout_backend: str | None = None
    # The objective never passes labels into the forward, so Liger CE/FLCE cannot fire.
    _loss_outside_model_forward = True

    # Defined before (or without) _setup_routing_replay so the tokenize/arm paths always read a value.
    _rollout_routing_replay = False
    _replay_row_spans = False

    def __init__(self, *args, **kwargs):
        _, kwargs = self._begin_on_policy_init(args, kwargs)

        async_config = kwargs.pop("async_config", None)
        environment_config = kwargs.pop("environment_config", None)
        environment_cls = kwargs.pop("environment_cls", None)
        environment_kwargs = kwargs.pop("environment_kwargs", None)

        self.async_config = async_config or AsyncTrainingConfig()
        # Read by ChunkedGRPOLogprobsMixin (avoids full [B,T,vocab] logits).
        self._use_chunked_grpo_logprobs = self.async_config.use_chunked_grpo_logprobs

        if environment_config is not None:
            self._environment_spec = environment_config.environment_type
            self._env_config_dict = environment_config.to_env_config()
        elif environment_cls is not None:
            if not (isinstance(environment_cls, type) and issubclass(environment_cls, BaseEnvironment)):
                raise TypeError(
                    f"environment_cls must be a subclass of BaseEnvironment, got {environment_cls}. "
                    "Ensure your environment class inherits from src.environments.base.BaseEnvironment."
                )
            self._environment_spec = (environment_cls, environment_kwargs or {})
            self._env_config_dict = {}
        else:
            raise ValueError(
                "Either environment_config (EnvironmentConfig from YAML) or "
                "environment_cls (custom BaseEnvironment subclass) is required."
            )

        # Read off the env instance the tokenize path also renders through, never re-derived from the
        # config dict: the class owns its defaults and its validation.
        self._group_random_effort = self._rollout_env.reasoning_effort == RANDOM_REASONING_EFFORT
        self._reject_unverified_carried_reasoning()

        # Rewards come from the environment, so the GRPOTrainer reward function is a stub.
        kwargs.setdefault("reward_funcs", env_reward_func)

        super().__init__(*args, **kwargs)

        # TRL exposes no self.{pad,eos}_token_id, and a VLM processing_class nests the real tokenizer;
        # TRL's __init__ has already set a missing pad token to eos on it.
        self._tokenizer = resolve_tokenizer(self.processing_class)
        self.pad_token_id = self._tokenizer.pad_token_id
        self.eos_token_id = self._tokenizer.eos_token_id
        self._batch_errors = BatchBuildFence()
        # Consecutive world-wide all-invalid rollout steps; _check_step_has_valid_episodes halts the
        # run at EMPTY_ROLLOUT_STEP_LIMIT.
        self._empty_rollout_steps: int = 0
        # Fail at construction, not mid-training: the tokenize-time overflow check needs a real limit.
        self._context_limit()

        # TRL only validates the global generation_batch_size % num_generations.
        per_rank_rows = self.args.per_device_train_batch_size * self.args.steps_per_generation
        if per_rank_rows % self.num_generations != 0:
            raise ValueError(
                f"per_device_train_batch_size * steps_per_generation ({per_rank_rows}) must be divisible "
                f"by num_generations ({self.num_generations}): environmental GRPO groups advantages "
                f"per rank, so a group cannot straddle ranks."
            )

        self._validate_eval_round()
        self._force_full_dataset_columns()
        self._reject_answerless_datasets()
        self._validate_reasoning_terms()

        self._train_on_sampled_tokens = self.async_config.train_on_sampled_tokens
        self._rollout_backend = self.async_config.rollout_backend
        self._rollout_template_kwargs = self.async_config.rollout_chat_template_kwargs
        self._rollout_stop_token_ids = resolve_rollout_stop_token_ids(
            self._tokenizer, self.async_config.rollout_stop_tokens
        )
        self._forced_close_ids = self._resolve_forced_close_ids()
        self._max_train_row_tokens = self.async_config.max_train_row_tokens
        self._rows_over_cap = 0
        # An eval round is the whole eval set; TRL's one-generation-batch buffer would keep its tail only.
        self._logs = unbounded_completion_logs()
        self._world_metrics = WorldMetrics()
        self._truncation_alarm_rate = self.async_config.truncation_alarm_rate
        self._drop_degenerate_groups = self.async_config.drop_degenerate_groups
        # Both range-validated by AsyncTrainingConfig._validate_ranges (finiteness included).
        self._scale_rewards_std_floor = self.async_config.scale_rewards_std_floor
        reject_inert_std_floor(self.args.scale_rewards, self._scale_rewards_std_floor)
        # Ahead of the balance gate, which refuses the mask beside it: here the mask is refused on its own.
        reject_off_policy_mask_threshold(self.args)
        self._balance_token_mass = self.async_config.balance_token_mass
        if self._balance_token_mass:
            validate_token_mass_balance(self.args)
        self._arm_update_breaker()
        # Keys of the once-per-run warnings already issued (warn_once), the tokenize mixin's included.
        self._warned_once: set[str] = set()

        # Mask/veto stages on the engine->trainer IS ratio (all default off).
        self._is_mask_config = self.async_config.build_is_mask_config()

        # Score log-probs at the engine's sampling temperature, or the IS ratio picks up a systematic bias.
        if self.temperature != self.async_config.rollout_temperature:
            logger.info(
                f"Scoring log-probs at the sampling temperature: temperature "
                f"{self.temperature} -> rollout_temperature {self.async_config.rollout_temperature}"
            )
            self.temperature = self.async_config.rollout_temperature
            self.args.temperature = self.async_config.rollout_temperature

        # Truncated per-token ratio vs the captured sampling logprobs, so it needs train_on_sampled_tokens.
        self._is_correction = self._train_on_sampled_tokens and self.vllm_importance_sampling_correction
        if self.vllm_importance_sampling_correction and not self._train_on_sampled_tokens:
            logger.warning(
                "vllm_importance_sampling_correction is ON but train_on_sampled_tokens is off, so the "
                "correction is DISABLED: the ratio needs the sampling log-probs the server returns "
                "alongside the sampled tokens. Every batch then trains uncorrected for the engine-vs-trainer "
                "numerics gap, and for a weight-sync lag where one exists (enable_prefetch, "
                "sync_weights_every_n_steps > 1). Set train_on_sampled_tokens: true (a vLLM server needs "
                "--return-tokens-as-token-ids), or set the correction to false deliberately."
            )
        self._require_forced_close_neutralized()
        if self._is_mask_config.any_stage_active and not self._is_correction:
            raise ValueError(
                "isr_geo_band/isr_veto/isr_opsm knobs require the importance-sampling "
                "correction (train_on_sampled_tokens + vllm_importance_sampling_correction) — "
                "without it they would silently do nothing."
            )
        if self._skip_update_masked_frac is not None and not (
            self._is_correction and self._is_mask_config.any_stage_active
        ):
            raise ValueError(
                "skip_update_masked_frac requires the importance-sampling correction AND at "
                "least one IS mask stage (isr_geo_band/isr_veto/isr_opsm) — without them no "
                "trajectory is ever masked and the circuit breaker would silently never fire."
            )
        early_stop = build_early_stop_callback(
            self.async_config.build_early_stop(), gap_key=LOGRATIO_MEAN_KEY, gap_logged=self._is_correction
        )
        if early_stop is not None:
            self.add_callback(early_stop)
        self._isr_engine_reference = self.async_config.isr_engine_reference
        if self._isr_engine_reference:
            self._validate_engine_reference(resolve_weight_sync_client(self._rollout_backend))
        if self._is_correction:
            self.use_vllm = True
        if self.args.vllm_importance_sampling_mode != "sequence_mask":
            logger.warning(
                f"vllm_importance_sampling_mode={self.args.vllm_importance_sampling_mode!r} is ignored: "
                "environmental GRPO always applies token-level truncated IS (clip_max only)."
            )
        if self.args.vllm_importance_sampling_clip_min:
            logger.warning(
                "vllm_importance_sampling_clip_min is ignored: environmental GRPO clips the IS ratio "
                "from above only (vllm_importance_sampling_clip_max)."
            )
        # Derived from the config class rather than a literal table, so a TRL default change cannot
        # turn this warning into a false positive.
        arg_defaults = {f.name: f.default for f in dataclasses.fields(type(self.args))}
        ignored_sampling = [
            name
            for name in _ROLLOUT_OWNED_SAMPLING_KNOBS
            if name in arg_defaults and getattr(self.args, name) != arg_defaults[name]
        ]
        if ignored_sampling:
            logger.warning(
                f"These GRPOConfig sampling knobs are IGNORED by environmental GRPO: "
                f"{sorted(ignored_sampling)}. Rollouts are generated by the environment actors from "
                f"the rollout server config, so set the rollout_* equivalents (rollout_top_p, rollout_top_k, "
                f"rollout_min_p, rollout_repetition_penalty) instead — these reach no sampler."
            )

        self._init_async_state()

        self._finish_on_policy_init()

        # After _setup_distributed_modes (needs the EP wrappers); eager, so an unsupported setup fails now.
        self._routing_injector = self._setup_routing_replay(self.async_config.routing_replay)

    def _loss_logits_width(self) -> LogitsWidth:
        """A per-turn row completes within one turn's cap; a whole-trajectory row runs to the row cap.
        One logit past the completion is kept for the next-token shift.

        Per-turn, this is the usual row, not a bound: a trajectory whose turns lost their captured ids
        falls back to one whole-trajectory row, which widens its micro-batch to that row.
        """
        cfg = self.async_config
        if self._train_on_sampled_tokens:
            return LogitsWidth(cfg.rollout_max_tokens + 1, "rollout_max_tokens")
        if cfg.max_train_row_tokens is not None:
            return LogitsWidth(cfg.max_train_row_tokens + 1, "max_train_row_tokens")
        context = self._context_limit()
        return LogitsWidth(context + 1, f"max_train_row_tokens (unset: the {context}-token context window)")

    def _reject_unverified_carried_reasoning(self) -> None:
        """``carry_reasoning`` sends an assistant message carrying ``reasoning_content`` back to the
        engine. vLLM forwards it to the chat template; what SGLang's chat schema does with it is
        unverified, and a silently dropped thought would train rows conditioned on context the
        engine never saw. Refused until verified rather than assumed."""
        if self._rollout_env.carry_reasoning and self.async_config.rollout_backend == SGLANG_BACKEND:
            raise ValueError(
                "carry_reasoning: true with rollout_backend: 'sglang' is not supported: whether SGLang's chat "
                "schema forwards an assistant message's reasoning_content to the chat template is unverified, "
                "so the carried thought could be dropped without a trace. Serve rollouts with vLLM, or turn "
                "carry_reasoning off."
            )

    def _force_full_dataset_columns(self) -> None:
        """Keep every dataset column: the rollout context IS the row minus ``prompt``.

        Column pruning keeps only what the model's forward signature names, which drops ``answer`` and
        every ``context_fields`` column — the environment would then grade episodes it was handed no
        ground truth for, at no error.
        """
        keep_all_dataset_columns(
            self.args,
            "Environmental GRPO requires remove_unused_columns=False (the rollout context is the row's "
            "non-prompt columns, which pruning would strip); forcing it off. Set it explicitly in your GRPOConfig.",
        )

    def _reject_answerless_datasets(self) -> None:
        """Refuse a dataset with no :data:`ANSWER_KEY` column when the environment grades against one.

        Every non-``prompt`` column of a row becomes that episode's rollout context, so an environment
        declaring ``requires_answer`` reads its ground truth from ``context[ANSWER_KEY]``. Without the
        column the whole run scores one constant — every GRPO group at zero advantage, nothing in the
        logs. Every rank holds the same schema, so this raises uniformly like the gates beside it.
        """
        if not self._rollout_env.requires_answer:
            return
        datasets = {}
        if self.train_dataset is not None:
            datasets["train_dataset"] = self.train_dataset
        if isinstance(self.eval_dataset, dict):
            datasets.update({f"eval_dataset[{name!r}]": split for name, split in self.eval_dataset.items()})
        elif self.eval_dataset is not None:
            datasets["eval_dataset"] = self.eval_dataset
        for name, dataset in datasets.items():
            # A plain torch Dataset exposes no column names (TRL refuses an IterableDataset before
            # this runs): warn rather than refuse a shape whose schema only a row would reveal.
            columns = getattr(dataset, "column_names", None)
            if columns is None:
                logger.warning(
                    f"{name} declares no columns, so the {ANSWER_KEY!r} column "
                    f"{type(self._rollout_env).__name__} grades against cannot be checked here."
                )
            elif ANSWER_KEY not in columns:
                raise ValueError(
                    f"{type(self._rollout_env).__name__} grades each episode against the dataset's "
                    f"{ANSWER_KEY!r} column (requires_answer), but {name} carries {sorted(columns)}. Point "
                    f"answer_field at the column holding the expected answer — it is carried into the "
                    f"row as {ANSWER_KEY!r} — or train this data on an environment that grades without one."
                )

    @property
    def _runs_logprob_recompute(self) -> bool:
        """Whether the no-grad log-prob recompute forward runs at all.

        ``_setup_routing_replay`` rejects ``routing_replay='recompute'`` against this gate at
        construction and ``_recompute_logps_and_routing_masks`` runs the pass under it; a drift
        between the two would capture nothing.
        """
        return self.num_iterations > 1 or self._misaligned_accumulation or self._is_correction or self.beta != 0.0

    def _setup_routing_replay(self, mode: str) -> RoutingReplayInjector | None:
        """Validate and build the routing-replay injector (``None`` when off).

        Fails fast on every unsupported shape: unknown mode, a model without EP MoE wrappers, a family
        that cannot re-derive gate weights, ``recompute`` in a config whose recompute pass never runs,
        and ``rollout`` without train-on-sampled-tokens.
        """
        if mode not in ROUTING_REPLAY_MODES:
            raise ValueError(f"routing_replay must be one of {list(ROUTING_REPLAY_MODES)}, got {mode!r}")
        self._rollout_routing_replay = mode == "rollout"
        if mode == "none":
            return None
        injector = build_routing_replay_injector(self.model)  # raises on no-EP / unsupported families
        # The dense per-row path trims each row to its real span, and capture/arm must tile the same layout.
        self._replay_row_spans = self._use_chunked_grpo_logprobs and rows_forward_densely(
            self.accelerator.unwrap_model(self.model), self.args.per_device_train_batch_size
        )
        if mode == "rollout":
            # Experimental: validate the engine's capture coverage on the serving shape before a run.
            if not self._train_on_sampled_tokens:
                raise ValueError(
                    "routing_replay='rollout' requires train_on_sampled_tokens: the engine's masks "
                    "align with the sampled token ids, not a chat-template re-render."
                )
        elif not self._runs_logprob_recompute:
            raise ValueError(
                "routing_replay='recompute' captures the mask in the no-grad logprob-recompute pass, "
                "which this config never runs (num_iterations=1, aligned accumulation, IS correction "
                "off, beta=0). Enable the IS correction (or beta/num_iterations>1) or set "
                "routing_replay='none'."
            )
        logger.info(f"Routing replay enabled (mode={mode}) over {injector.num_ep_layers} EP MoE layers")
        return injector

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Arm the routing-replay mask for this microbatch's gradient forward, then defer to TRL.

        The mask stays armed through the backward (reentrant-GC recompute reads the same selection);
        :meth:`training_step` disarms afterward.
        """
        if self._routing_injector is not None and model.training:
            masks = inputs.get(ROUTING_MASKS_KEY) if isinstance(inputs, dict) else None
            if masks is not None:
                self._routing_injector.arm(masks, row_spans=self._replay_spans_for(inputs))
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )

    def _replay_spans_for(self, inputs: dict) -> list[tuple[int, int]] | None:
        """Row spans for routing-replay capture/arm when the dense per-row path trims the forwards
        (FA4, or any attention path at ``per_device_train_batch_size`` 1).

        ``None`` on the padded-batch paths (batched rows without FA4, or chunked logprobs off), where
        the mask tiles ``rows × seq`` directly. Derived from the same padded masks the logprob forwards see.
        """
        if not self._replay_row_spans:
            return None
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        return dense_row_spans(attention_mask)

    def _resolve_reasoning_end_token_id(self) -> int | None:
        """The id of ``rollout_reasoning_end_token`` under the tokenizer, for the per-turn reasoning count the
        overlong charge reads; a run without the charge never reads it."""
        cfg = self.async_config
        if cfg.turn_overlong_penalty == 0:
            return None
        return resolve_reasoning_end_token_id(self._tokenizer, cfg.rollout_reasoning_end_token)

    def _require_forced_close_neutralized(self) -> None:
        """Forced reasoning closes are neutralized through the IS ratio, which only reaches the loss under the
        importance-sampling correction; without it they would train with the episode's advantage and teach the
        model to stop closing its reasoning."""
        if self._forced_close_ids is not None and not self._is_correction:
            raise ValueError(
                "a vLLM thinking budget can be enforced but the importance-sampling correction is off "
                "(train_on_sampled_tokens + vllm_importance_sampling_correction): the reasoning closes the "
                "engine forces at the budget would train with the episode's advantage."
            )

    @property
    def _any_episode_budgeted(self) -> bool:
        """Whether some episode the run draws carries a per-turn thinking cap: a drawable level's
        ``thinking_tokens``, or the run's ``rollout_max_thinking_tokens`` (:func:`thinking_caps_by_level`, which
        refuses a cap that fills the turn). A budgeted level the environment never draws caps no episode."""
        return any(cap is not None for cap in self._thinking_caps_by_level().values())

    def _thinking_caps_by_level(self) -> dict[str | None, int | None]:
        """The per-turn thinking cap of every level this run can draw (:func:`thinking_caps_by_level`), which
        refuses a level whose cap fills the turn and an output budget under one turn."""
        cfg = self.async_config
        return thinking_caps_by_level(
            self._rollout_env,
            max_tokens=cfg.rollout_max_tokens,
            max_thinking_tokens=cfg.rollout_max_thinking_tokens,
            max_episode_tokens=cfg.rollout_max_episode_tokens,
        )

    def _resolve_forced_close_ids(self) -> tuple[int, ...] | None:
        """The reasoning-end ids the engine forces when a turn reaches its thinking budget, whose forced
        occurrences the loss must not train on; ``None`` when no budget can be enforced (SGLang enforces none).

        A marker the tokenizer does not write (the ``</think>`` default on a family whose reasoning ends
        otherwise) keeps the forced closes in the loss, said loudly."""
        cfg = self.async_config
        if cfg.rollout_backend == SGLANG_BACKEND or not self._any_episode_budgeted:
            return None
        try:
            return resolve_reasoning_end_ids(self._tokenizer, cfg.rollout_reasoning_end_token)
        except ValueError as e:
            logger.warning(
                "%s The reasoning closes the engine forces at the thinking budget stay in the policy loss.", e
            )
            return None

    def train(self, *args, **kwargs):
        """Train with async rollout collection."""
        try:
            self._init_async_components()
            return super().train(*args, **kwargs)
        finally:
            self._cleanup_async_components()

    def _eval_loader_batch_size(self) -> int:
        """One eval batch is one synchronous rollout round (no prefetch), so its wall time is its slowest
        episode and ``per_device_eval_batch_size``-sized rounds idle the servers between them.
        ``eval_rollout_batch_size`` sizes the round; the loss forward still chunks by
        ``per_device_eval_batch_size``."""
        return self.async_config.eval_rollout_batch_size or super()._eval_loader_batch_size()

    def _validate_eval_round(self) -> None:
        """The eval round's geometry, checked at construction like the train round's. TRL validates
        only the GLOBAL eval batch against ``num_generations_eval``; groups are per rank here, so the
        per-rank round — ``eval_rollout_batch_size``, else ``per_device_eval_batch_size`` — must hold
        whole groups. A run that never evaluates has no eval round to check."""
        if not evaluation_runs(self.args):
            return
        rows = self.async_config.eval_rollout_batch_size
        if rows is None:
            per_rank = self.args.per_device_eval_batch_size
            if per_rank % self.num_generations_eval != 0:
                raise ValueError(
                    f"per_device_eval_batch_size ({per_rank}) must be divisible by num_generations_eval "
                    f"({self.num_generations_eval}): environmental GRPO groups advantages per rank, so an "
                    f"eval group cannot straddle rank batches (or set eval_rollout_batch_size to a multiple)."
                )
            return
        if rows % self.num_generations_eval != 0:
            raise ValueError(
                f"eval_rollout_batch_size ({rows}) must be divisible by num_generations_eval "
                f"({self.num_generations_eval}): environmental GRPO groups advantages per rank."
            )
        if self.args.dataloader_drop_last:
            # Accelerate's shard yields a round only once every rank holds a full batch; a round wider
            # than the split's share leaves fewer batches than ranks and an eval that scores nothing.
            raise ValueError(
                "eval_rollout_batch_size needs dataloader_drop_last=false: a dropped tail is the whole eval."
            )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """Generate completions and broadcast tensor results from TP rank 0 across the TP/ETP group.

        Each rank independently collects rollouts that may differ under sampling (temperature > 0),
        breaking the DTensor gradient consistency compute_loss expects.
        """
        result = self._generate_and_score_completions_base(inputs)

        return self._broadcast_tensors_from_tp_leader(result)

    def _generate_and_score_completions_base(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """Generate completions via Ray-actor rollouts and score with environment rewards."""
        device = self.accelerator.device
        mode = "eval" if not self.model.training else "train"
        # Rows past the split's own are never rolled out: they enter the batch as inert padding.
        episodes = self.eval_split_rows(len(inputs))

        # One rollout per row: TRL's RepeatSampler already expanded each prompt num_generations times.
        expanded_prompts, expanded_contexts = self._extract_prompts_and_contexts(inputs[:episodes])
        # Before the rollout, not after it: a prompt shape the trainer cannot turn into a task would
        # otherwise be submitted as an empty string, costing a full round, and an environment that
        # rejects an empty task raises inside the Ray actor first, replacing this config error with an
        # actor traceback.
        self._batch_errors.reject()

        # Timed as one phase: the step's generation wait (a prefetch hit is ~0, a miss pays the full
        # rollout latency of the round).
        with profiling_context(self, "rollout_acquire"):
            rollout_results = None
            if self._prefetch_enabled and mode == "train":
                rollout_results = self._try_get_prefetched_results()
                if rollout_results is None and self._prefetch_inflight > 0:
                    rollout_results = self._wait_for_inflight_prefetch()

            # A wedged prefetch pipeline is recorded, never raised (per-rank state), so fence it HERE
            # rather than after the fallback: peers that took the hit path are already waiting in
            # this fence, and letting the wedged rank first pay a full synchronous round plus a
            # blocking re-submit can outlast DIST_NCCL_TIMEOUT_MINUTES on a slow env. Every rank
            # reaches this point exactly once per round.
            self._batch_errors.reject()

            if rollout_results is None:
                rollout_results = self._loop.run_until_complete(
                    self._rollout_manager.collect_rollouts(expanded_prompts, expanded_contexts)
                )

        # Submit this round's prompts so the next round pops their one-round-stale rollouts (the IS
        # ratio corrects the staleness); a cold round primes the lag, rolling its prompts out twice.
        if self._prefetch_enabled and mode == "train":
            self._submit_for_prefetch(expanded_prompts, expanded_contexts)

        rollout_results = self._broadcast_rollouts_for_tp(rollout_results)
        # Counted once: the reasoning terms price these and the rollout metrics slice them.
        reasoning_tokens = self._episode_reasoning_tokens(rollout_results)

        with profiling_context(self, "build_training_tensors"):
            result = self._build_training_tensors(
                rollout_results, device, mode, len(inputs) - episodes, reasoning_tokens
            )

        self._log_rollout_metrics(rollout_results, mode, reasoning_tokens)

        return result

    def _episode_reasoning_tokens(self, rollout_results: list[RolloutResult]) -> list[list[int]]:
        """Each episode's per-assistant-turn reasoning token counts (:func:`reasoning_token_counts`)."""
        return [reasoning_token_counts(self._tokenizer, r.trajectory) for r in rollout_results]

    def _broadcast_rollouts_for_tp(self, rollout_results: list[RolloutResult]) -> list[RolloutResult]:
        """Replace each rank's rollouts with the TP/ETP-group leader's so all ranks tokenize identical
        trajectories; else independently-sampled rollouts deadlock the in-forward collectives. No-op outside TP/ETP."""
        return self._broadcast_object_from_tp_leader(rollout_results)

    def _extract_prompts_and_contexts(self, inputs: list[dict]) -> tuple[list[str], list[dict | None]]:
        """Extract prompts and contexts from the input batch.

        Prompts pass as raw strings to the environment, which constructs messages; the engine templates them.
        """
        prompts = []
        contexts = []

        for inp in inputs:
            prompt = inp["prompt"]
            try:
                prompt_text = task_prompt(prompt)
            except ValueError as exc:
                # The caller fences it before submitting anything to the environment.
                self._batch_errors.record(f"Environmental GRPO row: {exc}")
                prompt_text = ""

            prompts.append(prompt_text)
            ctx = {k: v for k, v in inp.items() if k != "prompt"}
            contexts.append(ctx if ctx else None)

        if self._group_random_effort:
            self._stamp_group_efforts(prompts, contexts)
        return prompts, contexts

    def _stamp_group_efforts(self, prompts: list[str], contexts: list[dict | None]) -> None:
        """One reasoning-effort draw per generation group, stamped into every member's rollout context.

        GRPO's group baseline compares the ``num_generations`` completions of one prompt against each
        other, which is only meaningful when they share the same conditioning. The env's
        ``reasoning_effort='random'`` is otherwise drawn per episode in the Ray actor, so the draw
        becomes part of the advantage and harder-conditioned members lose to their easier siblings
        regardless of policy quality. Rows arrive group-expanded (RepeatSampler), so consecutive
        blocks of the mode's group size are one group.
        Evaluation draws a group's level from its problem (:func:`stable_reasoning_effort`), so every eval
        scores each problem at the same level and checkpoints compare level for level.
        """
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

    def _build_training_tensors(
        self,
        rollout_results: list[RolloutResult],
        device: torch.device,
        mode: str,
        num_padding: int,
        reasoning_tokens: list[list[int]],
    ) -> dict[str, torch.Tensor]:
        """Build training tensors from rollout results.

        ``num_padding`` trajectories follow the rollouts: the rows that pad an eval split's final round
        (:meth:`eval_split_rows`). Each is an invalid, zero-reward trajectory of one inert row, so this
        rank's tensors keep the shape its peers gather and forward, and no metric reads them.
        ``reasoning_tokens`` is each rollout's per-turn reasoning count, priced by the reasoning terms.

        The phase helpers below run in this order on every rank, and every collective inside them
        sits behind a config- or mode-derived gate, never a local-data one: ``_assemble_rollout_routing``
        (the uniform raise), ``_recompute_logps_and_routing_masks`` (the recompute forward's EP
        dispatch) and ``_narrow_masks_and_normalizer`` (the normalizer gather). A rank-dependent skip
        or early-out anywhere in the sequence desyncs the world.
        """
        rewards = F.pad(self._build_rollout_rewards(rollout_results, reasoning_tokens, device), (0, num_padding))

        all_prompt_ids = []
        all_completion_ids = []
        all_prompt_masks = []
        all_completion_masks = []
        all_tool_masks = []
        all_sampling_logps: list[torch.Tensor | None] = []
        all_turn_routing: list[TurnRouting | None] = []
        all_negative_only: list[bool] = []

        # train_on_sampled_tokens splits a trajectory per assistant turn; turns_per_traj expands the advantage.
        turns_per_traj = []
        padding = [single_trajectory_row(self._masked_trajectory_tensors())] * num_padding
        for turn_rows in self._tokenize_step_rows(rollout_results) + padding:
            turns_per_traj.append(len(turn_rows))
            for row in turn_rows:
                # ``completion_mask`` is attention-valid (env/tool tokens conditioned sampling), ``tool_mask``
                # is the loss mask; conflating them hides tool output from attention. Bool: [rows, max_len] sink.
                loss_mask = row.completion_mask.to(device=device, dtype=torch.bool)
                # A fully-masked row (errored/empty rollout) stays invisible to attention too.
                attention_valid = torch.ones_like(loss_mask) if loss_mask.any() else loss_mask
                all_prompt_ids.append(row.prompt_ids.to(device))
                all_completion_ids.append(row.completion_ids.to(device))
                all_prompt_masks.append(torch.ones_like(row.prompt_ids, dtype=torch.bool, device=device))
                all_completion_masks.append(attention_valid)
                all_tool_masks.append(loss_mask)
                all_sampling_logps.append(row.sampling_logps.to(device) if row.sampling_logps is not None else None)
                all_turn_routing.append(row.turn_routing)
                all_negative_only.append(row.negative_only)

        # Config-derived, so every rank agrees on taking the correction branches below.
        use_is_correction = self._is_correction
        row_has_sampling = [o is not None for o in all_sampling_logps]

        # Fences the row-shape failures tokenization recorded (a context overflow, a malformed routing payload).
        self._batch_errors.reject()

        # After the uniform raise: the re-score never raises, but it must not sit between a rank's
        # collectives either. Train mode only: the engine re-score is the one IS stage gated to
        # training; eval still runs the ratio and mask stages against the trainer's own log-probs.
        all_engine_logps: list[torch.Tensor | None] | None = None
        if use_is_correction and self._isr_engine_reference and mode == "train":
            all_engine_logps = self._rescore_rows_on_engine(
                all_prompt_ids, all_completion_ids, row_has_sampling, turns_per_traj
            )

        # The per-turn split gives each rank a variable row count, padded with masked dummy rows across
        # ranks (unequal forwards deadlock) and to a multiple of steps_per_generation (TRL drops remainders).
        num_dummy_rows = 0
        if self._train_on_sampled_tokens:
            global_rows = torch.tensor([len(all_prompt_ids)], device=device)
            if is_multi_rank_run():
                dist.all_reduce(global_rows, op=dist.ReduceOp.MAX)
            chunks = self.args.steps_per_generation
            target_rows = math.ceil(int(global_rows.item()) / chunks) * chunks
            num_dummy_rows = target_rows - len(all_prompt_ids)
            # The inert row the tokenize path returns for a trajectory with nothing to train: one attended
            # prompt token (no NaN), a fully masked completion (zero loss).
            inert_prompt, inert_completion, inert_mask = (t.to(device) for t in self._masked_trajectory_tensors())
            inert_mask = inert_mask.bool()
            for _ in range(num_dummy_rows):
                all_prompt_ids.append(inert_prompt)
                all_completion_ids.append(inert_completion)
                all_prompt_masks.append(torch.ones_like(inert_prompt, dtype=torch.bool))
                all_completion_masks.append(inert_mask)
                all_tool_masks.append(inert_mask)
                all_sampling_logps.append(None)  # zeros below; row_has_sampling masks them
                all_turn_routing.append(None)  # dummy rows keep natural routing (-1 sentinel)
                all_negative_only.append(False)
                row_has_sampling.append(False)
                if all_engine_logps is not None:
                    all_engine_logps.append(None)

        rows = BatchRows(rollout_results, turns_per_traj, num_dummy_rows, self._train_on_sampled_tokens)

        # A turn missing sampling logprobs keeps ratio ≡ 1 for that row alone. Zeros are inert: row_has_sampling
        # masks them.
        all_sampling_logps = [
            o if o is not None else torch.zeros(len(c), device=device)
            for o, c in zip(all_sampling_logps, all_completion_ids, strict=True)
        ]

        prompt_ids = pad(all_prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        completion_ids = pad(all_completion_ids, padding_value=self.pad_token_id, padding_side="right")
        prompt_mask = pad(all_prompt_masks, padding_value=0, padding_side="left")
        completion_mask = pad(all_completion_masks, padding_value=0, padding_side="right")
        tool_mask = pad(all_tool_masks, padding_value=0, padding_side="right")
        sampling_logps = has_sampling = None
        if use_is_correction:
            sampling_logps = pad(all_sampling_logps, padding_value=0, padding_side="right")
            has_sampling = torch.tensor(row_has_sampling, device=device)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)

        rollout_routing_masks = self._assemble_rollout_routing(
            all_turn_routing, all_prompt_masks, all_completion_ids, tool_mask, prompt_ids, completion_ids, device, mode
        )

        with torch.no_grad():
            recompute_logps, routing_masks = self._recompute_logps_and_routing_masks(
                prompt_completion_ids, attention_mask, logits_to_keep, rollout_routing_masks, mode
            )
            # None on the aligned single-iteration path: TRL's detach() reuse makes the ratio exactly 1.
            old_per_token_logps = (
                recompute_logps if (self.num_iterations > 1 or self._misaligned_accumulation) else None
            )

            importance_sampling_ratio, logps_diff, corrected_mask, traj_row_ids = self._apply_is_correction(
                use_is_correction,
                recompute_logps,
                sampling_logps,
                has_sampling,
                completion_mask,
                rows,
                device,
                all_engine_logps,
            )
            if use_is_correction and self._forced_close_ids is not None:
                importance_sampling_ratio, forced = zero_engine_forced_closes(
                    importance_sampling_ratio,
                    sampling_logps,
                    completion_mask,
                    has_sampling,
                    completion_ids,
                    self._forced_close_ids,
                )
                # Over the tokens that carry sampling log-probs: no other token can be a forced close.
                sampled = sampled_token_mask(completion_mask, has_sampling)
                self._world_metrics.fraction("sampling/forced_close_frac", forced.sum(), sampled.sum())

            ref_per_token_logps = self._compute_ref_logps(prompt_completion_ids, attention_mask, logits_to_keep)
            kl_clamped = None
            if ref_per_token_logps is not None and recompute_logps is not None:
                ref_per_token_logps, kl_clamped = clamp_ref_logps(ref_per_token_logps, recompute_logps)

        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        # Per trajectory: 1 valid, 0 an invalid episode, -1 an eval round's padding, gathered as one tensor.
        episode_state = F.pad(rollout_valid_mask(rollout_results, device).to(torch.int8), (0, num_padding), value=-1)
        valid_mask = episode_state == 1
        traj_advantages = self._compute_advantages(rewards, num_generations, valid_mask=valid_mask)
        local_advantages = rows.to_rows(traj_advantages)
        # OPSM masks only negative-advantage trajectories, so it must run after the advantages exist.
        if self._is_mask_config.opsm_delta is not None and corrected_mask is not None:
            importance_sampling_ratio, opsm_masked = apply_opsm(
                importance_sampling_ratio,
                logps_diff,
                corrected_mask,
                traj_row_ids,
                local_advantages,
                self._is_mask_config.opsm_delta,
            )
            self._world_metrics.fraction("sampling/is_opsm_masked_frac", opsm_masked.sum(), opsm_masked.numel())
        gathered_rewards = gather(rewards)
        gathered_state = gather(episode_state)
        gathered_valid = gathered_state == 1
        if mode == "train":
            self._check_step_has_valid_episodes(rollout_results, gathered_valid)

        negative_only = torch.tensor(all_negative_only, device=device, dtype=torch.bool)
        completion_mask, tool_mask, loss_mask, num_items_in_batch = self._narrow_masks_and_normalizer(
            rows,
            valid_mask,
            num_generations,
            completion_mask,
            tool_mask,
            local_advantages,
            negative_only,
            device,
            mode,
        )
        if kl_clamped is not None:
            self._world_metrics.fraction(KL_CLAMP_FRAC_KEY, *kl_clamp_counts(kl_clamped, loss_mask))
        if mode == "train":
            balance = record_token_mass(
                local_advantages,
                loss_mask,
                importance_sampling_ratio,
                self.accelerator.gather,
                self._metrics[mode],
                self._balance_token_mass,
                negative_only=negative_only,
            )
            if balance is not None:
                # The per-trajectory values follow, so the completions record reports what the gradient carries.
                local_advantages, traj_advantages = balance.apply(local_advantages), balance.apply(traj_advantages)

        # Padding fills whole groups, so dropping its rows keeps every episode's group aligned.
        episodes = gathered_state >= 0
        self._log_headline_rewards(gathered_rewards[episodes], gathered_valid[episodes], num_generations, mode)
        self._metrics[mode]["sampling/is_correction_active"].append(float(use_is_correction))
        if corrected_mask is not None:
            eff_corrected = self._score_is_correction(importance_sampling_ratio, corrected_mask, loss_mask)
            if self._update_breaker_tripped(
                importance_sampling_ratio, eff_corrected, traj_row_ids, len(rollout_results), mode
            ):
                # Both tensors: the per-row one carries the zeroed policy gradient, the per-trajectory
                # one is what the completions record below writes.
                local_advantages = torch.zeros_like(local_advantages)
                traj_advantages = torch.zeros_like(traj_advantages)
        # After the breaker: the diagnostics read the advantages this step's gradient actually carries.
        self._record_step_diagnostics(
            rollout_results, num_generations, mode, recompute_logps, local_advantages, loss_mask
        )

        # After the breaker, not before it: the durable completions record must report the advantages
        # this step's gradient actually saw, which on a broken step are zeros.
        episodes = len(rollout_results)
        self._populate_completion_logs(rollout_results, rewards[:episodes], traj_advantages[:episodes], mode)
        # Every rank arrives here once per step: the one collective that turns the step's rank-local
        # counts into batch-level metrics.
        self._world_metrics.flush(self._metrics[mode])

        result = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "tool_mask": tool_mask,
            "advantages": local_advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "importance_sampling_ratio": importance_sampling_ratio,
            "num_items_in_batch": num_items_in_batch,
        }
        if routing_masks is not None:
            # Rides the batch dict like old_per_token_logps so TRL's shuffle/slice keep it row-aligned.
            result[ROUTING_MASKS_KEY] = routing_masks
        return result

    def _build_rollout_rewards(
        self, rollout_results: list[RolloutResult], reasoning_tokens: list[list[int]], device: torch.device
    ) -> torch.Tensor:
        """Per-trajectory environment rewards with the trainer-side shaping terms charged in place."""
        rewards = torch.tensor(
            [r.total_reward for r in rollout_results],
            device=device,
            dtype=torch.float32,
        )

        cfg = self.async_config
        length_terms_on = cfg.reasoning_price is not None or cfg.reasoning_floor > 0
        if length_terms_on or self._carry_reasoning:
            self._warn_if_no_reasoning_captured(rollout_results, length_terms_on)
        if length_terms_on:
            self._apply_reasoning_terms(rewards, rollout_results, reasoning_tokens)
        if cfg.turn_overlong_penalty > 0:
            self._apply_turn_overlong_charge(rewards, rollout_results)
        return rewards

    def _log_headline_rewards(
        self, gathered_rewards: torch.Tensor, gathered_valid: torch.Tensor, num_generations: int, mode: str
    ) -> None:
        """``reward`` / ``reward_std`` / ``reward/within_group_std`` over the gathered VALID episodes —
        the rows that train.

        An infra-errored or env-invalid episode carries a forced failure reward that says nothing
        about the policy; averaged in, a grader outage reads as a policy collapse, and its spread
        against valid siblings as a learning signal the advantage never sees. ``reward_std`` is the
        GLOBAL spread; ``reward/within_group_std`` is the one the GRPO advantage reads. A step with no
        valid episode logs none of them (the empty-step halt reports it).
        """
        valid_rewards = gathered_rewards[gathered_valid]
        if valid_rewards.numel() == 0:
            return
        self._metrics[mode]["reward"].append(valid_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(batch_reward_std(valid_rewards))
        if num_generations > 1 and gathered_rewards.numel() % num_generations == 0:
            self._metrics[mode]["reward/within_group_std"].append(
                within_group_reward_std(gathered_rewards, gathered_valid, num_generations)
            )

    def _assemble_rollout_routing(
        self,
        all_turn_routing: list[TurnRouting | None],
        all_prompt_masks: list[torch.Tensor],
        all_completion_ids: list[torch.Tensor],
        tool_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        device: torch.device,
        mode: str,
    ) -> torch.Tensor | None:
        """The engine-supplied routing mask under ``routing_replay='rollout'``, else ``None``.

        The gate is config- and mode-derived, so the uniform raise it wraps is reached by every rank
        or by none; the assembly's own failures are recorded for that raise, never thrown here.

        A batch carrying no engine routing is a capture failure only when ``tool_mask`` — the loss
        mask, read before the advantage-dependent narrowing, so a negative-only row counts — still has
        a trainable token. All-masked means every assistant turn yielded no row
        (:meth:`_tokenize_trajectory_turns`: no ids, zero tokens, over the row cap): there is no
        selection to replay, and every other mode trains that step as a no-op rather than ending the run.
        """
        rollout_routing_masks = None
        if self._rollout_routing_replay and mode == "train":
            if not any(m is not None for m in all_turn_routing):
                if not tool_mask.any():
                    logger.warning(
                        "routing_replay='rollout': nothing to replay this step — every training row is "
                        "masked (no assistant turn yielded a row), so the step contributes "
                        "no gradient."
                    )
                else:
                    self._batch_errors.record(
                        "routing_replay='rollout' but no rollout in this batch returned routed_experts "
                        "— serve vLLM >= 0.22 with --enable-return-routed-experts and a non-FlashInfer "
                        "MoE backend (--moe-backend triton), or SGLang with --enable-return-routed-experts "
                        "and --moe-runner-backend triton (the triton_kernel/flashinfer runners bypass "
                        "the capture hook)."
                    )
            else:
                try:
                    rollout_routing_masks, coverage = assemble_rollout_masks(
                        all_turn_routing,
                        [int(m.sum()) for m in all_prompt_masks],
                        [len(c) for c in all_completion_ids],
                        prompt_ids.size(1),
                        completion_ids.size(1),
                        self._routing_injector.num_ep_layers,
                        self._routing_injector.top_k,
                        self._routing_injector.num_experts,
                    )
                    rollout_routing_masks = rollout_routing_masks.to(device)
                    total = sum(coverage[k] for k in ROLLOUT_COVERAGE_SHAPES)
                    for key, count in coverage.items():
                        self._world_metrics.fraction(f"routing/rollout_{key}_frac", count, total)
                    if coverage["unresolved"] == total:
                        self._batch_errors.record(
                            f"routing_replay='rollout': no routed row on this rank ({total} in the "
                            f"batch) matched a known engine coverage convention, so every position "
                            f"would replay natural routing and R3 would be inert. The engine's "
                            f"routed_experts token count no longer lines up with the trainer's "
                            f"prompt + completion lengths — check the engine version against the "
                            f"conventions in assemble_rollout_masks."
                        )
                except ValueError as e:
                    self._batch_errors.record(f"routing_replay='rollout': {e}")
            self._batch_errors.reject()
        return rollout_routing_masks

    def _recompute_logps_and_routing_masks(
        self,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        rollout_routing_masks: torch.Tensor | None,
        mode: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The no-grad log-prob recompute and the routing mask the gradient forward replays.

        Runs inside the caller's ``torch.no_grad``. The recompute gate is config-derived, so the
        forward — and the EP dispatch inside it — is entered by every rank or by none.
        """
        # Recomputed once and chunked: a full-batch DeepEP dispatch overflows its 32-bit limit.
        recompute_logps = None
        # Either the recompute forward below captures the mask, or the engine already supplied it.
        routing_masks = rollout_routing_masks
        if self._runs_logprob_recompute:
            # Capture is scoped to this call: the ref-logps forward that follows is a different policy.
            replay_spans = (
                dense_row_spans(attention_mask)
                if self._routing_injector is not None and self._replay_row_spans and mode == "train"
                else None
            )
            with contextlib.ExitStack() as capture_stack:
                captured = (
                    capture_stack.enter_context(
                        self._routing_injector.capture(
                            prompt_completion_ids.size(0), prompt_completion_ids.size(1), row_spans=replay_spans
                        )
                    )
                    if self._routing_injector is not None and mode == "train" and not self._rollout_routing_replay
                    else None
                )
                if rollout_routing_masks is not None:
                    # The recompute must run under the same routing the update pass replays, or
                    # the ratio comes from a different token distribution than the policy trains on.
                    self._routing_injector.arm(rollout_routing_masks, row_spans=replay_spans)
                try:
                    recompute_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        compute_entropy=False,
                    )
                finally:
                    if rollout_routing_masks is not None:
                        self._routing_injector.disarm()
            if captured:
                routing_masks = captured[0]
        return recompute_logps, routing_masks

    def _apply_is_correction(
        self,
        use_is_correction: bool,
        recompute_logps: torch.Tensor | None,
        sampling_logps: torch.Tensor | None,
        row_has_sampling: torch.Tensor | None,
        completion_mask: torch.Tensor,
        rows: BatchRows,
        device: torch.device,
        all_engine_logps: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """The rollout-vs-trainer importance ratio and its mask stages.

        Returns ``(ratio, logps_diff, corrected_mask, traj_row_ids)``; the ratio is all-ones and
        the rest ``None`` with the correction off — a config-derived gate every rank takes alike, and
        the one under which the caller pads ``sampling_logps`` and builds ``row_has_sampling``.
        ``logps_diff`` is what the mask stages (and the caller's OPSM stage) read: the trainer
        diff, or under ``isr_engine_reference`` the engine's current-vs-sampling diff on every
        re-scored row (:func:`select_mask_logratio`). The IS weight itself is always trainer-vs-sampling.
        """
        importance_sampling_ratio = torch.ones_like(completion_mask, dtype=torch.float32)
        corrected_mask = None
        logps_diff = None
        traj_row_ids = None
        if use_is_correction:
            importance_sampling_ratio, logps_diff, corrected_mask = compute_is_ratio(
                recompute_logps,
                sampling_logps,
                completion_mask,
                row_has_sampling,
                self.vllm_importance_sampling_clip_max,
            )
            # Unclamped mean log-ratio (nats): ~0 when conditioning matches the engine's exact prompt.
            self._world_metrics.fraction(LOGRATIO_MEAN_KEY, logps_diff.sum(), corrected_mask.sum())
            clip_max = self.vllm_importance_sampling_clip_max
            # Past the truncation point in either direction, on the raw trainer-vs-sampler ratio; the
            # band [1/clip_max, clip_max] is empty at or below 1.
            if clip_max > 1:
                extreme = corrected_mask & (logps_diff.abs() > math.log(clip_max))
                self._world_metrics.fraction("sampling/is_ratio_extreme_frac", extreme.sum(), corrected_mask.sum())
            # Policy tokens the sampler emitted with probability 1 (budget-forced closes): uncorrected.
            certain = sampler_certain_mask(sampling_logps, completion_mask, row_has_sampling)
            sampled = sampled_token_mask(completion_mask, row_has_sampling)
            self._world_metrics.fraction("sampling/sampler_certain_frac", certain.sum(), sampled.sum())
            traj_row_ids = rows.traj_row_ids(device)
            if all_engine_logps is not None:
                engine_logps = torch.zeros_like(sampling_logps)
                for i, scored in enumerate(all_engine_logps):
                    if scored is not None:
                        engine_logps[i, : scored.numel()] = scored
                row_has_engine = torch.tensor([o is not None for o in all_engine_logps], device=device)
                logps_diff, engine_stats = select_mask_logratio(
                    logps_diff, recompute_logps, sampling_logps, engine_logps, corrected_mask, row_has_engine
                )
                for key, (numerator, denominator) in engine_stats.items():
                    self._world_metrics.fraction(key, numerator, denominator)
            if self._is_mask_config.any_mask_active:
                importance_sampling_ratio, mask_stats = apply_is_masks(
                    importance_sampling_ratio, logps_diff, corrected_mask, traj_row_ids, self._is_mask_config
                )
                for key, (masked, total) in mask_stats.items():
                    self._world_metrics.fraction(key, masked, total)
        return importance_sampling_ratio, logps_diff, corrected_mask, traj_row_ids

    def _validate_engine_reference(self, client_cls) -> None:
        """Construction gate for ``isr_engine_reference``: the re-score needs the sampling log-probs
        it is compared against, an engine that returns prefill log-probs of a given sequence, and a
        sampler whose recorded log-probs are the raw distribution the prefill returns (temperature
        and top-p of 1, every filter at its off default) — at any other setting the two references
        would not be the same function."""
        if not self._is_correction:
            raise ValueError(
                "isr_engine_reference requires the importance-sampling correction "
                "(train_on_sampled_tokens + vllm_importance_sampling_correction): the engine re-score "
                "is compared against the sampling log-probs that correction captures."
            )
        if not client_cls.SUPPORTS_ENGINE_RESCORE:
            raise ValueError(
                f"isr_engine_reference is not available on {client_cls.BACKEND_NAME}: its client declares "
                "no route that returns per-token log-probs of a given sequence under the current weights."
            )
        sources = self.async_config.rollout_field_sources()
        identity = {sources[name]: value for name, value in IDENTITY_SAMPLER.items()}
        moved = {
            name: current for name, value in identity.items() if (current := getattr(self.async_config, name)) != value
        }
        if moved:
            raise ValueError(
                "isr_engine_reference requires the identity sampler "
                f"({', '.join(f'{name} {value}' for name, value in identity.items())}), got {moved}: the engine "
                "echoes the raw distribution's log-probs while the sampling log-probs are the processed "
                "ones, so only at the identity sampler do the two share a reference."
            )

    def _rescore_rows_on_engine(
        self,
        all_prompt_ids: list[torch.Tensor],
        all_completion_ids: list[torch.Tensor],
        row_has_sampling: list[bool],
        turns_per_traj: list[int],
    ) -> list[torch.Tensor | None]:
        """Every sampled row's completion re-scored on the rollout engine under the weights synced for
        this step; ``None`` where the row carries no sampling log-probs or its request failed (that
        row's mask stages fall back to the trainer diff). Runs on every rank over its own rows through
        the rank's score-only clients (``_engine_rescore_clients``); a trajectory's rows go to one
        server so its turns share the prefix cache. Never raises: a rank-local raise here would desync
        the collectives that follow, so a failed step is reported (``sampling/engine_rescore_miss_frac``,
        a warning, an error when nothing succeeded) and trains on the trainer's reference.
        """
        clients = self._engine_rescore_clients
        if not clients:
            raise RuntimeError(
                "isr_engine_reference re-scored a batch before the rollout components started: no score-only "
                "client was built (_init_async_components builds them on every rank)."
            )
        row_traj = [traj for traj, turns in enumerate(turns_per_traj) for _ in range(turns)]
        indices = [i for i, has in enumerate(row_has_sampling) if has]

        def score(i: int) -> list[float]:
            client = clients[row_traj[i] % len(clients)]
            return client.score_completion_logprobs(all_prompt_ids[i].tolist(), all_completion_ids[i].tolist())

        scored: list[torch.Tensor | None] = [None] * len(all_prompt_ids)
        failures: list[Exception] = []
        with ThreadPoolExecutor(max_workers=_ENGINE_RESCORE_CONCURRENCY) as pool:
            futures = {i: pool.submit(score, i) for i in indices}
            for i, future in futures.items():
                try:
                    values = future.result()
                    if len(values) != all_completion_ids[i].numel():
                        raise ValueError(
                            f"row {i}: {len(values)} log-probs for {all_completion_ids[i].numel()} tokens"
                        )
                    scored[i] = torch.tensor(values, dtype=torch.float32, device=all_completion_ids[i].device)
                except Exception as e:  # noqa: BLE001 — a transport/shape failure on one row is that row's miss
                    failures.append(e)
        self._world_metrics.fraction("sampling/engine_rescore_miss_frac", len(failures), len(indices))
        if failures:
            report = logger.error if len(failures) == len(indices) else logger.warning
            # main_process_only=False: the failure is this rank's own requests, invisible to rank 0.
            report(
                f"[rank {get_global_rank()}] isr_engine_reference: {len(failures)}/{len(indices)} re-score "
                f"requests failed this step (those rows read the trainer's log-ratio instead); last error: "
                f"{failures[-1]}",
                main_process_only=False,
            )
        return scored

    def _narrow_masks_and_normalizer(
        self,
        rows: BatchRows,
        valid_mask: torch.Tensor,
        num_generations: int,
        completion_mask: torch.Tensor,
        tool_mask: torch.Tensor,
        advantages: torch.Tensor,
        negative_only: torch.Tensor,
        device: torch.device,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask this step's excluded trajectories out of both loss masks and take the DAPO normalizer.

        Returns ``(completion_mask, tool_mask, loss_mask, num_items_in_batch)``. A ``negative_only``
        row (an untrainable turn, per-row like ``advantages``) leaves ``tool_mask`` unless its
        advantage is negative, in either mode: the objective penalizes the turn, never rewards it. The
        drop set is train-mode only; the normalizer gather below it is world-wide, entered by every
        rank in every mode.
        """
        world = self._world_metrics
        (tool_mask,) = narrow_loss_masks(negative_only & (advantages >= 0), tool_mask)
        # Masking, not deleting: excluded rows still run the forward so every rank's collective sequence matches.
        if mode == "train":
            # An env-invalid episode's forced failure-reward advantage is noise, so its tokens go too.
            drop_traj = ~valid_mask
            world.fraction("sampling/invalid_episode_frac", drop_traj.sum(), drop_traj.numel())

            if self._drop_degenerate_groups:
                # Judged on the environment's reward: the trainer's length and overlong terms price every
                # episode differently, so no group would ever tie on the total it trains on.
                env_rewards = torch.tensor([r.total_reward for r in rows.rollout_results], device=device)
                degenerate = degenerate_group_mask(env_rewards, num_generations, valid_mask=valid_mask)
                drop_traj |= degenerate
                world.fraction(DEGENERATE_GROUP_FRAC_KEY, degenerate.sum(), degenerate.numel())

            if self.args.mask_truncated_completions:
                # TRL enforces this in its own generation path, which this trainer replaces.
                truncated = torch.tensor(
                    [bool(r.trajectory and r.trajectory.truncated) for r in rows.rollout_results], device=device
                )
                drop_traj |= truncated
                world.fraction("sampling/truncated_masked_frac", truncated.sum(), truncated.numel())

            if drop_traj.any():
                # TRL parity: tool_mask narrows with completion_mask, so the normalizer never counts them.
                completion_mask, tool_mask = narrow_loss_masks(rows.to_rows(drop_traj), completion_mask, tool_mask)

        rollout_rows = sum(rows.turns_per_traj[: len(rows.rollout_results)])
        world.fraction("sampling/untrainable_rows_frac", negative_only.sum(), rollout_rows)
        # After every drop: a kept row is one whose tokens reach the normalizer below.
        kept = negative_only & tool_mask.any(dim=1)
        world.fraction("sampling/untrainable_rows_trained_frac", kept.sum(), negative_only.sum())
        # DAPO normalizes by the gathered-global loss-token count / num_processes; a local count mis-scales.
        loss_mask = effective_loss_mask({"completion_mask": completion_mask, "tool_mask": tool_mask})
        num_items_in_batch = gathered_num_items(loss_mask, self.accelerator.gather)
        return completion_mask, tool_mask, loss_mask, num_items_in_batch

    def _score_is_correction(
        self,
        importance_sampling_ratio: torch.Tensor,
        corrected_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """The drop-narrowed corrected mask the trust-region breaker reads; also records the step's
        IS-correction coverage and surviving-ratio metrics."""
        # corrected_mask predates the drop narrowing; intersect so coverage cannot exceed 1.
        eff_corrected = corrected_mask & loss_mask
        world = self._world_metrics
        world.fraction("sampling/is_correction_coverage", eff_corrected.sum(), loss_mask.sum())
        # Masked ratios are zeroed in place, so a raw mean collapses as masking grows; report survivors.
        ratios = importance_sampling_ratio[eff_corrected]
        surviving = ratios[ratios > 0]
        world.fraction("sampling/is_masked_frac", ratios.numel() - surviving.numel(), ratios.numel())
        if surviving.numel():
            world.fraction("sampling/is_ratio_mean", surviving.sum(), surviving.numel())
            world.maximum("sampling/is_ratio_max", surviving.max())
        # Over the weights the loss applies, masked ones as zeros: a thrown-away token is a lost sample.
        world.effective_sample_frac("sampling/is_ess_frac", importance_sampling_ratio, eff_corrected)
        return eff_corrected

    def _record_step_diagnostics(
        self,
        rollout_results: list[RolloutResult],
        num_generations: int,
        mode: str,
        recompute_logps: torch.Tensor | None,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> None:
        """Record the step's solve-group split, eval success@k and log-prob/advantage covariance.

        The groups are this rank's consecutive ``num_generations`` blocks, judged on the environment's
        solve verdict: an episode outside the baseline or without a verdict does not vote, and no group
        metric is recorded where no episode carries one. success@k is an eval round's share of groups any
        member solved. The covariance pairs each loss token's pre-update log-prob with its row's
        advantage; it needs the recompute forward, a config-derived gate.
        """
        world = self._world_metrics
        solved = [solve_verdict(r.metrics) if r.counts_toward_baseline else None for r in rollout_results]
        groups, all_pass, all_fail, any_pass = group_solve_counts(solved, num_generations)
        if groups:
            world.fraction("outcome/all_pass_group_frac", all_pass, groups)
            world.fraction("outcome/all_fail_group_frac", all_fail, groups)
            if mode == "eval" and num_generations > 1:
                world.fraction(f"outcome/success@{num_generations}", any_pass, groups)
        if recompute_logps is not None:
            world.covariance(
                "logps/advantage_cov", recompute_logps, advantages.unsqueeze(1).expand_as(recompute_logps), loss_mask
            )

    def _validate_reasoning_terms(self) -> None:
        """A level whose cap fills the turn would be cut mid-reasoning every time, a price with no level to
        read would charge nothing (a level the table misses too, and one it invents is a typo), and a floor or
        an overlong charge in a run where no episode carries a thinking cap would never fire."""
        cfg = self.async_config
        budgeted = self._any_episode_budgeted
        if cfg.reasoning_price is not None:
            if self._rollout_env.reasoning_effort is None:
                raise ValueError(
                    "reasoning_price is set but the environment's reasoning_effort is unset, so no episode carries a "
                    "level to be priced at."
                )
            if set(cfg.reasoning_price) != set(VALID_REASONING_EFFORTS):
                levels, admitted = set(cfg.reasoning_price), set(VALID_REASONING_EFFORTS)
                raise ValueError(
                    f"reasoning_price must map exactly the effort levels {sorted(admitted)}; "
                    f"unknown {sorted(levels - admitted)}, missing {sorted(admitted - levels)}"
                )
        if (cfg.reasoning_floor > 0 or cfg.turn_overlong_penalty > 0) and not budgeted:
            raise ValueError(
                "reasoning_floor and turn_overlong_penalty price an episode against its per-turn thinking cap, but "
                "no episode of this run carries one: no drawable effort level sets thinking_tokens and "
                "rollout_max_thinking_tokens is unset."
            )
        if cfg.turn_overlong_penalty > 0:
            # At construction: the rollout start that reads it next runs inside train(), after Ray is up.
            self._resolve_reasoning_end_token_id()

    def _apply_reasoning_terms(
        self, rewards: torch.Tensor, rollout_results: list[RolloutResult], reasoning_tokens: list[list[int]]
    ) -> None:
        """Charge each episode its level's reasoning price and its under-use floor, in place.

        :func:`reasoning_price_term` prices the episode's reasoning tokens at its level's price;
        :func:`reasoning_floor_term` prices a shortfall against the per-turn thinking budget the episode ran
        under. An episode with no level is free of the price, one with no budget of the floor. Logs the batch
        means under ``reward/reasoning_price`` and ``reward/reasoning_floor``.
        """
        cfg = self.async_config
        price_on, floor_on = cfg.reasoning_price is not None, cfg.reasoning_floor > 0
        prices, floors = [], []
        for i, (r, tokens) in enumerate(zip(rollout_results, reasoning_tokens, strict=True)):
            traj = r.trajectory
            price = floor = 0.0
            if price_on and traj and traj.reasoning_effort in cfg.reasoning_price:
                price = reasoning_price_term(
                    tokens, cfg.reasoning_price[traj.reasoning_effort], cfg.reasoning_price_cap
                )
            if floor_on and traj and traj.reasoning_budget:
                floor = reasoning_floor_term(tokens, traj.reasoning_budget, cfg.reasoning_floor)
            rewards[i] += price + floor
            prices.append(price)
            floors.append(floor)
        # A config gate is rank-uniform, so every rank records the same keys.
        if price_on:
            self._world_metrics.fraction("reward/reasoning_price", sum(prices), len(prices))
        if floor_on:
            self._world_metrics.fraction("reward/reasoning_floor", sum(floors), len(floors))

    def _apply_turn_overlong_charge(self, rewards: torch.Tensor, rollout_results: list[RolloutResult]) -> None:
        """Charge each episode the turn that ran furthest into its thinking cap or the turn cap
        (:func:`turn_overlong_term`), in place. Logs the batch mean under ``reward/turn_overlong`` and the share
        of assistant turns charged under ``reward/turn_overlong_turn_frac``, both again per effort level
        (``effort/<level>/...``)."""
        penalty = self.async_config.turn_overlong_penalty
        # Per episode: its level (None without one), term, charged turns and assistant turns.
        rows = []
        for i, result in enumerate(rollout_results):
            term, charged, turns = turn_overlong_term(
                result.trajectory, penalty=penalty, turn_cap=self.async_config.rollout_max_tokens
            )
            rewards[i] += term
            rows.append(
                (result.trajectory.reasoning_effort if result.trajectory is not None else None, term, charged, turns)
            )
        # A config gate is rank-uniform, so every rank records the batch keys, a rank with no episode too; a
        # level's keys come from the ranks that held it, and the flush folds the union.
        for level in {None, *(row[0] for row in rows)}:
            prefix = "reward" if level is None else f"effort/{level}"
            held = rows if level is None else [row for row in rows if row[0] == level]
            self._world_metrics.fraction(f"{prefix}/turn_overlong", sum(row[1] for row in held), len(held))
            self._world_metrics.fraction(
                f"{prefix}/turn_overlong_turn_frac", sum(row[2] for row in held), sum(row[3] for row in held)
            )

    def _compute_ref_logps(
        self,
        prompt_completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor | None:
        """Reference-model log-probs for the KL penalty (``None`` when ``beta == 0``)."""
        if self.beta == 0.0:
            return None

        # Chunked: a whole-batch DeepEP dispatch exceeds its 32-bit limit on long trajectories.
        with reference_policy(self) as model:
            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                compute_entropy=False,
            )

        return ref_per_token_logps

    def _check_step_has_valid_episodes(
        self, rollout_results: list[RolloutResult], gathered_valid: torch.Tensor
    ) -> None:
        """Raise when no episode in the world survived the rollout.

        Such a step trains on an all-masked batch (zero gradient) while the log line still reads
        ``loss=0, reward=0``, so a wedged rollout server, a downed sandbox or an unreachable grader
        can go unnoticed. A single such step can be a grading blip, so the first is a warning and the
        second consecutive one raises, carrying the rollout layer's error text.

        Judged on ``gathered_valid``, the world-gathered validity mask every rank holds alike, so the
        check is rank-symmetric: a rank raising on its own local episodes would leave its peers waiting
        in the next collective.
        """
        if gathered_valid.any():
            self._empty_rollout_steps = 0
            return
        self._empty_rollout_steps += 1
        errors = sorted({r.error for r in rollout_results if r.error})
        untrainable = sorted(
            {
                r.trajectory.info[EPISODE_INVALID_REASON_KEY]
                for r in rollout_results
                if r.trajectory is not None and EPISODE_INVALID_REASON_KEY in r.trajectory.info
            }
        )
        if errors:
            detail = f" Rollout error: {errors[0]}"
        elif untrainable:
            detail = f" Episodes were dropped as untrainable: {untrainable[0]}"
        else:
            detail = " No rollout carried an error: the environment marked every episode invalid."
        if self._empty_rollout_steps < EMPTY_ROLLOUT_STEP_LIMIT:
            logger.warning(
                f"Every rollout in this step failed or was marked invalid — the step contributes no "
                f"gradient. Halting if the next step is empty too.{detail}"
            )
            return
        raise RuntimeError(
            f"{self._empty_rollout_steps} consecutive steps produced no valid episode anywhere in the "
            f"world, so the run is training on all-masked batches (zero gradient) while logging "
            f"loss=0.{detail}"
        )

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        num_generations: int,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GRPO group-relative advantages of each trajectory's total reward, shaping included."""
        return group_relative_advantages(
            rewards,
            num_generations,
            self.args.scale_rewards,
            valid_mask=valid_mask,
            std_floor=self._scale_rewards_std_floor,
        )

    def _update_breaker_tripped(
        self,
        importance_sampling_ratio: torch.Tensor,
        eff_corrected: torch.Tensor,
        traj_row_ids: torch.Tensor | None,
        num_trajectories: int,
        mode: str,
    ) -> bool:
        """Whether the trust-region circuit breaker fires on the IS-mask stages (``skip_update_masked_frac``).

        Two GLOBAL fractions feed it and either one above the threshold trips it: the share of
        IS-corrected trajectories whose every corrected loss token had its ratio zeroed by the
        band/veto/OPSM stages, and the share of corrected loss tokens zeroed — the masked
        trajectories are the long ones, so a trajectory count alone reads a step that lost most of
        its tokens as healthy. Past the threshold the survivors are a selection-biased sample of
        wherever the drifted policy still agrees with the rollouts — training on them amplifies
        the drift — so the caller zeroes the round's advantages and every optimizer step the
        generation round feeds (``num_iterations`` of them at the default ``steps_per_generation``)
        is skipped at pre-optimizer-step (:meth:`_skip_optimizer_step_if_breaker_tripped`); the
        next weight sync ships unchanged weights and the rollouts re-anchor. Any configured KL term
        is dropped with the round. The verdict is returned rather than applied because the per-row
        and the per-trajectory advantages both have to follow it — the second is what the durable
        completions record writes. Global fractions, so every DP rank acts identically.
        """
        if self._skip_update_masked_frac is None or mode != "train" or traj_row_ids is None:
            return False
        device = importance_sampling_ratio.device
        rows_valid = traj_row_ids >= 0
        corrected_tokens = eff_corrected & rows_valid.unsqueeze(1)
        masked_tokens = corrected_tokens & (importance_sampling_ratio <= 0)
        row_corrected = eff_corrected.any(dim=1) & rows_valid
        row_surviving = ((importance_sampling_ratio > 0) & eff_corrected).any(dim=1) & rows_valid
        traj_corrected = torch.zeros(num_trajectories, dtype=torch.bool, device=device)
        traj_surviving = torch.zeros(num_trajectories, dtype=torch.bool, device=device)
        traj_corrected[traj_row_ids[row_corrected]] = True
        traj_surviving[traj_row_ids[row_surviving]] = True
        masked = traj_corrected & ~traj_surviving
        traj_frac, token_frac = gathered_fractions(
            [(masked.sum(), traj_corrected.sum()), (masked_tokens.sum(), corrected_tokens.sum())],
            self.accelerator.gather,
        )
        self._metrics[mode]["sampling/is_masked_traj_frac"].append(traj_frac)
        self._metrics[mode]["sampling/is_masked_token_frac"].append(token_frac)
        skipped = max(traj_frac, token_frac) > self._skip_update_masked_frac
        self._metrics[mode][UPDATE_SKIPPED_KEY].append(float(skipped))
        # This round's verdict replaces the last one's: the flag stays up for every optimizer step
        # the round feeds and comes down only here.
        self._breaker_tripped_this_step = skipped
        if not skipped:
            return False
        logger.warning(
            f"IS trust region: {traj_frac:.0%} of corrected trajectories fully masked, {token_frac:.0%} of "
            f"corrected tokens masked (> skip_update_masked_frac={self._skip_update_masked_frac}) — zeroing "
            "this step's policy gradient and skipping its optimizer step; rollouts re-anchor at the next weight sync."
        )
        return True

    def _arm_update_breaker(self) -> None:
        """Take ``skip_update_masked_frac`` and, with it set, attach the pre-optimizer-step hook that acts on
        the breaker's verdict: without the knob no verdict trips, and the hook would run on every optimizer
        step for nothing. The verdict is set by :meth:`_update_breaker_tripped` once per generation round
        and held through every optimizer step that round feeds."""
        self._skip_update_masked_frac = self.async_config.skip_update_masked_frac
        self._breaker_tripped_this_step = False
        if self._skip_update_masked_frac is not None:
            self.add_callback(_BreakerOptimizerSkipCallback(self))

    def _skip_optimizer_step_if_breaker_tripped(self) -> None:
        """Drop the step's gradients while the breaker is tripped, so the optimizer steps no parameter.

        Every optimizer skips a parameter whose ``.grad`` is ``None``, moments included, while a
        zeroed loss still hands it zero gradients and Adam keeps stepping on momentum alone. Runs
        at pre-optimizer-step (after clipping) on every optimizer step of the round the verdict was
        taken for; the flag is not consumed here, since a round spans ``num_iterations`` optimizer
        steps and the zeroed advantages cover all of them.
        """
        if self._breaker_tripped_this_step:
            self.optimizer.zero_grad(set_to_none=True)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Training step with pre-generation weight synchronization and routing-replay disarm."""
        # Sync before delegating: this round's generation runs inside super().training_step, so a
        # tail-of-step sync would leave every rollout one optimizer step stale (train-begin force-syncs).
        # Gated on the attempt stamp, not the synced one: the cadence gate declines most steps, and
        # re-entering the path on every microbatch of a declined step costs a collective and a barrier.
        is_post_optimizer_step = (self.state.global_step > 0) and (
            self.state.global_step != self._last_sync_attempt_step
        )

        if is_post_optimizer_step:
            self._last_sync_attempt_step = self.state.global_step
            # The push holds every rank until it lands (sync_trainer_weights' barrier), so no step
            # starts its rollouts against a mid-update engine.
            self._sync_weights_to_engine_fenced()

        if self._routing_injector is None:
            return super().training_step(model, inputs, num_items_in_batch)
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        finally:
            # Disarm only after backward (the reentrant-GC recompute reads the armed mask during it).
            self._routing_injector.disarm()

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Log metrics including async-specific metrics (rollout means ride TRL's ``_metrics`` drain)."""
        mode = "eval" if not self.model.training else "train"

        # Both counter reads below are world reductions, gated on the config and the mode alone.
        if self._routing_injector is not None and mode == "train":
            # Drain the on-device flip counters once per logging step (never the per-microbatch hot path).
            flip = self._routing_injector.flip_rate()
            if flip is not None:
                logs["routing/replay_flip_rate"] = flip

        if self._rollout_manager:
            # Cumulative totals since train start, not this-batch means (those are _log_rollout_metrics).
            logs.update(self.cumulative_rollout_metrics())

        if self._prefetch_enabled:
            logs.update(self.prefetch_metrics())

        super().log(logs, start_time)
