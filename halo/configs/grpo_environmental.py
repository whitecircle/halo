"""Async environmental GRPO: the script arguments, ``EnvironmentConfig`` and ``AsyncRolloutConfig``."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import timedelta
from typing import Any, ClassVar, Literal

from halo.config_validation import (
    ValidatedConfig,
    differs_from_default,
    present,
    require_finite,
    require_int,
    require_text,
)
from halo.configs.grpo import (
    AdvantageShapingArguments,
    ChunkedLogprobsArguments,
    GRPOEarlyStopArguments,
    PackedMicrobatchArguments,
    PhaseTimerArguments,
    PromptDatasetArguments,
    RolloutWeightPrecisionArguments,
    ScoreCenteringArguments,
)
from halo.configs.rollout_arguments import RolloutBudgetArguments, RolloutSamplingArguments
from halo.configs.script_arguments import DEFAULT_ANSWER_FIELD, CommonScriptArguments
from halo.env import DEFAULT_NCCL_TIMEOUT_MINUTES, WATCHDOG_WARN_FRACTION, nccl_timeout
from halo.environments.rollout_config import (
    SGLANG_BACKEND,
    RolloutBackend,
    RolloutConfig,
)
from halo.inference.endpoints import DEFAULT_LOCAL_SERVER_URL
from halo.rewards.terms import ENVIRONMENT_REWARD_SOURCES, RewardTerm, parse_reward_terms, read_context_files
from halo.runtime.ranks import is_global_main_process
from halo.weight_sync.group_layout import SyncTopologySetting, validate_sync_topology

logger = logging.getLogger(__name__)

_ROLLOUT_SERVER_KEYS = frozenset({"url", "group_port", "group_host", "node"})


@dataclass(frozen=True)
class ISRMaskConfig:
    """Mask/veto stages on the truncated IS ratio, all off by default: the geometric-mean band (set both ends) masks a
    trajectory whose ``exp(mean log-ratio)`` over corrected tokens leaves it, the veto one with any corrected token's
    ratio below ``veto_min``, and OPSM (``opsm_delta``) a negative-advantage one past that many nats."""

    geo_band_min: float | None = None
    geo_band_max: float | None = None
    veto_min: float | None = None
    opsm_delta: float | None = None

    def __post_init__(self):
        lo, hi = self.geo_band_min, self.geo_band_max
        if (lo is None) != (hi is None):
            raise ValueError("isr_geo_band_min and isr_geo_band_max must be set together")
        owner = type(self).__name__
        require_finite(owner, gt=0, lt=1, **present(isr_geo_band_min=lo, isr_veto_min=self.veto_min))
        require_finite(owner, gt=1, **present(isr_geo_band_max=hi))
        require_finite(owner, gt=0, **present(isr_opsm_delta=self.opsm_delta))

    @property
    def any_mask_active(self) -> bool:
        """A stage :func:`apply_is_masks` applies (the geometric band or the veto) is set."""
        return self.geo_band_min is not None or self.veto_min is not None

    @property
    def any_stage_active(self) -> bool:
        return self.any_mask_active or self.opsm_delta is not None

    @property
    def sums_sequence_logratio(self) -> bool:
        """A stage reads the log-ratio summed over a trajectory (the band and OPSM); the veto reads single tokens. The
        sampler-logprob preflight holds such a run to engine log-probs not renormalized over the sampler's cut."""
        return self.geo_band_min is not None or self.opsm_delta is not None


@dataclass
class ISCorrectionArguments(ValidatedConfig):
    """The mask stages layered on the truncated importance-sampling ratio, and the trust-region breaker over them."""

    isr_geo_band_min: float | None = field(
        default=None,
        metadata={
            "help": "Trajectory geometric-mean band: mask a whole trajectory when exp(mean log-ratio) over its "
            "corrected tokens leaves [isr_geo_band_min, isr_geo_band_max]. Set both to activate. "
            "See agent-docs/training-methods/async-grpo/objective.md#trust-region-masks."
        },
    )
    isr_geo_band_max: float | None = field(
        default=None, metadata={"help": "Upper bound of the trajectory geometric-mean band (see isr_geo_band_min)."}
    )
    isr_veto_min: float | None = field(
        default=None,
        metadata={
            "help": "Catastrophic-token veto: mask a whole trajectory when any corrected token's raw IS ratio falls "
            "below this (verl and slime use about 1e-4). Unset = off."
        },
    )
    isr_opsm_delta: float | None = field(
        default=None,
        metadata={
            "help": "Off-policy sequence masking (DeepSeek-V3.2): mask negative-advantage trajectories whose |mean "
            "log-ratio| against the sampling policy exceeds this many nats; positives are never masked. Unset = "
            "off."
        },
    )
    isr_engine_reference: bool = field(
        default=False,
        metadata={
            "help": "Re-score every training row on the rollout engine under the just-synced weights, so the mask "
            "stages read the engine's current-vs-sampling log-ratio without the trainer-vs-engine disagreement. "
            "Requires the IS correction and plain sampling (temperature and top-p 1, no top-k, min-p or repetition "
            "penalty). See agent-docs/training-methods/async-grpo/objective.md#engine-reference."
        },
    )
    isr_engine_rescore_per_server: int = field(
        default=128,
        metadata={
            "help": "Re-score requests in flight on each rollout server under isr_engine_reference, summed over every "
            "rank: each rank keeps at most its share (this ÷ world size, at least one) in flight on each server, so "
            "the load on a server does not grow with the world size. "
            "See agent-docs/training-methods/async-grpo/objective.md#engine-reference."
        },
    )
    skip_update_masked_frac: float | None = field(
        default=None,
        metadata={
            "help": "Trust-region circuit breaker: when the mask stages fully mask more than this fraction of the "
            "IS-corrected trajectories, or mask more than this fraction of the corrected tokens, zero the round's "
            "advantages and skip its optimizer steps instead of training on the survivors. Unset = off; 0.3-0.5 is "
            "a sane range. See agent-docs/training-methods/async-grpo/objective.md#breaker."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        self.build_isr_mask_config()
        require_int(type(self).__name__, ge=1, isr_engine_rescore_per_server=self.isr_engine_rescore_per_server)
        if not self.isr_engine_reference and differs_from_default(self, "isr_engine_rescore_per_server"):
            raise ValueError(
                "isr_engine_rescore_per_server set with isr_engine_reference off: nothing re-scores on the engine. "
                "Remove it, or set isr_engine_reference."
            )
        # A fraction of the step's corrected trajectories/tokens; 0 would trip on every step, >1 never.
        require_finite(
            type(self).__name__, gt=0, le=1, **present(skip_update_masked_frac=self.skip_update_masked_frac)
        )

    def build_isr_mask_config(self) -> ISRMaskConfig:
        return ISRMaskConfig(
            geo_band_min=self.isr_geo_band_min,
            geo_band_max=self.isr_geo_band_max,
            veto_min=self.isr_veto_min,
            opsm_delta=self.isr_opsm_delta,
        )


@dataclass
class ReasoningCostArguments(ValidatedConfig):
    """What an episode pays for its reasoning length: the cap on the per-level price (each level's profile sets the
    price) and the under-use floor."""

    reasoning_price_cap: float = field(
        default=0.1,
        metadata={
            "help": "Cap of the reasoning price per episode, in reward units, so a long trace cannot outweigh the task "
            "reward. The price itself is per effort level: a level's profile (environment_kwargs."
            "reasoning_effort_profiles) charges reasoning_price per 1,000 reasoning tokens an episode spends past its "
            "reasoning_budget, summed over turns. Keep it below what the environment charges for the decisions it "
            "prices: an episode pays it or reasoning_floor, never both. Set while no drawable level prices reasoning, "
            "it is refused at trainer construction. Logged as reward/reasoning_price. "
            "See agent-docs/training-methods/async-grpo/reasoning.md#price."
        },
    )
    reasoning_floor: float = field(
        default=0.0,
        metadata={
            "help": "Weight of the reasoning under-use floor (0 = off): a baseline episode that earned no objective and "
            "reasoned, summed over its turns, under its level's reasoning_budget pays -weight × shortfall / budget; a "
            "solve pays nothing. The one term that pays for more reasoning. Keep it below what the environment "
            "charges for the decisions it prices: an episode pays it or the price, never both. Every level the "
            "environment draws needs a reasoning_budget above 0 (refused at trainer construction otherwise). Logged "
            "as reward/reasoning_floor. See agent-docs/training-methods/async-grpo/reasoning.md#floor."
        },
    )
    reasoning_terms_in_failed_groups: bool = field(
        default=False,
        metadata={
            "help": "Charge the reasoning price and reasoning_floor in a training group none of whose baseline episodes "
            "earned the objective (its reward/objective component, zeroed by a veto). Off by default: there the "
            "length terms are most of what tells the attempts apart, so the advantage would train reasoning length "
            "rather than the task. A group with a success is charged either way, and eval episodes always are. Off, "
            "it needs an environment term of positive weight (refused at trainer construction otherwise). Logged as "
            "outcome/reasoning_terms_waived_frac. "
            "See agent-docs/training-methods/async-grpo/reasoning.md#failed-groups."
        },
    )

    def _validate_fields(self) -> None:
        """A NaN passes every ordered comparison. Whether some level prices reasoning, which the cap and the
        failed-group switch need, is checked at trainer construction, beside the environment's profiles."""
        super()._validate_fields()
        owner = type(self).__name__
        require_finite(owner, gt=0, reasoning_price_cap=self.reasoning_price_cap)
        # A NaN or negative weight would parse and leave the floor off without a word.
        require_finite(owner, ge=0, why=" (0 = off)", reasoning_floor=self.reasoning_floor)


@dataclass
class AsyncRolloutConfig(
    RolloutSamplingArguments,
    RolloutBudgetArguments,
    ISCorrectionArguments,
    ReasoningCostArguments,
    AdvantageShapingArguments,
    GRPOEarlyStopArguments,
    ChunkedLogprobsArguments,
    PackedMicrobatchArguments,
    PhaseTimerArguments,
    RolloutWeightPrecisionArguments,
    ScoreCenteringArguments,
):
    """The async GRPO trainer's rollout side: Ray workers, rollout-server connections, weight sync, prefetch and
    timeouts, over the sampling, budget, IS-correction and reasoning-cost bundles."""

    rollout_actor_budget: int = field(
        default=256,
        metadata={
            "help": "Ray environment actors per node, split across that node's rollout ranks; with ray_address set, "
            "the cluster-wide total, split across every rollout rank. Each actor is async and runs many episodes, so "
            "this is no concurrency limit (max_concurrent_rollouts is). Size it to the CPU-side environment cost; a "
            "rank left under 4 actors warns. "
            "See agent-docs/infrastructure/ray.md#pool-sizing."
        },
    )
    max_concurrent_rollouts: int | None = field(
        default=None,
        metadata={
            "help": "Per-rank cap on one round's rollouts in flight, the generation-throughput throttle. Each round "
            "in flight holds its own cap: the prefetch_depth prefetched rounds, and a cold or eval round beside them. "
            "Unset = the larger of a train round's and an eval round's rows, so a round runs in one wave; a value "
            "below this rank's actor count is raised to it. "
            "See agent-docs/reference/configuration-reference.md#asyncrolloutconfig."
        },
    )
    eval_rollout_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Rows per rank in one evaluation rollout round (each prompt already repeated "
            "num_generations_eval times), a multiple of num_generations_eval and at most max_concurrent_rollouts. "
            "Unset leaves the round at the eval batch. "
            "See agent-docs/training-methods/async-grpo/evaluation.md#during-training."
        },
    )
    ray_address: str | None = field(
        default=None, metadata={"help": "Ray cluster address; unset starts a local Ray instance on each rank."}
    )
    rollout_backend: RolloutBackend = field(
        default=RolloutConfig.backend,
        metadata={
            "help": "Inference engine serving rollouts and receiving the weight updates. 'sglang' refuses "
            "rollout_max_thinking_tokens and must be served from the NCCL-aligned Dockerfile.sglang image with "
            "NCCL_CUMEM_ENABLE=1. "
            "See agent-docs/training-methods/async-grpo/setup.md#rollout-backend."
        },
    )
    rollout_server_url: str = field(
        default=DEFAULT_LOCAL_SERVER_URL,
        metadata={"help": "Primary rollout-server URL (vLLM or SGLang) for weight sync and generation."},
    )
    rollout_connection_timeout: float = field(
        default=120.0, metadata={"help": "Timeout in seconds to wait for the rollout server."}
    )
    rollout_server_configs: list[dict[str, Any]] | None = field(
        default=None,
        metadata={
            "help": "Rollout servers of a multi-server run, each a dict with 'url' and optional 'group_port' and "
            "'group_host' (the weight-transfer address the serving node dials back to), like "
            "[{'url': 'http://node1:8000', 'group_port': 51216}]. Any other key raises, as does a non-default "
            "rollout_server_url beside a non-empty list."
        },
    )
    rollout_group_port: int = field(
        default=51216,
        metadata={
            "help": "Trainer-side port of the NCCL weight-sync group, on either rollout backend; 0 takes a "
            "kernel-assigned port. With several servers it is the base: a rollout_server_configs entry without its "
            "own group_port binds rollout_group_port + index, or a kernel-assigned port of its own when this is 0. "
            "See agent-docs/training-methods/async-grpo/setup.md#weight-synchronization."
        },
    )
    rollout_sync_topology: SyncTopologySetting = field(
        default="per_server",
        metadata={
            "help": "How the trainer's weight-sync groups span the rollout servers: per_server (default) forms one "
            "NCCL group per server, single_group one group over all of them that broadcasts each chunk once, and auto "
            "picks single_group for two or more servers that report the same served model and weight-loading flags "
            "(and that one group can rank) and per_server otherwise. single_group and auto need every server to see "
            "every server's GPU (NCCL peer-to-peer), so servers that each see only their own GPU stay on per_server. "
            "single_group takes one rollout_group_port, one group_host, and an optional 'node' label per "
            "rollout_server_configs entry that orders the servers by node; auto refuses 'node'. See agent-docs/infrastructure/rollout-servers.md#one-group-over-every-server."
        },
    )
    rollout_adapter_sync: bool = field(
        default=False,
        metadata={
            "help": "Serve a LoRA policy as an adapter beside the frozen base instead of pushing it folded into the "
            "weights: the first sync pushes the base once, and each later sync loads the adapter alone under a new "
            "name that new episodes select. SGLang only, launched with --enable-lora; attention and dense-MLP "
            "projections only. A model it cannot serve is refused at startup with the reason. "
            "See agent-docs/infrastructure/rollout-servers.md#adapter-sync."
        },
    )
    sync_weights_every_n_steps: int = field(
        default=1, metadata={"help": "Sync weights to the rollout servers every N training steps, N >= 1."}
    )
    train_on_sampled_tokens: bool = field(
        default=True,
        metadata={
            "help": "Train on the server's sampled token ids, each assistant turn its own row sharing the "
            "trajectory's advantage, instead of re-tokenizing a chat-template re-render. vLLM needs "
            "--return-tokens-as-token-ids (SGLang no flag), and a trajectory with an uncaptured turn falls back to "
            "one re-tokenized row. See agent-docs/training-methods/async-grpo/objective.md#training-rows."
        },
    )
    routing_replay: Literal["none", "recompute", "rollout"] = field(
        default="none",
        metadata={
            "help": "MoE routing replay: pin the update pass's top-k expert selection to a recorded mask, "
            "re-deriving gate weights from live router scores. 'none' (the default) is off, 'recompute' (R2) "
            "records the trainer's own no-grad recompute pass and 'rollout' (R3, experimental) the rollout engine's "
            "routing. "
            "See agent-docs/training-methods/async-grpo/objective.md#routing-replay."
        },
    )
    length_ordered_microbatches: bool | None = field(
        default=None,
        metadata={
            "help": "Deal each training round's rows into its micro-batches longest first, every rank alike, in "
            "place of TRL's per-rank shuffle, and run the round's no-grad forwards in the same order: under ZeRO-3 "
            "or EP the ranks forward their rows in lock-step, so each row costs the longest one its peers forward "
            "with it. Only the row order changes. Unset = on wherever that holds (aligned accumulation, a loss that "
            "does not read micro-batch composition, rows forwarded one per call); true raises where it does not; "
            "false keeps TRL's shuffle. "
            "See agent-docs/training-methods/async-grpo/performance.md#length-ordered-micro-batches."
        },
    )
    loss_forward_importance_weights: bool = field(
        default=False,
        metadata={
            "help": "Take the importance weight (and the KL tail clamp) from the loss forward's own detached "
            "log-probs in place of a separate no-grad pass over the round's rows. The weight is the same function of "
            "the same weights on an aligned one-step round, built from a different forward program, so values differ "
            "by the forward's log-prob noise. Raises at construction where a consumer needs the round's log-probs "
            "before its first backward (token-mass balance, trajectory-level isr_* stages, the update breaker, "
            "routing_replay: recompute, num_iterations above 1, misaligned accumulation) or no pass would run. With "
            "early_stop_logratio_gap set, the logged log-ratio reads loss-mask tokens only. "
            "See agent-docs/training-methods/async-grpo/performance.md#weight-from-the-loss-forward."
        },
    )
    early_stop_on_skipped_updates: bool = field(
        default=False,
        metadata={
            "help": "Early stop: end training once the trust-region breaker skipped every update on "
            "early_stop_patience readings in a row. Requires skip_update_masked_frac. "
            "See agent-docs/training-methods/async-grpo/monitoring.md#early-stop."
        },
    )
    truncation_alarm_rate: float | None = field(
        default=0.25,
        metadata={
            "help": "Warn when a rollout round's episode/truncation_rate (episodes ended by the turn or token "
            "budget, not the task) rises over this, logging episode/truncation_alarm every round. In [0, 1); unset "
            "= off."
        },
    )
    zero_gradient_round_limit: int | None = field(
        default=50,
        metadata={
            "help": "Raise after this many train rounds in a row in which no token anywhere carries a policy gradient, "
            "while no earlier round of this launch has: every group tied or dropped, or every token masked, trains on "
            "nothing while logging loss 0. Past the first round that trained, a streak this long only warns. Logged as "
            "sampling/zero_gradient_round. Positive; null = off."
        },
    )
    episode_error_rate_limit: float | None = field(
        default=0.5,
        metadata={
            "help": "Raise once a train round's world-wide episode/error_rate (episodes that errored or timed out, "
            "which train as masked rows) is over this for episode_error_round_limit rounds in a row: the batch is "
            "then mostly the episodes that finished fast, a bias toward short answers, or the rollout path is down. "
            "In [0, 1); null = off. See agent-docs/training-methods/async-grpo/monitoring.md."
        },
    )
    episode_error_round_limit: int = field(
        default=3,
        metadata={"help": "Consecutive train rounds over episode_error_rate_limit that raise. Positive."},
    )
    tool_parser_watch_rounds: int | None = field(
        default=2,
        metadata={
            "help": "On a tool environment, raise after this many rounds hold the template's rendered tool-call opening "
            "in an assistant turn's text while the run has parsed no call anywhere: the server's tool parser does not "
            "read this model's format. A turn the engine cut is no evidence. Positive; null = off, for a model that "
            "writes the opening in prose or only malformed calls."
        },
    )
    # On by default here: on a sparse verifiable env reward the all-equal groups dominate the batch.
    drop_degenerate_groups: bool = field(
        default=True,
        metadata={
            "help": "Drop GRPO groups whose completions all settled the same environment reward (the reasoning "
            "price and floor left out), which carry no task contrast yet dilute the loss normalizer. Logged as "
            "sampling/degenerate_group_frac. See agent-docs/training-methods/async-grpo/objective.md#advantages."
        },
    )
    rollout_stop_tokens: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Special-token strings that end a turn, sent as the engine's stop_token_ids; list the model's "
            "tool-call terminator when it is no eos (gpt-oss <|call|> under harmony-disabled serving, refused without "
            "it on a tool environment). Empty = only eos stops a turn. "
            "See agent-docs/training-methods/async-grpo/episodes.md#tool-calls."
        },
    )
    rollout_chat_template_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Chat-template variables sent with every rollout request (chat_template_kwargs) and applied to "
            "the trainer's own renders, so both sides see one template state. reasoning_effort is refused here: "
            "it is per episode and travels as the request's top-level field. "
            "See agent-docs/training-methods/async-grpo/episodes.md#chat-template."
        },
    )
    enable_prefetch: bool | None = field(
        default=None,
        metadata={
            "help": "Roll the next rounds out while this one trains, so each round trains prefetch_depth rounds stale "
            "(the sampling-logprob IS ratio corrects it). Unset = on with two or more rollout servers, off with one; "
            "true prefetches on a single server too, false never. "
            "See agent-docs/training-methods/async-grpo/setup.md#multiple-servers-and-prefetch."
        },
    )
    prefetch_depth: int = field(
        default=1,
        metadata={
            "help": "Rounds prefetch collects ahead of the step that trains them, together on the engines and "
            "delivered in submission order: each round trains on samples up to this many rounds stale (one round "
            "per optimizer step at the default steps_per_generation), which the sampling-logprob IS ratio corrects. "
            "Raise it when the trainer waits on rollouts every step while the engines drain at each round's tail. "
            "Costs staleness, engine KV for every round in flight, and this many cold rounds when a run starts. "
            "An int >= 1; above 1 it needs a run that prefetches. "
            "See agent-docs/training-methods/async-grpo/setup.md#multiple-servers-and-prefetch."
        },
    )
    model_name: str | None = field(
        default=RolloutConfig.model_name,
        metadata={
            "help": "Model name sent in the rollout server's /v1/chat/completions requests. "
            "Unset is filled with model_name_or_path at script start, so a request always names one."
        },
    )
    request_timeout: float = field(
        default=RolloutConfig.request_timeout,
        metadata={
            "help": "HTTP timeout per rollout-server request in seconds of engine-serving time: a weight-sync "
            "pause is credited back, so a sync never expires a request."
        },
    )
    episode_timeout: float = field(
        default=RolloutConfig.episode_timeout,
        metadata={
            "help": (
                "Deadline in seconds for one whole rollout episode (generation, tools, grading), with "
                "weight-sync pauses credited back; a timed-out episode is canceled and counted in "
                "episode/error_rate. It must stay under the NCCL watchdog: the default is two thirds of the "
                f"{DEFAULT_NCCL_TIMEOUT_MINUTES}-min default, so raise DIST_NCCL_TIMEOUT_MINUTES before raising it."
            )
        },
    )
    max_retries: int = field(
        default=RolloutConfig.max_retries,
        metadata={
            "help": "Retries after a failed rollout-server request (total attempts = max_retries + 1). "
            "0 = one attempt, no retry."
        },
    )
    retry_base_wait: float = field(
        default=RolloutConfig.retry_base_wait,
        metadata={"help": "Base wait time in seconds for exponential backoff between retries."},
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        owner = type(self).__name__
        # A 0 divides by zero (the sync period, the empty actor list), reads as unset or fails inside the DataLoader.
        require_int(
            owner,
            ge=1,
            sync_weights_every_n_steps=self.sync_weights_every_n_steps,
            rollout_actor_budget=self.rollout_actor_budget,
            prefetch_depth=self.prefetch_depth,
            **present(
                max_concurrent_rollouts=self.max_concurrent_rollouts,
                eval_rollout_batch_size=self.eval_rollout_batch_size,
            ),
        )
        require_int(owner, ge=0, le=65535, rollout_group_port=self.rollout_group_port)
        self._validate_rollout_servers(owner)
        # Above 1 the depth sizes the prefetch pipeline; without one it would parse and hold no round.
        if self.prefetch_depth > 1 and not self.prefetch_active():
            why = (
                "enable_prefetch is false. Enable prefetch"
                if self.enable_prefetch is False
                else "enable_prefetch is unset, which prefetches on two or more rollout servers, and this run has "
                "one. Set enable_prefetch: true to prefetch on one server, configure two or more servers in "
                "rollout_server_configs"
            )
            raise ValueError(
                f"{owner}: prefetch_depth={self.prefetch_depth} sets how many rounds prefetch holds, but {why}, "
                "or remove prefetch_depth."
            )
        # A negative budget reaches backoff as no limit: a wedged server is retried until the NCCL watchdog fires.
        require_int(owner, ge=0, why=" (0 = one attempt, no retry)", max_retries=self.max_retries)
        # Wall-clock: a non-positive deadline cancels every episode on entry, halting the run as an empty batch.
        require_finite(
            owner,
            gt=0,
            request_timeout=self.request_timeout,
            episode_timeout=self.episode_timeout,
            rollout_connection_timeout=self.rollout_connection_timeout,
        )
        # A negative base shrinks the retry backoff instead of growing it.
        require_finite(owner, ge=0, retry_base_wait=self.retry_base_wait)
        # A rate is never above 1, so a threshold at 1 would never fire.
        require_finite(owner, ge=0, lt=1, **present(truncation_alarm_rate=self.truncation_alarm_rate))
        require_int(owner, ge=1, **present(zero_gradient_round_limit=self.zero_gradient_round_limit))
        require_finite(owner, ge=0, lt=1, **present(episode_error_rate_limit=self.episode_error_rate_limit))
        require_int(owner, ge=1, episode_error_round_limit=self.episode_error_round_limit)
        require_int(owner, ge=1, **present(tool_parser_watch_rounds=self.tool_parser_watch_rounds))
        # The breaker is the only writer of the skip flag: without it the condition could never fire.
        if self.early_stop_on_skipped_updates and self.skip_update_masked_frac is None:
            raise ValueError(
                "early_stop_on_skipped_updates needs skip_update_masked_frac: the trust-region breaker is "
                "what skips an update, so without it the condition never holds."
            )
        if not isinstance(self.rollout_chat_template_kwargs, Mapping):
            raise ValueError(
                "rollout_chat_template_kwargs must be a mapping of template variables, got "
                f"{type(self.rollout_chat_template_kwargs).__name__}"
            )
        if "reasoning_effort" in self.rollout_chat_template_kwargs:
            raise ValueError(
                "rollout_chat_template_kwargs must not carry 'reasoning_effort': the level is per episode and "
                "travels as the request's top-level field."
            )
        if self.rollout_weight_precision != "bf16" and self.rollout_adapter_sync:
            raise ValueError(
                f"rollout_weight_precision={self.rollout_weight_precision!r} cannot be combined with "
                "rollout_adapter_sync: the base is pushed once and the adapter is served apart from it, so no folded "
                "weight is ever quantized and the server would hold a different policy than the trainer. Set one of "
                "them back to its default."
            )
        # SGLang drops the field silently. The level still steers through the template; this only caps.
        if self.rollout_backend == SGLANG_BACKEND and self.rollout_max_thinking_tokens is not None:
            raise ValueError(
                "rollout_max_thinking_tokens is not supported with rollout_backend='sglang': the "
                "thinking_token_budget request field is vLLM-only and SGLang would silently ignore it, "
                "leaving reasoning uncapped. Steer with the environment's reasoning_effort instead."
            )

    def _validate_rollout_servers(self, owner: str) -> None:
        """The pool reads an entry by key: a misspelled ``group_port`` takes ``rollout_group_port + index`` instead, and
        a URL beside the list is never dialed."""
        servers = self.rollout_server_configs
        if servers is None:
            return
        if not isinstance(servers, list):
            raise ValueError(f"{owner}: rollout_server_configs must be a list of mappings, got {servers!r}")
        for index, server in enumerate(servers):
            where = f"{owner}: rollout_server_configs[{index}]"
            if not isinstance(server, Mapping):
                raise ValueError(f"{where} must be a mapping with a 'url', got {server!r}")
            unknown = sorted(map(str, set(server) - _ROLLOUT_SERVER_KEYS))
            if unknown:
                raise ValueError(f"{where} has unknown keys {unknown}; allowed: {sorted(_ROLLOUT_SERVER_KEYS)}")
            require_text(where, url=server.get("url"))
            if "group_port" in server:
                require_int(where, ge=0, le=65535, group_port=server["group_port"])
            if "group_host" in server:
                require_text(where, group_host=server["group_host"])
            if "node" in server:
                require_text(where, node=server["node"])
        validate_sync_topology(self.rollout_sync_topology, servers)
        if servers and differs_from_default(self, "rollout_server_url"):
            raise ValueError(
                f"{owner}: rollout_server_url={self.rollout_server_url!r} is set beside rollout_server_configs, whose "
                "entries alone name the servers, so it would never be dialed. Remove it, or add it to the list."
            )

    def _stops_on_skipped_updates(self) -> bool:
        return self.early_stop_on_skipped_updates

    def get_server_urls(self) -> list[str]:
        if self.rollout_server_configs:
            return [c["url"] for c in self.rollout_server_configs]
        return [self.rollout_server_url]

    def prefetch_active(self) -> bool:
        """``enable_prefetch`` where set, else two or more rollout servers. Every push pauses all servers together, so
        one server overlaps as several do; unset keeps a lone one on-policy."""
        if self.enable_prefetch is None:
            return len(self.get_server_urls()) > 1
        return self.enable_prefetch

    @classmethod
    def rollout_field_sources(cls) -> dict[str, str]:
        """Each ``RolloutConfig`` field -> its ``rollout_<name>`` spelling where declared, else the same name; fields
        with neither are derived."""
        declared = {f.name for f in fields(cls)}
        return {
            target.name: source
            for target in fields(RolloutConfig)
            for source in (target.name, f"rollout_{target.name}")
            if source in declared
        }

    def get_rollout_config(
        self,
        stop_token_ids: list[int] | None = None,
        reasoning_end_token_id: int | None = None,
        *,
        in_process_group: bool = True,
    ) -> RolloutConfig:
        """The caller owning the tokenizer resolves ``stop_token_ids`` and ``reasoning_end_token_id``; an eval joining
        no process group passes ``in_process_group=False``, skipping the NCCL-watchdog bound."""
        if in_process_group:
            self._validate_timeouts_against_nccl_watchdog()
        mirrored = {target: getattr(self, source) for target, source in self.rollout_field_sources().items()}
        mirrored["chat_template_kwargs"] = dict(self.rollout_chat_template_kwargs)
        return RolloutConfig(
            **mirrored,
            capture_token_ids=self.train_on_sampled_tokens,
            capture_candidate_logprobs=self.score_centering_top_k or 0,
            capture_routed_experts=self.routing_replay == "rollout",
            stop_token_ids=stop_token_ids,
            reasoning_end_token_id=reasoning_end_token_id,
        )

    def _validate_timeouts_against_nccl_watchdog(self):
        """Refuse an ``episode_timeout`` above the NCCL watchdog and warn near it: a straggler holds its peers at the
        per-step collective, so past the watchdog their collective aborts before the straggler is cancelled."""
        nccl_minutes = nccl_timeout() // timedelta(minutes=1)
        watchdog = nccl_minutes * 60
        if self.episode_timeout > watchdog:
            raise ValueError(
                f"episode_timeout ({self.episode_timeout:.0f}s) must be below the NCCL collective "
                f"watchdog ({watchdog:.0f}s = {nccl_minutes} min). A straggler rank holds its peers at "
                f"the per-step collective for up to episode_timeout, so an equal/greater value lets the "
                f"watchdog fire first and abort the run. Raise DIST_NCCL_TIMEOUT_MINUTES above "
                f"episode_timeout (keep ≥15 min margin so the cancelled rank can unwind and rejoin), or "
                f"lower episode_timeout."
            )
        if self.episode_timeout >= WATCHDOG_WARN_FRACTION * watchdog and is_global_main_process():
            logger.warning(
                f"episode_timeout ({self.episode_timeout:.0f}s) is within "
                f"{(1 - WATCHDOG_WARN_FRACTION) * 100:.0f}% of the NCCL watchdog "
                f"({watchdog:.0f}s): a near-deadline straggler risks tripping the peers' per-step "
                f"collective. Raise DIST_NCCL_TIMEOUT_MINUTES for more margin."
            )

        attempts = self.max_retries + 1
        backoff = self.retry_base_wait * (2**self.max_retries - 1)
        worst_case = attempts * self.request_timeout + backoff
        if worst_case >= WATCHDOG_WARN_FRACTION * watchdog and is_global_main_process():
            logger.warning(
                f"Rollout retry budget (~{worst_case:.0f}s = {attempts}×{self.request_timeout:.0f}s "
                f"request_timeout + backoff) is close to the {watchdog:.0f}s NCCL collective watchdog. "
                f"A stuck rollout will retry until the watchdog fires and HANG the per-step barrier "
                f"instead of giving up. Lower request_timeout or max_retries (a ≤16k-token turn "
                f"generates in ~110s, so request_timeout≈300 with max_retries=3 is ample), or raise "
                f"DIST_NCCL_TIMEOUT_MINUTES."
            )


@dataclass
class EnvironmentConfig(ValidatedConfig):
    """Environment selection, reward terms and turn budget, resolved through halo/environments/registry.py."""

    environment_type: str = field(
        default="react_math",
        metadata={
            "help": "Environment registry name: get_registered_environments() lists the live set, which "
            "custom register_environment() calls extend; an unknown name raises with the available names."
        },
    )
    rewards: list[dict[str, Any]] = field(
        default_factory=lambda: [{"source": "environment"}],
        metadata={
            "help": "Reward terms, each {source, name?, weight?, exponent?, ...}, summed into the episode reward as "
            "weight × score^exponent: 'environment' (the environment's own grade), 'judge' or 'reward_model'. The "
            "environment's per-turn and shaping terms add on top; YAML-only. "
            "See agent-docs/training-methods/grpo/rewards/README.md#environment-arm."
        },
    )
    max_turns: int | None = field(
        default=None,
        metadata={
            "help": "Maximum environment turns per episode, an int >= 1. Unset keeps the environment class's own "
            "default. See agent-docs/training-methods/async-grpo/environments/README.md#registry."
        },
    )
    environment_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Environment-specific kwargs passed to the registry factory, like {search_backend: duckduckgo} "
            "or {timeout_per_test: 10}. "
            "See agent-docs/training-methods/async-grpo/environments/README.md#configuration."
        },
    )

    def __post_init__(self):
        # Spelled as the registry keys it, so every consumer comparing the name reads the type the run resolves.
        if not isinstance(self.environment_type, str):
            raise ValueError(f"environment_type must be a registry name, got {self.environment_type!r}")
        self.environment_type = self.environment_type.lower()
        self.rewards = read_context_files(self.rewards)
        super().__post_init__()

    def _validate_fields(self) -> None:
        super()._validate_fields()
        # ``EnvironmentBase`` re-checks both, but only inside a Ray actor after the cluster and vLLM servers are up.
        require_int(type(self).__name__, ge=1, **present(max_turns=self.max_turns))
        self.reward_terms  # noqa: B018  parse at config time so a bad or empty list fails here
        if not isinstance(self.environment_kwargs, Mapping):
            raise ValueError(
                f"environment_kwargs must map option names to values, got {self.environment_kwargs!r}. A bare "
                f"`environment_kwargs:` header reads as null: comment it out, or list its options."
            )
        # Merged last in to_env_config, a key it writes from a top-level field would override it past the check above.
        shadowing = sorted(set(self.environment_kwargs) & set(self._core_env_config()))
        if shadowing:
            raise ValueError(
                f"environment_kwargs must not carry {shadowing}: the config writes them from its own validated "
                "top-level fields (rewards for reward_terms, max_turns). Set those instead."
            )

    @property
    def reward_terms(self) -> tuple[RewardTerm, ...]:
        return parse_reward_terms(self.rewards, ENVIRONMENT_REWARD_SOURCES)

    def _core_env_config(self) -> dict[str, Any]:
        return {"reward_terms": [dict(term) for term in self.rewards], "max_turns": self.max_turns}

    def to_env_config(self) -> dict[str, Any]:
        """The dict passed to ``build_environment(environment_type, config)``."""
        # Only an explicit turn cap is forwarded; the dataclass default would override the per-class defaults.
        config = {key: value for key, value in self._core_env_config().items() if value is not None}
        config.update(self.environment_kwargs)
        return config


@dataclass
class EnvironmentalGRPOScriptArguments(PromptDatasetArguments, CommonScriptArguments):
    """Dataset args; async infra is in AsyncRolloutConfig, env selection in EnvironmentConfig (parsed separately)."""

    PROJECT_NAME: ClassVar[str] = "environmental-grpo"

    answer_field: str | None = field(
        default=DEFAULT_ANSWER_FIELD,
        metadata={
            "help": "Dataset column holding the expected answer, carried when present. Whether an answer is needed "
            "is the environment's requires_answer."
        },
    )
    context_fields: list[str] | None = field(
        default=None, metadata={"help": "Further dataset columns passed to the environment as context."}
    )
