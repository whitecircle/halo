"""Configuration for async Environmental GRPO training (DistributedAsyncEnvironmentalGRPOTrainer)."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import timedelta
from math import isfinite
from typing import Any, Literal

from src.config_validation import require_finite, require_int, require_positive, require_positive_int
from src.configs.grpo import AdvantageShapingArguments, ChunkedLogprobsArguments, GRPOEarlyStopArguments
from src.env import DEFAULT_NCCL_TIMEOUT_MINUTES, WATCHDOG_WARN_FRACTION, nccl_timeout
from src.environments.rollout_config import (
    DEFAULT_EPISODE_TIMEOUT_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REASONING_END_TOKEN,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_RETRY_BASE_WAIT_SECONDS,
    DEFAULT_ROLLOUT_MAX_TOKENS,
    DEFAULT_ROLLOUT_MIN_P,
    DEFAULT_ROLLOUT_REPETITION_PENALTY,
    DEFAULT_ROLLOUT_TEMPERATURE,
    DEFAULT_ROLLOUT_TOP_K,
    DEFAULT_ROLLOUT_TOP_P,
    REASONING_BUDGET_SOFT_TEMPLATE_VAR,
    REASONING_BUDGET_TEMPLATE_VAR,
    REASONING_END_TOKEN_EXAMPLES,
    SGLANG_BACKEND,
    VLLM_BACKEND,
    RolloutBackend,
    RolloutConfig,
)
from src.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# The reasoning price's per-episode cap, read only while the price is on.
DEFAULT_REASONING_PRICE_CAP = 0.1

# Rollout knobs whose consumer has no meaning for a non-positive value (see _validate_ranges).
POSITIVE_ROLLOUT_FIELDS = (
    "rollout_temperature",
    "rollout_max_tokens",
    "request_timeout",
    "episode_timeout",
    "rollout_connection_timeout",
)


@dataclass(frozen=True)
class ISMaskConfig:
    """Mask/veto stages layered on the truncated IS ratio, applied by ``src/trainers/grpo/objective/logratio.py``.
    All default off.

    * ``geo_band_min``/``geo_band_max`` — trajectory geometric-mean band: mask the whole trajectory when
      ``exp(mean log-ratio over its corrected tokens)`` leaves the band. Both bounds must be set.
    * ``veto_min`` — catastrophic-token veto: mask the trajectory when any corrected token's ratio is below it.
    * ``opsm_delta`` — off-policy sequence masking (``apply_opsm``, applied separately once advantages exist).
    """

    geo_band_min: float | None = None
    geo_band_max: float | None = None
    veto_min: float | None = None
    opsm_delta: float | None = None

    def __post_init__(self):
        lo, hi = self.geo_band_min, self.geo_band_max
        if (lo is None) != (hi is None):
            raise ValueError("isr_geo_band_min and isr_geo_band_max must be set together")
        if lo is not None:
            require_finite(type(self).__name__, isr_geo_band_min=lo, isr_geo_band_max=hi)
            if not 0 < lo < 1 < hi:
                raise ValueError(f"isr_geo_band bounds must satisfy 0 < min < 1 < max, got [{lo}, {hi}]")
        if self.veto_min is not None and not 0 < self.veto_min < 1:
            raise ValueError(f"isr_veto_min must be in (0, 1), got {self.veto_min}")
        if self.opsm_delta is not None:
            require_positive(type(self).__name__, isr_opsm_delta=self.opsm_delta)

    @property
    def any_mask_active(self) -> bool:
        """Whether a stage :func:`apply_is_masks` applies (the geometric band or the veto) is set."""
        return self.geo_band_min is not None or self.veto_min is not None

    @property
    def any_stage_active(self) -> bool:
        """Whether any stage is set, OPSM included."""
        return self.any_mask_active or self.opsm_delta is not None

    @property
    def sums_sequence_logratio(self) -> bool:
        """Whether a stage reads the per-token log-ratio summed over a trajectory (the geometric band's
        and OPSM's per-trajectory mean); the veto reads single tokens. The sampler-logprob preflight
        holds such a run to engine log-probs not renormalized over the sampler's cut."""
        return self.geo_band_min is not None or self.opsm_delta is not None


@dataclass
class AsyncTrainingConfig(AdvantageShapingArguments, GRPOEarlyStopArguments, ChunkedLogprobsArguments):
    """Async training infrastructure: Ray workers, rollout-server connections, weight sync, rollout
    prefetch. Environment selection is in EnvironmentConfig, the trainer in
    src/trainers/grpo/environmental.py."""

    num_rollout_workers: int = field(
        default=64,
        metadata={
            "help": "Ray environment actors per training rank; each is async and runs many episodes, so this is no "
            "concurrency limit (max_concurrent_rollouts is). Size it to the CPU-side environment cost; with "
            "ray_address set each rank takes this // world_size, at least 1. "
            "See agent-docs/infrastructure/ray.md#pool-sizing."
        },
    )

    max_concurrent_rollouts: int | None = field(
        default=None,
        metadata={
            "help": "Per-rank cap on rollouts in flight, the generation-throughput throttle; size it to one "
            "generation cycle's per-rank demand with about 2× prefetch headroom. Unset = 4 × this rank's share of "
            "the rollout workers, and an explicit value below that share is raised to it. "
            "See agent-docs/reference/configuration-reference.md#asynctrainingconfig."
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
        default=VLLM_BACKEND,
        metadata={
            "help": "Inference engine serving rollouts and receiving the weight updates. 'sglang' refuses "
            "rollout_max_thinking_tokens and must be served from the NCCL-aligned Dockerfile.sglang image with "
            "NCCL_CUMEM_ENABLE=1. "
            "See agent-docs/training-methods/async-grpo/setup.md#rollout-backend."
        },
    )

    rollout_server_url: str = field(
        default="http://localhost:8000",
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
            "[{'url': 'http://node1:8000', 'group_port': 51216}]. Overrides rollout_server_url."
        },
    )

    sync_weights_every_n_steps: int = field(
        default=1,
        metadata={"help": "Sync weights to the rollout servers every N training steps, N >= 1."},
    )

    rollout_temperature: float = field(
        default=DEFAULT_ROLLOUT_TEMPERATURE, metadata={"help": "Temperature for rollout generation."}
    )

    rollout_top_p: float = field(
        default=DEFAULT_ROLLOUT_TOP_P, metadata={"help": "Top-p (nucleus) sampling for rollout generation."}
    )

    rollout_top_k: int = field(
        default=DEFAULT_ROLLOUT_TOP_K,
        metadata={
            "help": "Top-k sampling for rollouts: -1 (off) or an int >= 1, since SGLang rejects 0. Sent on every "
            "request, like rollout_min_p and rollout_repetition_penalty: both engines fill an omitted one from "
            "generation_config.json."
        },
    )

    rollout_min_p: float = field(
        default=DEFAULT_ROLLOUT_MIN_P,
        metadata={
            "help": "Min-p sampling for rollout generation, in [0, 1]; 0 = off. vLLM rejects every request "
            "with min_p > 0 on a server running speculative decoding (MTP)."
        },
    )

    rollout_repetition_penalty: float = field(
        default=DEFAULT_ROLLOUT_REPETITION_PENALTY,
        metadata={
            "help": "Repetition penalty for rollouts, in (0, 2]; 1 = off. Refused other than 1 under a "
            "sequence-level ratio (isr_geo_band_*, isr_opsm_delta), since the trainer's recomputed log-probs do not "
            "carry it."
        },
    )

    rollout_max_tokens: int = field(
        default=DEFAULT_ROLLOUT_MAX_TOKENS,
        metadata={
            "help": "Most tokens one turn generates (one /chat/completions call), checked against the server at "
            "startup. The trajectory accumulates across turns, bounded by rollout_max_episode_tokens where set and "
            "by the model's context window, which it fails past. "
            "See agent-docs/training-methods/async-grpo/episodes.md#trajectory-length."
        },
    )

    rollout_max_episode_tokens: int | None = field(
        default=None,
        metadata={
            "help": "Most tokens one episode may sample over all its turns, reasoning included; unset = max_turns × "
            "rollout_max_tokens. Enforced by the engine per turn, never stated to the model, and at least "
            "rollout_max_tokens. See agent-docs/training-methods/async-grpo/episodes.md#episode-output-budget."
        },
    )

    rollout_max_thinking_tokens: int | None = field(
        default=None,
        metadata={
            "help": "Per-turn reasoning-token cap (vLLM thinking_token_budget) that then forces the answer out of "
            "the rest of the turn; a level's smaller enforced thinking_tokens wins, and under a soft level it is "
            "the ceiling. Needs a reasoning parser on the vLLM server; unset leaves reasoning to the levels' "
            "budgets. See agent-docs/training-methods/async-grpo/reasoning.md#reasoning-budget."
        },
    )

    rollout_max_answer_tokens: int | None = field(
        default=None,
        metadata={
            "help": "Most a turn may generate past its reasoning cap, never stated to the model: an int in "
            "[1, rollout_max_tokens), unset = rollout_max_tokens alone bounds the turn. "
            "See agent-docs/training-methods/async-grpo/reasoning.md#answer-room."
        },
    )

    rollout_reasoning_end_token: str = field(
        default=DEFAULT_REASONING_END_TOKEN,
        metadata={
            "help": "The string the server's reasoning parser ends reasoning with, encoded as vLLM encodes it "
            f"({REASONING_END_TOKEN_EXAMPLES}). Wherever a vLLM thinking budget can bind, the trainer gives a "
            "forced run of its ids ratio 0 in the loss. "
            "See agent-docs/training-methods/async-grpo/objective.md#forced-reasoning-closes."
        },
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
    max_train_row_tokens: int | None = field(
        default=None,
        metadata={
            "help": "Longest training row (prompt + completion tokens) a training rank takes, above "
            "rollout_max_tokens; unset = no cap beyond the context check. A per-turn row over it leaves the batch, "
            "a whole-trajectory row trains at zero weight. "
            "See agent-docs/training-methods/async-grpo/episodes.md#training-row-cap."
        },
    )

    isr_geo_band_min: float | None = field(
        default=None,
        metadata={
            "help": "Trajectory geometric-mean band: mask a whole trajectory when exp(mean log-ratio) over its "
            "corrected tokens leaves [isr_geo_band_min, isr_geo_band_max]. Set both to activate. "
            "See agent-docs/training-methods/async-grpo/objective.md#trust-region-masks."
        },
    )
    isr_geo_band_max: float | None = field(
        default=None,
        metadata={"help": "Upper bound of the trajectory geometric-mean band (see isr_geo_band_min)."},
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

    skip_update_masked_frac: float | None = field(
        default=None,
        metadata={
            "help": "Trust-region circuit breaker: when the mask stages fully mask more than this fraction of the "
            "IS-corrected trajectories, or mask more than this fraction of the corrected tokens, zero the round's "
            "advantages and skip its optimizer steps instead of training on the survivors. Unset = off; 0.3-0.5 is "
            "a sane range. See agent-docs/training-methods/async-grpo/objective.md#breaker."
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

    # Default flipped vs AdvantageShapingArguments: on a sparse verifiable env reward the all-equal
    # groups dominate the batch.
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
            "tool-call terminator when it is no eos (gpt-oss <|call|> under harmony-disabled serving). Empty = only "
            "eos stops a turn. See agent-docs/training-methods/async-grpo/episodes.md#tool-calls."
        },
    )

    rollout_chat_template_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Chat-template variables sent with every rollout request (chat_template_kwargs) and applied to "
            "the trainer's own renders, so both sides see one template state. reasoning_effort, reasoning_budget "
            "and reasoning_budget_soft are refused here, being per episode. "
            "See agent-docs/training-methods/async-grpo/episodes.md#chat-template."
        },
    )

    reasoning_price: dict[str, float] | None = field(
        default=None,
        metadata={
            "help": "Per effort level (exactly low, medium and high), the reward units an episode pays per 1,000 "
            "reasoning tokens, capped by reasoning_price_cap; unset = off. Price the lowest level highest, so the "
            "same trace costs most where little reasoning was asked. "
            "See agent-docs/training-methods/async-grpo/reasoning.md#price."
        },
    )
    reasoning_price_past_budget: bool = field(
        default=False,
        metadata={
            "help": "Price only the reasoning tokens each turn spends past its level's soft budget instead of every "
            "reasoning token. Needs reasoning_price and a soft budget on every drawable level. "
            "See agent-docs/training-methods/async-grpo/reasoning.md#price-past-the-budget."
        },
    )
    reasoning_price_cap: float = field(
        default=DEFAULT_REASONING_PRICE_CAP,
        metadata={
            "help": "Cap of reasoning_price per episode, in reward units, so a long trace cannot outweigh the task "
            "reward. Keep it, plus reasoning_floor, below what the environment charges for the decisions it prices. "
            "See agent-docs/training-methods/async-grpo/reasoning.md#price."
        },
    )
    reasoning_floor: float = field(
        default=0.0,
        metadata={
            "help": "Weight of the reasoning under-use floor (0 = off): an episode reasoning less than three "
            "quarters of its per-turn thinking budget pays -weight × shortfall / that three-quarter target. Keep it, plus "
            "reasoning_price_cap, below what the environment charges for the decisions it prices. "
            "See agent-docs/training-methods/async-grpo/reasoning.md#floor."
        },
    )

    enable_prefetch: bool = field(
        default=True,
        metadata={
            "help": "Enable prefetching to overlap rollout collection with training. Multi-server "
            "only: with one rollout server it auto-disables, since that engine stops serving during "
            "weight sync and there is nothing to overlap against."
        },
    )

    model_name: str | None = field(
        default=None,
        metadata={
            "help": "Model name sent in the rollout server's /v1/chat/completions requests. "
            "Unset is filled with model_name_or_path at script start, so a request always names one."
        },
    )

    request_timeout: float = field(
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        metadata={
            "help": "HTTP timeout per rollout-server request in seconds of engine-serving time: a weight-sync "
            "pause is credited back, so a sync never expires a request."
        },
    )

    episode_timeout: float = field(
        default=DEFAULT_EPISODE_TIMEOUT_SECONDS,
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
        default=DEFAULT_MAX_RETRIES,
        metadata={
            "help": "Retries after a failed rollout-server request (total attempts = max_retries + 1). "
            "0 = one attempt, no retry."
        },
    )

    retry_base_wait: float = field(
        default=DEFAULT_RETRY_BASE_WAIT_SECONDS,
        metadata={"help": "Base wait time in seconds for exponential backoff between retries."},
    )

    def __post_init__(self):
        self._validate_ranges()

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        # A bare range check reads a bool as 0 or 1 and passes a NaN, so each knob goes through a finite or int
        # guard first.
        owner = type(self).__name__
        # 0 raises ZeroDivisionError at `global_step % sync_weights_every_n_steps`.
        require_positive_int(owner, sync_weights_every_n_steps=self.sync_weights_every_n_steps)
        self._validate_reasoning_price()
        # A NaN or negative weight would parse and leave the floor off without a word.
        require_finite(owner, reasoning_floor=self.reasoning_floor)
        if self.reasoning_floor < 0:
            raise ValueError(f"reasoning_floor must be a finite number >= 0 (0 = off), got {self.reasoning_floor}")
        # A negative budget reaches backoff as max_tries <= 0, which it treats as "no limit": a wedged
        # server is then retried until the NCCL watchdog kills the job.
        retries = self.max_retries
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError(f"max_retries must be an int >= 0 (0 = one attempt, no retry), got {retries!r}")
        self.build_is_mask_config()
        # 0 workers builds an empty actor list and then divides by it.
        require_positive_int(owner, num_rollout_workers=self.num_rollout_workers)
        # `max_concurrent_rollouts or default` reads 0 as "unset", so the cap would vanish.
        concurrency = self.max_concurrent_rollouts
        if concurrency is not None and (
            isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1
        ):
            raise ValueError(
                f"max_concurrent_rollouts must be an int >= 1 when set (null = derive from num_rollout_workers), "
                f"got {concurrency!r}"
            )
        # A non-positive batch raises inside the DataLoader, far from the knob; null is "unset".
        eval_batch = self.eval_rollout_batch_size
        if eval_batch is not None and (
            isinstance(eval_batch, bool) or not isinstance(eval_batch, int) or eval_batch < 1
        ):
            raise ValueError(
                f"eval_rollout_batch_size must be an int >= 1 when set (null = one round per eval batch), "
                f"got {eval_batch!r}"
            )
        # None of these consumers can express a non-positive value, and each swallows one far from
        # the knob: rollout_temperature divides the log-prob sweep (the trainer scores at the
        # sampling temperature, so a 0 is a ZeroDivisionError mid-step), rollout_max_tokens doubles
        # as the dr_grpo loss normalizer, and the deadlines are compared against wall-clock, where a
        # non-positive one cancels every episode on entry and halts the run as an empty batch.
        require_positive(owner, **{name: getattr(self, name) for name in POSITIVE_ROLLOUT_FIELDS})
        # The request's max_tokens, which no engine reads as a fraction.
        require_int(owner, rollout_max_tokens=self.rollout_max_tokens)
        # Sent verbatim on the wire; outside these ranges the server rejects every rollout request
        # (SGLang's, the narrower of the two engines': it refuses top_k 0 and a penalty above 2).
        require_finite(
            owner,
            rollout_top_p=self.rollout_top_p,
            rollout_min_p=self.rollout_min_p,
            rollout_repetition_penalty=self.rollout_repetition_penalty,
        )
        if not 0 < self.rollout_top_p <= 1:
            raise ValueError(f"rollout_top_p must be in (0, 1], got {self.rollout_top_p}")
        # True would pass as 1 and sample every rollout greedily; a fractional top-k fails every request.
        top_k = self.rollout_top_k
        if isinstance(top_k, bool) or not isinstance(top_k, int) or (top_k != -1 and top_k < 1):
            raise ValueError(f"rollout_top_k must be an int, -1 (off) or >= 1, got {top_k!r}")
        if not 0 <= self.rollout_min_p <= 1:
            raise ValueError(f"rollout_min_p must be in [0, 1], got {self.rollout_min_p}")
        if not 0 < self.rollout_repetition_penalty <= 2:
            raise ValueError(f"rollout_repetition_penalty must be in (0, 2], got {self.rollout_repetition_penalty}")
        # A negative base shrinks the retry backoff instead of growing it.
        require_finite(owner, retry_base_wait=self.retry_base_wait)
        if self.retry_base_wait < 0:
            raise ValueError(
                f"retry_base_wait must be a finite number >= 0 (0 = retry immediately), got {self.retry_base_wait}"
            )
        # A turn's answer room is `rollout_max_tokens - rollout_max_thinking_tokens` (or rollout_max_answer_tokens
        # where smaller), none where the caps meet: the turn would then spend its whole cap on reasoning and stop
        # before the answer or tool call it exists to produce. A cap of 0 would still be sent (as 1) while counting
        # as no cap.
        cap = self.rollout_max_thinking_tokens
        if cap is not None and (
            isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap < self.rollout_max_tokens
        ):
            raise ValueError(
                f"rollout_max_thinking_tokens must be an int in [1, rollout_max_tokens={self.rollout_max_tokens}) when "
                f"set (null = no run-wide cap): rollout_max_tokens bounds the whole turn, and at or above it the turn "
                f"has no answer room and is cut mid-reasoning, got {cap!r}"
            )
        # The turn total is min(rollout_max_tokens, reasoning cap + this): at or above rollout_max_tokens it
        # shrinks no turn, and below 1 a turn the engine force-closed has no room left to answer.
        answer = self.rollout_max_answer_tokens
        if answer is not None and (
            isinstance(answer, bool) or not isinstance(answer, int) or not 1 <= answer < self.rollout_max_tokens
        ):
            raise ValueError(
                f"rollout_max_answer_tokens must be an int in [1, rollout_max_tokens={self.rollout_max_tokens}) when "
                f"set (null = rollout_max_tokens alone bounds the turn): at or above it the bound shrinks no turn, "
                f"got {answer!r}"
            )
        # Below one turn's cap the first turn could never use the per-turn budget the run states. A turn's
        # total is at most rollout_max_tokens, so this also holds one reasoning cap plus its answer room.
        if self.rollout_max_episode_tokens is not None and (
            isinstance(self.rollout_max_episode_tokens, bool)
            or not isinstance(self.rollout_max_episode_tokens, int)
            or self.rollout_max_episode_tokens < self.rollout_max_tokens
        ):
            raise ValueError(
                f"rollout_max_episode_tokens must be an int >= rollout_max_tokens ({self.rollout_max_tokens}), so "
                f"one whole turn fits the episode, or null, got {self.rollout_max_episode_tokens!r}"
            )
        # A row is a turn's prompt plus its completion, and the completion alone may run to
        # rollout_max_tokens: a cap at or below it leaves out every turn that used its budget — a
        # length bias against long turns, not the memory bound the knob is.
        row_cap = self.max_train_row_tokens
        if row_cap is not None and (
            isinstance(row_cap, bool) or not isinstance(row_cap, int) or row_cap <= self.rollout_max_tokens
        ):
            raise ValueError(
                f"max_train_row_tokens ({row_cap!r}) must be an int above rollout_max_tokens "
                f"({self.rollout_max_tokens}): a row is prompt + completion, so a cap at or below the per-turn "
                f"generation budget drops every turn that runs to it."
            )
        # A fraction of the step's corrected trajectories/tokens; 0 would trip on every step, >1 never.
        if self.skip_update_masked_frac is not None:
            require_finite(owner, skip_update_masked_frac=self.skip_update_masked_frac)
            if not 0.0 < self.skip_update_masked_frac <= 1.0:
                raise ValueError(f"skip_update_masked_frac must be in (0, 1], got {self.skip_update_masked_frac}")
        # A rate is never above 1, so a threshold at 1 would never fire.
        if self.truncation_alarm_rate is not None:
            require_finite(owner, truncation_alarm_rate=self.truncation_alarm_rate)
            if not 0.0 <= self.truncation_alarm_rate < 1.0:
                raise ValueError(f"truncation_alarm_rate must be in [0, 1) or null, got {self.truncation_alarm_rate}")
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
        per_episode = {
            "reasoning_effort",
            REASONING_BUDGET_TEMPLATE_VAR,
            REASONING_BUDGET_SOFT_TEMPLATE_VAR,
        } & set(self.rollout_chat_template_kwargs)
        if per_episode:
            raise ValueError(
                f"rollout_chat_template_kwargs must not carry {sorted(per_episode)}: the level and its thinking "
                "budget are per episode; the level travels as the request's top-level field and the budget is "
                "added to the nested form per request."
            )
        self._validate_backend_capabilities()

    def _stops_on_skipped_updates(self) -> bool:
        return self.early_stop_on_skipped_updates

    def build_is_mask_config(self) -> ISMaskConfig:
        """The IS mask stages the ``isr_*`` knobs set, validated by :class:`ISMaskConfig` itself."""
        return ISMaskConfig(
            geo_band_min=self.isr_geo_band_min,
            geo_band_max=self.isr_geo_band_max,
            veto_min=self.isr_veto_min,
            opsm_delta=self.isr_opsm_delta,
        )

    def _validate_reasoning_price(self) -> None:
        """A NaN passes every ordered comparison, a negative price would pay for the length it prices, and a
        cap set while the price is off parses and changes nothing. The levels a price must map are checked at
        trainer construction, beside the environment that draws them."""
        if self.reasoning_price is None and self.reasoning_price_cap != DEFAULT_REASONING_PRICE_CAP:
            raise ValueError(
                "reasoning_price_cap set with reasoning_price unset: nothing reads it until the price is on. Remove "
                "it, or set reasoning_price."
            )
        if self.reasoning_price is None and self.reasoning_price_past_budget:
            raise ValueError(
                "reasoning_price_past_budget set with reasoning_price unset: it chooses what the price counts. Set "
                "reasoning_price, or remove it."
            )
        if self.reasoning_price is not None:
            if not isinstance(self.reasoning_price, Mapping) or not self.reasoning_price:
                raise ValueError(
                    f"reasoning_price must map each effort level to a price, got {self.reasoning_price!r}"
                )
            for level, price in self.reasoning_price.items():
                if isinstance(price, bool) or not isinstance(price, int | float) or not isfinite(price) or price < 0:
                    raise ValueError(f"reasoning_price[{level!r}] must be a finite number >= 0, got {price!r}")
        require_positive(type(self).__name__, reasoning_price_cap=self.reasoning_price_cap)

    def _validate_backend_capabilities(self) -> None:
        """Reject request knobs the selected engine does not implement.

        SGLang ignores unknown request fields rather than rejecting them, so an unimplemented knob
        would otherwise no-op with only a log line.

        Sampled-token training and rollout routing replay are supported: SGLang carries the sampled
        ids in the ``meta_info`` echoed on each choice and publishes routed experts response-level
        (raw-int32 wire format, handled by ``decode_rollout_routing``).

        Neither is the environment's per-effort ``thinking_tokens`` profile, whose budget reaches the
        same request field: the level it belongs to still reaches the chat template and the reasoning
        terms, so on an engine without the field the level keeps steering and only the hard
        cap is lost. The rollout actor warns once per process that it is unenforced. The knob refused
        here is the one whose point is the enforced cap: ``rollout_max_thinking_tokens`` is a cap and
        nothing else.
        """
        if self.rollout_backend != SGLANG_BACKEND:
            return
        if self.rollout_max_thinking_tokens is not None:
            raise ValueError(
                "rollout_max_thinking_tokens is not supported with rollout_backend='sglang': the "
                "thinking_token_budget request field is vLLM-only and SGLang would silently ignore it, "
                "leaving reasoning uncapped. Steer with the environment's reasoning_effort instead."
            )

    def get_server_urls(self) -> list[str]:
        """Get list of rollout-server URLs for generation."""
        if self.rollout_server_configs:
            return [c["url"] for c in self.rollout_server_configs]
        return [self.rollout_server_url]

    def prefetch_active(self) -> bool:
        """Whether the run prefetches: ``enable_prefetch`` with two or more rollout servers. A weight sync
        pauses a lone engine for its whole push, so with one server there is nothing to overlap against."""
        return self.enable_prefetch and len(self.get_server_urls()) > 1

    @classmethod
    def rollout_field_sources(cls) -> dict[str, str]:
        """Map each ``RolloutConfig`` field to the field of this config :meth:`get_rollout_config`
        copies into it.

        Derived from the two declarations rather than listed, so a knob added to both sides forwards
        itself: the YAML surface spells a rollout knob ``rollout_<name>`` where the bare name would be
        ambiguous (that spelling wins where both exist) and identically otherwise. ``RolloutConfig``
        fields with no counterpart here are derived by the builder from other state.
        """
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
    ):
        """Build RolloutConfig from this config. ``stop_token_ids`` (from ``rollout_stop_tokens``) and
        ``reasoning_end_token_id`` (from ``rollout_reasoning_end_token``) are resolved by the caller that
        owns the tokenizer; the latter ends the per-turn reasoning count ``episode/thinking_cap_turns``
        reads. ``in_process_group`` says the rollout runs inside a training
        process group, whose NCCL collective watchdog its timeouts must stay under (the trainer, the
        default); an eval sampling under a training contract joins none and passes False."""
        if in_process_group:
            self._validate_timeouts_against_nccl_watchdog()
        mirrored = {target: getattr(self, source) for target, source in self.rollout_field_sources().items()}
        mirrored["chat_template_kwargs"] = dict(self.rollout_chat_template_kwargs)
        return RolloutConfig(
            **mirrored,
            # Derived from other state rather than mirrored from a same-named knob.
            capture_token_ids=self.train_on_sampled_tokens,
            capture_routed_experts=self.routing_replay == "rollout",
            stop_token_ids=stop_token_ids,
            reasoning_end_token_id=reasoning_end_token_id,
        )

    def _validate_timeouts_against_nccl_watchdog(self):
        """Guard rollout timeouts against the NCCL collective watchdog.

        A straggler rank holds its peers at the per-step FSDP collective for as long as its slowest
        episode runs; if ``episode_timeout`` or the retry budget reaches the watchdog, the peers'
        collective aborts before the straggler is cancelled. Reads :func:`~src.env.nccl_timeout`, so the
        bound checked here is the one ``init_process_group`` installs.
        """
        nccl_minutes = nccl_timeout() // timedelta(minutes=1)
        watchdog = nccl_minutes * 60

        # Above the watchdog, a straggler cannot be cancelled before the peers' collective aborts;
        # values merely close to it only warn, below.
        if self.episode_timeout > watchdog:
            raise ValueError(
                f"episode_timeout ({self.episode_timeout:.0f}s) must be below the NCCL collective "
                f"watchdog ({watchdog:.0f}s = {nccl_minutes} min). A straggler rank holds its peers at "
                f"the per-step collective for up to episode_timeout, so an equal/greater value lets the "
                f"watchdog fire first and abort the run. Raise DIST_NCCL_TIMEOUT_MINUTES above "
                f"episode_timeout (keep ≥15 min margin so the cancelled rank can unwind and rejoin), or "
                f"lower episode_timeout."
            )
        # Every rank builds a rollout config; a warning about the config itself is said once.
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
