"""Configuration for async Environmental GRPO training (DistributedAsyncEnvironmentalGRPOTrainer)."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from math import isfinite
from typing import Any, Literal

from src.args.mixins import AdvantageShapingArguments, ChunkedLogprobsArguments, GRPOEarlyStopArguments
from src.args.validation import require_finite, require_int, require_positive, require_positive_int
from src.configs.rollout_config import (
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
    REASONING_BUDGET_TEMPLATE_VAR,
    REASONING_END_TOKEN_EXAMPLES,
    SGLANG_BACKEND,
    VLLM_BACKEND,
    RolloutBackend,
    RolloutConfig,
)
from src.distributed.runtime import is_global_main_process
from src.env import DEFAULT_NCCL_TIMEOUT_MINUTES, WATCHDOG_WARN_FRACTION, resolve_nccl_timeout_minutes

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
            "help": "Ray environment actors (per training rank). The actors are async, so one handles "
            "many concurrent episodes — this is NOT the HTTP-concurrency limit (max_concurrent_rollouts "
            "is). Size it to the CPU-side env cost (tool execution, verifier grading). Created per rank, "
            "so cluster-wide total = world_size × this (divided by world_size when ray_address is set); "
            "an actor needs 1 free CPU at placement only — the sandbox gate, not Ray, bounds CPU use."
        },
    )

    max_concurrent_rollouts: int | None = field(
        default=None,
        metadata={
            "help": "Per-rank asyncio-semaphore cap on rollouts in flight — the real generation-throughput "
            "throttle. Server-pool load = this × world_size ÷ num_servers (under TP every rank collects a full batch). Size it to the per-rank "
            "rollout demand of one generation cycle (per_device_train_batch_size × steps_per_generation, "
            "which itself defaults to gradient_accumulation_steps) with ~2× headroom for prefetch; raising "
            "it past the actual rollout count does nothing. "
            "Default: 4 × this rank's share of the rollout workers (num_rollout_workers ÷ world_size on a shared Ray "
            "cluster, all of them locally), clamped to ≥ that share."
        },
    )
    eval_rollout_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Rows per rank in one evaluation rollout round (rows, not prompts: the eval sampler has already "
            "repeated each prompt num_generations_eval times). Eval rounds run without prefetch, so a round's wall "
            "time is its slowest episode and per_device_eval_batch_size-sized rounds idle the servers between them; "
            "size it to what the servers sustain: rows × world_size requests are in flight at once, and a turn "
            "that decodes slower than request_timeout allows fails the episode; a multiple of num_generations_eval, at "
            "most max_concurrent_rollouts (a wider round runs in serial waves). The final round's padding is never "
            "rolled out or scored. It bounds the loader's batch, "
            "not eval peak memory: the loss forward still chunks per_device_eval_batch_size rows, but the round's "
            "widest completion sets the padded width. None leaves the round at the eval batch."
        },
    )

    ray_address: str | None = field(default=None, metadata={"help": "Ray cluster address. None for local mode."})

    rollout_backend: RolloutBackend = field(
        default=VLLM_BACKEND,
        metadata={
            "help": "Inference engine serving rollouts and receiving weight updates. Both support "
            "generation, NCCL weight sync, `train_on_sampled_tokens` and `routing_replay: rollout`. "
            "'sglang' does not support `rollout_max_thinking_tokens` (the trainer wires neither of SGLang's "
            "budget mechanisms; harmony models have none server-side), needs cuMem parity on "
            "the server (NCCL_CUMEM_ENABLE=1, which docker-compose.sglang.yml sets), and must be served "
            "from the NCCL-aligned Dockerfile.sglang image — the stock upstream image ships a "
            "different NCCL than the trainer and cannot form the weight-sync group."
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
            "help": "List of rollout-server configs for multi-server setup (engine per rollout_backend). "
            "Each config is a dict with 'url' and optional 'group_port' and 'group_host' "
            "(the weight-transfer master address the serving node dials back to; defaults to the "
            "resolution chain arg → VLLM_GROUP_HOST/SGLANG_GROUP_HOST → loopback-if-local → "
            "default-route NIC). Example: [{'url': 'http://node1:8000', 'group_port': 51216}]. "
            "If set, overrides rollout_server_url for weight sync."
        },
    )

    sync_weights_every_n_steps: int = field(
        default=1,
        metadata={
            "help": "Sync weights to the rollout server(s) every N training steps. Must be >= 1 (1 = every step)."
        },
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
            "help": "Top-k sampling for rollout generation: an int, -1 (off) or >= 1; 0 is refused, since SGLang "
            "rejects it, and so are a bool and a float. Sent on every request, as are rollout_min_p and "
            "rollout_repetition_penalty: both engines fill an omitted one from the model's generation_config.json."
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
            "help": "Repetition penalty for rollout generation, in (0, 2]; 1 = off. The sampling log-probs "
            "carry it and the trainer's recomputed ones do not, so a value other than 1 is refused at "
            "startup under a sequence-level ratio (isr_geo_band_*, isr_opsm_delta)."
        },
    )

    rollout_max_tokens: int = field(
        default=DEFAULT_ROLLOUT_MAX_TOKENS,
        metadata={
            "help": "Max tokens per single-turn generation (one /chat/completions call). The whole "
            "multi-turn trajectory accumulates across turns, bounded by rollout_max_episode_tokens where set and "
            "by the model's context window (shared with vLLM); a trajectory exceeding the context fails. This "
            "per-turn bound is verified against the server at startup."
        },
    )

    rollout_max_episode_tokens: int | None = field(
        default=None,
        metadata={
            "help": "The most tokens one episode may sample over all its assistant turns, reasoning and visible "
            "output together, recoveries included (null = unbounded: max_turns x rollout_max_tokens). Enforced "
            "by the engine per turn, never stated to the model: a turn's max_tokens is the smaller of its own "
            "cap and what the episode has left, its reasoning cap shrinks with it so the turn keeps its answer "
            "room (rollout_max_tokens less its reasoning cap, or rollout_max_answer_tokens where smaller), and an "
            "episode left with less than that room starts no further turn and ends truncated, priced like a "
            "max_turns overflow. Must be >= rollout_max_tokens, so one whole turn fits. Logged as "
            "episode/output_budget_exhausted: the share of episodes whose budget held no further turn when they "
            "ended, by the budget or done."
        },
    )

    rollout_max_thinking_tokens: int | None = field(
        default=None,
        metadata={
            "help": "Per-turn reasoning-token cap for reasoning models (vLLM thinking_token_budget): caps the "
            "chain-of-thought, then forces the model to answer with the rest of max_tokens. A level whose profile "
            "sets a smaller thinking_tokens runs under that instead. Requires a reasoning parser on the vLLM "
            "server (--reasoning-parser qwen3 for Qwen3.x; the openai_gptoss plugin for gpt-oss). None = only a "
            "level's thinking_tokens caps reasoning, or nothing does."
        },
    )

    rollout_max_answer_tokens: int | None = field(
        default=None,
        metadata={
            "help": "The most a turn may generate past its reasoning cap (null = rollout_max_tokens alone bounds "
            "the turn). A turn's max_tokens becomes the smaller of rollout_max_tokens and its reasoning cap plus "
            "this, the cap being its level's or, on a retry, the reserve; under rollout_max_episode_tokens a turn "
            "starts only while the episode has its answer room left (the smaller of this and rollout_max_tokens less "
            "the level's cap). Enforced per turn, never stated to the model. Every "
            "episode needs a reasoning cap: a drawable level with no thinking_tokens while "
            "rollout_max_thinking_tokens is unset is refused. On SGLang, which enforces no reasoning cap, it bounds "
            "the whole turn at the cap plus this. Must be an int in [1, rollout_max_tokens)."
        },
    )

    rollout_reasoning_end_token: str = field(
        default=DEFAULT_REASONING_END_TOKEN,
        metadata={
            "help": "The string the server's reasoning parser ends reasoning with, encoded as vLLM encodes it "
            f"({REASONING_END_TOKEN_EXAMPLES}). "
            "Wherever a vLLM thinking budget can bind, a forced run of its ids gets ratio 0 in the loss (a marker "
            "holding none of the tokenizer's added tokens only warns). Where it is one token, a turn's reasoning "
            "is counted as its sampled ids up to and including it, the count episode/thinking_cap_turns reads."
        },
    )

    train_on_sampled_tokens: bool = field(
        default=True,
        metadata={
            "help": "Train env-GRPO on the ACTUAL sampled generation token ids (captured from the server's "
            "logprobs) rather than re-tokenizing a chat-template re-render of the parsed trajectory. "
            "Model-agnostic and fully faithful — it eliminates every re-tokenization mismatch (tool-call "
            "rendering, argument whitespace, reasoning re-render). Each assistant turn is its own training "
            "row (prompt = the history the server built for that turn, completion = that turn's sampled "
            "ids), sharing the trajectory's advantage. On vLLM the server must run with "
            "`--return-tokens-as-token-ids` (docker-compose.vllm.yml passes it); SGLang captures per "
            "request and needs no flag. All-or-nothing per trajectory: one uncaptured turn falls that "
            "whole trajectory back to a single re-tokenized row. Default on."
        },
    )
    max_train_row_tokens: int | None = field(
        default=None,
        metadata={
            "help": "Longest training row (prompt + completion tokens) a training rank takes; None = no cap beyond "
            "the context check. A per-turn row over it is left out of the batch (the episode's other turns still "
            "train); a whole-trajectory row over it trains as a zero-weight row. Logged as "
            "sampling/rows_over_cap_frac: rows left out over rows left out plus rows that train. A memory bound "
            "for the training ranks — must be above rollout_max_tokens — below the context check, which rejects "
            "rows the served model could not have produced."
        },
    )

    isr_geo_band_min: float | None = field(
        default=None,
        metadata={
            "help": "TRAJECTORY geometric-mean band: mask a whole trajectory when exp(mean log-ratio "
            "over its corrected tokens) leaves [isr_geo_band_min, isr_geo_band_max] (NeMo-RL seq-mask-tis, "
            "slime MIS). Aggregated per trajectory across all its turn rows (drift compounds over the "
            "episode). Set both to activate; start [0.99, 1.01]."
        },
    )
    isr_geo_band_max: float | None = field(
        default=None,
        metadata={"help": "Upper bound of the trajectory geometric-mean band (see isr_geo_band_min)."},
    )
    isr_veto_min: float | None = field(
        default=None,
        metadata={
            "help": "Catastrophic-token veto: mask a whole trajectory when ANY corrected token's raw IS "
            "ratio falls below this (verl/slime use ~1e-4 — such a token marks a sequence the ratio can "
            "no longer honestly correct). None (default) = off."
        },
    )
    isr_opsm_delta: float | None = field(
        default=None,
        metadata={
            "help": "DeepSeek-V3.2 Off-Policy Sequence Masking: mask NEGATIVE-advantage trajectories "
            "whose |mean log-ratio| vs the sampling policy exceeds this many nats (positives never "
            "masked). None (default) = off."
        },
    )
    isr_engine_reference: bool = field(
        default=False,
        metadata={
            "help": "Re-score every training row on the rollout engine under the weights just synced "
            "(one prefill per row through the completions prompt_logprobs echo) and feed the mask "
            "stages (isr_geo_band/isr_veto/isr_opsm) the log-ratio between the engine's current and "
            "sampling log-probs instead of the trainer-vs-engine one. That removes the trainer<->engine "
            "disagreement from the stages; the engine's own decode-vs-prefill disagreement stays in "
            "the reference (measured -0.003 nats/token at sampling entropy 0.36 with identical weights, "
            "against -0.0003 for trainer-vs-engine-prefill), and like every such floor it grows with "
            "entropy, so the band still needs room above it. The IS weight itself stays "
            "trainer-vs-sampling. Logged as sampling/engine_logratio_mean (staleness plus the "
            "decode-vs-prefill floor), sampling/numerics_logratio_mean (trainer minus engine-prefill) "
            "and sampling/engine_rescore_coverage. Requires the IS correction, rollout_temperature "
            "and rollout_top_p of 1.0 and rollout_top_k / rollout_min_p / rollout_repetition_penalty off "
            "(prefill log-probs are the raw distribution); vLLM re-scores through "
            "the completions prompt_logprobs echo, SGLang through /generate with logprob_start_len. The "
            "server must hold headroom for that pass: vLLM materializes an fp32 log-softmax over the "
            "vocabulary for every prefill chunk of a prompt_logprobs request (max_num_batched_tokens x "
            "vocab x 4 B — 8 GB at 8192 x 248k, outside its memory profile), so serve with "
            "--gpu-memory-utilization <= 0.80 or a smaller --max-num-batched-tokens; at 0.90 the "
            "engine dies of CUDA OOM under load."
        },
    )

    routing_replay: Literal["none", "recompute", "rollout"] = field(
        default="none",
        metadata={
            "help": "MoE routing replay: pin the update pass's top-k expert selection to a recorded mask, "
            "re-deriving gate weights from live router scores (removes the discontinuous routing-flip "
            "component of the pass-to-pass policy divergence; Qwen 'Routing Replay' / DeepSeek 'Keep "
            "Routing'). 'none' (default) = off; 'recompute' (R2) = capture the mask in the trainer's own "
            "no-grad logprob-recompute pass and replay it in the update + GC-recompute forwards; "
            "'rollout' (R3, EXPERIMENTAL) = replay the rollout ENGINE's mask, removing the full "
            "cross-engine routing discontinuity — requires train_on_sampled_tokens and a capture-capable "
            "server: vLLM >= 0.22 with --enable-return-routed-experts and a non-FlashInfer MoE backend "
            "(--moe-backend triton), or SGLang with --enable-return-routed-experts and "
            "--moe-runner-backend triton (the triton_kernel/flashinfer runners bypass the capture hook); "
            "positions the engine did not cover (e.g. prefix-cache hits) keep natural routing. Validate "
            "capture coverage on your serving shape before a long run. "
            "MoE-with-EP-wrappers only; Gemma4 and Zaya are rejected (see _supports_routing_replay)."
        },
    )

    skip_update_masked_frac: float | None = field(
        default=None,
        metadata={
            "help": "Trust-region circuit breaker (KL-free): when the fraction of IS-corrected "
            "trajectories fully masked by the geo-band/veto/OPSM stages, OR the fraction of corrected "
            "tokens masked (the masked trajectories are the long ones), exceeds this, ZERO the step's "
            "advantages and SKIP its optimizer step (gradients dropped to None, so Adam's momentum "
            "cannot step the weights either) instead of training on the unmasked survivors. At high "
            "masked fractions the survivors are a selection-biased sample (exactly the rows where the "
            "drifted policy still agrees with the rollout), so continuing to train amplifies the drift; "
            "skipping holds the policy still until the next weight-sync re-anchors the rollouts. Logged "
            "as `sampling/update_skipped` with `sampling/is_masked_traj_frac` / "
            "`sampling/is_masked_token_frac`. None (default) = off; 0.3-0.5 is a sane range."
        },
    )
    early_stop_on_skipped_updates: bool = field(
        default=False,
        metadata={
            "help": "Early stop (GRPOEarlyStopArguments): end training once the trust-region breaker "
            "(`skip_update_masked_frac`) skipped every update on `early_stop_patience` readings in a row (one "
            "generation round each at `logging_steps: 1`). Past that the policy is frozen where the rollouts no "
            "longer agree with it, and the run only spends compute. Raises without `skip_update_masked_frac`."
        },
    )

    truncation_alarm_rate: float | None = field(
        default=0.25,
        metadata={
            "help": "Warn when the share of a rollout round's episodes that ended truncated "
            "(`episode/truncation_rate`: the max_turns cap, the episode output budget, or a cut turn past its "
            "recoveries) rises over "
            "this, and log `episode/truncation_alarm` (1 over, 0 under) every round. Past it the turn or "
            "token budget, not the task, ends a large share of episodes. The warning repeats only after the rate "
            "has dropped back under. In [0, 1); None = off."
        },
    )

    # Default flipped vs AdvantageShapingArguments: on a sparse verifiable env reward the all-equal
    # groups dominate the batch.
    drop_degenerate_groups: bool = field(
        default=True,
        metadata={
            "help": "Drop GRPO groups whose completions ALL settled the same environment reward (grade, "
            "shaping and external scores; the trainer's reasoning price and floor excluded: length regularizers, "
            "with which a group tied on everything else would train on its members' reasoning lengths alone). Such "
            "a group has no task contrast to learn from, and its tokens would still inflate the loss normalizer and "
            "dilute the groups that do carry signal. "
            "Masking them restores the effective batch size (the cheap half of DAPO's dynamic sampling: "
            "drop, without resampling replacements). Logged as `sampling/degenerate_group_frac`. Default on."
        },
    )

    rollout_stop_tokens: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Special-token strings that end a turn's generation, resolved to ids via the tokenizer "
            "and sent as the engine's `stop_token_ids`. Set to the model's tool-call terminator so a turn "
            "stops when the model emits its call and the environment runs it — without this a model whose "
            "terminator is not an eos (e.g. gpt-oss `<|call|>` under harmony-disabled serving) keeps "
            "generating, hallucinating the tool result and playing out the whole episode in one turn "
            "(huge, off-policy-noisy completions). Empty (default) = only the model's eos stops a turn."
        },
    )

    rollout_chat_template_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Chat-template variables sent with every rollout request as `chat_template_kwargs` and applied "
            "to the trainer's own renders, so both sides see one template state. The stock Qwen3.x template reads "
            "`preserve_thinking`: with the env's `carry_reasoning`, every prior turn's reasoning stays rendered "
            "even after a user message (a recovery nudge). `reasoning_effort` and `reasoning_budget` are refused "
            "here: both are per episode; the level travels as the request's top-level field (SGLang also gets an exact "
            "nested copy, the spelling it lets override that field), the budget is added to the nested form per request."
        },
    )

    reasoning_price: dict[str, float] | None = field(
        default=None,
        metadata={
            "help": "Per effort level, the reward units an episode pays per 1,000 reasoning tokens, summed over its "
            "assistant turns (null = off). Must map exactly the three effort levels, low, medium and high (refused at "
            "trainer construction otherwise, as is an environment whose reasoning_effort is unset); price the lowest "
            "level highest, so the same trace costs most where little reasoning was asked. Capped by "
            "reasoning_price_cap. Logged as reward/reasoning_price."
        },
    )
    reasoning_price_cap: float = field(
        default=DEFAULT_REASONING_PRICE_CAP,
        metadata={
            "help": "Cap of reasoning_price per episode, in reward units: a long trace cannot outweigh the task "
            "reward. Keep it, plus reasoning_floor, below what the environment charges for the decisions it "
            "prices (a resubmission, in code contests)."
        },
    )
    reasoning_floor: float = field(
        default=0.0,
        metadata={
            "help": "Weight of the reasoning under-use floor (0 = off): an episode whose reasoning tokens, summed "
            "over its turns, fall short of three quarters of its per-turn thinking cap (its level's thinking_tokens, "
            "clamped by rollout_max_thinking_tokens) pays -weight x shortfall / that reference. The one term that "
            "pays for more reasoning; it resists reasoning shrinking toward nothing. Keep it, plus "
            "reasoning_price_cap while the price is on, below what the environment charges for the decisions it "
            "prices (a resubmission, in code contests). An episode with no thinking budget is free of it, and a "
            "run where no drawable level sets one and rollout_max_thinking_tokens is unset is refused at trainer "
            "construction. Logged as reward/reasoning_floor."
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
                "Deadline for one rollout episode in seconds of engine-serving time (a weight-sync pause is "
                "credited back). Bounds the WHOLE episode "
                "(generation + tool execution + grading), unlike request_timeout which bounds a single "
                "HTTP call. Without it a wedged tool/sandbox blocks its rank forever, and the other ranks "
                "block behind it at the next collective. A timed-out episode is cancelled and counted in "
                "episode/error_rate. The default sits at two thirds of the "
                f"{DEFAULT_NCCL_TIMEOUT_MINUTES}-min default NCCL watchdog so a straggler is cancelled "
                f"with ~{DEFAULT_NCCL_TIMEOUT_MINUTES // 3} min of margin; raise DIST_NCCL_TIMEOUT_MINUTES "
                "before raising this."
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
        per_episode = {"reasoning_effort", REASONING_BUDGET_TEMPLATE_VAR} & set(self.rollout_chat_template_kwargs)
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
        collective aborts before the straggler is cancelled. Shares ``get_nccl_timeout()``'s
        resolver, so the bound checked here is the one ``init_process_group`` installs.
        """
        nccl_minutes = resolve_nccl_timeout_minutes()
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
