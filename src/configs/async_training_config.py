"""Configuration for async Environmental GRPO training (DistributedAsyncEnvironmentalGRPOTrainer)."""

import logging
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from src.args.mixins import AdvantageShapingArguments, ChunkedLogprobsArguments
from src.configs.rollout_config import (
    DEFAULT_EPISODE_TIMEOUT_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_RETRY_BASE_WAIT_SECONDS,
    DEFAULT_ROLLOUT_MAX_TOKENS,
    DEFAULT_ROLLOUT_TEMPERATURE,
    DEFAULT_ROLLOUT_TOP_P,
    RolloutConfig,
)
from src.env import WATCHDOG_WARN_FRACTION, resolve_nccl_timeout_minutes

logger = logging.getLogger(__name__)


def rollout_field_sources(config_cls) -> dict[str, str]:
    """``RolloutConfig`` field -> the ``config_cls`` field :meth:`AsyncTrainingConfig.get_rollout_config`
    copies into it.

    Derived from the two declarations instead of listed, so a knob added to both sides forwards
    itself: the YAML surface spells a rollout knob ``rollout_<name>`` where the bare name would be
    ambiguous (that spelling wins where both exist) and identically otherwise. ``RolloutConfig``
    fields with no counterpart here are the ones the builder derives from other state.

    Takes the config class as an argument because it is declared below this module-level mapping.
    """
    declared = {f.name for f in fields(config_cls)}
    return {
        target.name: source
        for target in fields(RolloutConfig)
        for source in (target.name, f"rollout_{target.name}")
        if source in declared
    }


@dataclass
class AsyncTrainingConfig(AdvantageShapingArguments, ChunkedLogprobsArguments):
    """Async training infrastructure (Ray workers, rollout-server connections, weight sync, rollout/prefetch). Environment selection lives in EnvironmentConfig; trainer in src/trainers/grpo/environmental.py."""

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
            "throttle. Server-pool load = this × data_parallel_size ÷ num_servers. Size it to the per-rank "
            "rollout demand of one generation cycle (per_device_train_batch_size × gradient_accumulation_steps) "
            "with ~2× headroom for prefetch; raising it past the actual rollout count does nothing. "
            "Default: 4 × this rank's share of the rollout workers (num_rollout_workers ÷ world_size on a shared Ray "
            "cluster, all of them locally), clamped to ≥ that share."
        },
    )

    ray_address: str | None = field(default=None, metadata={"help": "Ray cluster address. None for local mode."})

    rollout_backend: Literal["vllm", "sglang"] = field(
        default="vllm",
        metadata={
            "help": "Inference engine serving rollouts and receiving weight updates. Both support "
            "generation, NCCL weight sync, `train_on_sampled_tokens` and `routing_replay: rollout`. "
            "'sglang' does not support `rollout_max_thinking_tokens` (the trainer wires neither of "
            "SGLang's budget mechanisms; harmony models have none server-side), needs "
            "fsdp_reshard_after_backward=False (its sync forces socket NCCL process-global, making "
            "FSDP2's per-microstep reshard the dominant step cost otherwise), and must be served "
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
        metadata={"help": "Sync weights to vLLM server(s) every N training steps. Must be >= 1 (1 = every step)."},
    )

    rollout_temperature: float = field(
        default=DEFAULT_ROLLOUT_TEMPERATURE, metadata={"help": "Temperature for rollout generation."}
    )

    rollout_top_p: float = field(
        default=DEFAULT_ROLLOUT_TOP_P, metadata={"help": "Top-p (nucleus) sampling for rollout generation."}
    )

    rollout_max_tokens: int = field(
        default=DEFAULT_ROLLOUT_MAX_TOKENS,
        metadata={
            "help": "Max tokens per single-turn generation (one /chat/completions call). The whole "
            "multi-turn trajectory accumulates across turns and is bounded only by the model's context "
            "window (shared with vLLM); env-GRPO does not truncate it — a trajectory exceeding the context "
            "fails. This per-turn budget is the active generation knob; it is verified against the server "
            "at startup."
        },
    )

    rollout_max_thinking_tokens: int | None = field(
        default=None,
        metadata={
            "help": "Per-turn reasoning-token budget for reasoning models (vLLM thinking_token_budget): "
            "caps the chain-of-thought, then forces the model to answer with the rest of max_tokens. "
            "Requires a reasoning parser on the vLLM server (--reasoning-parser qwen3 for Qwen3.x; the "
            "openai_gptoss plugin for gpt-oss). None = unbounded reasoning."
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
            "ids), sharing the trajectory's advantage. Requires the vLLM server to run with "
            "`--return-tokens-as-token-ids`; falls back to re-tokenization for any turn whose ids were not "
            "captured. Default on."
        },
    )

    isr_band_min: float | None = field(
        default=None,
        metadata={
            "help": "Bidirectional TOKEN band on the vLLM->trainer IS ratio: a corrected token whose "
            "raw ratio leaves [isr_band_min, isr_band_max] is MASKED (ratio 0, gradient removed) instead "
            "of merely truncated — the production-convergent MoE-mismatch treatment (GLM-5, IcePop). "
            "Set both bounds to activate; start [0.5, 2]. None (default) = truncation only."
        },
    )
    isr_band_max: float | None = field(
        default=None,
        metadata={"help": "Upper bound of the token band (see isr_band_min)."},
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
            "trajectories masked by the geo-band/veto/OPSM stages exceeds this, ZERO the whole step's "
            "policy gradient instead of training on the unmasked survivors. At high masked fractions the "
            "surviving trajectories are a selection-biased sample (exactly the rows where the drifted "
            "policy still agrees with the rollout), so continuing to train amplifies the drift; skipping "
            "holds the policy still until the next weight-sync re-anchors the rollouts. Logged as "
            "`sampling/update_skipped`. None (default) = off; 0.3-0.5 is a sane range."
        },
    )

    # Flipped vs AdvantageShapingArguments: on a sparse verifiable env reward the dead all-equal
    # groups dominate the batch, so dropping them is the right default here.
    drop_degenerate_groups: bool = field(
        default=True,
        metadata={
            "help": "Drop GRPO groups whose completions ALL scored the same reward. Their advantage is "
            "already 0 (no policy gradient), but their tokens would still inflate the loss normalizer and "
            "dilute the groups that do carry signal — on a sparse verifiable reward these dead groups "
            "dominate the batch. Masking them restores the effective batch size (the cheap half of DAPO's "
            "dynamic sampling: drop, without resampling replacements). Logged as "
            "`sampling/degenerate_group_frac`. Default on."
        },
    )

    rollout_stop_tokens: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Special-token strings that end a turn's generation, resolved to ids via the tokenizer "
            "and sent as vLLM `stop_token_ids`. Set to the model's tool-call terminator so a turn stops "
            "when the model emits its call and the environment runs it — without this a model whose "
            "terminator is not an eos (e.g. gpt-oss `<|call|>` under harmony-disabled serving) keeps "
            "generating, hallucinating the tool result and playing out the whole episode in one turn "
            "(huge, off-policy-noisy completions). Empty (default) = only the model's eos stops a turn."
        },
    )

    reasoning_compliance_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight of the reasoning-budget calibration reward (0 = off). When > 0 and an "
            "episode has a CoT budget (reasoning_effort set), the trainer adds an ASYMMETRIC per-turn "
            "calibration term (reasoning_calibration_penalty): no penalty in [0.3,0.9]x the budget, a "
            "mild penalty below (under-use), a strong penalty above / on truncation (over-use, up to "
            "-weight). Trains the model to match the requested effort. ~0.15 shapes without dominating "
            "the verifier reward."
        },
    )

    enable_prefetch: bool = field(
        default=True, metadata={"help": "Enable prefetching to overlap rollout collection with training."}
    )

    num_prefetch_batches: int = field(
        default=1,
        metadata={"help": "Number of batches to prefetch ahead. 1 provides good overlap without excessive memory."},
    )

    model_name: str | None = field(
        default=None,
        metadata={
            "help": "Model name sent in vLLM /v1/chat/completions requests. "
            "Optional — the server answers with its loaded model when omitted."
        },
    )

    request_timeout: float = field(
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS, metadata={"help": "HTTP timeout per vLLM request in seconds."}
    )

    episode_timeout: float = field(
        default=DEFAULT_EPISODE_TIMEOUT_SECONDS,
        metadata={
            "help": (
                "Wall-clock deadline for one rollout episode in seconds. Bounds the WHOLE episode "
                "(generation + tool execution + grading), unlike request_timeout which bounds a single "
                "HTTP call. Without it a wedged tool/sandbox blocks its rank forever, and the other ranks "
                "block behind it at the next collective. A timed-out episode is cancelled and counted in "
                "episode/error_rate. The default sits at two thirds of the 30-min default NCCL watchdog "
                "so a straggler is cancelled with ~10 min of margin; raise DIST_NCCL_TIMEOUT_MINUTES "
                "before raising this."
            )
        },
    )

    max_retries: int = field(
        default=DEFAULT_MAX_RETRIES,
        metadata={
            "help": "Retries after a failed vLLM request (total attempts = max_retries + 1). "
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
        # `global_step % sync_weights_every_n_steps` — 0 raises ZeroDivisionError mid-training, after
        # the rollout workers and vLLM servers are already up.
        if self.sync_weights_every_n_steps < 1:
            raise ValueError(
                f"sync_weights_every_n_steps must be >= 1 (1 = every step), got {self.sync_weights_every_n_steps}"
            )
        # A negative budget reaches backoff as max_tries <= 0, which it treats as "no limit": a wedged
        # server is then retried until the NCCL watchdog kills the job.
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0 (0 = one attempt, no retry), got {self.max_retries}")
        # queue.Queue treats maxsize <= 0 as UNBOUNDED, so `num_prefetch_batches: 0` — the natural
        # spelling of "no prefetch" — would silently buffer rollouts until the host runs out of RAM.
        if self.num_prefetch_batches < 1:
            raise ValueError(
                f"num_prefetch_batches must be >= 1, got {self.num_prefetch_batches}; set "
                f"enable_prefetch: false to turn prefetching off."
            )
        # 0 workers builds an empty actor list and divides by it once Ray and the servers are already up.
        if self.num_rollout_workers < 1:
            raise ValueError(f"num_rollout_workers must be >= 1, got {self.num_rollout_workers}")
        # `max_concurrent_rollouts or default` reads 0 as "unset", so the cap would silently vanish.
        if self.max_concurrent_rollouts is not None and self.max_concurrent_rollouts < 1:
            raise ValueError(
                f"max_concurrent_rollouts must be >= 1 when set (null = derive from "
                f"num_rollout_workers), got {self.max_concurrent_rollouts}"
            )
        self._validate_backend_capabilities()

    def _validate_backend_capabilities(self) -> None:
        """Reject knobs the selected engine cannot honour, rather than letting them no-op silently.

        SGLang ignores unknown request fields instead of rejecting them, so a knob it does not
        implement would otherwise degrade quietly — the run ignores the thinking budget with only a
        log line to show for it.

        Sampled-token training and rollout routing replay are NOT among those: SGLang carries the
        sampled ids in the ``meta_info`` it echoes on each choice, and publishes routed experts
        response-level (raw-int32 wire format, handled by ``decode_rollout_routing``).

        Neither is the environment's per-effort ``thinking_tokens`` profile, whose budget reaches the
        same request field: it is also stamped on the trajectory, where ``reasoning_compliance_weight``
        can price CoT against it, so on an engine without the field it degrades to a soft target
        instead of vanishing. The rollout actor warns once per process that it is unenforced. This
        knob has no such second consumer, so rejecting it is the only honest answer here.
        """
        if self.rollout_backend != "sglang":
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

    def get_rollout_config(self, stop_token_ids: list[int] | None = None):
        """Build RolloutConfig from this config. ``stop_token_ids`` is resolved from
        ``rollout_stop_tokens`` by the trainer (which owns the tokenizer)."""
        self._validate_timeouts_against_nccl_watchdog()
        return RolloutConfig(
            **{target: getattr(self, source) for target, source in rollout_field_sources(type(self)).items()},
            # Derived from other state rather than mirrored from a same-named knob.
            capture_token_ids=self.train_on_sampled_tokens,
            capture_routed_experts=self.routing_replay == "rollout",
            stop_token_ids=stop_token_ids,
        )

    def _validate_timeouts_against_nccl_watchdog(self):
        """Guard rollout timeouts against the NCCL collective watchdog.

        A straggler rank holds its peers at the per-step FSDP collective for as long as its slowest
        episode runs; if ``episode_timeout`` or the retry budget reaches the watchdog, the peers'
        collective aborts before the straggler is cancelled. Shares ``get_nccl_timeout()``'s resolver,
        so the bound checked here is the one ``init_process_group`` actually installs.
        """
        nccl_minutes = resolve_nccl_timeout_minutes()
        watchdog = nccl_minutes * 60

        # episode_timeout above the watchdog is a guaranteed footgun (straggler can't be cancelled
        # before the peers' collective aborts); equal only warns below.
        if self.episode_timeout > watchdog:
            raise ValueError(
                f"episode_timeout ({self.episode_timeout:.0f}s) must be below the NCCL collective "
                f"watchdog ({watchdog:.0f}s = {nccl_minutes} min). A straggler rank holds its peers at "
                f"the per-step collective for up to episode_timeout, so an equal/greater value lets the "
                f"watchdog fire first and abort the run. Raise DIST_NCCL_TIMEOUT_MINUTES above "
                f"episode_timeout (keep ≥15 min margin so the cancelled rank can unwind and rejoin), or "
                f"lower episode_timeout."
            )
        if self.episode_timeout >= WATCHDOG_WARN_FRACTION * watchdog:
            logger.warning(
                f"episode_timeout ({self.episode_timeout:.0f}s) is within "
                f"{(1 - WATCHDOG_WARN_FRACTION) * 100:.0f}% of the NCCL watchdog "
                f"({watchdog:.0f}s): a near-deadline straggler risks tripping the peers' per-step "
                f"collective. Raise DIST_NCCL_TIMEOUT_MINUTES for more margin."
            )

        attempts = self.max_retries + 1
        backoff = self.retry_base_wait * (2**self.max_retries - 1)
        worst_case = attempts * self.request_timeout + backoff
        if worst_case >= WATCHDOG_WARN_FRACTION * watchdog:
            logger.warning(
                f"Rollout retry budget (~{worst_case:.0f}s = {attempts}×{self.request_timeout:.0f}s "
                f"request_timeout + backoff) is close to the {watchdog:.0f}s NCCL collective watchdog. "
                f"A stuck rollout will retry until the watchdog fires and HANG the per-step barrier "
                f"instead of giving up. Lower request_timeout or max_retries (a ≤16k-token turn "
                f"generates in ~110s, so request_timeout≈300 with max_retries=3 is ample), or raise "
                f"DIST_NCCL_TIMEOUT_MINUTES."
            )
