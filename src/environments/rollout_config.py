"""Generation and retry configuration for rollout collection, shared by the trainer config and the Ray rollout
actors. Free of Ray and rollout-engine imports: ``AsyncTrainingConfig`` builds one and the actors receive it pickled.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from src.env import DEFAULT_NCCL_TIMEOUT_MINUTES

# The rollout engines a run selects between: the type both config surfaces declare, and the spellings
# every consumer branching on the engine without its weight-sync client (whose ``BACKEND_KEY`` carries
# the same value) compares against.
RolloutBackend = Literal["vllm", "sglang"]
VLLM_BACKEND, SGLANG_BACKEND = get_args(RolloutBackend)

# The sampling filters' off values by RolloutConfig field, sent on every request: both engines fill an omitted
# one from the model's generation_config.json. -1 is the top_k off value both accept (SGLang refuses 0).
SAMPLER_FILTERS_OFF = {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}
# The sampler whose reported log-probs are the raw model distribution's: every filter off, temperature and top-p 1.
IDENTITY_SAMPLER = {"temperature": 1.0, "top_p": 1.0, **SAMPLER_FILTERS_OFF}

# Chat-template variable carrying an episode's per-turn thinking budget, the pair of the request's
# top-level ``reasoning_effort``; both are per episode, never run-wide.
REASONING_BUDGET_TEMPLATE_VAR = "reasoning_budget"
# Set true beside it when that budget is soft (the engine does not enforce it), so a template states it
# as a target rather than a cut; absent for an enforced budget.
REASONING_BUDGET_SOFT_TEMPLATE_VAR = "reasoning_budget_soft"
# The chat-template reasoning-effort steers a config accepts: a level, or "random" (one drawn per episode).
ReasoningEffortChoice = Literal["low", "medium", "high", "random"]
# Example end strings of the families' reasoning parsers, for the knob's help and its refusal.
REASONING_END_TOKEN_EXAMPLES = (
    "Qwen3.x '</think>', Gemma 4 '<channel|>', gpt-oss '<|start|>assistant<|channel|>final<|message|>'"
)


@dataclass
class RolloutConfig:
    """Generation and retry configuration for rollout collection.

    Its defaults are the ones ``AsyncTrainingConfig``, the validated YAML surface, reads for the fields it
    mirrors; the fields that config derives rather than mirrors default to their off state.
    """

    backend: RolloutBackend = VLLM_BACKEND
    """Rollout engine; SGLang drops unknown request keys, so the payload builder gates the vLLM-only fields on it."""
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = SAMPLER_FILTERS_OFF["top_k"]
    min_p: float = SAMPLER_FILTERS_OFF["min_p"]
    repetition_penalty: float = SAMPLER_FILTERS_OFF["repetition_penalty"]
    max_tokens: int = 32768
    """Max tokens per single-turn generation."""
    max_episode_tokens: int | None = None
    """The most tokens one episode may sample over all its turns; None = only the per-turn caps and max_turns."""
    max_thinking_tokens: int | None = None
    """Per-turn reasoning-token cap (vLLM ``thinking_token_budget``); None = only a level's own budget caps it."""
    max_answer_tokens: int | None = None
    """The most a turn may generate past its reasoning cap; None = ``max_tokens`` alone bounds the turn."""
    reasoning_end_token_id: int | None = None
    """The id closing reasoning, resolved from ``rollout_reasoning_end_token`` by the caller owning the tokenizer:
    the per-turn reasoning count ``episode/thinking_cap_turns`` reads ends at it. None = no count."""
    capture_token_ids: bool = False
    """Capture the sampled token ids and vLLM's rendered prompt ids, so training uses what the server rendered and
    the model emitted. Off in a hand-built config; ``get_rollout_config`` derives it from ``train_on_sampled_tokens``."""
    capture_routed_experts: bool = False
    """Capture the engine's per-token MoE routing (the raw ``routed_experts`` payload) for R3 routing replay; needs
    ``--enable-return-routed-experts`` and a non-FlashInfer MoE backend."""
    stop_token_ids: list[int] | None = None
    """Token ids that end a turn: the model's tool-call terminator, without which a non-eos terminator lets the model
    play the whole episode in one turn."""
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    """Chat-template variables sent with every request; never the per-episode effort or thinking-budget keys."""
    model_name: str | None = None
    """Model name for /v1/chat/completions; the server's loaded model when omitted."""
    request_timeout: float = 120.0
    """HTTP timeout per request in seconds of engine-serving time: a weight-sync pause is credited back."""
    episode_timeout: float = DEFAULT_NCCL_TIMEOUT_MINUTES * 60 * 2 / 3
    """Deadline for one episode in seconds of engine-serving time. Two thirds of the default NCCL watchdog, so a
    straggler episode is cancelled before its peers' per-step collective aborts."""
    max_retries: int = 3
    """Max retry attempts for a transient engine failure."""
    retry_base_wait: float = 1.0
    """Base wait time in seconds for exponential backoff."""
