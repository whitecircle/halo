"""Generation and retry configuration for rollout collection, shared by the trainer config and the
Ray rollout actors.

Kept free of Ray and rollout-engine imports: ``AsyncTrainingConfig`` builds one and the actors receive
it pickled, so those imports must not be pulled into every ``import src.configs``.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

# ``AsyncTrainingConfig`` is the validated YAML surface and supplies every mirrored field below, so a
# directly-built RolloutConfig defaults to what that path would produce; one shared constant per pair
# keeps the two in step. The fields that path derives rather than mirrors (``capture_token_ids``,
# ``capture_routed_experts``, ``stop_token_ids``, ``reasoning_end_token_id``) default to the off state
# instead — see each field.
DEFAULT_ROLLOUT_TEMPERATURE = 0.7
DEFAULT_ROLLOUT_TOP_P = 0.95
# The three filters' off values, sent on every request: both engines fill an omitted one from the
# model's generation_config.json. -1 is the top_k off value both accept (SGLang refuses 0).
DEFAULT_ROLLOUT_TOP_K = -1
DEFAULT_ROLLOUT_MIN_P = 0.0
DEFAULT_ROLLOUT_REPETITION_PENALTY = 1.0
# Those off values by RolloutConfig field, and the sampler whose reported log-probs are the raw model
# distribution's: every filter off, temperature and top-p at 1.
SAMPLER_FILTERS_OFF = {
    "top_k": DEFAULT_ROLLOUT_TOP_K,
    "min_p": DEFAULT_ROLLOUT_MIN_P,
    "repetition_penalty": DEFAULT_ROLLOUT_REPETITION_PENALTY,
}
IDENTITY_SAMPLER = {"temperature": 1.0, "top_p": 1.0, **SAMPLER_FILTERS_OFF}
DEFAULT_ROLLOUT_MAX_TOKENS = 32768
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_WAIT_SECONDS = 1.0

# Two thirds of the 30-min default NCCL watchdog, so a straggler episode is cancelled before its
# peers' per-step collective aborts. Shared with ``AsyncTrainingConfig.episode_timeout`` so a
# directly built RolloutConfig does not default above what that validated path allows.
DEFAULT_EPISODE_TIMEOUT_SECONDS = 1200.0

# Chat-template variable carrying an episode's per-turn thinking budget, the pair of the request's
# top-level ``reasoning_effort``; both are per episode, never run-wide.
REASONING_BUDGET_TEMPLATE_VAR = "reasoning_budget"
DEFAULT_REASONING_END_TOKEN = "</think>"
# Example end strings of the families' reasoning parsers, for the knob's help and its refusal.
REASONING_END_TOKEN_EXAMPLES = (
    "Qwen3.x '</think>', Gemma 4 '<channel|>', gpt-oss '<|start|>assistant<|channel|>final<|message|>'"
)


@dataclass
class RolloutConfig:
    """Generation and retry configuration for rollout collection."""

    backend: Literal["vllm", "sglang"] = "vllm"
    """Rollout engine serving these requests. Both speak OpenAI chat completions, but SGLang drops
    unknown request keys rather than rejecting them, so the payload builder gates the vLLM-only
    fields below on this value. Mirrors ``AsyncTrainingConfig.rollout_backend``, which validates
    it."""

    temperature: float = DEFAULT_ROLLOUT_TEMPERATURE
    top_p: float = DEFAULT_ROLLOUT_TOP_P
    top_k: int = DEFAULT_ROLLOUT_TOP_K
    min_p: float = DEFAULT_ROLLOUT_MIN_P
    repetition_penalty: float = DEFAULT_ROLLOUT_REPETITION_PENALTY
    max_tokens: int = DEFAULT_ROLLOUT_MAX_TOKENS
    """Max tokens per single-turn generation. See ``AsyncTrainingConfig.rollout_max_tokens``, the
    validated surface this mirrors."""
    max_episode_tokens: int | None = None
    """The most tokens one episode may sample over all its turns, reasoning and visible output
    together: a turn's ``max_tokens`` narrows to what the episode has left, and an episode with no
    room for a further turn ends truncated. None = only the per-turn caps and ``max_turns`` bound the
    episode. Mirrors ``AsyncTrainingConfig.rollout_max_episode_tokens``."""

    max_thinking_tokens: int | None = None
    """Per-turn reasoning-token cap (vLLM ``thinking_token_budget``): caps CoT, then forces an answer
    within the rest of ``max_tokens``. A level's ``thinking_tokens`` below it is the turn's cap instead.
    Requires a server-side reasoning parser. None = only a level's ``thinking_tokens`` caps reasoning, or
    nothing does."""

    reasoning_end_token_id: int | None = None
    """The id of the token that closes reasoning, resolved from ``rollout_reasoning_end_token`` by the
    caller that owns the tokenizer: a turn's reasoning is counted as the sampled ids up to and including
    it, the count the overlong charge reads. None = no count."""

    capture_token_ids: bool = False
    """Request per-token logprobs so the sampled generation token ids can be captured (needs the
    server flag ``--return-tokens-as-token-ids``), for training on exactly what the model emitted.
    Off in a hand-built config, since the logprobs payload is large. An env-GRPO run does not take
    this default: ``AsyncTrainingConfig.get_rollout_config`` derives the value from
    ``train_on_sampled_tokens``, which is on by default, so a stock YAML run captures ids.

    Also sets vLLM's ``return_token_ids`` request flag, which returns the engine's rendered prompt
    ids (top-level ``prompt_token_ids``, prefix-cache-neutral). The trainer uses them as each
    per-turn row's prompt verbatim, on the same principle as the sampled completion ids: the server
    template render is the ground truth, and a trainer-side re-render can drift from it."""

    capture_routed_experts: bool = False
    """Capture the engine's per-token MoE routing (``routed_experts`` on each choice — needs the server
    flag ``--enable-return-routed-experts`` and a non-FlashInfer MoE backend) for R3 routing replay.
    Kept as the raw base64 payload; the trainer decodes at tokenization."""

    stop_token_ids: list[int] | None = None
    """Token ids that end a turn (vLLM ``stop_token_ids``). Set to the model's tool-call terminator so a
    turn stops when the model emits its call; otherwise a non-eos terminator keeps the model generating,
    hallucinating the tool result and playing the whole episode in one turn."""

    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    """Chat-template variables sent with every request (``chat_template_kwargs``), e.g. Qwen3.x's
    ``preserve_thinking`` so reasoning the env carries stays rendered across a later user message. Never
    the per-episode keys: the reasoning effort travels top-level and the level's thinking budget is added
    per request (``generation_control_fields``)."""

    model_name: str | None = None
    """Model name for /v1/chat/completions. Optional — vllm-serve uses the loaded model when omitted."""

    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    """HTTP timeout per request in seconds of engine-serving time: a weight-sync pause is credited back."""

    episode_timeout: float = DEFAULT_EPISODE_TIMEOUT_SECONDS
    """Deadline for one episode in seconds of engine-serving time (vs ``request_timeout`` per HTTP
    call). A wedged tool or sandbox otherwise hangs its rank forever, blocking peers at the next
    collective. Timed-out episodes are cancelled. See :data:`DEFAULT_EPISODE_TIMEOUT_SECONDS` for how
    the default is sized."""

    max_retries: int = DEFAULT_MAX_RETRIES
    """Max retry attempts for transient vLLM failures."""

    retry_base_wait: float = DEFAULT_RETRY_BASE_WAIT_SECONDS
    """Base wait time in seconds for exponential backoff."""
