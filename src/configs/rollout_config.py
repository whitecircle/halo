"""Generation and retry configuration for rollout collection, shared by the trainer config and the
Ray rollout actors.

Deliberately a leaf: ``AsyncTrainingConfig`` builds one, and the actors receive it pickled, so it must
not carry the Ray (or any other engine) import into every ``import src.configs``.
"""

from dataclasses import dataclass
from typing import Literal

# ``AsyncTrainingConfig`` is the validated YAML surface and supplies every MIRRORED field below, so a
# directly-built RolloutConfig must default to what that path would have produced. One constant per
# pair is the only way the two cannot drift. The three fields that path DERIVES rather than mirrors
# (``capture_token_ids``, ``capture_routed_experts``, ``stop_token_ids``) are outside that rule: they
# default to the off state a hand-built config wants, which for ``capture_token_ids`` is NOT what the
# YAML path produces — see the field.
DEFAULT_ROLLOUT_TEMPERATURE = 0.7
DEFAULT_ROLLOUT_TOP_P = 0.95
DEFAULT_ROLLOUT_MAX_TOKENS = 32768
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_WAIT_SECONDS = 1.0

# Two thirds of the 30-min default NCCL watchdog, so a straggler episode is cancelled before its
# peers' per-step collective aborts. Shared with ``AsyncTrainingConfig.episode_timeout``: a directly
# built RolloutConfig must not default above what that validated path allows.
DEFAULT_EPISODE_TIMEOUT_SECONDS = 1200.0


@dataclass
class RolloutConfig:
    """Generation and retry configuration for rollout collection."""

    backend: Literal["vllm", "sglang"] = "vllm"
    """Rollout engine serving these requests. Both speak OpenAI chat completions, but the vLLM-only
    request fields below are *silently ignored* by SGLang (its request model drops unknown keys
    rather than rejecting them), so the payload builder gates them on this rather than sending them
    hopefully. Mirrors ``AsyncTrainingConfig.rollout_backend``, which validates the value."""

    temperature: float = DEFAULT_ROLLOUT_TEMPERATURE
    top_p: float = DEFAULT_ROLLOUT_TOP_P
    max_tokens: int = DEFAULT_ROLLOUT_MAX_TOKENS
    """Max tokens per single-turn generation. See ``AsyncTrainingConfig.rollout_max_tokens``, the
    validated surface this mirrors."""

    max_thinking_tokens: int | None = None
    """Per-turn reasoning-token budget (vLLM ``thinking_token_budget``): caps CoT, then forces an
    answer. Requires a server-side reasoning parser. None = only ``max_tokens`` caps the turn."""

    capture_token_ids: bool = False
    """Request per-token logprobs so the sampled generation token ids can be captured (needs the server
    flag ``--return-tokens-as-token-ids``), for training on exactly what the model emitted. Off in a
    hand-built config — the logprobs payload is large and only a consumer that reads the ids wants it.
    An env-GRPO run does not take this default: ``AsyncTrainingConfig.get_rollout_config`` derives the
    value from ``train_on_sampled_tokens``, which is ON by default, so a stock YAML run captures ids.

    Also sets vLLM's ``return_token_ids`` request flag, which returns the ENGINE's rendered prompt ids
    (top-level ``prompt_token_ids``, prefix-cache-neutral). The trainer uses them as each per-turn
    row's prompt verbatim — the same no-re-render principle as the sampled completion ids: the server
    template render is the ground truth, a trainer-side re-render can only drift from it."""

    capture_routed_experts: bool = False
    """Capture the engine's per-token MoE routing (``routed_experts`` on each choice — needs the server
    flag ``--enable-return-routed-experts`` and a non-FlashInfer MoE backend) for R3 routing replay.
    Kept as the raw base64 payload; the trainer decodes at tokenization."""

    stop_token_ids: list[int] | None = None
    """Token ids that end a turn (vLLM ``stop_token_ids``). Set to the model's tool-call terminator so a
    turn stops when the model emits its call; otherwise a non-eos terminator keeps the model generating,
    hallucinating the tool result and playing the whole episode in one turn."""

    model_name: str | None = None
    """Model name for /v1/chat/completions. Optional — vllm-serve uses the loaded model when omitted."""

    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    """HTTP timeout per request in seconds."""

    episode_timeout: float = DEFAULT_EPISODE_TIMEOUT_SECONDS
    """Wall-clock deadline for one episode in seconds (vs ``request_timeout`` per HTTP call). A wedged
    tool or sandbox otherwise hangs its rank forever, blocking peers at the next collective. Timed-out
    episodes are cancelled. See :data:`DEFAULT_EPISODE_TIMEOUT_SECONDS` for how the default is sized."""

    max_retries: int = DEFAULT_MAX_RETRIES
    """Max retry attempts for transient vLLM failures."""

    retry_base_wait: float = DEFAULT_RETRY_BASE_WAIT_SECONDS
    """Base wait time in seconds for exponential backoff."""
