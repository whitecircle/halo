"""Generation and retry configuration for rollout collection. Free of Ray and rollout-engine imports:
``AsyncRolloutConfig`` builds one and the actors receive it pickled."""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from halo.env import DEFAULT_NCCL_TIMEOUT_MINUTES

logger = logging.getLogger(__name__)

# The rollout engines a run selects between; a weight-sync client's ``BACKEND_KEY`` carries the same values.
RolloutBackend = Literal["vllm", "sglang"]
VLLM_BACKEND, SGLANG_BACKEND = get_args(RolloutBackend)

# The sampling filters' off values by RolloutConfig field, sent on every request: both engines fill an omitted
# one from the model's generation_config.json. -1 is the top_k off value both accept (SGLang refuses 0).
SAMPLER_FILTERS_OFF = {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}
# The sampler whose reported log-probs are the raw model distribution's: every filter off, temperature and top-p 1.
IDENTITY_SAMPLER = {"temperature": 1.0, "top_p": 1.0, **SAMPLER_FILTERS_OFF}

ReasoningEffortChoice = Literal["low", "medium", "high", "random"]
RANDOM_REASONING_EFFORT = "random"
VALID_REASONING_EFFORTS = tuple(level for level in get_args(ReasoningEffortChoice) if level != RANDOM_REASONING_EFFORT)
# Example end strings of the families' reasoning parsers, for the knob's help and its refusal.
REASONING_END_TOKEN_EXAMPLES = (
    "Qwen3.x '</think>', Gemma 4 '<channel|>', gpt-oss '<|end|><|start|>assistant<|channel|>final<|message|>'"
)


@dataclass
class RolloutConfig:
    """Its defaults are the ones ``AsyncRolloutConfig``, the validated YAML surface, reads for the fields it mirrors;
    fields that config derives default to their off state."""

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
    """Per-turn reasoning-token cap (vLLM ``thinking_token_budget``), the same at every effort level; None =
    ``max_tokens`` less ``max_answer_tokens`` caps it where that is set, else nothing does."""
    max_answer_tokens: int | None = None
    """The most a turn may generate past its reasoning cap; None = ``max_tokens`` alone bounds the turn."""
    reasoning_end_token_id: int | None = None
    """The id closing reasoning, resolved from ``rollout_reasoning_end_token`` by the caller owning the tokenizer:
    the per-turn reasoning count ``episode/thinking_cap_turns`` reads ends at it. None = no count."""
    capture_token_ids: bool = False
    """Capture the sampled token ids and the engine's rendered prompt ids, so training uses what the server rendered
    and the model emitted. Off in a hand-built config; ``get_rollout_config`` derives it from
    ``train_on_sampled_tokens``."""
    capture_routed_experts: bool = False
    """Capture the engine's per-token MoE routing (the raw ``routed_experts`` payload) for R3 routing replay; needs
    ``--enable-return-routed-experts`` and the triton MoE backend the weight sync also needs."""
    stop_token_ids: list[int] | None = None
    """Token ids that end a turn: the model's tool-call terminator, without which a non-eos terminator lets the model
    play the whole episode in one turn."""
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    """Chat-template variables sent with every request; never the per-episode effort level."""
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


def resolve_rollout_stop_token_ids(tokenizer, names: list[str]) -> list[int] | None:
    """Every name must resolve, since a dropped terminator runs the turn past the call it ends."""
    if not names:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    ids = {name: tokenizer.convert_tokens_to_ids(name) for name in names}
    unresolved = [name for name, tid in ids.items() if tid is None or tid == unk]
    if unresolved:
        raise ValueError(
            f"rollout_stop_tokens {unresolved} are not tokens of "
            f"{getattr(tokenizer, 'name_or_path', 'the tokenizer')!r}, so no turn would stop where the config "
            "says. Check the spellings against its special tokens, or drop them."
        )
    return list(ids.values())


def resolve_reasoning_end_ids(tokenizer, marker: str) -> tuple[int, ...]:
    """Encoded without special tokens, as vLLM forces its parser's end string; an encoding holding none of the
    tokenizer's added tokens is a marker this model does not write, and raises."""
    ids = tuple(tokenizer.encode(marker, add_special_tokens=False))
    if not set(ids) & set(tokenizer.added_tokens_decoder):
        raise ValueError(
            f"rollout_reasoning_end_token {marker!r} is not a reasoning marker of this tokenizer: it encodes to "
            f"{list(ids)}, none of them one of its added tokens. Name the end string the server's reasoning "
            f"parser forces ({REASONING_END_TOKEN_EXAMPLES})."
        )
    return ids


def find_reasoning_end_ids(
    tokenizer, marker: str, *, logger: logging.Logger | None = logger
) -> tuple[int, ...] | None:
    """``None`` for a marker this tokenizer does not write, warned unless ``logger`` is ``None``; a turn's reasoning
    then goes uncounted."""
    try:
        return resolve_reasoning_end_ids(tokenizer, marker)
    except ValueError as e:
        if logger is not None:
            logger.warning("%s A turn's reasoning goes uncounted.", e)
        return None


def resolve_reasoning_end_token_id(reasoning_end_ids: tuple[int, ...] | None) -> int | None:
    """``None`` without the ids or for a marker of several tokens (gpt-oss's final-channel opener)."""
    return reasoning_end_ids[0] if reasoning_end_ids is not None and len(reasoning_end_ids) == 1 else None
