"""OpenAI chat-completions wire format for the rollout engines (vLLM and SGLang).

Builds a turn's request payload and reads back the three facts a trainer may capture (sampled ids,
their logprobs, the engine's prompt ids). The two engines spell the capture flags differently and
return those facts in different places. Transport and the episode loop belong to the rollout drivers.
"""

import logging
from typing import Any, get_args, get_type_hints

from src.configs.rollout_config import RolloutConfig
from src.log import warn_once

logger = logging.getLogger(__name__)

# Backends already warned that a per-effort thinking budget reaches no engine field (once per process).
_THINKING_BUDGET_UNENFORCED_WARNED: set[str] = set()


def _logprob_entries(choice: dict[str, Any]) -> list[dict[str, Any]] | None:
    """A choice's per-token ``logprobs.content`` entries, or None when the server returned none."""
    return (choice.get("logprobs") or {}).get("content") or None


def _extract_token_ids(choice: dict[str, Any]) -> list[int] | None:
    """Recover the sampled generation token ids from a choice's per-token logprobs. With
    ``--return-tokens-as-token-ids`` each entry's ``token`` is ``"token_id:<N>"``; parse the ints.
    Returns None if logprobs are absent or malformed, so the caller falls back to re-tokenization."""
    content = _logprob_entries(choice)
    if content is None:
        return None
    ids: list[int] = []
    for entry in content:
        tok = entry.get("token", "")
        if not isinstance(tok, str) or not tok.startswith("token_id:"):
            return None
        try:
            ids.append(int(tok[len("token_id:") :]))
        except ValueError:
            return None
    return ids or None


def _extract_token_logprobs(choice: dict[str, Any]) -> list[float] | None:
    """Recover the per-token sampling log-probs from a choice's ``logprobs.content``. Returns None if
    logprobs are absent or any entry lacks a numeric ``logprob``. Diagnostic only (entropy-collapse
    early warning)."""
    content = _logprob_entries(choice)
    if content is None:
        return None
    lps: list[float] = []
    for entry in content:
        lp = entry.get("logprob")
        if not isinstance(lp, (int, float)):
            return None
        lps.append(float(lp))
    return lps or None


def _sglang_meta_triples(choice: dict[str, Any]) -> list[list[Any]] | None:
    """A choice's ``meta_info.output_token_logprobs`` — ``[logprob, token_id, token_text]`` per sampled
    token, present when the request set both ``logprobs`` and ``return_meta_info``.

    SGLang's OpenAI-compat ``logprobs`` field drops the token id while converting (it reports the
    token as text), but ``return_meta_info`` echoes the pre-conversion dict on the choice, which still
    carries it. Returns None when absent or malformed, so the caller falls back to re-tokenization.
    """
    triples = (choice.get("meta_info") or {}).get("output_token_logprobs")
    if not isinstance(triples, list) or not triples:
        return None
    if not all(isinstance(t, (list, tuple)) and len(t) >= 2 for t in triples):
        return None
    return triples


def _capture_vllm(
    choice: dict[str, Any], data: dict[str, Any]
) -> tuple[list[int] | None, list[float] | None, list[int] | None]:
    """Sampled ids, their logprobs, and the engine's prompt ids, in vLLM's response shape."""
    return (
        _extract_token_ids(choice),
        _extract_token_logprobs(choice),
        data.get("prompt_token_ids"),
    )


def _capture_sglang(
    choice: dict[str, Any], data: dict[str, Any]
) -> tuple[list[int] | None, list[float] | None, list[int] | None]:
    """Same three, in SGLang's response shape: both live on the choice, not the response root."""
    triples = _sglang_meta_triples(choice)
    if triples is None:
        return None, None, choice.get("prompt_token_ids")
    try:
        ids = [int(t[1]) for t in triples]
        logprobs = [float(t[0]) for t in triples]
    except (TypeError, ValueError):
        return None, None, choice.get("prompt_token_ids")
    return ids or None, logprobs or None, choice.get("prompt_token_ids")


# Per-backend response readers: the engines return the same three facts in different places.
_TOKEN_CAPTURE = {"vllm": _capture_vllm, "sglang": _capture_sglang}

# Every selectable backend needs a reader, checked at import rather than on the first captured turn.
_SELECTABLE_BACKENDS = frozenset(get_args(get_type_hints(RolloutConfig)["backend"]))
if not _SELECTABLE_BACKENDS:
    raise RuntimeError("RolloutConfig.backend no longer declares a Literal roster; the reader table cannot be checked")
_UNREADABLE_BACKENDS = sorted(_SELECTABLE_BACKENDS - set(_TOKEN_CAPTURE))
if _UNREADABLE_BACKENDS:
    raise RuntimeError(
        f"rollout backend(s) {_UNREADABLE_BACKENDS} are selectable on RolloutConfig but have no "
        f"response reader in {__name__}: ship the reader with the backend's payload branch."
    )


def capture_generation_tokens(
    choice: dict[str, Any], data: dict[str, Any], backend: str
) -> tuple[list[int] | None, list[float] | None, list[int] | None]:
    """The turn's sampled ids, their logprobs and the engine's prompt ids, read in ``backend``'s shape."""
    return _TOKEN_CAPTURE[backend](choice, data)


def capture_routing_mask(choice: dict[str, Any], data: dict[str, Any]) -> str | None:
    """The turn's raw base64 MoE routing payload, wherever this response put it.

    vLLM attaches it per-choice; SGLang publishes it response-level under ``sglext``.
    """
    return choice.get("routed_experts") or (data.get("sglext") or {}).get("routed_experts")


def generation_control_fields(config: RolloutConfig, reasoning_effort: str | None = None) -> dict[str, Any]:
    """The request fields carrying a turn's generation contract: the reasoning level, the CoT budget
    bound to that level, and the turn terminator. Shared by both drivers (the training payload below
    and the eval runner's SDK call), so a knob reaches them alike.

    ``reasoning_effort`` is sent top-level only. Both engines derive their thinking toggles from that
    spelling (vLLM ``enable_thinking``, SGLang ``thinking`` + ``enable_thinking``) and not from the
    nested ``chat_template_kwargs`` form, which vLLM merges under the top-level field anyway. Sending
    both is ambiguous: vLLM resolves a disagreement to the top-level value, SGLang to the nested one.
    """
    fields: dict[str, Any] = {}
    if config.stop_token_ids:
        # Stop at the tool-call terminator, else the model hallucinates the tool result itself.
        fields["stop_token_ids"] = config.stop_token_ids
    if config.max_thinking_tokens is not None:
        if config.backend == "vllm":
            # vLLM forces the reasoning-end marker; needs a server-side reasoning parser.
            fields["thinking_token_budget"] = config.max_thinking_tokens
        else:
            # A budget reaching here came from the env's per-effort profile (a global one is rejected
            # at config time); it still reaches the reward side, so dropping it is not a full no-op.
            warn_once(
                logger,
                _THINKING_BUDGET_UNENFORCED_WARNED,
                config.backend,
                "Per-effort thinking budget of %d tokens is NOT enforced on rollout_backend=%r "
                "(thinking_token_budget is a vLLM-only request field): nothing caps reasoning "
                "below max_tokens=%d. The budget still reaches the trajectory, so a run with "
                "reasoning_compliance_weight > 0 prices CoT against it as a soft target; with "
                "that weight at its default 0 the level steers only the chat template.",
                config.max_thinking_tokens,
                config.backend,
                config.max_tokens,
            )
    if reasoning_effort is not None:
        fields["reasoning_effort"] = reasoning_effort
    return fields


def build_payload(
    messages: list[dict],
    config: RolloutConfig,
    reasoning_effort: str | None = None,
    tools: list[dict] | None = None,
) -> dict[str, Any]:
    """Build the OpenAI chat-completions request payload. ``reasoning_effort`` is the per-episode
    resolved level; ``config.max_thinking_tokens`` carries that level's budget when the env binds one."""
    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
    }
    # SGLang ignores unknown request keys rather than rejecting them, so vLLM's spelling of the
    # capture flags would be a no-op there; gate them on the engine.
    is_vllm = config.backend == "vllm"
    if config.capture_token_ids:
        # Sampled ids are recovered from the logprobs; top_logprobs=0 keeps one entry per token.
        payload["logprobs"] = True
        payload["top_logprobs"] = 0
        if is_vllm:
            # The server's own render is authoritative; a client-side re-render can differ.
            payload["return_token_ids"] = True
        else:
            # SGLang's OpenAI `logprobs` drops the token id; the meta_info it echoes still
            # carries it, and the prompt ids come back on the choice (non-streaming only).
            payload["return_meta_info"] = True
            payload["return_prompt_token_ids"] = True
    if config.capture_routed_experts and not is_vllm:
        # SGLang additionally needs a per-request opt-in. start_len stays 0 so rows cover the full
        # sequence, which is the convention assemble_rollout_masks replays prompt spans from.
        payload["return_routed_experts"] = True
    if config.model_name:
        payload["model"] = config.model_name
    if tools:
        payload["tools"] = tools
    payload.update(generation_control_fields(config, reasoning_effort))
    return payload
