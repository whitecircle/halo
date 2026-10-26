"""OpenAI chat-completions wire format for the rollout engines (vLLM and SGLang): a turn's request fields and the
turn a reply reports, the captures included, which the two engines request and return in different places.
Transport belongs to the rollout drivers."""

import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import replace
from typing import Any

from src.environments.episode import TurnGeneration, describe_exception
from src.environments.rollout_config import (
    REASONING_BUDGET_SOFT_TEMPLATE_VAR,
    REASONING_BUDGET_TEMPLATE_VAR,
    SGLANG_BACKEND,
    VLLM_BACKEND,
    RolloutConfig,
)
from src.inference.response import FINISH_REASON_LENGTH, get_finish_reason, get_reasoning_text
from src.log import warn_once

logger = logging.getLogger(__name__)

# Model types vLLM serves with its mistral_common tokenizer, whose encode() turns the budget's "[THINK]"
# and "[/THINK]" markers into plain-text pieces while the model writes them as control tokens.
_VLLM_UNMATCHED_REASONING_MARKER_MODEL_TYPES = frozenset({"mistral4"})

# Backends already warned (once per process) that a per-effort thinking budget reaches no engine field, that
# their completions carry no ``usage.completion_tokens``, and that a cut call's text could not be decoded.
_thinking_budget_unenforced_warned: set[str] = set()
_completion_tokens_missing_warned: set[str] = set()
_cut_call_decode_failed_warned: set[str] = set()


def _logprob_entries(choice: dict[str, Any]) -> list[dict[str, Any]] | None:
    """A choice's per-token ``logprobs.content`` entries, or None when the server returned none; ``[]`` is a
    zero-token completion whose capture succeeded."""
    content = (choice.get("logprobs") or {}).get("content")
    return content if isinstance(content, list) else None


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
    return ids


def _extract_token_logprobs(choice: dict[str, Any]) -> list[float] | None:
    """Recover the sampled tokens' behavior-policy log-probs (the IS correction's input) from a choice's
    ``logprobs.content``. Returns None if logprobs are absent or any entry lacks a numeric ``logprob``."""
    content = _logprob_entries(choice)
    if content is None:
        return None
    lps: list[float] = []
    for entry in content:
        lp = entry.get("logprob")
        if not isinstance(lp, (int, float)):
            return None
        lps.append(float(lp))
    return lps


def _sglang_meta_triples(choice: dict[str, Any]) -> list[list[Any]] | None:
    """A choice's ``meta_info.output_token_logprobs`` (``[logprob, token_id, token_text]`` per sampled token), or
    None when absent or malformed: SGLang's OpenAI ``logprobs`` drops the id, the ``return_meta_info`` echo keeps it."""
    triples = (choice.get("meta_info") or {}).get("output_token_logprobs")
    if not isinstance(triples, list):
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
    return ids, logprobs, choice.get("prompt_token_ids")


# Per-backend response readers: the engines return the same three facts in different places.
_TOKEN_CAPTURE = {VLLM_BACKEND: _capture_vllm, SGLANG_BACKEND: _capture_sglang}


def capture_generation_tokens(
    choice: dict[str, Any], data: dict[str, Any], backend: str
) -> tuple[list[int] | None, list[float] | None, list[int] | None]:
    """The turn's sampled ids, their logprobs and the engine's prompt ids, read in ``backend``'s shape."""
    return _TOKEN_CAPTURE[backend](choice, data)


def turn_from_chat_completion(data: dict[str, Any], config: RolloutConfig) -> TurnGeneration:
    """One turn as a chat-completions response (a raw JSON body, or an SDK reply's ``model_dump()``) reports it:
    text, tool calls, reasoning, the sampled-token count and why it stopped, plus the captures ``config``
    requested (sampled ids, their logprobs, the engine's prompt ids, the MoE routing)."""
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    msg = choice["message"]
    text = msg.get("content") or ""
    if config.capture_token_ids:
        token_ids, token_logprobs, prompt_token_ids = capture_generation_tokens(choice, data, config.backend)
    else:
        token_ids = token_logprobs = prompt_token_ids = None
    tokens = usage.get("completion_tokens") or 0
    if not tokens and text:
        # The captured ids are the only honest length; a word count of the text is not one.
        tokens = len(token_ids) if token_ids else 0
        warn_once(
            logger,
            _completion_tokens_missing_warned,
            config.backend,
            "%s returned a completion without usage.completion_tokens; generation-token metrics "
            "count the captured token ids instead (0 when none are captured).",
            config.backend,
        )
    routing_mask = None
    if config.capture_routed_experts:
        # The raw base64 MoE routing. vLLM attaches it per choice. SGLang leaves it in the ``meta_info`` it echoes
        # per choice when the request asks for that (every request capturing sampled ids does), and publishes it
        # response-level under ``sglext`` otherwise.
        routing_mask = (
            choice.get("routed_experts")
            or (choice.get("meta_info") or {}).get("routed_experts")
            or (data.get("sglext") or {}).get("routed_experts")
        )
    return TurnGeneration(
        text=text,
        tool_calls=msg.get("tool_calls") or [],
        reasoning=get_reasoning_text(msg) or "",
        tokens=tokens,
        token_ids=token_ids,
        token_logprobs=token_logprobs,
        routing_mask=routing_mask,
        # The engine's prompt length anchors the mask; the trainer's re-render can differ by a token.
        routing_prompt_tokens=usage.get("prompt_tokens") if routing_mask else None,
        prompt_token_ids=prompt_token_ids,
        finish_reason=get_finish_reason(
            choice, completion_tokens=usage.get("completion_tokens"), max_tokens=config.max_tokens
        ),
    )


async def with_cut_call_as_written(
    gen: TurnGeneration,
    config: RolloutConfig,
    detokenize: Callable[[list[int]], Awaitable[str]],
    log_prefix: str,
) -> TurnGeneration:
    """``gen`` with the call its token cap cut as the policy wrote it: the ids past the reasoning close, decoded by
    the driver's ``detokenize``, replace the arguments vLLM's parser salvaged. The salvage stands where no
    single-token close splits the ids or the decode fails."""
    ids, end = gen.token_ids, config.reasoning_end_token_id
    cut_in_call = gen.finish_reason == FINISH_REASON_LENGTH and gen.tool_calls and ids
    if not cut_in_call or config.backend != VLLM_BACKEND or end is None or end not in ids:
        return gen
    try:
        written = await detokenize(ids[len(ids) - ids[::-1].index(end) :])
    except Exception as e:  # what a judge reads of a cut call never fails the episode
        warn_once(
            logger,
            _cut_call_decode_failed_warned,
            config.backend,
            "%s: decoding a cut call's text failed (%s); the judge reads the call as the parser salvaged it",
            log_prefix,
            describe_exception(e),
        )
        return gen
    first = gen.tool_calls[0]
    return replace(gen, tool_calls=[{**first, "function": {**first.get("function", {}), "arguments": written}}])


def _sampler_fields(config: RolloutConfig) -> dict[str, Any]:
    # Sent at every value, off included: both engines fill an omitted filter from the model's
    # generation_config.json, and the eval's OpenAI SDK call has no keyword for any of them.
    fields: dict[str, Any] = {
        "top_k": config.top_k,
        "min_p": config.min_p,
        "repetition_penalty": config.repetition_penalty,
    }
    if config.stop_token_ids:
        # Stop at the tool-call terminator, else the model hallucinates the tool result itself.
        fields["stop_token_ids"] = config.stop_token_ids
    return fields


def _thinking_cap_fields(config: RolloutConfig) -> dict[str, Any]:
    if config.max_thinking_tokens is None:
        return {}
    if config.backend == VLLM_BACKEND:
        # vLLM forces the reasoning-end marker; needs a server-side reasoning parser.
        return {"thinking_token_budget": config.max_thinking_tokens}
    # A cap reaching here came from the env's profile — a level's budget or a soft level's ceiling (a
    # global one is rejected at config time); the budget still reaches the reward side.
    warn_once(
        logger,
        _thinking_budget_unenforced_warned,
        config.backend,
        "Per-turn thinking cap of %d tokens is NOT enforced on rollout_backend=%r "
        "(thinking_token_budget is a vLLM-only request field): nothing caps reasoning "
        "below max_tokens=%d. The level still reaches the chat template and the reasoning "
        "terms (reasoning_price, reasoning_floor), which price reasoning per level "
        "without the engine's help.",
        config.max_thinking_tokens,
        config.backend,
        config.max_tokens,
    )
    return {}


def _template_fields(
    config: RolloutConfig, reasoning_effort: str | None, reasoning_budget: int | None, budget_soft: bool
) -> dict[str, Any]:
    """The level top-level, where both engines derive their thinking toggles from it, and the run's template
    variables nested, joined by the level's budget under :data:`REASONING_BUDGET_TEMPLATE_VAR`."""
    fields: dict[str, Any] = {}
    template_kwargs = dict(config.chat_template_kwargs or {})
    if reasoning_budget is not None:
        # Episode-constant: the trainer renders every turn with the one value the trajectory carries, never the
        # narrower cap an output budget leaves a late turn.
        template_kwargs[REASONING_BUDGET_TEMPLATE_VAR] = reasoning_budget
        if budget_soft:
            template_kwargs[REASONING_BUDGET_SOFT_TEMPLATE_VAR] = True
    if reasoning_effort is not None:
        fields["reasoning_effort"] = reasoning_effort
        if config.backend != VLLM_BACKEND:
            # SGLang pops a nested copy into the top-level field before rendering, so a disagreement
            # would resolve to the nested value; the exact copy keeps one value either way.
            template_kwargs["reasoning_effort"] = reasoning_effort
    if template_kwargs:
        fields["chat_template_kwargs"] = template_kwargs
    return fields


def generation_control_fields(
    config: RolloutConfig,
    reasoning_effort: str | None = None,
    reasoning_budget: int | None = None,
    budget_soft: bool = False,
) -> dict[str, Any]:
    """The request fields carrying a turn's generation contract: the sampler filters, the reasoning cap the engine
    enforces this turn, the level, and the template variables with the budget the template states; one owner
    for the training payload and the eval's SDK call alike."""
    return {
        **_sampler_fields(config),
        **_thinking_cap_fields(config),
        **_template_fields(config, reasoning_effort, reasoning_budget, budget_soft),
    }


def capture_request_fields(config: RolloutConfig) -> dict[str, Any]:
    """The request fields asking the engine for the captures ``config`` requests, in its own spelling: SGLang
    ignores unknown keys rather than rejecting them, so vLLM's would be a no-op there."""
    fields: dict[str, Any] = {}
    is_vllm = config.backend == VLLM_BACKEND
    if config.capture_token_ids:
        # Sampled ids are recovered from the logprobs; top_logprobs=0 keeps one entry per token.
        fields["logprobs"] = True
        fields["top_logprobs"] = 0
        if is_vllm:
            # The server's own render is authoritative; a client-side re-render can differ.
            fields["return_token_ids"] = True
        else:
            # SGLang's OpenAI `logprobs` drops the token id; the meta_info it echoes still
            # carries it, and the prompt ids come back on the choice (non-streaming only).
            fields["return_meta_info"] = True
            fields["return_prompt_token_ids"] = True
    if config.capture_routed_experts and not is_vllm:
        # SGLang additionally needs a per-request opt-in. start_len stays 0 so rows cover the full
        # sequence, which is the convention assemble_rollout_masks replays prompt spans from.
        fields["return_routed_experts"] = True
    return fields


def build_payload(
    messages: list[dict],
    config: RolloutConfig,
    reasoning_effort: str | None = None,
    tools: list[dict] | None = None,
    reasoning_budget: int | None = None,
    budget_soft: bool = False,
) -> dict[str, Any]:
    """Build the OpenAI chat-completions request payload. ``reasoning_effort`` is the per-episode
    resolved level; ``config.max_thinking_tokens`` carries the turn's engine cap and ``reasoning_budget``
    the level's budget the template states, soft or not (see :func:`generation_control_fields`)."""
    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        **capture_request_fields(config),
    }
    if config.model_name:
        payload["model"] = config.model_name
    if tools:
        payload["tools"] = tools
    payload.update(generation_control_fields(config, reasoning_effort, reasoning_budget, budget_soft))
    return payload


def reject_unenforceable_thinking_budget(model_types: Iterable[str], backend: str) -> None:
    """Refuse a thinking budget vLLM accepts but can never enforce for these ``model_types``."""
    unmatched = sorted(_VLLM_UNMATCHED_REASONING_MARKER_MODEL_TYPES.intersection(model_types))
    if backend == VLLM_BACKEND and unmatched:
        raise ValueError(
            f"vLLM cannot enforce a thinking budget for model_type {unmatched}: it serves this model with its "
            "mistral_common tokenizer, which encodes the [THINK] and [/THINK] markers as plain text while the "
            "model writes them as control tokens, so the budget never starts counting and reasoning runs "
            "uncapped. Unset rollout_max_thinking_tokens and the thinking_tokens of every effort level the "
            "environment draws, and steer with reasoning_effort instead."
        )
