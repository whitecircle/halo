"""OpenAI chat-completions wire format for the rollout engines: a turn's request fields and the turn a reply reports,
which vLLM and SGLang request and return in different places."""

import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

from halo.environments.episodes.candidate_logprobs import parse_sglang_candidates
from halo.environments.rollout_config import (
    SGLANG_BACKEND,
    VLLM_BACKEND,
    RolloutConfig,
)
from halo.environments.rollouts.episode_loop import TurnGeneration, describe_exception
from halo.inference.response import FINISH_REASON_LENGTH, get_finish_reason, get_reasoning_text
from halo.log import warn_once

logger = logging.getLogger(__name__)

_thinking_budget_unenforced_warned: set[str] = set()
_completion_tokens_missing_warned: set[str] = set()
_cut_call_decode_failed_warned: set[str] = set()


def _logprob_entries(choice: dict[str, Any]) -> list[dict[str, Any]] | None:
    """``None`` when the server returned none; ``[]`` is a zero-token completion whose capture succeeded."""
    content = (choice.get("logprobs") or {}).get("content")
    return content if isinstance(content, list) else None


def _extract_token_ids(choice: dict[str, Any]) -> list[int] | None:
    """With ``--return-tokens-as-token-ids`` each entry's ``token`` is ``"token_id:<N>"``. ``None`` if absent or
    malformed, so the caller falls back to re-tokenization."""
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
    """Behavior-policy log-probs, the IS correction's input. ``None`` if absent or any entry lacks a number."""
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
    """``[logprob, token_id, token_text]`` per token, or ``None``: SGLang's OpenAI ``logprobs`` drops the id, the
    ``return_meta_info`` echo keeps it."""
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


_TOKEN_CAPTURE = {VLLM_BACKEND: _capture_vllm, SGLANG_BACKEND: _capture_sglang}


def capture_generation_tokens(
    choice: dict[str, Any], data: dict[str, Any], backend: str
) -> tuple[list[int] | None, list[float] | None, list[int] | None]:
    return _TOKEN_CAPTURE[backend](choice, data)


def turn_from_chat_completion(data: dict[str, Any], config: RolloutConfig) -> TurnGeneration:
    """One turn as a chat-completions response reports it (a raw JSON body, or an SDK reply's ``model_dump()``), plus
    the captures ``config`` requested."""
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
    candidates = None
    if config.capture_candidate_logprobs and token_ids is not None:
        # SGLang only: a construction refusal keeps vLLM, which would return none here, off the knob.
        candidates = parse_sglang_candidates(
            choice.get("meta_info") or {}, len(token_ids), config.capture_candidate_logprobs
        )
    routing_mask = None
    if config.capture_routed_experts:
        # vLLM attaches it per choice; SGLang in the echoed ``meta_info`` when asked, else under ``sglext``.
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
        candidates=candidates,
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
    """``gen`` with the cut call as the policy wrote it: the decoded ids past the reasoning close replace the arguments
    vLLM's parser salvaged. The salvage stands where no single-token close splits the ids or the decode fails."""
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
    # The turn less its answer room (a global cap is rejected at config time); the profile budgets still reach the
    # reward side.
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


def _template_fields(config: RolloutConfig, reasoning_effort: str | None) -> dict[str, Any]:
    """The level goes top-level, where both engines derive their thinking toggles from it; the run's template
    variables go nested."""
    fields: dict[str, Any] = {}
    template_kwargs = dict(config.chat_template_kwargs or {})
    if reasoning_effort is not None:
        fields["reasoning_effort"] = reasoning_effort
        if config.backend != VLLM_BACKEND:
            # SGLang pops a nested copy into the top-level field, so a disagreement would resolve to the nested value.
            template_kwargs["reasoning_effort"] = reasoning_effort
    if template_kwargs:
        fields["chat_template_kwargs"] = template_kwargs
    return fields


def generation_control_fields(config: RolloutConfig, reasoning_effort: str | None = None) -> dict[str, Any]:
    """Sampler filters, the engine's reasoning cap, the level and the template variables; one owner for the training
    payload and the eval's SDK call alike."""
    return {
        **_sampler_fields(config),
        **_thinking_cap_fields(config),
        **_template_fields(config, reasoning_effort),
    }


def capture_request_fields(config: RolloutConfig) -> dict[str, Any]:
    """In the engine's own spelling: SGLang ignores unknown keys, so vLLM's would be a no-op there."""
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
            # SGLang's OpenAI `logprobs` drops the id; its meta_info keeps it, and prompt ids come back on the choice.
            fields["return_meta_info"] = True
            fields["return_prompt_token_ids"] = True
            if config.capture_candidate_logprobs:
                # The candidates ride flat base64 arrays in meta_info instead of one JSON object per token.
                fields["top_logprobs"] = config.capture_candidate_logprobs
                fields["return_flat_raw_output_top_logprobs"] = True
    if config.capture_routed_experts and not is_vllm:
        # A per-request opt-in. start_len stays 0 so rows cover the full sequence, as build_rollout_routing expects.
        fields["return_routed_experts"] = True
    return fields


def build_payload(
    messages: list[dict],
    config: RolloutConfig,
    reasoning_effort: str | None = None,
    tools: list[dict] | None = None,
) -> dict[str, Any]:
    """``config.max_thinking_tokens`` carries the turn's engine cap (:func:`generation_control_fields`)."""
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
    payload.update(generation_control_fields(config, reasoning_effort))
    return payload


def reject_unenforceable_thinking_budget(model_types: Iterable[str], unenforceable: Mapping[str, str]) -> None:
    """Refuse a thinking budget the rollout engine accepts but can never enforce for these ``model_types``."""
    unmatched = sorted(set(model_types) & unenforceable.keys())
    if unmatched:
        facts = "; ".join(unenforceable[model_type] for model_type in unmatched)
        raise ValueError(
            f"The rollout engine cannot enforce a thinking budget for model_type {unmatched}: {facts}, so the "
            "budget never starts counting and reasoning runs uncapped. Unset rollout_max_thinking_tokens and "
            "rollout_max_answer_tokens, and steer with reasoning_effort instead."
        )
