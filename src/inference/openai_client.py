"""The OpenAI-compatible chat client: the async client factory, one request with the retry an
aggregator needs, and the single-request helper that normalizes a reply.

Targets any OpenAI-compatible endpoint — a locally served vLLM/SGLang rollout server or a hosted
aggregator — not just OpenAI. Endpoint defaults and key resolution live in
:mod:`src.inference.endpoints`; the parallel, resumable path in :mod:`src.inference.batch_requests`.
"""

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any

from openai import AsyncOpenAI

from src.env import env_str
from src.inference.endpoints import DEFAULT_LOCAL_BASE_URL
from src.inference.response import OpenAIResponse, get_finish_reason, get_reasoning_text

# Client-error statuses that report a transient server condition, not a bad request: retried here and
# by the rollout drivers (:func:`src.environments.episode.is_terminal_client_status`).
TRANSIENT_CLIENT_ERROR_CODES = frozenset({408, 429})
# An aggregator can answer 200 with no choices and an error body the SDK does not retry (OpenRouter
# spells an upstream rate limit this way); these codes are retried here with this backoff.
RETRYABLE_UPSTREAM_CODES = TRANSIENT_CLIENT_ERROR_CODES | {500, 502, 503, 504}
UPSTREAM_RETRIES = 4
UPSTREAM_BACKOFF_SECONDS = 2.0
# The SDK's own transport retries (connection errors, 408/409/429, 5xx), enough for a long eval to
# survive transient errors under load.
SDK_MAX_RETRIES = 4

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class EmptyChoicesError(RuntimeError):
    """A chat completion that carried no choices: the error body it carried instead, if any."""

    def __init__(self, model: str, error: Any):
        super().__init__(f"chat completion for {model!r} carried no choices: {error!r}")
        self.error = error


def create_openai_client(base_url: str | None = None, api_key_override: str | None = None) -> AsyncOpenAI:
    """Create an ``AsyncOpenAI`` client with optional base URL and API key override.

    An omitted ``base_url`` targets :data:`DEFAULT_LOCAL_BASE_URL`, never the public OpenAI API. No key (an
    empty override and an unset or blank ``OPENAI_API_KEY``) raises ``ValueError``: the SDK re-reads a
    blank variable itself, and the empty key it builds with fails only later, as an opaque 401
    (``.env.example`` ships the variable blank).
    """
    api_key = api_key_override or env_str("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("no API key for the OpenAI-compatible client: pass one, or set OPENAI_API_KEY")
    return AsyncOpenAI(base_url=base_url or DEFAULT_LOCAL_BASE_URL, api_key=api_key, max_retries=SDK_MAX_RETRIES)


async def chat_completion(client: AsyncOpenAI, **request: Any):
    """``client.chat.completions.create(**request)``, with a reply that carries no choices retried
    on a retryable upstream code (:data:`RETRYABLE_UPSTREAM_CODES`, exponential backoff) and raised as
    :class:`EmptyChoicesError` otherwise, so no caller indexes an empty ``choices``."""
    for attempt in range(UPSTREAM_RETRIES + 1):
        completion = await client.chat.completions.create(**request)
        if completion.choices:
            return completion
        error = getattr(completion, "error", None)
        code = error.get("code") if isinstance(error, Mapping) else None
        # An aggregator may spell the code as a string.
        code = int(code) if isinstance(code, int | str) and str(code).isdigit() else None
        if code not in RETRYABLE_UPSTREAM_CODES or attempt == UPSTREAM_RETRIES:
            raise EmptyChoicesError(request.get("model", "?"), error)
        await asyncio.sleep(UPSTREAM_BACKOFF_SECONDS * 2**attempt)
    raise AssertionError("unreachable")


def json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """The ``response_format`` request field asking for a reply under a strict JSON schema."""
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


async def generate_openai_response(
    model: str,
    messages: str | list[dict],
    *,
    client: AsyncOpenAI,
    temperature: float | None = 0.0,
    max_tokens: int = 512,
    tools: list[dict] | None = None,
    request_timeout: float = 180,
    extra_body: dict[str, Any] | None = None,
    top_p: float | None = None,
) -> OpenAIResponse:
    """One chat completion from an OpenAI-compatible API, as an :class:`OpenAIResponse`.

    ``messages`` is a message list or one user message. ``temperature`` ``None`` keeps the served
    default, which reasoning models require. ``extra_body`` rides the request body verbatim, for
    fields outside the OpenAI chat schema — the rollout engines' generation contract comes from
    :func:`~src.environments.engine_wire.generation_control_fields`, the one owner of those
    spellings, and OpenRouter's ``reasoning`` field from its caller. ``top_p`` is sent only when set.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "timeout": request_timeout,
    }
    if temperature is not None:
        request["temperature"] = temperature
    if top_p is not None:
        request["top_p"] = top_p
    if tools is not None:
        request["tools"] = tools
    if extra_body:
        request["extra_body"] = extra_body

    completion = await chat_completion(client, **request)
    choice = completion.choices[0]
    message = choice.message
    # Outside the OpenAI schema, so the SDK keeps it as an extra attribute of the choice.
    token_ids = getattr(choice, "token_ids", None)
    return OpenAIResponse(
        answer=message.content,
        reasoning=get_reasoning_text(message),
        finish_reason=get_finish_reason(choice) or "",
        tool_calls=message.tool_calls,
        prompt_tokens=_usage_field(completion, "prompt_tokens"),
        completion_tokens=_usage_field(completion, "completion_tokens"),
        total_tokens=_usage_field(completion, "total_tokens"),
        token_ids=token_ids if isinstance(token_ids, list) else None,
    )


def _usage_field(completion, field: str) -> int:
    """A usage field from a completion response; 0 when the endpoint reported no usage."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return 0
    return getattr(usage, field, 0) or 0


def parse_json_object(content: str) -> Any | None:
    """The JSON value ``content`` is, or the first JSON object it embeds in prose; ``None`` when
    neither parses."""
    match = _JSON_OBJECT.search(content)
    for candidate in (content, match.group(0) if match else None):
        if candidate is None:
            continue
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None
