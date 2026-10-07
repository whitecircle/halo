"""The OpenAI-compatible chat client: the async client factory, one request with the retry an
aggregator needs, and the single-request helper that parses a structured reply.

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
from pydantic import BaseModel, ValidationError

from src.env import env_str
from src.inference.endpoints import DEFAULT_LOCAL_BASE_URL
from src.inference.response import OpenAIResponse, get_finish_reason, get_reasoning_text

# An aggregator can answer 200 with no choices and an error body the SDK does not retry (OpenRouter
# spells an upstream rate limit this way); these codes are retried here with this backoff.
RETRYABLE_UPSTREAM_CODES = (408, 429, 500, 502, 503, 504)
UPSTREAM_RETRIES = 4
UPSTREAM_BACKOFF_SECONDS = 2.0

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class EmptyChoicesError(RuntimeError):
    """A chat completion that carried no choices: the error body it carried instead, if any."""

    def __init__(self, model: str, error: Any):
        super().__init__(f"chat completion for {model!r} carried no choices: {error!r}")
        self.error = error


def create_openai_client(
    base_url: str | None = None, api_key_override: str | None = None, max_retries: int = 4
) -> AsyncOpenAI:
    """Create an ``AsyncOpenAI`` client with optional base URL and API key override.

    An omitted ``base_url`` targets :data:`DEFAULT_LOCAL_BASE_URL`, never the public OpenAI API.
    ``max_retries`` defaults to 4 so a long eval survives transient 429/5xx errors under load.
    """
    # `or None`, never "": the SDK raises its own named "api_key must be set" error on None, while an
    # empty string passes construction and fails later as an opaque 401 (`.env.example` ships it blank).
    client_api_key = api_key_override or env_str("OPENAI_API_KEY") or None

    return AsyncOpenAI(base_url=base_url or DEFAULT_LOCAL_BASE_URL, api_key=client_api_key, max_retries=max_retries)


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


def json_schema_response_format(name: str, schema: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """The ``response_format`` request field asking for a reply under a strict JSON schema."""
    return {"type": "json_schema", "json_schema": {"name": name, "strict": strict, "schema": schema}}


async def generate_openai_response(
    model: str,
    messages: str | list[dict],
    *,
    client: AsyncOpenAI,
    response_format: type[BaseModel] | None = None,
    system_prompt: str | None = None,
    temperature: float | None = 0.0,
    max_tokens: int = 512,
    use_native_json_schema: bool = True,
    tools: list[dict] | None = None,
    request_timeout: float = 180,
    extra_body: dict[str, Any] | None = None,
    top_p: float | None = None,
) -> OpenAIResponse:
    """One chat completion from an OpenAI-compatible API, as an :class:`OpenAIResponse`.

    ``messages`` is a message list or one user message. ``response_format`` asks for a structured
    reply: a pydantic model, whose JSON the reply is parsed into (``answer`` is the model instance).
    Without ``use_native_json_schema`` its schema is injected as prompt text instead of sent as a
    request field. ``temperature`` ``None`` keeps the
    served default, which reasoning models require. ``extra_body`` rides the request body verbatim,
    for fields outside the OpenAI chat schema — the rollout engines' generation contract comes from
    :func:`~src.environments.engine_wire.generation_control_fields`, the one owner of those
    spellings, and OpenRouter's ``reasoning`` field from its caller. ``top_p`` is sent only when set.
    """
    # Shallow-copy so appending a system prompt never mutates the caller's message list.
    messages = list(messages) if isinstance(messages, list) else [{"role": "user", "content": messages}]
    model_format = response_format

    if model_format is not None and not use_native_json_schema:
        instruction = f"\n\nAnswer using only following JSON schema:\n{model_format.model_json_schema()}"
        if system_prompt is not None:
            system_prompt += instruction
        elif messages and isinstance(messages[0].get("content"), str):
            # New dict — the shallow copy shares the caller's dicts, so in-place += would corrupt them.
            messages[0] = {**messages[0], "content": messages[0]["content"] + instruction}

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

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
    if model_format is not None and use_native_json_schema:
        request["response_format"] = json_schema_response_format(
            model_format.__name__, model_format.model_json_schema(), strict=False
        )
    if tools is not None:
        request["tools"] = tools
    if extra_body:
        request["extra_body"] = extra_body

    completion = await chat_completion(client, **request)
    choice = completion.choices[0]
    message = choice.message
    # Outside the OpenAI schema, so the SDK keeps it as an extra attribute of the choice.
    token_ids = getattr(choice, "token_ids", None)
    response = OpenAIResponse(
        answer=message.content,
        reasoning=get_reasoning_text(message),
        finish_reason=get_finish_reason(choice) or "",
        tool_calls=message.tool_calls,
        prompt_tokens=_usage_field(completion, "prompt_tokens"),
        completion_tokens=_usage_field(completion, "completion_tokens"),
        total_tokens=_usage_field(completion, "total_tokens"),
        token_ids=token_ids if isinstance(token_ids, list) else None,
    )
    if model_format is None:
        return response
    # content is None on a pure tool-call turn; normalize so parsing fails into the ValueError below.
    content = message.content or ""
    candidates = [content] if use_native_json_schema else []
    match = _JSON_OBJECT.search(content)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        parsed = _parse_structured(model_format, candidate)
        if parsed is not None:
            return response.model_copy(update={"answer": parsed})
    raise ValueError(f"Response does not contain valid JSON: {message.content}")


def _usage_field(completion, field: str) -> int:
    """A usage field from a completion response; 0 when the endpoint reported no usage."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return 0
    return getattr(usage, field, 0) or 0


def _parse_structured(response_format: type[BaseModel], payload: str) -> BaseModel | None:
    """``payload`` as ``response_format``, or ``None`` when it is not that model's JSON.

    Narrow by type: a malformed or off-schema payload is the expected outcome the caller falls back
    around (native parse → regex extraction → raise), while any other exception is a bug and
    propagates.
    """
    try:
        return response_format.model_validate_json(payload)
    except ValidationError:
        return None


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
