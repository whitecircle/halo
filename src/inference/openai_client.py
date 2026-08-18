"""OpenAI-compatible client surface: endpoint/key defaults, the async client factory and the
single/parallel request helpers.

Targets any OpenAI-compatible endpoint — a locally served vLLM/SGLang rollout server or a hosted
aggregator — not just OpenAI. The parallel path resumes through
:mod:`src.inference.resume_store`.
"""

import asyncio
import hashlib
import json
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tqdm.asyncio import tqdm as async_tqdm

from src.env import env_str
from src.inference.response import OpenAIResponse, get_finish_reason, get_reasoning_text
from src.inference.resume_store import append_openai_checkpoint, load_openai_checkpoint

logger = logging.getLogger(__name__)

# Endpoint every OpenAI-compatible caller targets when its URL flag is omitted. One spelling, because
# the alternative default — the public OpenAI API — ships user conversations off-site on a
# forgotten flag. Lives here, beside the client plumbing, so ``src`` never imports it from ``scripts``.
DEFAULT_LOCAL_BASE_URL = "http://localhost:8000/v1"

# The hosted aggregator every off-site caller targets when a hosted model id is requested.
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The key a keyless local rollout server (vLLM / SGLang without --api-key) accepts. The SDK refuses to
# construct a client without SOME key, so a None default would crash the documented local invocation.
_LOCAL_SERVER_API_KEY = "EMPTY"


@dataclass(slots=True)
class RequestTools:
    raw: list[dict[str, object]] | list[list[dict[str, object]] | None] | None
    per_message: bool

    def for_message(self, index: int) -> list[dict[str, object]] | None:
        if self.raw is None:
            return None
        if self.per_message:
            return self.raw[index]
        return self.raw

    def hash_sample(self) -> str | None:
        """Identity of the WHOLE tool declaration set for the resume-checkpoint key.

        A key built from a prefix of the declarations lets a run with unchanged prompts but edited
        tool schemas resolve to another run's checkpoint and replay its completions.
        """
        if self.raw is None:
            return None
        return _sequence_digest(self.raw)


def resolve_local_api_key() -> str:
    """Key for the local rollout server: ``VLLM_API_KEY`` → ``OPENAI_API_KEY`` → placeholder.

    The CLI default for every ``--openai_api_key`` flag, so an explicit key arrives as the flag's
    value and never through here. ``VLLM_API_KEY`` first because it is the server-side ``--api-key``
    convention.
    """
    return env_str("VLLM_API_KEY") or env_str("OPENAI_API_KEY") or _LOCAL_SERVER_API_KEY


def resolve_external_api_key(explicit: str | None = None) -> str | None:
    """Key for a hosted (non-local) endpoint: explicit → ``OPENROUTER_API_KEY`` → ``OPENAI_API_KEY``.

    ``None`` when nothing is set — each caller decides whether that is fatal.
    """
    explicit = explicit.strip() if explicit else None
    return explicit or env_str("OPENROUTER_API_KEY") or env_str("OPENAI_API_KEY") or None


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


async def generate_openai_response(
    model: str,
    user_message: str | list[dict],
    response_format: type[BaseModel] | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    use_native_json_schema: bool = True,
    custom_client: AsyncOpenAI | None = None,
    tools: list[dict] | None = None,
    request_timeout: float = 180,
    extra_body: dict[str, Any] | None = None,
    top_p: float | None = None,
) -> OpenAIResponse:
    """Generate a response from an OpenAI-compatible API.

    ``extra_body`` rides the request body verbatim, for fields outside the OpenAI chat schema — the
    rollout engines' generation contract comes from
    :func:`~src.environments.engine_wire.generation_control_fields`, the one owner of those spellings.
    ``top_p`` is sent only when set, so the served default stands otherwise. Without
    ``use_native_json_schema`` the schema is injected as prompt text.
    """
    # Shallow-copy so appending a system prompt never mutates the caller's message list.
    messages = list(user_message) if isinstance(user_message, list) else [{"role": "user", "content": user_message}]

    if response_format and not use_native_json_schema:
        if system_prompt is not None:
            system_prompt += f"\n\nAnswer using only following JSON schema:\n{response_format.model_json_schema()}"
        elif messages and isinstance(messages[0].get("content"), str):
            # New dict — the shallow copy shares the caller's dicts, so in-place += would corrupt them.
            new_first = dict(messages[0])
            new_first["content"] += (
                f"\n\nAnswer using only following JSON schema:\n{response_format.model_json_schema()}"
            )
            messages[0] = new_first

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    if response_format and use_native_json_schema:
        response_format_arg = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
            },
        }
    else:
        response_format_arg = None

    if custom_client is None:
        raise ValueError(
            "custom_client is required: build one with create_openai_client(base_url=..., "
            "api_key_override=...). There is no implicit default — an endpoint and credentials "
            "guessed from the environment fail as a 401 mid-run instead of as a config error here."
        )

    api_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": request_timeout,
    }

    if top_p is not None:
        api_kwargs["top_p"] = top_p

    if response_format_arg is not None:
        api_kwargs["response_format"] = response_format_arg

    if tools is not None:
        api_kwargs["tools"] = tools

    if extra_body:
        api_kwargs["extra_body"] = extra_body

    try:
        completion = await custom_client.chat.completions.create(**api_kwargs)
    except TimeoutError:
        logger.error("Timeout error for model %s", model)
        raise
    except Exception as e:
        logger.error("Error calling OpenAI API: %s: %s", type(e).__name__, e)
        raise

    message = completion.choices[0].message
    finish_reason = get_finish_reason(completion.choices[0]) or ""
    tool_calls = message.tool_calls
    reasoning = get_reasoning_text(message)

    prompt_tokens = _get_usage_field(completion, "prompt_tokens")
    completion_tokens = _get_usage_field(completion, "completion_tokens")
    total_tokens = _get_usage_field(completion, "total_tokens")

    def _resp(answer):
        return OpenAIResponse(
            answer=answer,
            reasoning=reasoning,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    if response_format is not None:
        # content is None on a pure tool-call turn; normalize so parsing fails into the ValueError below.
        content = message.content or ""
        if use_native_json_schema:
            parsed = _parse_structured(response_format, content)
            if parsed is not None:
                return _resp(parsed)

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            parsed = _parse_structured(response_format, json_match.group(0))
            if parsed is not None:
                return _resp(parsed)
        raise ValueError(f"Response does not contain valid JSON: {message.content}")
    else:
        return _resp(message.content)


async def parallel_openai_requests(
    model: str,
    user_messages: list[str] | list[list[dict]],
    response_format: type[BaseModel] | None = None,
    use_native_json_schema: bool = True,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_workers: int = 8,
    checkpoint_file: str | None = None,
    checkpoint_interval: int = 10,
    disable_checkpoints: bool = False,
    custom_client: AsyncOpenAI | None = None,
    tools: list[dict] | list[list[dict] | None] | None = None,
    request_timeout: float = 180,
) -> list[OpenAIResponse]:
    """Process multiple OpenAI requests in parallel with incremental checkpointing (completed results
    appended so a re-run resumes; failed requests are not checkpointed so they retry).

    ``tools`` is either a single list applied to every message, or one list per message.
    ``request_timeout`` bounds each individual request, exactly as in :func:`generate_openai_response`.
    """
    request_tools = resolve_request_tools(tools, len(user_messages))

    results = [None] * len(user_messages)
    semaphore = asyncio.Semaphore(max_workers)

    async def process_message_wrapper(idx, message):
        try:
            async with semaphore:
                response = await generate_openai_response(
                    model,
                    message,
                    response_format,
                    system_prompt,
                    temperature,
                    max_tokens,
                    use_native_json_schema,
                    custom_client,
                    request_tools.for_message(idx),
                    request_timeout=request_timeout,
                )
                return idx, response
        except Exception:
            # Non-fatal so one bad row cannot end the batch, but never silent: the row is left
            # un-checkpointed, comes back as None, and its cause is logged with the traceback.
            logger.warning("Request %d failed; leaving it unprocessed for a re-run", idx, exc_info=True)
            return idx, None

    processed_indices = set()
    # Declared unconditionally: the completion loop appends to them under the same
    # ``disable_checkpoints`` gate, and a branch-local definition is one edit from a NameError.
    new_results_buffer = []
    buffer_lock = asyncio.Lock()
    if not disable_checkpoints:
        file_lock = asyncio.Lock()
        checkpoint_file = resolve_checkpoint_file(
            checkpoint_file,
            model=model,
            user_messages=user_messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            request_tools=request_tools,
            response_format=response_format,
        )

        logger.info("Checkpoint file path: %s", checkpoint_file)

        checkpoint = load_openai_checkpoint(
            checkpoint_file,
            result_count=len(results),
            response_format=response_format,
        )
        results = checkpoint.results
        processed_indices = checkpoint.processed_indices
        if checkpoint.skipped_records:
            logger.warning("Skipped %d invalid checkpoint records", checkpoint.skipped_records)
        logger.info("Loaded %d results from %s", len(processed_indices), checkpoint_file)

        async def save_checkpoint_incremental():
            """Append new results since the last checkpoint."""
            try:
                async with buffer_lock:
                    if not new_results_buffer:
                        return

                    results_to_save = [(idx, result) for idx, result in new_results_buffer if result is not None]
                    new_results_buffer.clear()

                async with file_lock:
                    append_openai_checkpoint(checkpoint_file, results_to_save)
            except Exception:
                # A lost append means the completed rows re-run on resume — costly, not corrupting.
                logger.error("Error saving incremental results", exc_info=True)

    indices_to_process = [i for i in range(len(user_messages)) if i not in processed_indices]

    if not indices_to_process:
        logger.info("All requests have already been processed.")
        return results

    tasks = [asyncio.create_task(process_message_wrapper(idx, user_messages[idx])) for idx in indices_to_process]

    completed_count = 0

    progress_bar = async_tqdm(total=len(tasks), desc="Processing requests")

    # process_message_wrapper never raises: a failure yields (idx, None), left un-checkpointed so a re-run retries it.
    for future in asyncio.as_completed(tasks):
        idx, result = await future
        results[idx] = result

        if not disable_checkpoints:
            async with buffer_lock:
                new_results_buffer.append((idx, result))
            processed_indices.add(idx)
            completed_count += 1
            if completed_count % checkpoint_interval == 0:
                await save_checkpoint_incremental()

        progress_bar.update(1)

    progress_bar.close()

    if not disable_checkpoints:
        await save_checkpoint_incremental()

    return results


def resolve_request_tools(
    tools: list[dict[str, object]] | list[list[dict[str, object]] | None] | None,
    message_count: int,
) -> RequestTools:
    per_message = _is_per_message_tools(tools)
    if per_message and tools is not None and len(tools) != message_count:
        raise ValueError(
            "If tools is per-message, it must have the same length as user_messages. "
            f"Got {len(tools)} tools and {message_count} messages."
        )
    return RequestTools(raw=tools, per_message=per_message)


def resolve_checkpoint_file(
    checkpoint_file: str | None,
    *,
    model: str,
    user_messages: list[str] | list[list[dict]],
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
    request_tools: RequestTools,
    response_format: type[BaseModel] | None,
) -> str:
    if checkpoint_file is not None:
        return checkpoint_file

    hash_content = {
        "model": model,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": request_tools.hash_sample(),
        "per_message_tools": request_tools.per_message,
        "response_format": response_format.__name__ if response_format else None,
        "messages_sample": _sequence_digest(user_messages),
    }
    task_hash = hashlib.md5(json.dumps(hash_content, sort_keys=True).encode()).hexdigest()
    return str(Path(tempfile.gettempdir()) / f"openai_requests_{task_hash}.jsonl")


def _get_usage_field(completion, field: str) -> int:
    """A usage field from a completion response; 0 when the endpoint reported no usage."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return 0
    return getattr(usage, field, 0) or 0


def _parse_structured(response_format: type[BaseModel], payload: str) -> BaseModel | None:
    """``payload`` as ``response_format``, or ``None`` when it is not that model's JSON.

    Narrow by type: a malformed or off-schema payload is the expected outcome the caller falls back
    around (native parse → regex extraction → raise), while anything else is a bug that must surface.
    """
    try:
        return response_format.model_validate_json(payload)
    except ValidationError:
        return None


def _is_per_message_tools(
    tools: list[dict[str, object]] | list[list[dict[str, object]] | None] | None,
) -> bool:
    if not tools:
        return False

    first = tools[0]
    if first is None or isinstance(first, list):
        return True

    if isinstance(first, dict) and "type" not in first:
        return not any(isinstance(tool, dict) and "type" in tool for tool in tools)

    return False


def _sequence_digest(items: list) -> str:
    """Stable identity of a WHOLE request sequence (prompt rows or tool definitions), for the
    resume-checkpoint key.

    Every element reaches the digest because callers pass the post-resume PENDING subset: two
    disjoint subsets of one dataset can differ only past a prefix a long shared system prompt fills,
    and a colliding filename replays the earlier run's results onto unrelated rows by index. Fed
    element by element (NUL-delimited) so a 100k-row dataset is never materialized twice.
    """
    digest = hashlib.sha256()
    for item in items:
        digest.update(repr(item).encode())
        digest.update(b"\0")
    return digest.hexdigest()
