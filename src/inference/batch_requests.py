"""Many chat requests in parallel, resumable: completed rows land in a JSONL store named by the request
identity, so a re-run replays them instead of re-issuing them; a failed row is never written, so it retries."""

import asyncio
import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

from src.inference.openai_client import generate_openai_response
from src.inference.response import OpenAIResponse

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _RequestTools:
    """The tool declarations of a batch: one list shared by every message, or one list per message."""

    raw: list[dict[str, object]] | list[list[dict[str, object]] | None] | None
    per_message: bool

    def for_message(self, index: int) -> list[dict[str, object]] | None:
        if self.raw is None:
            return None
        if self.per_message:
            return self.raw[index]
        return self.raw


@dataclass(slots=True)
class _ResumeLoad:
    """A resume store read back: one slot per request (``None`` where nothing replays), the replayed
    indices and the count of lines skipped as unreadable or out of range."""

    results: list[OpenAIResponse | None]
    processed_indices: set[int]
    skipped_records: int = 0


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
    """Identity of a WHOLE request sequence (prompt rows or tool definitions) for the resume-store key.

    Callers pass the post-resume pending subset, and two subsets sharing a long system prompt agree on
    any prefix: a prefix key would replay one run's results onto the other's rows by index. Hashed
    element by element, so a large dataset is never serialized whole.
    """
    digest = hashlib.sha256()
    for item in items:
        digest.update(repr(item).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _optional_int(value: object) -> int:
    return value if type(value) is int else 0


def _response_from_record(result: dict[str, object]) -> OpenAIResponse:
    answer = result.get("answer")
    # A null answer is legitimate with tool_calls; only neither-present is genuinely corrupt.
    if answer is None and not result.get("tool_calls"):
        raise ValueError("resume record is missing answer")

    token_ids = result.get("token_ids")
    if token_ids is not None and not (isinstance(token_ids, list) and all(type(t) is int for t in token_ids)):
        raise TypeError("resume record token_ids must be a list of integers")

    finish_reason = result.get("finish_reason")
    reasoning = result.get("reasoning")
    return OpenAIResponse(
        answer=answer,
        reasoning=reasoning if isinstance(reasoning, str) else None,
        finish_reason=finish_reason if isinstance(finish_reason, str) else "",
        tool_calls=result.get("tool_calls"),
        prompt_tokens=_optional_int(result.get("prompt_tokens")),
        completion_tokens=_optional_int(result.get("completion_tokens")),
        total_tokens=_optional_int(result.get("total_tokens")),
        token_ids=token_ids,
    )


def _parse_resume_line(line: str) -> tuple[int, OpenAIResponse]:
    data = json.loads(line)
    if not isinstance(data, dict):
        raise TypeError("resume record must be an object")

    index = data.get("index")
    if type(index) is not int:
        raise TypeError("resume record index must be an integer")

    result = data.get("result")
    if result is None:
        raise ValueError("empty resume record result")
    if not isinstance(result, dict):
        raise TypeError("resume record result must be an object")

    return index, _response_from_record(result)


def resolve_request_tools(
    tools: list[dict[str, object]] | list[list[dict[str, object]] | None] | None,
    message_count: int,
) -> _RequestTools:
    per_message = _is_per_message_tools(tools)
    if per_message and tools is not None and len(tools) != message_count:
        raise ValueError(
            "If tools is per-message, it must have the same length as messages. "
            f"Got {len(tools)} tools and {message_count} messages."
        )
    return _RequestTools(raw=tools, per_message=per_message)


def resume_store_path(
    *,
    model: str,
    messages: list[str] | list[list[dict]],
    temperature: float | None,
    max_tokens: int,
    request_tools: _RequestTools,
) -> str:
    """The resume-store path under the temp dir, keyed by the request identity."""
    hash_content = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": None if request_tools.raw is None else _sequence_digest(request_tools.raw),
        "per_message_tools": request_tools.per_message,
        "messages_sample": _sequence_digest(messages),
    }
    task_hash = hashlib.md5(json.dumps(hash_content, sort_keys=True).encode()).hexdigest()
    return str(Path(tempfile.gettempdir()) / f"openai_requests_{task_hash}.jsonl")


def load_resume_records(store_path: str, *, result_count: int) -> _ResumeLoad:
    results: list[OpenAIResponse | None] = [None] * result_count
    processed_indices: set[int] = set()
    skipped_records = 0
    path = Path(store_path)

    if not path.exists():
        return _ResumeLoad(results=results, processed_indices=processed_indices)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                index, response = _parse_resume_line(line)
            except (TypeError, ValueError):
                skipped_records += 1
                continue

            if index < 0 or index >= result_count:
                skipped_records += 1
                continue

            results[index] = response
            processed_indices.add(index)

    return _ResumeLoad(
        results=results,
        processed_indices=processed_indices,
        skipped_records=skipped_records,
    )


def append_resume_records(
    store_path: str,
    records: list[tuple[int, OpenAIResponse]],
) -> None:
    if not records:
        return

    path = Path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for index, response in records:
            file.write(
                json.dumps(
                    {"index": index, "result": response.model_dump()},
                    ensure_ascii=False,
                )
                + "\n"
            )


async def parallel_openai_requests(
    model: str,
    messages: list[str] | list[list[dict]],
    *,
    client: AsyncOpenAI,
    flush_interval: int,
    temperature: float | None = 0.0,
    max_tokens: int = 512,
    max_workers: int = 8,
    resume: bool = True,
    tools: list[dict] | list[list[dict] | None] | None = None,
) -> list[OpenAIResponse | None]:
    """One :func:`generate_openai_response` per entry of ``messages``, ``max_workers`` in flight. With
    ``resume``, completed results are appended to the resume store every ``flush_interval`` completions
    so a re-run resumes, and a failed request is left ``None`` and unstored, so it retries on the next run;
    without it nothing is stored and ``flush_interval`` is unused.

    ``tools`` is either a single list applied to every message, or one list per message.
    """
    request_tools = resolve_request_tools(tools, len(messages))

    results: list[OpenAIResponse | None] = [None] * len(messages)
    semaphore = asyncio.Semaphore(max_workers)

    async def process(index: int) -> tuple[int, OpenAIResponse | None]:
        try:
            async with semaphore:
                response = await generate_openai_response(
                    model,
                    messages[index],
                    client=client,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=request_tools.for_message(index),
                )
                return index, response
        except Exception:
            # Non-fatal so one bad row cannot end the batch.
            logger.warning("Request %d failed; leaving it unprocessed for a re-run", index, exc_info=True)
            return index, None

    processed: set[int] = set()
    pending_records: list[tuple[int, OpenAIResponse]] = []
    store_path = None
    if resume:
        store_path = resume_store_path(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            request_tools=request_tools,
        )
        logger.info("Checkpoint file path: %s", store_path)
        stored = load_resume_records(store_path, result_count=len(results))
        results = stored.results
        processed = stored.processed_indices
        if stored.skipped_records:
            logger.warning("Skipped %d invalid checkpoint records", stored.skipped_records)
        logger.info("Loaded %d results from %s", len(processed), store_path)

    def flush_resume_store() -> None:
        """Append the results that landed since the last flush; a lost append means those rows re-run
        on resume — costly, not corrupting."""
        records, pending_records[:] = list(pending_records), []
        try:
            append_resume_records(store_path, records)
        except Exception:
            logger.error("Error saving incremental results", exc_info=True)

    todo = [index for index in range(len(messages)) if index not in processed]
    if not todo:
        logger.info("All requests have already been processed.")
        return results

    tasks = [asyncio.create_task(process(index)) for index in todo]
    progress_bar = async_tqdm(total=len(tasks), desc="Processing requests")
    completed = 0
    # process() never raises; a failure arrives as (index, None).
    for future in asyncio.as_completed(tasks):
        index, result = await future
        results[index] = result
        progress_bar.update(1)
        if not resume:
            continue
        completed += 1
        if result is not None:
            pending_records.append((index, result))
        if completed % flush_interval == 0:
            flush_resume_store()
    progress_bar.close()
    if resume:
        flush_resume_store()
    return results
