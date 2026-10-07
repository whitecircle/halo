"""Many chat requests in parallel, resumable: completed rows land in a JSONL store
(:mod:`src.inference.resume_store`) named by the request identity, so a re-run replays them."""

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
from src.inference.resume_store import append_openai_checkpoint, load_openai_checkpoint

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RequestTools:
    """The tool declarations of a batch: one list shared by every message, or one list per message."""

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


def resolve_request_tools(
    tools: list[dict[str, object]] | list[list[dict[str, object]] | None] | None,
    message_count: int,
) -> RequestTools:
    per_message = _is_per_message_tools(tools)
    if per_message and tools is not None and len(tools) != message_count:
        raise ValueError(
            "If tools is per-message, it must have the same length as messages. "
            f"Got {len(tools)} tools and {message_count} messages."
        )
    return RequestTools(raw=tools, per_message=per_message)


def resolve_checkpoint_file(
    *,
    model: str,
    messages: list[str] | list[list[dict]],
    temperature: float | None,
    max_tokens: int,
    request_tools: RequestTools,
) -> str:
    """The resume-store path under the temp dir, keyed by the request identity."""
    hash_content = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": request_tools.hash_sample(),
        "per_message_tools": request_tools.per_message,
        "messages_sample": _sequence_digest(messages),
    }
    task_hash = hashlib.md5(json.dumps(hash_content, sort_keys=True).encode()).hexdigest()
    return str(Path(tempfile.gettempdir()) / f"openai_requests_{task_hash}.jsonl")


async def parallel_openai_requests(
    model: str,
    messages: list[str] | list[list[dict]],
    *,
    client: AsyncOpenAI,
    temperature: float | None = 0.0,
    max_tokens: int = 512,
    max_workers: int = 8,
    checkpoint_interval: int = 10,
    disable_checkpoints: bool = False,
    tools: list[dict] | list[list[dict] | None] | None = None,
) -> list[OpenAIResponse | None]:
    """One :func:`generate_openai_response` per entry of ``messages``, ``max_workers`` in flight, with
    incremental checkpointing: completed results are appended so a re-run resumes, and a failed
    request is left ``None`` and un-checkpointed, so it retries on the next run.

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
            # Non-fatal so one bad row cannot end the batch, but never silent: the row is left
            # un-checkpointed, comes back as None, and its cause is logged with the traceback.
            logger.warning("Request %d failed; leaving it unprocessed for a re-run", index, exc_info=True)
            return index, None

    processed: set[int] = set()
    pending_records: list[tuple[int, OpenAIResponse]] = []
    checkpoint_file = None
    if not disable_checkpoints:
        checkpoint_file = resolve_checkpoint_file(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            request_tools=request_tools,
        )
        logger.info("Checkpoint file path: %s", checkpoint_file)
        checkpoint = load_openai_checkpoint(checkpoint_file, result_count=len(results))
        results = checkpoint.results
        processed = checkpoint.processed_indices
        if checkpoint.skipped_records:
            logger.warning("Skipped %d invalid checkpoint records", checkpoint.skipped_records)
        logger.info("Loaded %d results from %s", len(processed), checkpoint_file)

    def flush_checkpoint() -> None:
        """Append the results that landed since the last flush; a lost append means those rows re-run
        on resume — costly, not corrupting."""
        records, pending_records[:] = list(pending_records), []
        try:
            append_openai_checkpoint(checkpoint_file, records)
        except Exception:
            logger.error("Error saving incremental results", exc_info=True)

    todo = [index for index in range(len(messages)) if index not in processed]
    if not todo:
        logger.info("All requests have already been processed.")
        return results

    tasks = [asyncio.create_task(process(index)) for index in todo]
    progress_bar = async_tqdm(total=len(tasks), desc="Processing requests")
    completed = 0
    # process never raises: a failure yields (index, None), left un-checkpointed so a re-run retries it.
    for future in asyncio.as_completed(tasks):
        index, result = await future
        results[index] = result
        progress_bar.update(1)
        if disable_checkpoints:
            continue
        completed += 1
        if result is not None:
            pending_records.append((index, result))
        if completed % checkpoint_interval == 0:
            flush_checkpoint()
    progress_bar.close()
    if not disable_checkpoints:
        flush_checkpoint()
    return results


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
