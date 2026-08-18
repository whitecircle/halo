"""JSONL resume store for a batch of OpenAI-compatible requests.

Completed rows are appended as they land, so a re-run replays them instead of re-billing them; a
failed row is never written, so it retries. The file itself is named by the caller
(``resolve_checkpoint_file``), which alone knows the request identity.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from src.inference.response import OpenAIResponse

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CheckpointLoad:
    results: list[OpenAIResponse | None]
    processed_indices: set[int]
    skipped_records: int = 0


def load_openai_checkpoint(
    checkpoint_file: str,
    *,
    result_count: int,
    response_format: type[BaseModel] | None,
) -> CheckpointLoad:
    results: list[OpenAIResponse | None] = [None] * result_count
    processed_indices: set[int] = set()
    skipped_records = 0
    path = Path(checkpoint_file)

    if not path.exists():
        return CheckpointLoad(results=results, processed_indices=processed_indices)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                index, response = _checkpoint_line_to_response(line, response_format)
            except (TypeError, ValueError, json.JSONDecodeError):
                skipped_records += 1
                continue

            if index < 0 or index >= result_count:
                skipped_records += 1
                continue

            results[index] = response
            processed_indices.add(index)

    return CheckpointLoad(
        results=results,
        processed_indices=processed_indices,
        skipped_records=skipped_records,
    )


def append_openai_checkpoint(
    checkpoint_file: str,
    records: list[tuple[int, OpenAIResponse]],
) -> None:
    if not records:
        return

    path = Path(checkpoint_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for index, response in records:
            file.write(
                json.dumps(
                    {"index": index, "result": _response_to_checkpoint(response)},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _response_to_checkpoint(response: OpenAIResponse) -> dict[str, object]:
    result = response.model_dump()
    if isinstance(response.answer, BaseModel):
        result["answer"] = response.answer.model_dump()
    return result


def _checkpoint_line_to_response(
    line: str,
    response_format: type[BaseModel] | None,
) -> tuple[int, OpenAIResponse]:
    data = json.loads(line)
    if not isinstance(data, dict):
        raise TypeError("checkpoint record must be an object")

    index = data.get("index")
    if type(index) is not int:
        raise TypeError("checkpoint index must be an integer")

    result = data.get("result")
    if result is None:
        raise ValueError("empty checkpoint result")
    if not isinstance(result, dict):
        raise TypeError("checkpoint result must be an object")

    return index, _openai_response_from_checkpoint(result, response_format)


def _openai_response_from_checkpoint(
    result: dict[str, object],
    response_format: type[BaseModel] | None,
) -> OpenAIResponse:
    answer = result.get("answer")
    if response_format is not None and isinstance(answer, dict):
        try:
            answer = response_format.model_validate(answer)
        except Exception as e:
            logger.warning(
                "Structured response failed %s validation; returning the raw unvalidated dict: %s",
                response_format.__name__,
                e,
            )

    # A null answer is legitimate with tool_calls; only neither-present is genuinely corrupt.
    if answer is None and not result.get("tool_calls"):
        raise ValueError("checkpoint result is missing answer")

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
    )


def _optional_int(value: object) -> int:
    return value if type(value) is int else 0
