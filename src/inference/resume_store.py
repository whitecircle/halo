"""JSONL resume store for a batch of OpenAI-compatible requests.

Completed rows are appended as they land, so a re-run replays them instead of re-issuing them; a
failed row is never written, so it retries. The caller names the file (``resolve_checkpoint_file``),
since only it knows the request identity.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from src.inference.response import OpenAIResponse


@dataclass(slots=True)
class CheckpointLoad:
    results: list[OpenAIResponse | None]
    processed_indices: set[int]
    skipped_records: int = 0


def load_openai_checkpoint(checkpoint_file: str, *, result_count: int) -> CheckpointLoad:
    results: list[OpenAIResponse | None] = [None] * result_count
    processed_indices: set[int] = set()
    skipped_records = 0
    path = Path(checkpoint_file)

    if not path.exists():
        return CheckpointLoad(results=results, processed_indices=processed_indices)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                index, response = _checkpoint_line_to_response(line)
            except (TypeError, ValueError):
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
                    {"index": index, "result": response.model_dump()},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _checkpoint_line_to_response(line: str) -> tuple[int, OpenAIResponse]:
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

    return index, _openai_response_from_checkpoint(result)


def _openai_response_from_checkpoint(result: dict[str, object]) -> OpenAIResponse:
    answer = result.get("answer")
    # A null answer is legitimate with tool_calls; only neither-present is genuinely corrupt.
    if answer is None and not result.get("tool_calls"):
        raise ValueError("checkpoint result is missing answer")

    token_ids = result.get("token_ids")
    if token_ids is not None and not (isinstance(token_ids, list) and all(type(t) is int for t in token_ids)):
        raise TypeError("checkpoint token_ids must be a list of integers")

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


def _optional_int(value: object) -> int:
    return value if type(value) is int else 0
