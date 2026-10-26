"""The rows a row map emits for a dropped row, the predicate that drops them, and the report of how much of a
corpus a filter dropped."""

import logging
from typing import Any

from datasets import Dataset, DatasetDict

from halo.runtime.ranks import is_local_main_process

logger = logging.getLogger(__name__)

# The dropped share of a corpus past which the drop is a likely config bug, reported as a WARNING.
_HIGH_REJECTION_WARN_FRACTION = 0.5


def build_rejection_row(tokenizer=None, **tokenizer_kwargs) -> dict[str, list]:
    """The typed row a tokenizing map emits for a dropped row: ``tokenizer``'s output keys (probed once), else
    ``input_ids``/``attention_mask``, each a one-element list of the real element type.

    Never ``None`` or ``[]``: a worker whose first writer batch is all rejected would infer a ``null`` column and
    fail casting the real batches. The ``[0]`` attention mask is what :func:`is_valid_example` drops it on.
    """
    if tokenizer is None:
        return {"input_ids": [0], "attention_mask": [0]}
    sample = tokenizer("sample", **tokenizer_kwargs)
    return {key: [0] if key == "attention_mask" else (list(values[:1]) or [0]) for key, values in sample.items()}


def _blank_text(content: Any) -> Any:
    """``content`` with its text emptied, in its own shape; a non-text part would stay content-bearing, so it is
    refused."""
    if not isinstance(content, list):
        return ""
    if any(part.get("type") != "text" for part in content):
        raise ValueError(f"Only text content can be blanked, got parts of type {[p.get('type') for p in content]}.")
    return [{**part, "text": ""} for part in content]


def blank_conversation(messages: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The rejection value of a message-list column: ``messages`` with every text blanked, or one blank user
    turn. Keys and content shapes stay, so the Arrow struct matches the real rows'."""
    if not messages:
        return [{"role": "user", "content": ""}]
    return [{**message, "content": _blank_text(message.get("content"))} for message in messages]


def _is_content_bearing(value: Any) -> bool:
    """Whether a filter-field value carries content: not ``None``, a blank string or an empty list, and a message
    or text-part list only when some text in it is non-blank. Anything else (token ids, image parts) does."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        if not value:
            return False
        if all(isinstance(item, dict) and ("role" in item or "content" in item) for item in value):
            return any(_is_content_bearing(item.get("content")) for item in value)
        if all(isinstance(item, dict) and item.get("type") == "text" for item in value):
            return any(_is_content_bearing(item.get("text")) for item in value)
    return True


def is_valid_example(row: dict[str, Any], filter_field: str = "input_ids") -> bool:
    """Whether ``row`` is a real example rather than a rejection row: a content-bearing ``filter_field`` and an
    ``attention_mask``, where present, that is not all zero."""
    if not _is_content_bearing(row.get(filter_field)):
        return False
    mask = row.get("attention_mask")
    return mask is None or any(mask)


def dataset_total_size(dataset: Dataset | DatasetDict) -> int:
    """The number of rows across every split."""
    if isinstance(dataset, Dataset):
        return len(dataset)
    return sum(len(ds) for ds in dataset.values())


def report_rejected_rows(original_size: int, kept_size: int, context: str) -> None:
    """Log, on the local main process, how much of a corpus a post-map filter dropped: a WARNING past
    :data:`_HIGH_REJECTION_WARN_FRACTION`, where a vanishing corpus is a config bug, INFO below it."""
    if not is_local_main_process():
        return
    rejected = original_size - kept_size
    fraction = rejected / original_size if original_size > 0 else 0.0
    message = f"Filtered out {rejected}/{original_size} examples ({100 * fraction:.1f}%) rejected during {context}"
    if fraction >= _HIGH_REJECTION_WARN_FRACTION:
        logger.warning(
            f"{message}. A rejection rate this high usually means a config bug — an "
            f"assistant_message_template that never matches the rendered turns, a wrong "
            f"conversation/text field, or a max_length below the typical sequence length."
        )
    else:
        logger.info(message)
