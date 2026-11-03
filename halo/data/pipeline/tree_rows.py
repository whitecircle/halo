"""Tree rows: conversations sharing a prefix up to a configured marker, built into packed-ready trees. Every row is
rendered and tokenized on its own, as the flat pipeline does, so a tree path holds the flat example's tokens; the key
only decides which rows are examined together."""

import functools
import hashlib
import json
from collections.abc import Callable
from itertools import groupby
from typing import Any

import numpy as np
import pyarrow.compute as pc
from datasets import Dataset, DatasetDict, Features, List, Value
from transformers import PreTrainedTokenizer

from halo.data.chat_render import tokenize_rendered
from halo.data.completion_spans import PACKED_SPAN_POLICY, build_span_labels, resolve_spans_or_warn
from halo.data.conversation import resolve_json_value
from halo.data.pipeline.coordinated_map import coordinated_filter, coordinated_map
from halo.data.pipeline.map_cache_key import carry_cache_key
from halo.data.pipeline.rejected_rows import dataset_total_size, report_rejected_rows
from halo.data.pipeline.text_rows import render_row_conversation
from halo.data.tree_layout import (
    DEPTH_COLUMN,
    POSITION_COLUMN,
    TRAINED_COLUMN,
    build_trie,
    serialize_tree,
    share_limit,
    split_to_budget,
)
from halo.data.turn_markers import tokenize_response_template
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.log import info_logger
from halo.runtime.consensus import agree_probe_across_ranks
from halo.runtime.ranks import is_global_main_process

logger = info_logger(__name__)

SHARE_KEY_COLUMN = "tree_key"

# A shared span under this many tokens is repeated per row, not split off.
MIN_SHARED_TOKENS = 16

# All per token, at the narrowest types: inferred, the two layout columns would be 8-byte integers.
TREE_ROW_FEATURES = Features(
    {
        "input_ids": List(Value("int32")),
        TRAINED_COLUMN: List(Value("bool")),
        POSITION_COLUMN: List(Value("int32")),
        DEPTH_COLUMN: List(Value("int16")),
    }
)

# One per dropped conversation, so the filter removing them also counts them.
_DROPPED_ROW = dict.fromkeys(TREE_ROW_FEATURES, [])


def _shared_part(conversation: list[dict[str, Any]], markers: tuple[str, ...], requires: str | None) -> Any:
    """Up to and including a marker in the message before the final turn, or the whole conversation if no cut."""
    if len(conversation) < 2 or not isinstance(conversation[-2].get("content"), str):
        return conversation
    content = conversation[-2]["content"]
    found = [marker for marker in markers if content.count(marker) == 1]
    if len(found) != 1:
        return conversation
    cut = content.index(found[0]) + len(found[0])
    if requires is not None and requires not in content[:cut]:
        return conversation
    return [conversation[:-2], {**conversation[-2], "content": content[:cut]}]


def share_key(conversation: list[dict[str, Any]], markers: tuple[str, ...], requires: str | None) -> str:
    """Ends at a marker when exactly one of ``markers`` occurs exactly once in the message before the final turn, after
    ``requires`` if set; any other conversation keys on its whole content, so only an exact duplicate joins it."""
    shared = _shared_part(conversation, markers, requires)
    encoded = json.dumps(shared, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8", "surrogatepass")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def build_share_key_map(
    conversation_field: str, markers: tuple[str, ...], requires: str | None
) -> Callable[[dict[str, list]], dict[str, list]]:
    def key_batch(batch: dict[str, list]) -> dict[str, list]:
        return {
            SHARE_KEY_COLUMN: [
                share_key(resolve_json_value(conversation), markers, requires)
                for conversation in batch[conversation_field]
            ]
        }

    return key_batch


def is_kept_tree(depth: list[int]) -> bool:
    """Filter predicate on the depth column: False for the empty rows that stand for over-length conversations."""
    return bool(depth)


def _trained_tokens(
    input_ids: list[int],
    *,
    tokenizer: PreTrainedTokenizer,
    response_token_ids: list[int],
    eos_token_ids: frozenset[int],
    assistant_message_template: str,
    train_on_last_assistant_only: bool,
) -> list[bool]:
    """Which tokens of a flat example the packing completion collator trains: each span through its closing token,
    minus pad-valued tokens where pad is not the EOS, never the document's first token."""
    trained = [False] * len(input_ids)
    spans = resolve_spans_or_warn(
        input_ids,
        response_token_ids,
        eos_token_ids,
        train_on_last_assistant_only=train_on_last_assistant_only,
        span_policy=PACKED_SPAN_POLICY,
        response_prompt_template=assistant_message_template,
        tokenizer=tokenizer,
        row_label="tree row",
    )
    untrained_pad = None if tokenizer.pad_token_id == tokenizer.eos_token_id else tokenizer.pad_token_id
    values = [LABEL_IGNORE_INDEX if token == untrained_pad else token for token in input_ids]
    labels = build_span_labels(values, input_ids, spans, restore_end=True)
    trained = [label != LABEL_IGNORE_INDEX for label in labels]
    if trained:
        trained[0] = False
    return trained


def build_tree_row_processor(
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    *,
    conversation_field: str,
    share_markers: tuple[str, ...],
    share_requires: str | None,
    eos_token_ids: frozenset[int],
    assistant_message_template: str,
    train_on_last_assistant_only: bool,
    **render_kwargs,
) -> Callable[[dict[str, list]], dict[str, list]]:
    """Batched map turning runs of adjacent rows with one share key into tree columns: one row per tree, more when a
    run overflows ``max_length``, several roots in one row when they share under :data:`MIN_SHARED_TOKENS`. A
    conversation over ``max_length`` is dropped; a run a batch or worker boundary cuts becomes two trees and changes no
    example."""
    # Partials: the map's cache key reads a partial's bound arguments and skips a dict a closure captures.
    render = functools.partial(
        render_row_conversation,
        tokenizer=tokenizer,
        conversation_field=conversation_field,
        **render_kwargs,
    )
    trained_tokens = functools.partial(
        _trained_tokens,
        tokenizer=tokenizer,
        response_token_ids=tokenize_response_template(assistant_message_template, tokenizer),
        eos_token_ids=eos_token_ids,
        assistant_message_template=assistant_message_template,
        train_on_last_assistant_only=train_on_last_assistant_only,
    )

    def process_batch(batch: dict[str, list]) -> dict[str, list]:
        out: dict[str, list] = {column: [] for column in TREE_ROW_FEATURES}
        rows = [{column: values[i] for column, values in batch.items()} for i in range(len(batch[conversation_field]))]
        keys = [share_key(resolve_json_value(row[conversation_field]), share_markers, share_requires) for row in rows]
        for _, run in groupby(range(len(rows)), key=keys.__getitem__):
            paths = [
                tokenize_rendered(tokenizer, render(rows[i]), truncation=False, padding=False)["input_ids"]
                for i in run
            ]
            trained = [trained_tokens(path) for path in paths]
            nodes = build_trie(paths, [share_limit(mask) for mask in trained], MIN_SHARED_TOKENS)
            trees, dropped = split_to_budget(nodes, max_length)
            for tree_row in [*(serialize_tree(tree, paths, trained) for tree in trees), *[_DROPPED_ROW] * dropped]:
                for column in TREE_ROW_FEATURES:
                    out[column].append(tree_row[column])
        return out

    return process_batch


def _share_key_order(keys: Dataset) -> np.ndarray:
    """Sorted in Arrow: as Python strings the column takes about a minute and 1.5 GB per rank at 10M rows. int64, since
    ``select`` fingerprints the indices, dtype included, and an unstamped in-memory split keys its next map on that."""
    return pc.sort_indices(keys.with_format("arrow")[SHARE_KEY_COLUMN]).to_numpy().astype(np.int64)


def _order_by_share_key(
    dataset: DatasetDict, conversation_field: str, markers: tuple[str, ...], requires: str | None, num_proc: int
) -> DatasetDict:
    """Rows of one share key adjacent, keys in digest order, which doubles as the shuffle."""
    keys = coordinated_map(
        dataset,
        build_share_key_map(conversation_field, markers, requires),
        desc="Keying rows by shared prefix",
        num_proc=num_proc,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    step = f"tree-order-{hashlib.md5(repr((markers, requires)).encode()).hexdigest()[:12]}"
    return DatasetDict(
        {
            split: carry_cache_key(dataset[split], dataset[split].select(_share_key_order(keys[split])), step)
            for split in dataset
        }
    )


def prepare_tree_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    *,
    conversation_field: str,
    share_markers: tuple[str, ...],
    share_requires: str | None,
    num_proc: int,
    **processor_kwargs,
) -> DatasetDict:
    """Raw conversations to packed-ready tree rows, every split: key, order, build trees, drop and report overlong."""
    for split in dataset:
        # A map over no rows returns its input untouched, so the tree columns would not exist below.
        if agree_probe_across_ranks(len(dataset[split]) == 0, f"the {split} split", "empty split"):
            raise ValueError(f"tree packing got an empty {split} split on at least one data-parallel rank.")
    processed = coordinated_map(
        _order_by_share_key(dataset, conversation_field, share_markers, share_requires, num_proc),
        build_tree_row_processor(
            tokenizer,
            max_length,
            conversation_field=conversation_field,
            share_markers=share_markers,
            share_requires=share_requires,
            **processor_kwargs,
        ),
        desc="Processing tree rows",
        num_proc=num_proc,
        batched=True,
        remove_columns=dataset["train"].column_names,
        features=TREE_ROW_FEATURES,
    )
    trees = coordinated_filter(
        processed, is_kept_tree, desc="dropping over-length rows", num_proc=num_proc, input_columns=[DEPTH_COLUMN]
    )
    rows, tree_count = dataset_total_size(dataset), dataset_total_size(trees)
    kept = rows - (dataset_total_size(processed) - tree_count)
    report_rejected_rows(rows, kept, f"tree processing (longer than max_length={max_length})")
    if is_global_main_process():
        logger.info(f"Tree packing: {kept} rows in {tree_count} trees")
    return trees
