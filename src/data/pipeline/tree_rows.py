"""Tree rows: conversations that share a prefix up to a configured marker, ordered side by side and built into
packed-ready trees (:mod:`src.data.tree_layout`); :func:`prepare_tree_dataset` is the whole of it. Every row is
rendered and tokenized on its own, exactly as the flat pipeline does it, so a tree path holds the flat example's
tokens and trains the tokens the packing collator trains for it; the key only decides which rows are examined
together."""

import functools
import hashlib
import json
from collections.abc import Callable
from itertools import groupby
from typing import Any

import numpy as np
from datasets import DatasetDict, Features, List, Value
from transformers import PreTrainedTokenizer

from src.data.completion_spans import PACKED_SPAN_POLICY, build_span_labels, resolve_spans_or_warn
from src.data.pipeline.chat_render import tokenize_rendered
from src.data.pipeline.conversation import resolve_json_value
from src.data.pipeline.coordinated_map import coordinated_filter, coordinated_map
from src.data.pipeline.map_cache_key import carry_cache_key
from src.data.pipeline.rejected_rows import dataset_total_size, report_rejected_rows
from src.data.pipeline.text_rows import render_row_conversation
from src.data.tree_layout import (
    DEPTH_COLUMN,
    POSITION_COLUMN,
    TRAINED_COLUMN,
    build_trie,
    serialize_tree,
    share_limit,
    split_to_budget,
)
from src.data.turn_markers import tokenize_response_template
from src.kernels.logprobs import LABEL_IGNORE_INDEX
from src.log import info_logger
from src.runtime.consensus import agree_probe_across_ranks
from src.runtime.ranks import is_global_main_process

logger = info_logger(__name__)

# The one column of the share-key map's output.
SHARE_KEY_COLUMN = "tree_key"

# Columns of a processed tree row, all per token, at the narrowest types that hold them: inferred,
# the two layout columns would be 8-byte integers.
TREE_ROW_FEATURES = Features(
    {
        "input_ids": List(Value("int32")),
        TRAINED_COLUMN: List(Value("bool")),
        POSITION_COLUMN: List(Value("int32")),
        DEPTH_COLUMN: List(Value("int16")),
    }
)

# The row emitted in place of a conversation that does not fit the length budget. One per dropped
# conversation, so the filter that removes them also counts them.
_DROPPED_ROW = dict.fromkeys(TREE_ROW_FEATURES, [])


def _shared_part(conversation: list[dict[str, Any]], markers: tuple[str, ...], requires: str | None) -> Any:
    """The part of a conversation other rows may repeat verbatim: up to and including a marker in the message
    before the final turn, or the whole conversation where no single cut exists."""
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
    """Digest of :func:`_shared_part`: it ends at a marker when exactly one of ``markers`` occurs exactly once in
    the message before the final turn, after ``requires`` if set; any other conversation keys on its whole
    content, so only an exact duplicate joins it."""
    shared = _shared_part(conversation, markers, requires)
    encoded = json.dumps(shared, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8", "surrogatepass")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def build_share_key_map(
    conversation_field: str, markers: tuple[str, ...], requires: str | None
) -> Callable[[dict[str, list]], dict[str, list]]:
    """Build the batched map that reduces every row to its :func:`share_key`."""

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
    """Which tokens of one flat example ``DataCollatorForCompletionOnlyLMWithPacking`` trains: each completion
    span through its closing token, minus its pad-valued tokens where pad is not the EOS, and never the
    document's first token, which nothing precedes."""
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
    min_shared_tokens: int,
    eos_token_ids: frozenset[int],
    assistant_message_template: str,
    train_on_last_assistant_only: bool,
    **render_kwargs,
) -> Callable[[dict[str, list]], dict[str, list]]:
    """Build the batched map that turns runs of adjacent rows with one share key into tree columns: one row per
    tree, more when a run overflows ``max_length`` and splits into trees that each repeat the shared part, and
    several roots in one row when they share under ``min_shared_tokens``. A conversation over ``max_length`` alone
    is dropped, as on the flat path; a run a batch boundary cuts becomes two trees and changes no example.
    ``render_kwargs`` are the flat row processor's, so a row renders by the same call."""
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
            nodes = build_trie(paths, [share_limit(mask) for mask in trained], min_shared_tokens)
            trees, dropped = split_to_budget(nodes, max_length)
            for tree_row in [*(serialize_tree(tree, paths, trained) for tree in trees), *[_DROPPED_ROW] * dropped]:
                for column in TREE_ROW_FEATURES:
                    out[column].append(tree_row[column])
        return out

    return process_batch


def _order_by_share_key(
    dataset: DatasetDict, conversation_field: str, markers: tuple[str, ...], requires: str | None, num_proc: int
) -> DatasetDict:
    """Every split with the rows of one share key adjacent, the keys in digest order, which doubles as the
    shuffle."""
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
            split: carry_cache_key(
                dataset[split],
                dataset[split].select(np.argsort(np.asarray(keys[split][SHARE_KEY_COLUMN]), kind="stable")),
                step,
            )
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
    """Raw conversations to packed-ready tree rows, every split: key, order, build trees, drop the conversations
    longer than ``max_length`` and report them; ``processor_kwargs`` are :func:`build_tree_row_processor`'s."""
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
