"""Preparation of preference corpora (DPO, SMPO, reward): row normalization, chat-template maps, SMPO's tokenizers."""

from collections.abc import Mapping
from functools import partial
from typing import Any

from datasets import Dataset, Features, Sequence, Value
from datasets import Image as ImageFeature
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, ProcessorMixin

from halo.data.chat_render import render_chat_template
from halo.data.conversation import as_conversation, chat_template_kwargs, reject_image_content
from halo.data.pipeline.coordinated_map import DATASET_NUM_PROC, coordinated_filter, coordinated_map
from halo.data.pipeline.prompt_completion import (
    render_prompt_completions,
    tokenize_completions_jointly,
    tokenize_prompt_completions,
)
from halo.data.pipeline.rejected_rows import report_rejected_rows
from halo.data.pipeline.text_rows import prepare_generative_row
from halo.data.pipeline.vlm_rows import normalize_vlm_conversation
from halo.data.vlm_detection import VLM_RAW_IMAGE_COLUMNS
from halo.data.vlm_render import extract_conversation_images, render_vlm_text
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.runtime.consensus import agree_probe_across_ranks

VLM_PREFERENCE_COLUMNS = ("chosen_text", "rejected_text", "images")

# TRL's optional per-pair margin, carried through the map unchanged (its collator reads this name).
MARGIN_COLUMN = "margin"

# How many of a VLM SMPO row's last prompt tokens its completions start with; the collator drops them from the prompt.
PROMPT_OVERLAP_COLUMN = "prompt_overlap"

# Pinned: a writer batch or map shard of text-only rows would infer ``images`` as ``List(null)``, refusing image rows.
VLM_SMPO_FEATURES = Features(
    {
        "prompt_text": Value("string"),
        PROMPT_OVERLAP_COLUMN: Value("int64"),
        "images": Sequence(ImageFeature()),
        "chosen_input_ids": Sequence(Value("int64")),
        "rejected_input_ids": Sequence(Value("int64")),
    }
)


def _shared_message_prefix_len(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x.get("role") != y.get("role") or x.get("content") != y.get("content"):
            break
        n += 1
    return n


def is_empty_reply(message: Mapping[str, Any]) -> bool:
    """An assistant turn with no tool call whose content is ``None``, empty or blank, as a string or as text chunks."""
    if message.get("tool_calls"):
        return False
    content = message.get("content")
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    return all(
        isinstance(chunk, Mapping) and chunk.get("type") == "text" and not (chunk.get("text") or "").strip()
        for chunk in content
    )


def has_a_reply_on_both_sides(row: Mapping[str, Any]) -> bool:
    """False for a pair whose chosen or rejected reply, the side's last assistant turn, is empty
    (:func:`is_empty_reply`). Against a real reply an empty one teaches when to end the turn, not which reply is
    better, and a template that checks content (Mistral's) refuses to render it. A side that is missing or does not
    end on an assistant turn passes, for :func:`normalize_preference_row` to refuse by name."""
    for field in ("chosen", "rejected"):
        side = row.get(field)
        last = side[-1] if isinstance(side, list) and side else None
        if isinstance(last, Mapping) and last.get("role") == "assistant" and is_empty_reply(last):
            return False
    return True


def drop_empty_reply_pairs(dataset: Dataset, split_name: str, num_proc: int = DATASET_NUM_PROC) -> Dataset:
    """COLLECTIVE: the pairs :func:`has_a_reply_on_both_sides` keeps, with the count line, before any template map. A
    split the filter empties raises on every rank."""
    before = len(dataset)
    dataset = coordinated_filter(
        dataset,
        has_a_reply_on_both_sides,
        desc=f"Dropping {split_name} preference pairs with an empty reply",
        num_proc=num_proc,
    )
    # A per-rank fact on a presharded corpus, agreed so the raise lands on every rank at once.
    if agree_probe_across_ranks(
        before > 0 and len(dataset) == 0, f"the {split_name} split", "split emptied by the empty-reply filter"
    ):
        raise ValueError(
            f"All {before} pairs of the {split_name} split have an empty or blank chosen or rejected reply on at "
            f"least one data-parallel rank, so none is left to train on. Check the dataset's chosen and rejected "
            f"columns."
        )
    report_rejected_rows(before, len(dataset), f"the {split_name} split (an empty or blank chosen or rejected reply)")
    return dataset


def normalize_preference_row(row: dict[str, Any]) -> dict[str, Any]:
    """``prompt`` a message list, ``chosen``/``rejected`` continuation-only lists. A string prompt becomes a user turn,
    a repeated prompt prefix is stripped, and a missing prompt (Skywork-Reward style) is the shared leading span; a
    continuation that is empty or does not end on an assistant turn raises as a mis-mapped dataset."""
    chosen, rejected = row["chosen"], row["rejected"]
    prompt = as_conversation(row.get("prompt"))

    if not prompt:
        prompt = list(chosen[: _shared_message_prefix_len(chosen, rejected)])
        if not prompt:
            raise ValueError(
                "Preference row has no 'prompt' and chosen/rejected share no leading messages to extract one from."
            )

    for field in ("chosen", "rejected"):
        completion = row[field]
        # A completion that repeats every prompt turn is a full conversation, so strip the prefix.
        if _shared_message_prefix_len(completion, prompt) == len(prompt):
            completion = completion[len(prompt) :]
        if not completion or completion[-1].get("role") != "assistant":
            raise ValueError(
                f"Preference row '{field}' is not an assistant continuation after prompt normalization: "
                f"roles={[m.get('role') for m in row[field]]}"
            )
        row[field] = completion
    row["prompt"] = prompt
    return row


def split_vlm_preference_row(
    features: dict[str, Any], subject: str
) -> tuple[list[dict[str, Any]], list, dict[str, list[dict[str, Any]]]]:
    """Raw row to prompt history, images and completions. Images may sit only in the prompt, the one region both sides
    render identically; ``subject`` names the route in that refusal."""
    row = normalize_preference_row({key: features[key] for key in ("prompt", "chosen", "rejected") if key in features})

    images = next((features[column] for column in VLM_RAW_IMAGE_COLUMNS if features.get(column) is not None), None)
    prompt_history, pil_images = extract_conversation_images(normalize_vlm_conversation(row["prompt"], images))

    completions = {}
    for field in ("chosen", "rejected"):
        completion_history, completion_images = extract_conversation_images(row[field])
        if completion_images:
            raise ValueError(
                f"{subject} carries images inside the '{field}' completion — completions must be "
                f"text-only (images belong in the prompt, which both sides share)."
            )
        completions[field] = completion_history
    return prompt_history, pil_images, completions


def vlm_preference_features(dataset: Dataset) -> Features:
    """Pinned like :func:`vlm_map_features` (an all-text shard infers ``images`` as ``List(null)``); the ``Image``
    sequence also hands back PIL objects. ``margin`` is declared exactly when the dataset carries it."""
    chosen, rejected, images = VLM_PREFERENCE_COLUMNS
    features = {chosen: Value("string"), rejected: Value("string"), images: Sequence(ImageFeature())}
    if MARGIN_COLUMN in dataset.column_names:
        features[MARGIN_COLUMN] = Value("float32")
    return Features(features)


def render_vlm_preference_row(features: dict[str, Any], processing_class: ProcessorMixin) -> dict[str, Any]:
    """Each side chat-templated whole; images travel as PIL objects, so no pixels reach the Arrow cache."""
    prompt_history, pil_images, completions = split_vlm_preference_row(features, "VLM preference row")
    rendered = {
        f"{field}_text": render_vlm_text(processing_class, prompt_history + history)
        for field, history in completions.items()
    }
    return {**rendered, "images": pil_images}


def tokenize_preference_row(
    features: dict[str, str],
    processing_class: PreTrainedTokenizerBase,
    *,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    truncation_mode: str,
    eos_token_ids: frozenset[int] = frozenset(),
) -> dict[str, list[int]]:
    """Tokenized jointly, so a token merged across the boundary splits the same way on both sides."""
    prompt_ids, (chosen_ids, rejected_ids) = tokenize_prompt_completions(
        features["prompt"],
        (features["chosen"], features["rejected"]),
        processing_class,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        truncation_mode=truncation_mode,
        joint_boundary=True,
        eos_token_ids=eos_token_ids,
    )
    return {"prompt_input_ids": prompt_ids, "chosen_input_ids": chosen_ids, "rejected_input_ids": rejected_ids}


def tokenize_vlm_preference_row(
    features: dict[str, Any],
    processing_class: ProcessorMixin,
    *,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    truncation_mode: str,
    eos_token_ids: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Completions split off by the text pipeline's prefix strip and tokenized as :func:`tokenize_preference_row` does;
    the prompt stays text plus PIL images for the collator, which refuses an over-length one, so ``max_prompt_length``
    and ``truncation_mode`` bound nothing here."""
    prompt_history, pil_images, completions = split_vlm_preference_row(features, "VLM SMPO row")
    prompt_text, completion_texts = render_prompt_completions(
        partial(render_vlm_text, processing_class), prompt_history, completions.items()
    )
    (chosen_ids, rejected_ids), prompt_overlap = tokenize_completions_jointly(
        prompt_text,
        completion_texts,
        resolve_tokenizer(processing_class),
        max_completion_length=max_completion_length,
        eos_token_ids=eos_token_ids,
    )
    return {
        "prompt_text": prompt_text,
        PROMPT_OVERLAP_COLUMN: prompt_overlap,
        "images": pil_images,
        "chosen_input_ids": chosen_ids,
        "rejected_input_ids": rejected_ids,
    }


def render_preference_row(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    tools_field: str | None = None,
    reasoning_effort_field: str | None = None,
) -> dict[str, str]:
    row = normalize_preference_row(row)

    # DPO's run-intent dispatch reads image columns only, so an embedded-image row reaches here undetected.
    for field in ("prompt", "chosen", "rejected"):
        reject_image_content(row[field], f"preference field '{field}'")

    template_kwargs = chat_template_kwargs(
        row, interleaved_thinking=False, tools_field=tools_field, reasoning_effort_field=reasoning_effort_field
    )
    row["prompt"], (row["chosen"], row["rejected"]) = render_prompt_completions(
        partial(render_chat_template, tokenizer, **template_kwargs),
        row["prompt"],
        [(field, row[field]) for field in ("chosen", "rejected")],
    )
    return row


def prepare_preference_datasets(
    train_dataset,
    eval_dataset,
    tokenizer: PreTrainedTokenizer,
    num_proc: int = DATASET_NUM_PROC,
    tools_field: str | None = None,
    reasoning_effort_field: str | None = None,
):
    process_fn = partial(
        render_preference_row,
        tokenizer=tokenizer,
        tools_field=tools_field,
        reasoning_effort_field=reasoning_effort_field,
    )
    train_dataset = coordinated_map(
        train_dataset, process_fn, desc="Applying chat template to train dataset", num_proc=num_proc
    )
    eval_dataset = coordinated_map(
        eval_dataset, process_fn, desc="Applying chat template to eval dataset", num_proc=num_proc
    )
    return train_dataset, eval_dataset


def _normalized_generative_row(row, tokenizer, max_length, tools_field=None, reasoning_effort_field=None):
    """Normalized first: the raw test split still carries hub shapes :func:`prepare_generative_row` cannot template."""
    return prepare_generative_row(
        normalize_preference_row(row),
        tokenizer=tokenizer,
        max_length=max_length,
        tools_field=tools_field,
        reasoning_effort_field=reasoning_effort_field,
    )


def prepare_generative_dataset(
    dataset,
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int,
    num_proc: int = DATASET_NUM_PROC,
    tools_field: str | None = None,
    reasoning_effort_field: str | None = None,
):
    return coordinated_map(
        dataset,
        partial(
            _normalized_generative_row,
            tokenizer=tokenizer,
            max_length=max_prompt_length,
            tools_field=tools_field,
            reasoning_effort_field=reasoning_effort_field,
        ),
        num_proc=num_proc,
        desc="Preparing dataset for generation",
    )
