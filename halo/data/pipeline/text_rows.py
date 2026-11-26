"""Text row processors: the per-row chat render and tokenization a coordinated map runs over a text corpus."""

import functools
from collections.abc import Callable
from typing import Any

from transformers import PreTrainedTokenizer

from halo.data.chat_render import (
    render_chat_template,
    render_conversation,
    tokenize_rendered,
    tokenize_training_conversation,
)
from halo.data.conversation import (
    TEMPLATE_COLUMN_KNOBS,
    chat_template_kwargs,
    reject_image_content,
    resolve_json_value,
)
from halo.data.pipeline.rejected_rows import build_rejection_row
from halo.models.introspection.context_window import is_bounded_length

_TEXT_RENDER_KNOBS = (
    "conversation_field",
    "system_prompt",
    "model_supports_system_role",
    "interleaved_thinking",
    *TEMPLATE_COLUMN_KNOBS,
)


def text_render_kwargs(args) -> dict[str, Any]:
    """One dict for a text renderer and its map-cache key, so the key cannot miss a knob the render reads."""
    return {name: getattr(args, name) for name in _TEXT_RENDER_KNOBS}


def render_row_conversation(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    conversation_field: str,
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
    add_generation_prompt: bool = False,
    interleaved_thinking: bool = False,
    tools_field: str | None = None,
    reasoning_effort_field: str | None = None,
) -> str:
    """``add_generation_prompt`` drops the final message: on the generation-eval path the trailing assistant turn is
    the reference answer to regenerate."""
    conversation = resolve_json_value(row[conversation_field])
    if add_generation_prompt:
        conversation = conversation[:-1]

    return render_conversation(
        tokenizer,
        conversation,
        row,
        conversation_field=conversation_field,
        system_prompt=system_prompt,
        model_supports_system_role=model_supports_system_role,
        add_generation_prompt=add_generation_prompt,
        interleaved_thinking=interleaved_thinking,
        tools_field=tools_field,
        reasoning_effort_field=reasoning_effort_field,
    )


def tokenize_training_row(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    eos_token_ids: frozenset[int],
    *,
    conversation_field: str,
    **render_kwargs,
):
    """One row's whole conversation as an untruncated training row (:func:`tokenize_training_conversation`)."""
    return tokenize_training_conversation(
        tokenizer,
        resolve_json_value(row[conversation_field]),
        row,
        eos_token_ids,
        conversation_field=conversation_field,
        **render_kwargs,
    )


def prepare_generative_row(row, tokenizer, max_length, tools_field=None, reasoning_effort_field=None):
    """``row["prompt"]`` messages to a tokenized, padded generation prompt, capped at a bounded ``max_length``."""
    reject_image_content(row["prompt"], "prompt field 'prompt'")
    constructed_prompt = render_chat_template(
        tokenizer,
        row["prompt"],
        add_generation_prompt=True,
        **chat_template_kwargs(
            row, interleaved_thinking=False, tools_field=tools_field, reasoning_effort_field=reasoning_effort_field
        ),
    )
    # An unset cap means no cap: HF resolves `truncation=True, max_length=None` to the model_max_length.
    return tokenize_rendered(
        tokenizer,
        constructed_prompt,
        for_generation=True,
        truncation=is_bounded_length(max_length),
        padding=True,
        max_length=max_length,
    )


def build_chat_row_processor(
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    conversation_field: str,
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
    add_generation_prompt: bool = False,
    interleaved_thinking: bool = False,
    tools_field: str | None = None,
    reasoning_effort_field: str | None = None,
    *,
    eos_token_ids: frozenset[int],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """A row map that chat-templates and tokenizes; a row over ``max_length`` becomes the typed rejection row.
    ``eos_token_ids`` are the model's terminators, which decide whether a training row's last turn is closed."""
    none_example = build_rejection_row(tokenizer, truncation=True, padding=False, max_length=max_length)
    knobs = {
        "conversation_field": conversation_field,
        "system_prompt": system_prompt,
        "model_supports_system_role": model_supports_system_role,
        "interleaved_thinking": interleaved_thinking,
        "tools_field": tools_field,
        "reasoning_effort_field": reasoning_effort_field,
    }
    # Partials: the map's cache key reads a partial's bound arguments and skips a dict a closure captures.
    render_prompt = functools.partial(
        render_row_conversation, tokenizer=tokenizer, add_generation_prompt=True, **knobs
    )
    tokenize_row = functools.partial(tokenize_training_row, tokenizer=tokenizer, eos_token_ids=eos_token_ids, **knobs)

    def process_row(row: dict[str, Any]) -> dict[str, Any]:
        if add_generation_prompt:
            # Generation prompts must not end with a tokenizer-appended turn terminator (Zaya-style).
            tokenized = tokenize_rendered(
                tokenizer, render_prompt(row), for_generation=True, truncation=False, padding=False
            )
        else:
            tokenized = tokenize_row(row)
        if len(tokenized["input_ids"]) > max_length:
            return none_example.copy()
        return tokenized

    return process_row


def build_text_row_processor(
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    text_field: str = "text",
    append_eos: bool = True,
    truncate: bool = True,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Row map for raw-text (pretraining) documents: a trailing EOS per document so packing keeps boundaries, the typed
    rejection row for empty text. ``truncate=False`` when packing leaves the overflow to the strategy (``bfd_split``
    keeps it, ``bfd`` drops it)."""
    eos_id = tokenizer.eos_token_id

    def process_row(row: dict[str, Any]) -> dict[str, Any]:
        text = row.get(text_field)
        if not isinstance(text, str) or not text:
            return build_rejection_row()
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        add_eos = append_eos and eos_id is not None
        # Reserve the last slot for EOS before appending, so an at/over-max_length doc keeps its EOS.
        if truncate:
            cap = max_length - 1 if add_eos else max_length
            if len(ids) > cap:
                ids = ids[:cap]
        if add_eos:
            ids = ids + [eos_id]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    return process_row
