"""Text row processors: the per-row chat render and tokenization a coordinated map runs over a text corpus."""

from collections.abc import Callable
from typing import Any

from transformers import PreTrainedTokenizer

from halo.data.pipeline.chat_render import render_conversation, tokenize_rendered
from halo.data.pipeline.conversation import chat_template_kwargs, reject_image_content, resolve_json_value
from halo.data.pipeline.rejected_rows import build_rejection_row
from halo.models.context_window import is_bounded_length

# The chat-render knobs a text run's script args carry, named as the text renderers take them.
_TEXT_RENDER_KNOBS = (
    "conversation_field",
    "system_prompt",
    "model_supports_system_role",
    "interleaved_thinking",
    "tools_field",
)


def text_render_kwargs(args) -> dict[str, Any]:
    """The run's :data:`_TEXT_RENDER_KNOBS`, passed to a text renderer and to its map-cache key from
    this one dict, so the key cannot miss a knob the render reads."""
    return {name: getattr(args, name) for name in _TEXT_RENDER_KNOBS}


def render_row_conversation(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    conversation_field: str = "messages",
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
    add_generation_prompt: bool = False,
    interleaved_thinking: bool = False,
    tools_field: str | None = None,
) -> str:
    """Chat-template the conversation at ``row[conversation_field]`` through :func:`render_conversation`.
    ``add_generation_prompt`` drops the final message: on the generation-eval path the trailing assistant turn is
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
    )


def prepare_generative_row(row, tokenizer, max_length, tools_field=None):
    """``row["prompt"]`` messages to a tokenized, padded generation prompt, capped at a bounded ``max_length``."""
    reject_image_content(row["prompt"], "prompt field 'prompt'")
    constructed_prompt = tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs(row, interleaved_thinking=False, tools_field=tools_field),
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
    conversation_field: str = "messages",
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
    add_generation_prompt: bool = False,
    use_padding: bool = True,
    interleaved_thinking: bool = False,
    tools_field: str | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the row map that chat-templates and tokenizes a conversation; a row over ``max_length`` becomes the
    typed rejection row. ``use_padding=False`` for padding-free collation."""
    none_example = build_rejection_row(tokenizer, truncation=True, padding=False, max_length=max_length)

    def process_row(row: dict[str, Any]) -> dict[str, Any]:
        constructed_prompt = render_row_conversation(
            row,
            tokenizer,
            conversation_field=conversation_field,
            system_prompt=system_prompt,
            model_supports_system_role=model_supports_system_role,
            add_generation_prompt=add_generation_prompt,
            interleaved_thinking=interleaved_thinking,
            tools_field=tools_field,
        )

        # Generation prompts must not end with a tokenizer-appended turn terminator (Zaya-style).
        tokenized = tokenize_rendered(
            tokenizer, constructed_prompt, for_generation=add_generation_prompt, truncation=False, padding=False
        )
        if len(tokenized["input_ids"]) > max_length:
            return none_example.copy()

        if not use_padding:  # the row fits and nothing pads, so a second pass would change nothing
            return tokenized

        return tokenize_rendered(
            tokenizer,
            constructed_prompt,
            for_generation=add_generation_prompt,
            truncation=True,
            padding=use_padding,
            max_length=max_length,
        )

    return process_row


def build_text_row_processor(
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    text_field: str = "text",
    append_eos: bool = True,
    truncate: bool = True,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the row map for raw-text (pretraining) documents: no chat template, a trailing EOS per document
    (``append_eos``) so packing keeps the boundaries, and the typed rejection row for empty text.
    ``truncate=False`` when packing leaves the overflow to the strategy (``bfd_split`` keeps it, ``bfd`` drops it).
    """
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
