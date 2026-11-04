"""Chat-template rendering and the tokenization of rendered text, with probe-based special-token ownership."""

import weakref
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizer

from halo.data.conversation import (
    chat_template_kwargs,
    fold_system_into_conversation,
    reject_image_content,
)

# Weak: a dropped tokenizer must not be kept alive.
_tokenizer_specials_cache: "weakref.WeakKeyDictionary[Any, _TokenizerSpecials]" = weakref.WeakKeyDictionary()


def _strip_leading_bos(text: str, tokenizer: PreTrainedTokenizer) -> str:
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        return text[len(tokenizer.bos_token) :]
    return text


@dataclass(frozen=True)
class _TokenizerSpecials:
    """What ``add_special_tokens=True`` adds around plain text; ``trailing_special_text`` decodes
    ``trailing_special_ids``."""

    adds_leading_bos: bool
    trailing_special_ids: tuple[int, ...]
    trailing_special_text: str = ""


def probe_tokenizer_specials(tokenizer: PreTrainedTokenizer) -> _TokenizerSpecials:
    """Probed once per tokenizer on the empty string: a leading ``bos_token_id`` means the post-processor owns BOS, the
    rest is the suffix. ``tokenizer.bos_token`` alone misleads: gpt-oss/Bailing name a BOS their post-processor never
    emits, and Zaya's appends ``<|im_end|>`` instead of prepending one."""
    cached = _tokenizer_specials_cache.get(tokenizer)
    if cached is not None:
        return cached
    ids = list(tokenizer("", add_special_tokens=True)["input_ids"])
    bos_id = getattr(tokenizer, "bos_token_id", None)
    adds_leading_bos = bool(ids) and bos_id is not None and ids[0] == bos_id
    trailing = tuple(ids[1:] if adds_leading_bos else ids)
    specials = _TokenizerSpecials(
        adds_leading_bos=adds_leading_bos,
        trailing_special_ids=trailing,
        trailing_special_text=tokenizer.decode(list(trailing)) if trailing else "",
    )
    _tokenizer_specials_cache[tokenizer] = specials
    return specials


def lacks_emitted_bos(ids: list[int], tokenizer: PreTrainedTokenizer) -> bool:
    """Whether ``ids`` lack the BOS the post-processor emits; prepending a nominal BOS it never emits would train on a
    token the policy never sees."""
    bos_id = tokenizer.bos_token_id
    return (
        bos_id is not None and probe_tokenizer_specials(tokenizer).adds_leading_bos and (not ids or ids[0] != bos_id)
    )


def _render_terminator_survived(encoded: Any, tokenizer: PreTrainedTokenizer, specials: _TokenizerSpecials) -> bool:
    """Whether the render's own terminator sits just before the appended copy, at most whitespace apart, which makes
    that copy a duplicate. Length cannot decide it: a render filling ``max_length`` measures the same as a cut one."""
    ids = encoded["input_ids"]
    n = len(specials.trailing_special_ids)
    end = len(ids) - n
    while end >= n:
        if tuple(ids[end - n : end]) == specials.trailing_special_ids:
            return True
        if tokenizer.decode(ids[end - 1 : end]).strip():
            return False  # a content token sits here, so the render's terminator did not survive
        end -= 1  # template whitespace between the terminator and the appended copy
    return False


def _strip_trailing_special_ids(encoded: Any, trailing_special_ids: tuple[int, ...]) -> None:
    """In place, on every list column as long as ``input_ids``."""
    ids = encoded["input_ids"]
    n_trailing = len(trailing_special_ids)
    orig_len = len(ids)
    if orig_len < n_trailing or tuple(ids[-n_trailing:]) != trailing_special_ids:
        return  # truncation already cut the suffix
    keep = orig_len - n_trailing
    for key in list(encoded.keys()):
        value = encoded[key]
        if isinstance(value, list) and len(value) == orig_len:
            encoded[key] = value[:keep]


def render_conversation(
    tokenizer: PreTrainedTokenizer,
    conversation: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    conversation_field: str,
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
    add_generation_prompt: bool = False,
    interleaved_thinking: bool = False,
    tools_field: str | None = None,
) -> str:
    """Chat-template one text conversation; ``conversation`` comes apart from ``row`` because the SDPG collator builds
    its own. BOS stays as rendered: :func:`tokenize_rendered` owns the specials, and gemma-4's template emits a BOS its
    post-processor does not add."""
    reject_image_content(conversation, f"conversation_field '{conversation_field}'")
    conversation = fold_system_into_conversation(conversation, system_prompt, model_supports_system_role)
    return tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **chat_template_kwargs(row, interleaved_thinking, tools_field),
    )


def render_generation_prompt(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, Any]],
    *,
    max_prompt_length: int | None = None,
    **template_kwargs,
) -> str | None:
    """A GRPO generation prompt, or ``None`` past ``max_prompt_length``: cutting it would remove the question the
    verifier grades. The rendered BOS is stripped only where the post-processor prepends its own."""
    reject_image_content(messages, "generation prompt")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
    if probe_tokenizer_specials(tokenizer).adds_leading_bos:
        text = _strip_leading_bos(text, tokenizer)
    if max_prompt_length is not None:
        # The row's own generation-prompt tokenization, so the specials it owns are counted exactly.
        tokenized = tokenize_rendered(tokenizer, text, for_generation=True, truncation=False, padding=False)
        if len(tokenized["input_ids"]) > max_prompt_length:
            return None
    return text


def tokenize_rendered(tokenizer: PreTrainedTokenizer, text: str, *, for_generation: bool = False, **tokenizer_kwargs):
    """Tokenize so each special appears once: a post-processor prepending BOS gets the text with its rendered BOS
    stripped, any other keeps the text verbatim and adds specials only when it appends a suffix (Zaya's EOS).
    ``for_generation`` strips the appended suffix so a prompt never ends on a terminator; a training row drops it only
    as a duplicate of a render terminator that survived truncation."""
    specials = probe_tokenizer_specials(tokenizer)
    if specials.adds_leading_bos:
        encoded = tokenizer(_strip_leading_bos(text, tokenizer), add_special_tokens=True, **tokenizer_kwargs)
    else:
        add_special_tokens = bool(specials.trailing_special_ids)
        encoded = tokenizer(text, add_special_tokens=add_special_tokens, **tokenizer_kwargs)
    render_terminated = (
        bool(specials.trailing_special_ids)
        and text.rstrip().endswith(specials.trailing_special_text)
        and _render_terminator_survived(encoded, tokenizer, specials)
    )
    if specials.trailing_special_ids and (for_generation or render_terminated):
        _strip_trailing_special_ids(encoded, specials.trailing_special_ids)
    return encoded
