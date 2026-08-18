"""Chat-template rendering and the tokenization of rendered text: probe-based special-token ownership.

A leaf over :mod:`src.data.pipeline.conversation`, so the row maps, the offline tokenizer and the
self-distillation collator reach the render seam without pulling in the coordinated-map machinery.
"""

import weakref
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizer

from src.data.pipeline.conversation import (
    chat_template_kwargs,
    fold_system_into_conversation,
    reject_image_content,
)

# Probe results cached per tokenizer instance (weak: a dropped tokenizer must not be kept alive).
_TOKENIZER_SPECIALS_CACHE: "weakref.WeakKeyDictionary[Any, TokenizerSpecials]" = weakref.WeakKeyDictionary()


def _strip_leading_bos(text: str, tokenizer: PreTrainedTokenizer) -> str:
    """Strip a leading BOS token (the tokenizer re-adds it on tokenization)."""
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        return text[len(tokenizer.bos_token) :]
    return text


@dataclass(frozen=True)
class TokenizerSpecials:
    """What a tokenizer's ``add_special_tokens=True`` post-processor adds around plain text.

    ``trailing_special_text`` is the decoded form of ``trailing_special_ids`` (empty when nothing
    is appended), used to detect renders that already terminate with the appended sequence.
    """

    adds_leading_bos: bool
    trailing_special_ids: tuple[int, ...]
    trailing_special_text: str = ""


def probe_tokenizer_specials(tokenizer: PreTrainedTokenizer) -> TokenizerSpecials:
    """Probe (once per tokenizer instance) what ``add_special_tokens=True`` actually adds.

    ``tokenizer.bos_token`` alone is unreliable: gpt-oss/Bailing define a nominal ``bos_token`` their
    post-processor never emits, while Zaya's post-processor appends a trailing ``<|im_end|>`` instead
    of prepending BOS. Tokenizing the empty string with ``add_special_tokens=True`` reveals the real
    contract: a leading ``bos_token_id`` means the post-processor owns BOS; everything after it (or
    the whole output when there is no leading BOS) is the appended trailing-special suffix.
    """
    cached = _TOKENIZER_SPECIALS_CACHE.get(tokenizer)
    if cached is not None:
        return cached
    ids = list(tokenizer("", add_special_tokens=True)["input_ids"])
    bos_id = getattr(tokenizer, "bos_token_id", None)
    adds_leading_bos = bool(ids) and bos_id is not None and ids[0] == bos_id
    trailing = tuple(ids[1:] if adds_leading_bos else ids)
    specials = TokenizerSpecials(
        adds_leading_bos=adds_leading_bos,
        trailing_special_ids=trailing,
        trailing_special_text=tokenizer.decode(list(trailing)) if trailing else "",
    )
    _TOKENIZER_SPECIALS_CACHE[tokenizer] = specials
    return specials


def _render_terminator_survived(encoded: Any, tokenizer: PreTrainedTokenizer, specials: TokenizerSpecials) -> bool:
    """Whether the RENDER's own turn terminator is still present just before the tokenizer-appended
    copy, separated at most by whitespace (templates close the final turn with ``<terminator>\\n``).

    Decides whether the appended copy is a duplicate to strip or the row's ONLY terminator. A
    length-vs-``max_length`` comparison cannot: a render that exactly fills the budget is never cut,
    yet measures the same as a cut one.
    """
    ids = encoded["input_ids"]
    n = len(specials.trailing_special_ids)
    end = len(ids) - n  # position just before the tokenizer-appended copy
    while end >= n:
        if tuple(ids[end - n : end]) == specials.trailing_special_ids:
            return True
        if tokenizer.decode(ids[end - 1 : end]).strip():
            return False  # a content token sits here, so the render's terminator did not survive
        end -= 1  # template whitespace between the terminator and the appended copy
    return False


def _strip_trailing_special_ids(encoded: Any, trailing_special_ids: tuple[int, ...]) -> None:
    """Trim the tokenizer-appended trailing specials from every same-length list column in place."""
    ids = encoded["input_ids"]
    n_trailing = len(trailing_special_ids)
    orig_len = len(ids)
    if orig_len < n_trailing or tuple(ids[-n_trailing:]) != trailing_special_ids:
        return  # truncation already cut the suffix — nothing to strip
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
    """Chat-template one TEXT conversation — the single home for the render the text paths share.

    ``row`` supplies the per-row template kwargs (interleaved thinking, tools); ``conversation`` is
    passed separately because the SDPG collator builds its own histories. BOS is left exactly as the
    template rendered it — :func:`tokenize_rendered` owns the specials contract, and a strip here
    would delete BOS for families whose template emits it while their post-processor adds nothing
    (gemma-4).
    """
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
    """Chat-template ``messages`` as a generation prompt, or ``None`` when it busts the budget.

    The prompt stage the GRPO family shares (RLVR, environmental). BOS ownership follows the probed
    contract of :func:`tokenize_rendered`: the rendered leading BOS is stripped ONLY when the
    post-processor prepends one of its own, else gemma-4-style templates lose theirs for good.
    ``max_prompt_length`` is a FILTER, not a truncation — cutting a prompt removes the very question
    the verifier grades against — so an over-budget prompt returns ``None``; the default disables it.
    GRPO is text-only, so an image part is refused per row (:func:`reject_image_content`).
    """
    reject_image_content(messages, "generation prompt")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
    if probe_tokenizer_specials(tokenizer).adds_leading_bos:
        text = _strip_leading_bos(text, tokenizer)
    if max_prompt_length is not None:
        # Measure with the generation-prompt tokenization the row will actually get, so the budget is
        # not off by the specials this render does or does not own.
        tokenized = tokenize_rendered(tokenizer, text, for_generation=True, truncation=False, padding=False)
        if len(tokenized["input_ids"]) > max_prompt_length:
            return None
    return text


def tokenize_rendered(tokenizer: PreTrainedTokenizer, text: str, *, for_generation: bool = False, **tokenizer_kwargs):
    """Tokenize rendered chat-template text so specials appear exactly once, owned by whoever
    really emits them.

    Templates disagree about BOS — some emit it, some rely on ``add_special_tokens``, gemma-4 emits it
    while its post-processor adds nothing — so :func:`probe_tokenizer_specials` decides:

    - post-processor prepends BOS → strip any rendered BOS, tokenize with ``add_special_tokens=True``;
    - it does not → keep the text verbatim (the template owns BOS), with ``add_special_tokens=True``
      only when the post-processor appends trailing specials (Zaya-style trailing EOS).

    ``for_generation=True`` also strips those trailing specials, so a generation prompt never ends on
    a turn terminator. A training row keeps them, EXCEPT when the render already ends with the same
    sequence (modulo whitespace) and it SURVIVED into the ids (:func:`_render_terminator_survived`):
    then the appended copy is a duplicate. Truncation that cut the render's own terminator skips the
    dedup — there the appended copy is the row's only one.
    """
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
