"""Chat-template rendering and the tokenization of rendered text, with probe-based special-token ownership."""

import json
import logging
import weakref
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizer

from halo.data.conversation import (
    chat_template_kwargs,
    fold_system_into_conversation,
    has_wire_form_arguments,
    parse_tool_call_arguments,
    reject_image_content,
    resolve_row_tools,
)
from halo.data.turn_markers import lacks_terminator

logger = logging.getLogger(__name__)

# Weak: a dropped tokenizer must not be kept alive.
_tokenizer_specials_cache: "weakref.WeakKeyDictionary[Any, _TokenizerSpecials]" = weakref.WeakKeyDictionary()
# Each probe's result beside the text of the template it was probed on, as a run sets its template on a loaded
# tokenizer.
_wire_form_support_cache: "weakref.WeakKeyDictionary[Any, tuple[str, dict[bool, bool]]]" = weakref.WeakKeyDictionary()
_turn_closer_cache: "weakref.WeakKeyDictionary[Any, tuple[str, dict[tuple, int | None]]]" = weakref.WeakKeyDictionary()


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


def _template_probe_results(cache: weakref.WeakKeyDictionary, owner: Any) -> dict:
    """``owner``'s probe results in ``cache`` under its current chat template, emptied when the template changed."""
    template = str(getattr(owner, "chat_template", None))
    cached = cache.get(owner)
    if cached is None or cached[0] != template:
        cached = cache[owner] = (template, {})
    return cached[1]


def _render_probe(template_owner: Any, conversation: list[dict[str, Any]], *, with_tools: bool) -> str | None:
    """``conversation`` as the chat template of ``template_owner`` writes it, declaring the probe tool under
    ``with_tools``; ``None`` where the template raises."""
    tool = {
        "type": "function",
        "function": {
            "name": "probe",
            "description": "Probe.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string", "description": "Probe."}},
                "required": ["value"],
            },
        },
    }
    try:
        return template_owner.apply_chat_template(
            conversation, tokenize=False, **({"tools": [tool]} if with_tools else {})
        )
    except Exception as exc:  # a template's own raise on a turn it does not render
        logger.debug(f"Chat-template probe render failed: {exc}")
        return None


def _probe_tool_round(arguments: Any) -> list[dict[str, Any]]:
    """An assistant turn calling the probe tool once with ``arguments``, then the tool's reply."""
    call = {"type": "function", "id": "call_0", "function": {"name": "probe", "arguments": arguments}}
    return [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_0", "content": "r"},
    ]


def render_tool_round_probe(template_owner: Any, arguments: Any, *, with_tools: bool, answer: str = "a") -> str | None:
    """The chat template's text for a user turn, one tool call carrying ``arguments``, its reply and the assistant's
    ``answer``, declaring the tool under ``with_tools``; ``None`` where it raises. ``template_owner`` is the tokenizer
    or processor whose template renders the rows."""
    conversation = [
        {"role": "user", "content": "q"},
        *_probe_tool_round(arguments),
        {"role": "assistant", "content": answer},
    ]
    return _render_probe(template_owner, conversation, with_tools=with_tools)


def supports_wire_form_arguments(tokenizer: PreTrainedTokenizer, *, with_tools: bool) -> bool:
    """Whether the chat template writes a tool call whose ``arguments`` is JSON text (the OpenAI wire form) as that
    text, or as it writes the mapping. False where it renders the mapping and raises on the text, as a template writing
    one block per argument does (Qwen3.5/3.6, Gemma 4, GLM), or re-encodes the text as a JSON string, as one applying
    ``tojson`` to it does; false too where it raises on both probes, which then tell nothing (Mistral v0.3's checks a
    call id's length): such a template takes the mapping a serving engine hands it. Probed once per tokenizer, template
    and ``with_tools``: a tokenizer holding named templates renders a conversation that declares tools with its
    ``tool_use`` one."""
    verdicts = _template_probe_results(_wire_form_support_cache, tokenizer)
    if with_tools not in verdicts:
        arguments = {"value": "1"}
        # Compact, so a template writing the text as stored is told from one writing the mapping.
        wire = json.dumps(arguments, separators=(",", ":"))
        as_text = render_tool_round_probe(tokenizer, wire, with_tools=with_tools)
        as_mapping = render_tool_round_probe(tokenizer, arguments, with_tools=with_tools)
        verdicts[with_tools] = as_text is not None and (as_mapping is None or as_text == as_mapping or wire in as_text)
    return verdicts[with_tools]


def resolve_template_conversation(tokenizer: PreTrainedTokenizer, conversation: Any, *, with_tools: bool) -> Any:
    """``conversation`` as the tokenizer's chat template takes it: wire-form tool-call arguments parsed into mappings
    where the template does not write the text as stored, which raises on arguments that are not a JSON object. A
    template that writes the text keeps the row's own spelling. ``with_tools`` is whether the render declares tools."""
    if not has_wire_form_arguments(conversation) or supports_wire_form_arguments(tokenizer, with_tools=with_tools):
        return conversation
    return parse_tool_call_arguments(conversation)


def render_chat_template(
    tokenizer: PreTrainedTokenizer, conversation: Any, *, add_generation_prompt: bool = False, **template_kwargs
) -> str:
    """The chat template's text for ``conversation`` under ``template_kwargs``, its tool calls' arguments as the
    template takes them (:func:`resolve_template_conversation`)."""
    return tokenizer.apply_chat_template(
        resolve_template_conversation(tokenizer, conversation, with_tools=template_kwargs.get("tools") is not None),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )


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
    reasoning_effort_field: str | None = None,
) -> str:
    """Chat-template one text conversation; ``conversation`` comes apart from ``row`` because the SDPG collator builds
    its own. BOS stays as rendered: :func:`tokenize_rendered` owns the specials, and gemma-4's template emits a BOS its
    post-processor does not add."""
    reject_image_content(conversation, f"conversation_field '{conversation_field}'")
    conversation = fold_system_into_conversation(conversation, system_prompt, model_supports_system_role)
    return render_chat_template(
        tokenizer,
        conversation,
        add_generation_prompt=add_generation_prompt,
        **chat_template_kwargs(row, interleaved_thinking, tools_field, reasoning_effort_field),
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
    text = render_chat_template(tokenizer, messages, add_generation_prompt=True, **template_kwargs)
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


def has_open_final_assistant_turn(
    input_ids, conversation: list[dict[str, Any]], tokenizer: PreTrainedTokenizer, eos_token_ids: frozenset[int]
) -> bool:
    """Whether a training row ends on an assistant turn its template leaves without a terminator. A template that ends
    a turn only at the next role marker (GLM's native ones) closes no last turn, whose span would then never end and
    train no stop. ``eos_token_ids`` are the model's terminators."""
    return (
        bool(input_ids)
        and bool(conversation)
        and conversation[-1].get("role") == "assistant"
        and lacks_terminator(input_ids, tokenizer, eos_token_ids)
    )


def _probe_turn_closer(
    template_owner: Any,
    tokenizer: PreTrainedTokenizer,
    eos_token_ids: frozenset[int],
    *,
    calls_tool: bool,
    with_tools: bool,
) -> int | None:
    """The token the chat template writes right after an assistant turn that calls a tool (``calls_tool``) or answers,
    once a tool reply or a user turn follows it, where that token is one of ``eos_token_ids``; ``None`` where a render
    raises, the turn renders otherwise once followed, or the template writes another token first."""
    user, answer = {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}
    turn, follow_up = _probe_tool_round({"value": "1"}) if calls_tool else (answer, user)
    closed = _render_probe(template_owner, [user, turn], with_tools=with_tools)
    continued = _render_probe(template_owner, [user, turn, follow_up], with_tools=with_tools)
    if closed is None or continued is None:
        return None
    closed_ids = tokenizer(closed, add_special_tokens=False)["input_ids"]
    continued_ids = tokenizer(continued, add_special_tokens=False)["input_ids"]
    end = len(closed_ids)
    if len(continued_ids) <= end or continued_ids[:end] != closed_ids:
        return None
    return continued_ids[end] if continued_ids[end] in eos_token_ids else None


def resolve_final_turn_closer(
    tokenizer: PreTrainedTokenizer,
    message: dict[str, Any],
    eos_token_ids: frozenset[int],
    *,
    with_tools: bool,
    template_owner: Any = None,
) -> int:
    """The token closing a training row's last assistant turn ``message`` where the template leaves it open: the one
    the chat template of ``template_owner`` (the tokenizer by default; a VLM run's processor) writes after such a turn,
    a tool call or an answer, when the conversation goes on (GLM's ``<|observation|>`` after a call, ``<|user|>`` after
    an answer), where that token is one of ``eos_token_ids``; else the tokenizer's eos. Probed once per template,
    terminator set, kind of turn and ``with_tools``, whether the row declares tools."""
    owner = tokenizer if template_owner is None else template_owner
    closers = _template_probe_results(_turn_closer_cache, owner)
    calls_tool = bool(message.get("tool_calls"))
    key = (frozenset(eos_token_ids), calls_tool, with_tools)
    if key not in closers:
        closers[key] = _probe_turn_closer(owner, tokenizer, key[0], calls_tool=calls_tool, with_tools=with_tools)
    return tokenizer.eos_token_id if closers[key] is None else closers[key]


def resolve_final_turn_closers(
    tokenizer: PreTrainedTokenizer, eos_token_ids: frozenset[int], *, template_owner: Any = None
) -> dict[str, int]:
    """:func:`resolve_final_turn_closer` for each kind of last turn, keyed ``answer`` or ``call``, suffixed
    ``_with_tools`` where the row declares tools: what the training renders close an open last turn with, which a chat
    bake records and a run is checked against."""
    turns = {"answer": {"role": "assistant", "content": "a"}, "call": _probe_tool_round({"value": "1"})[0]}
    return {
        f"{kind}_with_tools" if with_tools else kind: resolve_final_turn_closer(
            tokenizer, turn, eos_token_ids, with_tools=with_tools, template_owner=template_owner
        )
        for kind, turn in turns.items()
        for with_tools in (False, True)
    }


def close_final_assistant_turn(
    encoded,
    conversation: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    eos_token_ids: frozenset[int],
    *,
    with_tools: bool,
) -> None:
    """Append the closing token :func:`resolve_final_turn_closer` names, in place, to a training row
    :func:`has_open_final_assistant_turn` finds open; ``with_tools`` is whether the row declares tools."""
    input_ids = encoded["input_ids"]
    if not has_open_final_assistant_turn(input_ids, conversation, tokenizer, eos_token_ids):
        return
    closer = resolve_final_turn_closer(tokenizer, conversation[-1], eos_token_ids, with_tools=with_tools)
    width = len(input_ids)
    for name, column in encoded.items():
        if isinstance(column, list) and len(column) == width:
            column.append(closer if name == "input_ids" else column[-1])


def tokenize_training_conversation(
    tokenizer: PreTrainedTokenizer,
    conversation: list[dict[str, Any]],
    row: dict[str, Any],
    eos_token_ids: frozenset[int],
    **render_kwargs,
):
    """``conversation`` rendered by :func:`render_conversation` under ``render_kwargs`` and tokenized as an untruncated
    training row, the last assistant turn closed where the template leaves it open (:func:`close_final_assistant_turn`).
    ``eos_token_ids`` are the model's terminators."""
    rendered = render_conversation(tokenizer, conversation, row, **render_kwargs)
    encoded = tokenize_rendered(tokenizer, rendered, truncation=False, padding=False)
    with_tools = resolve_row_tools(row, render_kwargs.get("tools_field")) is not None
    close_final_assistant_turn(encoded, conversation, tokenizer, eos_token_ids, with_tools=with_tools)
    return encoded
