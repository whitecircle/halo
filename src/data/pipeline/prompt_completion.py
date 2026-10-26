"""One prompt and its completions, rendered and tokenized the same way wherever a trainer stores them as text: the
preference pipeline (SMPO; DPO, KTO and reward dataset prep share the render) and offline GRPO."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from transformers import PreTrainedTokenizerBase

from src.data.pipeline.chat_render import lacks_emitted_bos
from src.data.turn_markers import lacks_terminator
from src.models.context_window import is_bounded_length


def render_prompt_completions(
    render: Callable[[list[dict[str, Any]]], str],
    prompt_messages: list[dict[str, Any]],
    completions: Iterable[tuple[str, list[dict[str, Any]]]],
) -> tuple[str, list[str]]:
    """``render(prompt_messages)``, and each ``(field, messages)`` completion as ``render(prompt + completion)``
    minus that prefix. A completion never renders alone: strict templates (Qwen3.5) refuse assistant-only message
    lists, and a BOS-emitting template would put a BOS mid-sequence."""
    prompt_text = render(prompt_messages)
    completion_texts = []
    for field, messages in completions:
        full_text = render(prompt_messages + messages)
        if not full_text.startswith(prompt_text):
            raise ValueError(
                f"Chat template broke the prefix invariant: template(prompt + {field}) does not start "
                "with template(prompt), so the completion cannot be split off. "
                f"prompt head: {prompt_text[:120]!r} / full head: {full_text[:120]!r}"
            )
        completion_texts.append(full_text[len(prompt_text) :])
    return prompt_text, completion_texts


def tokenize_prompt_completions(
    prompt: str,
    completions: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    truncation_mode: str,
    joint_boundary: bool,
    eos_token_ids: frozenset[int],
) -> tuple[list[int], list[list[int]]]:
    """Token ids of a rendered prompt and its completions under the two length budgets (unset or non-positive =
    no cap). ``joint_boundary`` tokenizes each ``prompt + completion`` whole and splits all at one prompt length,
    one token earlier when a token merged across the boundary on any side, so a preference pair conditions on one
    prompt; otherwise each completion tokenizes alone, as a stored rollout was sampled. BOS goes where the
    post-processor would emit one, before the prompt is cut by ``truncation_mode``; EOS closes a completion that
    lacks a terminator, before the plain cut at ``max_completion_length``: an EOS where the text went on would teach
    stopping there."""
    completion_cap = max_completion_length if is_bounded_length(max_completion_length) else None
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if joint_boundary:
        full_ids = [tokenizer(prompt + text, add_special_tokens=False)["input_ids"] for text in completions]
        split = len(prompt_ids)
        if any(ids[:split] != prompt_ids for ids in full_ids):
            split -= 1  # a non-merging side still matches the prompt up to here
        prompt_ids, completion_ids = prompt_ids[:split], [ids[split:] for ids in full_ids]
    else:
        # Tokenizer-side, so truncation_side holds, and only under a real cap: HF cuts `max_length=None` at
        # model_max_length.
        completion_ids = [
            tokenizer(
                text, truncation=completion_cap is not None, max_length=completion_cap, add_special_tokens=False
            )["input_ids"]
            for text in completions
        ]

    if lacks_emitted_bos(prompt_ids, tokenizer):
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    if is_bounded_length(max_prompt_length):
        keep_start = truncation_mode == "keep_start"
        prompt_ids = prompt_ids[:max_prompt_length] if keep_start else prompt_ids[-max_prompt_length:]
    eos = [tokenizer.eos_token_id]
    completion_ids = [ids + eos if lacks_terminator(ids, tokenizer, eos_token_ids) else ids for ids in completion_ids]
    return prompt_ids, [ids[:completion_cap] for ids in completion_ids]
