"""What delimits an assistant turn: the response marker a completion starts at, the token ids that end
it, and the construction-time refusals of a marker no path could find."""

import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Any

from transformers import PreTrainedTokenizerBase

from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def reject_missing_response_marker(
    response_prompt_template: str | list[int] | None,
    train_on_completions_only: bool,
    subject: str,
) -> None:
    """Refuse completion-only masking with no assistant marker, where the pair is first accepted rather than
    at the first batch, after the model load and the dataset map; ``subject`` names the caller."""
    if train_on_completions_only and response_prompt_template is None:
        raise ValueError(
            f"{subject}: train_on_completions_only=True requires assistant_message_template (the "
            f"response marker the model's chat template renders) so completion-only labels can be "
            f"built. Pass the marker, or set train_on_completions_only=False to train on the full "
            f"sequence."
        )


def tokenize_response_template(
    response_prompt_template: str | list[int],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """A response template's token ids, refusing an empty one, which would match every position."""
    if isinstance(response_prompt_template, str):
        token_ids = tokenizer.encode(response_prompt_template, add_special_tokens=False)
    else:
        token_ids = list(response_prompt_template)

    if not token_ids:
        raise ValueError(
            f"response_prompt_template produced empty token IDs. "
            f"Template: {response_prompt_template!r}. "
            f"An empty template would match every position, training on all tokens."
        )
    return token_ids


def verify_marker_renders_in_chat_template(
    tokenizer: PreTrainedTokenizerBase, assistant_message_template: str
) -> None:
    """Raise when ``assistant_message_template`` occurs in no rendered probe, which would mask every label.

    Probes with and without ``thinking`` (gpt-oss harmony emits the marker only for a thinking message).
    """
    probe_conversations = (
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "thinking": "t"}],
    )
    verify_marker_renders(
        lambda conversation: tokenizer.apply_chat_template(conversation, tokenize=False),
        probe_conversations,
        assistant_message_template,
    )


def verify_marker_renders(
    render: Callable[[list[dict[str, Any]]], Any],
    probe_conversations: Sequence[list[dict[str, Any]]],
    assistant_message_template: str,
) -> list[str]:
    """Raise when ``assistant_message_template`` occurs in none of ``probe_conversations`` as ``render`` renders
    them, which would mask every label; returns the rendered probes, empty (with a warning) where none rendered.

    Probes the rendered text, since templates assemble the marker from pieces.
    """
    rendered_texts = []
    probe_failures = []
    for conversation in probe_conversations:
        try:
            rendered = render(conversation)
        except Exception as exc:  # template requires inputs this synthetic variant lacks; skip it
            probe_failures.append(f"{type(exc).__name__}: {exc}")
            continue
        # isinstance: only actual text can be probed; anything else is skipped.
        if isinstance(rendered, str):
            rendered_texts.append(rendered)
    if not rendered_texts and is_global_main_process():
        # Nothing rendered, so the check below never ran: silence would read as a passing probe.
        logger.warning(
            "Could not verify assistant_message_template %r against this tokenizer's chat "
            "template: no probe conversation rendered (%s). Completion-only masking is UNCHECKED "
            "for this run — a marker the template never emits masks EVERY label and trains zero "
            "tokens; watch for the per-batch all-masked warnings.",
            assistant_message_template,
            "; ".join(probe_failures) or "apply_chat_template returned a non-text value",
        )
    if rendered_texts and all(assistant_message_template not in text for text in rendered_texts):
        raise ValueError(
            f"assistant_message_template {assistant_message_template!r} does not occur in this "
            f"tokenizer's rendered chat template, so completion-only masking would mask EVERY "
            f"label and the run would train zero tokens at loss ~0. Set the marker to the exact "
            f"substring the template emits before assistant content (inspect "
            f"tokenizer.apply_chat_template on one conversation), or set "
            f"train_on_completions_only: false."
        )
    return rendered_texts


def reject_unusable_response_marker(
    tokenizer: PreTrainedTokenizerBase,
    response_prompt_template: str | None,
    train_on_completions_only: bool,
    subject: str,
) -> None:
    """A text completion-only collator's construction-time marker refusals: no marker at all, then, where
    the tokenizer carries a chat template, a marker the template never renders."""
    reject_missing_response_marker(response_prompt_template, train_on_completions_only, subject)
    if train_on_completions_only and getattr(tokenizer, "chat_template", None):
        verify_marker_renders_in_chat_template(tokenizer, response_prompt_template)


def resolve_eos_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    model_config: Any = None,
) -> frozenset[int]:
    """Every token id that ends an assistant turn: the tokenizer's eos and the model config's ``eos_token_id``
    (or its ``text_config``'s), where GLM-4 lists the role markers that delimit its turns. A distinct pad is
    no terminator, else a turn missing its own would train the model to predict the trailing pad.
    """
    ids: set[int] = set()

    def _collect(value: Any) -> None:
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, (list, tuple)):
            ids.update(v for v in value if isinstance(v, int))

    _collect(tokenizer.eos_token_id)
    if model_config is not None:
        _collect(getattr(model_config, "eos_token_id", None))
        text_config = getattr(model_config, "text_config", None)
        if text_config is not None:
            _collect(getattr(text_config, "eos_token_id", None))

    return frozenset(ids)


def ends_with_terminator(input_ids, tokenizer, eos_token_ids) -> bool:
    """Whether ``input_ids`` already ends a turn, ignoring the trailing whitespace templates close it with
    (``<terminator>\\n``), so no second terminator is appended; walks back to the first content token."""
    for token_id in reversed(input_ids):
        if token_id in eos_token_ids:
            return True
        if tokenizer.decode([token_id]).strip():
            return False
    return False


def lacks_terminator(input_ids, tokenizer, eos_token_ids) -> bool:
    """Whether ``input_ids`` need ``tokenizer.eos_token_id`` appended to close the turn: any of
    ``eos_token_ids`` (the tokenizer's eos when empty) already counts, since GLM-4 and Gemma close turns with
    a role marker the config lists; False when the tokenizer has no eos."""
    eos_id = tokenizer.eos_token_id
    return eos_id is not None and not ends_with_terminator(input_ids, tokenizer, eos_token_ids or {eos_id})


def warn_if_pad_equals_eos(tokenizer: PreTrainedTokenizerBase) -> None:
    """Warn when pad == eos: completion masking must then restore the turn-ending EOS from
    ``input_ids``, since the LM collator masks every pad position in labels."""
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        warnings.warn(
            "The pad_token_id and eos_token_id values of this tokenizer are identical. "
            "The toolkit collators restore turn-ending EOS labels from input_ids at real-token "
            "positions (attention_mask == 1), so EOS is still trained and padding stays masked — "
            "no action needed with these collators; third-party LM collators may still mask "
            "every EOS-valued label.",
            stacklevel=3,
        )
