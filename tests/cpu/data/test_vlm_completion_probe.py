#!/usr/bin/env python
"""CPU tests for the VLM collators' construction-time completion probe: completion-only masking that would train no
token is refused where the collator is built, before the model load and the dataset map.

Two setups train nothing: a marker the processor's chat template never renders, and assistant turns that end in no
turn terminator, as under a Gemma template borrowed for PaliGemma, whose turns close with ``<end_of_turn>`` while the
model's eos is ``<eos>``. The probe masks under the collator's own span policy, so the self-distillation collator,
whose policy runs a terminator-less span to the row end, is not refused.

Run: ``python tests/cpu/data/test_vlm_completion_probe.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest

from halo.data.collators.self_distill import SelfDistillVLMDataCollator
from halo.data.collators.vlm import VLMDataCollator
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from tests.common.prefix_lm import digit_row, tiny_paligemma_config, tiny_paligemma_processor

# Gemma's turns over the tiny vocabulary: each closes with ``<end_of_turn>``, never the tokenizer's ``<eos>``.
_GEMMA_TURN_TEMPLATE = (
    "{% for m in messages %}<start_of_turn>{{ m['role'] }} {% for p in m['content'] %}"
    "{% if p['type'] == 'image' %}<image>{% else %}{{ p['text'] }} {% endif %}{% endfor %}<end_of_turn> {% endfor %}"
)
# Turns that end at the next turn's role marker, which the model config lists as an eos id (GLM-4).
_ROLE_DELIMITED_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {% for p in m['content'] %}"
    "{% if p['type'] == 'image' %}<image>{% else %}{{ p['text'] }} {% endif %}{% endfor %}{% endfor %}"
)


def _processor(template: str, special_tokens: list[str]):
    processor = tiny_paligemma_processor()
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}, replace_extra_special_tokens=False
    )
    processor.chat_template = template
    return processor


def _gemma_turn_processor():
    return _processor(_GEMMA_TURN_TEMPLATE, ["<start_of_turn>", "<end_of_turn>"])


def _completion_collator(processor, model_config=None, marker: str = "assistant", cls=VLMDataCollator, **kwargs):
    return cls(
        processor,
        processor.tokenizer,
        256,
        train_on_completions_only=True,
        response_prompt_template=marker,
        model_config=tiny_paligemma_config(processor) if model_config is None else model_config,
        **kwargs,
    )


def test_turns_that_end_in_no_terminator_are_refused_at_construction():
    processor = _gemma_turn_processor()
    with pytest.raises(ValueError, match=r"no assistant turn .* ends in a turn terminator \['<eos>'\]"):
        _completion_collator(processor)


def test_naming_the_turn_end_token_as_eos_trains_the_answer():
    """Anti-vacuity: the refusal is the terminator's doing. With ``eos_token`` set to the token the template closes a
    turn with, the same collator builds and trains each answer through its ``<end_of_turn>``."""
    processor = _gemma_turn_processor()
    processor.tokenizer.eos_token = "<end_of_turn>"
    batch = _completion_collator(processor)([digit_row()])
    trained = processor.tokenizer.convert_ids_to_tokens(batch["labels"][0][batch["labels"][0] != LABEL_IGNORE_INDEX])
    assert trained == ["assistant", "The", "digit", "is", "3", ".", "<end_of_turn>"]


def test_a_marker_the_template_never_renders_is_refused_at_construction():
    processor = tiny_paligemma_processor()
    with pytest.raises(ValueError, match="does not occur in this tokenizer's rendered chat template"):
        _completion_collator(processor, marker="model")


def test_a_turn_closed_by_the_next_role_marker_passes():
    """Only the last assistant turn of a role-delimited template lacks a terminator; the probe's first closes."""
    processor = _processor(_ROLE_DELIMITED_TEMPLATE, ["<|user|>", "<|assistant|>"])
    config = tiny_paligemma_config(processor)
    config.eos_token_id = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<|user|>")]
    _completion_collator(processor, config, marker="<|assistant|>")


def test_the_self_distill_policy_is_not_refused_for_a_terminator_less_turn():
    processor = _gemma_turn_processor()
    _completion_collator(
        processor,
        cls=SelfDistillVLMDataCollator,
        hint_template="{answer}",
        answer_field="answer",
        solution_field=None,
        confidence_power=1.0,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
