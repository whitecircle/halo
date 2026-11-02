#!/usr/bin/env python
"""CPU tests for the prefix-LM (PaliGemma) batches: every trained token lands in the causal suffix of
``token_type_ids``, image and prompt in the bidirectional prefix, a batch or template the processor would corrupt is
refused, and the newline the processor appends trains nowhere.

PaliGemma's processor marks no suffix in a rendered conversation (all 0: one bidirectional prefix), puts ``<bos>``
after each row's last ``<image>`` (into the text of a row without one) and appends ``\\n`` to each row.

Run: ``python tests/cpu/data/test_prefix_lm_collators.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
from transformers import Gemma3Config

from halo.data.collators.self_distill import SelfDistillVLMDataCollator
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models.attention.prefix_lm import has_prefix_lm_attention, prefix_lm_token_type_ids
from tests.common.prefix_lm import (
    digit_row,
    first_scored_position,
    paligemma_collator,
    text_row,
    tiny_paligemma_config,
    tiny_paligemma_processor,
)

IGNORE = LABEL_IGNORE_INDEX


def _assert_suffix_from_first_scored(labels: torch.Tensor, token_type_ids: torch.Tensor) -> None:
    start = first_scored_position(labels)
    assert token_type_ids[:start].eq(0).all() and token_type_ids[start:].eq(1).all()


def test_token_type_ids_open_the_suffix_at_the_first_scored_label():
    labels = torch.tensor(
        [
            [5, IGNORE, IGNORE, 7, 8, IGNORE],
            [IGNORE] * 6,
            [IGNORE, 3, IGNORE, 4, IGNORE, IGNORE],
        ]
    )
    assert prefix_lm_token_type_ids(labels).tolist() == [[0, 0, 0, 1, 1, 1], [0] * 6, [0, 1, 1, 1, 1, 1]]


@pytest.mark.parametrize("completions_only", [True, False], ids=["completions-only", "all-tokens"])
def test_the_vlm_collator_puts_every_trained_token_in_the_suffix(completions_only):
    processor = tiny_paligemma_processor()
    batch = paligemma_collator(processor, tiny_paligemma_config(processor), completions_only=completions_only)(
        [digit_row(3), digit_row(7)]
    )
    for row in range(2):
        _assert_suffix_from_first_scored(batch["labels"][row], batch["token_type_ids"][row])
    # The image placeholders sit in the bidirectional prefix in both modes.
    assert batch["token_type_ids"][batch["input_ids"] == processor.image_token_id].eq(0).all()


def test_a_model_that_is_no_prefix_lm_keeps_the_processor_token_type_ids():
    """Anti-vacuity: the rewrite is keyed on the model's family, so Gemma 3's image markers are never replaced. The
    processor's own ``token_type_ids`` for a rendered conversation are all 0."""
    processor = tiny_paligemma_processor()
    assert not has_prefix_lm_attention(Gemma3Config())
    batch = paligemma_collator(processor, Gemma3Config(), completions_only=True)([digit_row()])
    assert batch["token_type_ids"].eq(0).all()


def test_the_self_distill_teacher_branch_gets_its_own_suffix():
    processor = tiny_paligemma_processor()
    collator = paligemma_collator(
        processor,
        tiny_paligemma_config(processor),
        completions_only=True,
        cls=SelfDistillVLMDataCollator,
        hint_template="{answer}",
        answer_field="answer",
        solution_field=None,
        confidence_power=1.0,
    )
    batch = collator([{**digit_row(3), "answer": "3"}])
    for prefix in ("", "teacher_"):
        _assert_suffix_from_first_scored(batch[f"{prefix}labels"][0], batch[f"{prefix}token_type_ids"][0])
    assert batch["teacher_input_ids"].shape[1] > batch["input_ids"].shape[1], "the hint lengthens the teacher"


def test_a_row_without_an_image_is_refused_on_a_prefix_lm():
    processor = tiny_paligemma_processor()
    collator = paligemma_collator(processor, tiny_paligemma_config(processor), completions_only=False)
    with pytest.raises(ValueError, match="without an image"):
        collator([digit_row(), text_row()])


def test_the_processor_splices_bos_into_a_text_row_of_an_image_batch():
    """Anti-vacuity: the refusal guards a real corruption. On a model that is no prefix LM the same mixed batch
    collates, and the text row's ``What`` comes out split around a spliced ``<bos>``."""
    processor = tiny_paligemma_processor()
    batch = paligemma_collator(processor, Gemma3Config(), completions_only=False)([digit_row(), text_row()])
    tokens = processor.tokenizer.convert_ids_to_tokens(batch["input_ids"][1].tolist())
    assert tokens[:3] == ["user", "<unk>", "<bos>"]


def _bos_rendering_processor():
    """The tiny processor under a template that opens with ``<bos>``, as a Gemma template does."""
    processor = tiny_paligemma_processor()
    processor.chat_template = "{{ bos_token }}" + processor.chat_template
    return processor


@pytest.mark.parametrize("completions_only", [True, False], ids=["completions-only", "all-tokens"])
def test_a_template_rendering_bos_is_refused_at_construction(completions_only):
    processor = _bos_rendering_processor()
    with pytest.raises(ValueError, match="Remove the BOS from the chat template"):
        paligemma_collator(processor, tiny_paligemma_config(processor), completions_only=completions_only)


def test_a_row_holds_the_one_bos_the_processor_puts_after_the_last_image():
    processor = tiny_paligemma_processor()
    batch = paligemma_collator(processor, tiny_paligemma_config(processor), completions_only=False)([digit_row()])
    ids = batch["input_ids"][0]
    bos_positions = (ids == processor.tokenizer.bos_token_id).nonzero().flatten().tolist()
    assert bos_positions == [int((ids == processor.image_token_id).nonzero().max()) + 1]


def test_a_rendered_bos_reaches_the_row_beside_the_processors_own():
    """Anti-vacuity: the refusal guards a real duplicate. On a model that is no prefix LM the same template builds,
    and its row holds the rendered ``<bos>`` and the processor's."""
    processor = _bos_rendering_processor()
    batch = paligemma_collator(processor, Gemma3Config(), completions_only=False)([digit_row()])
    assert int((batch["input_ids"][0] == processor.tokenizer.bos_token_id).sum()) == 2


def _newline_keeping_processor():
    """The tiny processor with a tokenizer that keeps ``\\n`` as a token, as a real Gemma tokenizer does."""
    processor = tiny_paligemma_processor()
    processor.tokenizer.add_tokens(["\n"])
    return processor


@pytest.mark.parametrize("completions_only", [True, False], ids=["completions-only", "all-tokens"])
def test_the_newline_the_processor_appends_trains_nowhere(completions_only):
    processor = _newline_keeping_processor()
    collator = paligemma_collator(processor, tiny_paligemma_config(processor), completions_only=completions_only)
    batch = collator([digit_row(3), digit_row(7)])
    newline_id = processor.tokenizer.convert_tokens_to_ids("\n")
    for row in range(2):
        assert batch["input_ids"][row, -1] == newline_id
        assert batch["labels"][row, -1] == IGNORE
        assert batch["labels"][row, -2] == processor.tokenizer.eos_token_id, "the answer's own terminator trains"


def test_the_processor_appends_a_newline_after_the_conversation():
    """Anti-vacuity: on a model that is no prefix LM the appended newline closes the row and trains."""
    processor = _newline_keeping_processor()
    batch = paligemma_collator(processor, Gemma3Config(), completions_only=False)([digit_row()])
    assert batch["labels"][0, -1] == processor.tokenizer.convert_tokens_to_ids("\n")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
