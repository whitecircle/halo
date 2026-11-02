#!/usr/bin/env python
"""CPU tests for the image layout of every VLM processor call: one image list per text.

Gemma 3 and Gemma 4 processors read a flat image list as one text's images and refuse a batch of several texts, so a
flat list breaks SFT above batch 1 and every reward pair, whose chosen and rejected sides are two texts. A tiny
offline Gemma 3 (the real processor class over a word-level tokenizer, a 28-pixel SigLIP tower) runs the SFT and
reward collators and a forward.

Run: ``python tests/cpu/data/test_vlm_batch_images.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import math

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3ImageProcessor,
    Gemma3Processor,
    PreTrainedTokenizerFast,
)

from halo.data.collators.vlm import VLMDataCollator
from halo.data.collators.vlm_preference import DataCollatorForVLMPreference
from halo.data.vlm_render import extract_conversation_images, render_vlm_text
from tests.common.datasets import digit_image
from tests.common.models import TINY_SIGLIP_VISION_CONFIG
from tests.common.tiny_models import padded_vocab_size

_SPECIAL_TOKENS = ("<unk>", "<eos>", "<pad>", "<bos>", "<start_of_turn>", "<end_of_turn>")
# Gemma 3's image markers: the placeholder a template renders, the soft tokens it expands to, the closing marker.
_IMAGE_TOKENS = {"boi_token": "<start_of_image>", "image_token": "<image_soft_token>", "eoi_token": "<end_of_image>"}
_WORDS = "user model What digit is shown ? The . 0 1 2 3 4 5 6 7 8 9".split()
_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<start_of_turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }} "
    "{% for p in m['content'] %}{% if p['type'] == 'image' %}<start_of_image>{% else %}{{ p['text'] }} {% endif %}"
    "{% endfor %}<end_of_turn> {% endfor %}"
)
_MARKER = "<start_of_turn>model"
# Soft tokens per image: the tiny tower's 2x2 patches pooled to one.
_IMAGE_SEQ_LENGTH = 1


def _tiny_gemma3_processor() -> Gemma3Processor:
    vocab = {token: index for index, token in enumerate((*_SPECIAL_TOKENS, *_IMAGE_TOKENS.values(), *_WORDS))}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<pad>",
        bos_token="<bos>",
        extra_special_tokens=_IMAGE_TOKENS,
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]}, replace_extra_special_tokens=False
    )
    return Gemma3Processor(
        image_processor=Gemma3ImageProcessor(size={"height": 28, "width": 28}),
        tokenizer=tokenizer,
        chat_template=_CHAT_TEMPLATE,
        image_seq_length=_IMAGE_SEQ_LENGTH,
    )


def _tiny_gemma3_config(processor: Gemma3Processor) -> Gemma3Config:
    tokenizer = processor.tokenizer
    return Gemma3Config(
        text_config={
            "vocab_size": padded_vocab_size(tokenizer),
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "layer_types": ["sliding_attention", "full_attention"],
            "pad_token_id": tokenizer.pad_token_id,
        },
        vision_config=dict(TINY_SIGLIP_VISION_CONFIG),
        mm_tokens_per_image=_IMAGE_SEQ_LENGTH,
        boi_token_index=tokenizer.convert_tokens_to_ids(_IMAGE_TOKENS["boi_token"]),
        eoi_token_index=tokenizer.convert_tokens_to_ids(_IMAGE_TOKENS["eoi_token"]),
        image_token_index=tokenizer.convert_tokens_to_ids(_IMAGE_TOKENS["image_token"]),
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")],
    )


def _row(digit: int, *, images: int = 1) -> dict:
    content = [{"type": "image", "image": digit_image(digit)} for _ in range(images)]
    content.append({"type": "text", "text": "What digit is shown ?"})
    history, row_images = extract_conversation_images(
        [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": f"The digit is {digit} ."}]},
        ]
    )
    return {"history": history, "images": row_images}


def _sft_collator(processor: Gemma3Processor) -> VLMDataCollator:
    return VLMDataCollator(
        processor,
        processor.tokenizer,
        256,
        train_on_completions_only=True,
        response_prompt_template=_MARKER,
        model_config=_tiny_gemma3_config(processor),
    )


@pytest.mark.parametrize("images_per_row", [(1, 1), (2, 1), (1, 0)], ids=["one-each", "two-and-one", "image-and-text"])
def test_an_sft_batch_of_several_rows_pairs_each_row_with_its_images(images_per_row):
    processor = _tiny_gemma3_processor()
    batch = _sft_collator(processor)([_row(3 + i, images=n) for i, n in enumerate(images_per_row)])

    assert batch["pixel_values"].shape[0] == sum(images_per_row)
    soft_tokens = (batch["input_ids"] == processor.tokenizer.image_token_id).sum(dim=-1)
    assert soft_tokens.tolist() == [n * _IMAGE_SEQ_LENGTH for n in images_per_row]
    assert (batch["labels"] != -100).any(dim=-1).all(), "every row trains its answer"


def test_a_reward_pair_collates_on_a_processor_that_batches_images_per_text():
    processor = _tiny_gemma3_processor()
    row = _row(3)
    text = render_vlm_text(processor, row["history"])
    batch = DataCollatorForVLMPreference(processor=processor)(
        [{"chosen_text": text, "rejected_text": text, "images": row["images"]}]
    )

    assert batch["input_ids"].shape[0] == 2 and batch["pixel_values"].shape[0] == 2
    assert torch.equal(batch["input_ids"][0], batch["input_ids"][1])


def test_the_tiny_gemma3_trains_on_a_mixed_batch():
    """The collated batch is one the model accepts: a finite loss that the image positions reach. The image
    projection starts at zero, so its own gradient, not the tower's, shows the image features enter the loss."""
    processor = _tiny_gemma3_processor()
    batch = _sft_collator(processor)([_row(3), _row(5, images=0)])
    torch.manual_seed(0)
    model = Gemma3ForConditionalGeneration(_tiny_gemma3_config(processor)).train()

    loss = model(**batch).loss
    loss.backward()

    assert math.isfinite(loss.item())
    assert model.model.multi_modal_projector.mm_input_projection_weight.grad.abs().sum() > 0


def test_a_flat_image_list_is_what_the_processor_refuses():
    """Anti-vacuity: the same two-row batch with its images flattened is refused by the processor itself."""
    processor = _tiny_gemma3_processor()
    rows = [_row(3), _row(5)]
    texts = [render_vlm_text(processor, row["history"]) for row in rows]
    with pytest.raises(ValueError, match="inconsistently sized batches"):
        processor(text=texts, images=[image for row in rows for image in row["images"]], return_tensors="pt")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
