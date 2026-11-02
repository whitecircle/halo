"""A tiny real PaliGemma (processor, configs, read-a-digit rows, the VLM collator over them) for the prefix-LM
attention tests: a word-level tokenizer and a 28-pixel SigLIP tower, built offline with no hub checkpoint."""

from __future__ import annotations

import torch
from transformers import PaliGemmaConfig, PaliGemmaProcessor, SiglipImageProcessor

from halo.data.collators.vlm import VLMDataCollator
from halo.data.vlm_render import extract_conversation_images
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from tests.common.datasets import digit_image
from tests.common.models import TINY_SIGLIP_VISION_CONFIG
from tests.common.tiny_models import padded_vocab_size, word_level_tokenizer

# Gemma 2's two softcaps, which PaliGemma 2 carries and transformers' SDPA path drops.
GEMMA2_SOFTCAPS = {"attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0}
# Renders each turn as ``<role> <parts> <eos>``; the processor puts ``<bos>`` after the last ``<image>``.
_CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }} {% for p in m['content'] %}{% if p['type'] == 'image' %}<image>"
    "{% else %}{{ p['text'] }} {% endif %}{% endfor %}<eos> {% endfor %}"
)
_WORDS = "user assistant What digit is shown ? The . 0 1 2 3 4 5 6 7 8 9".split()
# Image placeholder tokens each image expands to: (28 // 14) ** 2 patches of the tiny tower.
_IMAGE_SEQ_LENGTH = 4


def tiny_paligemma_processor() -> PaliGemmaProcessor:
    """A PaliGemma processor over :data:`_WORDS` with a chat template, so the toolkit's VLM render applies."""
    tokenizer = word_level_tokenizer(_WORDS)
    tokenizer.add_special_tokens({"bos_token": "<bos>"})
    image_processor = SiglipImageProcessor(size={"height": 28, "width": 28})
    image_processor.image_seq_length = _IMAGE_SEQ_LENGTH
    return PaliGemmaProcessor(image_processor=image_processor, tokenizer=tokenizer, chat_template=_CHAT_TEMPLATE)


def tiny_paligemma_config(processor: PaliGemmaProcessor, *, text_model_type: str = "gemma") -> PaliGemmaConfig:
    """PaliGemma (``gemma``) or PaliGemma 2 (``gemma2``, softcapped) over ``processor``'s vocabulary, leaving
    ``use_bidirectional_attention`` unset as every published checkpoint does."""
    tokenizer = processor.tokenizer
    extra = GEMMA2_SOFTCAPS if text_model_type == "gemma2" else {}
    text_config = {
        "model_type": text_model_type,
        "vocab_size": padded_vocab_size(tokenizer),
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "pad_token_id": tokenizer.pad_token_id,
        **extra,
    }
    return PaliGemmaConfig(
        text_config=text_config,
        vision_config=dict(TINY_SIGLIP_VISION_CONFIG),
        projection_dim=64,
        image_token_id=processor.image_token_id,
    )


def digit_row(digit: int = 3) -> dict:
    """A mapped VLM row (``history``/``images``): one image turn asking for the digit, and the answer."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": digit_image(digit)},
                {"type": "text", "text": "What digit is shown ?"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": f"The digit is {digit} ."}]},
    ]
    history, images = extract_conversation_images(conversation)
    return {"history": history, "images": images}


def text_row(digit: int = 3) -> dict:
    """A mapped VLM row with no image."""
    history, images = extract_conversation_images(
        [
            {"role": "user", "content": "What digit is shown ?"},
            {"role": "assistant", "content": f"The digit is {digit} ."},
        ]
    )
    return {"history": history, "images": images}


def paligemma_collator(processor, model_config, *, completions_only: bool, cls=VLMDataCollator, **kwargs):
    """``cls`` (the SFT VLM collator by default) over ``processor``; ``completions_only`` trains the answer alone."""
    return cls(
        processor,
        processor.tokenizer,
        256,
        train_on_completions_only=completions_only,
        response_prompt_template="assistant" if completions_only else None,
        model_config=model_config,
        **kwargs,
    )


def first_scored_position(labels: torch.Tensor) -> int:
    """The first position of a ``labels`` row the shifted loss scores (position 0 never is)."""
    return int((labels[1:] != LABEL_IGNORE_INDEX).nonzero()[0]) + 1
