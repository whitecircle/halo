"""A local processor for a tiny composite: the family's own processor and image processor, at the tiny tower's geometry,
over a word-level tokenizer. A row built on one needs no released processor in the cache."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import (
    FeatureExtractionMixin,
    MiniMaxM3VLImageProcessor,
    MiniMaxM3VLProcessor,
    MiniMaxM3VLVideoProcessor,
    NemotronH_Omni_Reasoning_V3ImageProcessor,
    NemotronH_Omni_Reasoning_V3Processor,
    NemotronH_Omni_Reasoning_V3VideoProcessor,
    ProcessorMixin,
    Qwen2VLImageProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from tests.common.tiny_models import word_level_tokenizer

_WORDS = "user assistant Describe the image . A square of noise".split()
_IMAGE_SIDE = 56


@dataclass(frozen=True)
class _Recipe:
    processor: type[ProcessorMixin]
    image_processor: Callable[[], object]
    # The markers the processor reads off the tokenizer, each one token, and the prompt's image placeholder.
    tokens: tuple[str, ...]
    placeholder: str
    # The processor's other members, which it refuses to build without; an image row never reads them.
    members: dict[str, Callable[[], object]]


_RECIPES = {
    "minimax_m3_vl": _Recipe(
        MiniMaxM3VLProcessor,
        lambda: MiniMaxM3VLImageProcessor(min_pixels=_IMAGE_SIDE**2, max_pixels=_IMAGE_SIDE**2),
        (
            MiniMaxM3VLProcessor.IMAGE_TOKEN,
            MiniMaxM3VLProcessor.VIDEO_TOKEN,
            MiniMaxM3VLProcessor.VISION_START_TOKEN,
            MiniMaxM3VLProcessor.VISION_END_TOKEN,
        ),
        MiniMaxM3VLProcessor.IMAGE_TOKEN,
        {"video_processor": MiniMaxM3VLVideoProcessor},
    ),
    "nemotron_h_omni": _Recipe(
        NemotronH_Omni_Reasoning_V3Processor,
        lambda: NemotronH_Omni_Reasoning_V3ImageProcessor(min_num_patches=16, max_num_patches=16),
        ("<image>", "<video>", "<so_embedding>", "<img>", "</img>"),
        "<image>",
        {"video_processor": NemotronH_Omni_Reasoning_V3VideoProcessor, "feature_extractor": FeatureExtractionMixin},
    ),
    "qwen3_5_moe": _Recipe(
        Qwen3VLProcessor,
        lambda: Qwen2VLImageProcessor(patch_size=16, merge_size=2, min_pixels=64**2, max_pixels=64**2),
        ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"),
        "<|vision_start|><|image_pad|><|vision_end|>",
        {"video_processor": Qwen3VLVideoProcessor},
    ),
}
TINY_PROCESSOR_MODEL_TYPES = tuple(sorted(_RECIPES))


def tiny_processor(model_type: str) -> ProcessorMixin:
    recipe = _RECIPES[model_type]
    tokenizer = word_level_tokenizer(_WORDS)
    tokenizer.add_special_tokens({"additional_special_tokens": list(recipe.tokens)})
    template = (
        "{% for m in messages %}{{ m['role'] }} {% for p in m['content'] %}{% if p['type'] == 'image' %}"
        f"{recipe.placeholder}"
        "{% else %}{{ p['text'] }} {% endif %}{% endfor %}<eos> {% endfor %}"
    )
    members = {name: build() for name, build in recipe.members.items()}
    return recipe.processor(
        image_processor=recipe.image_processor(), tokenizer=tokenizer, chat_template=template, **members
    )


def tiny_image_batch(processor: ProcessorMixin) -> dict[str, torch.Tensor]:
    """One image-bearing conversation over ``processor``, its labels masked at the image placeholders."""
    image = Image.fromarray((np.random.RandomState(0).rand(_IMAGE_SIDE, _IMAGE_SIDE, 3) * 255).astype("uint8"))
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image ."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A square of noise ."}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False)
    batch = processor(text=[text], images=[image], return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[labels == processor.image_token_id] = -100
    return {key: value for key, value in batch.items() if torch.is_tensor(value)} | {"labels": labels}
