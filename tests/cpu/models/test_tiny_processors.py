#!/usr/bin/env python
"""A local processor reaches its tiny composite's vision tower: the placeholders it expands are the ones the composite
fills, and the pixels move the loss. A row on it that never reached the tower would pass with no images at all."""

import pytest
import torch

from tests.common.tiny_models import (
    TINY_MOE_VLM_FAMILIES,
    TINY_UNWRAPPED_COMPOSITE_FAMILIES,
    tiny_family_model,
)
from tests.common.tiny_processors import TINY_PROCESSOR_MODEL_TYPES, tiny_image_batch, tiny_processor

_FAMILIES = {**TINY_MOE_VLM_FAMILIES, **TINY_UNWRAPPED_COMPOSITE_FAMILIES}


def _composite(model_type: str, processor):
    torch.manual_seed(0)
    model = tiny_family_model(_FAMILIES[model_type], processor.tokenizer)
    model.config.image_token_id = processor.image_token_id
    # The composites that read the id once, at construction.
    if hasattr(model, "image_token_id"):
        model.image_token_id = processor.image_token_id
    model.config._attn_implementation = "eager"
    return model.eval()


@pytest.mark.parametrize("model_type", TINY_PROCESSOR_MODEL_TYPES)
def test_the_batch_carries_placeholders_and_masks_them_from_the_labels(model_type):
    processor = tiny_processor(model_type)
    batch = tiny_image_batch(processor)
    placeholders = batch["input_ids"] == processor.image_token_id
    assert placeholders.any(), "the processor expanded no image placeholder"
    assert (batch["labels"][placeholders] == -100).all()
    assert (batch["labels"][~placeholders] != -100).any()


@pytest.mark.parametrize("model_type", TINY_PROCESSOR_MODEL_TYPES)
@torch.no_grad()
def test_zeroed_pixels_move_the_composites_loss(model_type):
    processor = tiny_processor(model_type)
    batch = tiny_image_batch(processor)
    model = _composite(model_type, processor)
    loss = model(**batch, use_cache=False).loss
    blank = model(**batch | {"pixel_values": torch.zeros_like(batch["pixel_values"])}, use_cache=False).loss
    assert torch.isfinite(loss)
    assert not torch.equal(loss, blank), "the images never reached the loss"


@pytest.mark.parametrize("model_type", TINY_PROCESSOR_MODEL_TYPES)
@torch.no_grad()
def test_a_composite_refuses_a_batch_whose_placeholders_it_does_not_recognize(model_type):
    """The other half of the check above: the tower's output has nowhere to land under a wrong token id."""
    processor = tiny_processor(model_type)
    batch = tiny_image_batch(processor)
    model = _composite(model_type, processor)
    wrong = processor.image_token_id + 1
    model.config.image_token_id = wrong
    if hasattr(model, "image_token_id"):
        model.image_token_id = wrong
    if hasattr(model, "model") and hasattr(model.model, "image_token_id"):
        model.model.image_token_id = wrong
    with pytest.raises(ValueError):
        model(**batch, use_cache=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
