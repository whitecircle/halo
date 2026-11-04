#!/usr/bin/env python
"""``Qwen4ExpForConditionalGeneration`` under the family patches, loaded as a run loads it: its text tower runs the
document forwards behind the vision tower."""

import pytest
import torch
from transformers import AutoModelForImageTextToText

from tests.common.qwen4_exp import (
    load_host_table_model,
    ngram_table,
    packed_markers,
    packed_positions,
    transformers_forwards,
)
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, tiny_family_model

TOLERANCE = 1e-6
LEAK_FLOOR = 1e-3
DOCUMENTS = [9, 30, 21]
# Image placeholder ids inside the tiny vocabulary.
IMAGE_TOKEN, VISION_START = 5, 4
VOCAB = 1000


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_VLM_FAMILIES["qwen4_exp"])
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.normal_(std=0.3)
    directory = str(tmp_path_factory.mktemp("tiny") / "qwen4_exp_composite")
    model.save_pretrained(directory)
    return directory


@pytest.fixture
def model(checkpoint):
    model = load_host_table_model(checkpoint, model_class=AutoModelForImageTextToText).eval()
    assert type(model).__name__ == "Qwen4ExpForConditionalGeneration" and ngram_table(model).is_attached
    model.config.image_token_id, model.config.vision_start_token_id = IMAGE_TOKEN, VISION_START
    return model


def _image_batch(model) -> dict:
    """Two rows around one image each in the Qwen patch layout, the first right-padded."""
    vision = model.config.vision_config
    merge = vision.spatial_merge_size
    grid = (1, 4 * merge, 2 * merge)
    patches = grid[0] * grid[1] * grid[2]
    generator = torch.Generator().manual_seed(1)
    text = torch.randint(8, VOCAB, (40,), generator=generator).tolist()
    ids = torch.tensor([text[:7] + [VISION_START] + [IMAGE_TOKEN] * (patches // merge**2) + text[7:]] * 2)
    mask = torch.ones_like(ids)
    mask[0, -9:] = 0
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "pixel_values": torch.randn(2 * patches, patch_dim, generator=generator),
        "image_grid_thw": torch.tensor([grid, grid]),
        "mm_token_type_ids": (ids == IMAGE_TOKEN).long(),
    }


def _logits(model, **batch) -> torch.Tensor:
    with torch.no_grad():
        return model(use_cache=False, **batch).logits


def test_an_image_batch_computes_transformers_logits(model):
    batch = _image_batch(model)
    patched = _logits(model, **batch)
    with transformers_forwards():
        stock = _logits(model, **batch)
    real = batch["attention_mask"].bool()
    assert (patched - stock)[real].abs().max() <= TOLERANCE


def test_a_packed_text_row_keeps_each_document_to_itself(model):
    positions = packed_positions([DOCUMENTS])
    ids = torch.randint(8, VOCAB, (1, sum(DOCUMENTS)), generator=torch.Generator().manual_seed(2))
    markers = {"position_ids": positions, **packed_markers(model.config, positions)}
    packed = _logits(model, input_ids=ids, **markers)
    start = 0
    for length in DOCUMENTS:
        alone = _logits(model, input_ids=ids[:, start : start + length])
        assert (packed[:, start : start + length] - alone).abs().max() <= TOLERANCE, (start, length)
        start += length
    with transformers_forwards():
        leaked = _logits(model, input_ids=ids, **markers)
    assert (leaked - packed).abs().max() > LEAK_FLOOR, "transformers' forwards no longer leak across this row"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
