#!/usr/bin/env python
"""The image-run options of the scripts with a vision path: which parameters ``freeze_vision_tower`` holds on every
composite, what ``min_pixels`` / ``max_pixels`` do to a real image processor, and what each refuses."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Gemma4ImageProcessor, Qwen2VLImageProcessor

from halo.configs.sft import SFTScriptArguments
from halo.data.vlm_render import apply_image_pixel_bounds
from halo.entrypoints.data_setup import apply_vision_options
from halo.models.introspection.module_tree import backbone_with_layers, media_module_names
from tests.common.tiny_models import TINY_DENSE_FAMILY, TINY_MOE_VLM_FAMILIES, TINY_QWEN3_VL_FAMILY, tiny_family_model

_COMPOSITES = {**TINY_MOE_VLM_FAMILIES, "qwen3_vl": TINY_QWEN3_VL_FAMILY}


def _options(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{"freeze_vision_tower": False, "min_pixels": None, "max_pixels": None, **overrides})


def _no_adapters() -> SimpleNamespace:
    return SimpleNamespace(use_peft=False, lora_target_modules=None)


def _image_processing() -> SimpleNamespace:
    return SimpleNamespace(image_processor=Qwen2VLImageProcessor())


def _text_tower_names(model) -> set[str]:
    backbone = backbone_with_layers(model)
    # Nemotron's composite exposes no output embedding of its own.
    output_embeddings = model.get_output_embeddings() or model.language_model.get_output_embeddings()
    kept = {id(param) for param in backbone.parameters()} | {id(param) for param in output_embeddings.parameters()}
    return {name for name, param in model.named_parameters() if id(param) in kept}


@pytest.mark.parametrize("family", list(_COMPOSITES))
def test_the_freeze_holds_exactly_what_lies_outside_the_text_tower(family):
    """Anything a text row never reaches would otherwise train only on some ranks, and anything it reaches must stay."""
    torch.manual_seed(0)
    model = tiny_family_model(_COMPOSITES[family])
    apply_vision_options(_options(freeze_vision_tower=True), model, _no_adapters(), None, is_vlm=True)

    text_tower = _text_tower_names(model)
    frozen = {name for name, param in model.named_parameters() if not param.requires_grad}
    assert frozen, "the premise: this composite has a tower"
    assert frozen == {name for name, _ in model.named_parameters()} - text_tower


def test_a_full_fine_tune_keeps_training_its_tower():
    model = tiny_family_model(TINY_QWEN3_VL_FAMILY)
    apply_vision_options(_options(), model, _no_adapters(), None, is_vlm=True)
    assert all(param.requires_grad for param in model.parameters())


def test_a_checkpoint_with_no_tower_refuses_the_freeze():
    model = tiny_family_model(TINY_DENSE_FAMILY)
    with pytest.raises(ValueError, match="holds no vision or audio tower"):
        apply_vision_options(_options(freeze_vision_tower=True), model, _no_adapters(), None, is_vlm=False)


def test_adapters_refuse_the_freeze():
    model = tiny_family_model(TINY_QWEN3_VL_FAMILY)
    with pytest.raises(ValueError, match="adapters"):
        apply_vision_options(
            _options(freeze_vision_tower=True), model, SimpleNamespace(use_peft=True), None, is_vlm=True
        )


def test_the_module_walk_names_each_tower_once():
    """SigLIP nests a ``vision_model`` inside the ``vision_tower`` the walk already named."""
    siglip = torch.nn.Module()
    siglip.vision_model = torch.nn.Linear(1, 1)
    model = torch.nn.Module()
    model.vision_tower = siglip
    model.embed_vision = torch.nn.Linear(1, 1)
    assert media_module_names(model) == ["vision_tower", "embed_vision"]


def _grid(processing, side: int = 400) -> list[list[int]]:
    image = Image.fromarray((np.random.RandomState(0).rand(side, side, 3) * 255).astype("uint8"))
    return processing.image_processor(images=[image], return_tensors="pt")["image_grid_thw"].tolist()


def test_max_pixels_lowers_the_patches_each_image_costs():
    unbounded, bounded = _image_processing(), _image_processing()
    apply_image_pixel_bounds(bounded, None, 56 * 56 * 4)
    assert np.prod(_grid(bounded)) < np.prod(_grid(unbounded))


def test_min_pixels_raises_them_and_each_bound_leaves_the_other():
    unbounded, bounded = _image_processing(), _image_processing()
    original_cap = bounded.image_processor.size.longest_edge
    apply_image_pixel_bounds(bounded, 1200 * 1200, None)
    assert np.prod(_grid(bounded)) > np.prod(_grid(unbounded))
    assert bounded.image_processor.size.longest_edge == original_cap


def test_the_export_carries_the_bounds(tmp_path):
    processing = _image_processing()
    apply_image_pixel_bounds(processing, 1000, 2000)
    processing.image_processor.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "preprocessor_config.json").read_text())
    assert (saved["size"]["shortest_edge"], saved["size"]["longest_edge"]) == (1000, 2000)


def test_a_processor_that_does_not_resize_by_pixel_count_refuses_the_bounds():
    """Gemma 4's would ignore them, as ``from_pretrained`` does when handed them."""
    with pytest.raises(ValueError, match="would ignore them"):
        apply_image_pixel_bounds(SimpleNamespace(image_processor=Gemma4ImageProcessor()), None, 1000)


def test_a_tokenizer_refuses_the_bounds():
    with pytest.raises(ValueError, match="would ignore them"):
        apply_image_pixel_bounds(SimpleNamespace(), 10, None)


def test_a_text_run_refuses_the_bounds():
    with pytest.raises(ValueError, match="take"):
        apply_vision_options(_options(max_pixels=10), None, _no_adapters(), None, is_vlm=False)


def test_bounds_left_unset_touch_nothing():
    apply_image_pixel_bounds(SimpleNamespace(), None, None)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"min_pixels": 0}, "min_pixels"),
        ({"max_pixels": 1.5}, "max_pixels"),
        ({"min_pixels": 9, "max_pixels": 3}, "over"),
    ],
)
def test_the_arguments_refuse_a_bad_bound(overrides, match):
    with pytest.raises(ValueError, match=match):
        SFTScriptArguments(dataset=["dummy"], **overrides)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
