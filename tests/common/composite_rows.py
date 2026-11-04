"""A tiny composite with rows its tower really reads: one holding an image, and one holding only text. The family's own
processor builds them where it has a local one, a Qwen-shaped grid or the family's pixel layout elsewhere."""

import inspect

import torch

from tests.common.tiny_models import (
    TINY_LLAMA4_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_QWEN3_5_COMPOSITE_FAMILY,
    TINY_QWEN3_VL_DEEPSTACK_FAMILY,
    TINY_QWEN3_VL_FAMILY,
    TINY_UNWRAPPED_COMPOSITE_FAMILIES,
    tiny_family_model,
)
from tests.common.tiny_processors import TINY_PROCESSOR_MODEL_TYPES, tiny_image_batch, tiny_processor

_IMAGE_TOKEN, _VISION_START = 5, 4
_TEXT_ROW = "Describe the image . A square of noise ."
_IMAGE_SIDE_IN_MERGES = 2

COMPOSITE_FAMILIES = {
    "qwen3_vl": TINY_QWEN3_VL_FAMILY,
    "qwen3_vl_deepstack": TINY_QWEN3_VL_DEEPSTACK_FAMILY,
    "qwen3_5": TINY_QWEN3_5_COMPOSITE_FAMILY,
    **TINY_MOE_VLM_FAMILIES,
    **TINY_UNWRAPPED_COMPOSITE_FAMILIES,
    "glm5_next": TINY_MOE_FAMILIES["glm5_next"],
    "llama4": TINY_LLAMA4_FAMILY,
    "step3p7": TINY_MOE_FAMILIES["step3p7"],
}


def text_row(ids: list[int] | None = None) -> dict[str, torch.Tensor]:
    ids = torch.tensor([ids or [6, 7, 8, 9, 10, 11, 12, 13]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def _grid_row(config) -> dict[str, torch.Tensor]:
    vision = config.vision_config
    merge = vision.spatial_merge_size
    grid = (1, _IMAGE_SIDE_IN_MERGES * merge, _IMAGE_SIDE_IN_MERGES * merge)
    patches = grid[0] * grid[1] * grid[2]
    ids = torch.tensor([[6, _VISION_START] + [_IMAGE_TOKEN] * (patches // merge**2) + [8, 9]])
    patch_width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": ids.clone(),
        "pixel_values": torch.randn(patches, patch_width, generator=torch.Generator().manual_seed(1)),
        "image_grid_thw": torch.tensor([grid]),
        "mm_token_type_ids": (ids == _IMAGE_TOKEN).long(),
    }


def _patch_positions(side: int) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    return torch.stack([xs.flatten(), ys.flatten()], -1).unsqueeze(0)


def _pixels(family: str, config) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One image in the layout ``family``'s tower reads, with the tensors beside it."""
    vision = config.vision_config
    generator = torch.Generator().manual_seed(1)
    if family == "mistral3":
        side = vision.patch_size * config.spatial_merge_size * _IMAGE_SIDE_IN_MERGES
        pixels = torch.randn(1, vision.num_channels, side, side, generator=generator)
        return pixels, {"image_sizes": torch.tensor([[side, side]])}
    if family == "gemma4":
        patches = vision.pooling_kernel_size**2 * _IMAGE_SIDE_IN_MERGES**2
        pixels = torch.randn(1, patches, 3 * vision.patch_size**2, generator=generator)
        return pixels, {"image_position_ids": _patch_positions(int(patches**0.5))}
    if family == "lfm2_vl":
        patches = vision.num_patches
        side = int(patches**0.5)
        pixels = torch.randn(1, patches, vision.num_channels * vision.patch_size**2, generator=generator)
        return pixels, {
            "spatial_shapes": torch.tensor([[side, side]]),
            "pixel_attention_mask": torch.ones(1, patches, dtype=torch.long),
        }
    if family == "inkling_mm_model":
        layout = (vision.temporal_patch_size, vision.patch_size, vision.patch_size, 3)
        return torch.randn(_IMAGE_SIDE_IN_MERGES**2, *layout, generator=generator), {}
    side = vision.image_size
    return torch.randn(1, vision.num_channels, side, side, generator=generator), {}


def _feature_count(model, pixels: torch.Tensor, extra: dict[str, torch.Tensor]) -> int:
    parameters = inspect.signature(model.get_image_features).parameters
    with torch.no_grad():
        output = model.get_image_features(
            pixels, **{key: value for key, value in extra.items() if key in parameters or "kwargs" in parameters}
        )
    pooled = getattr(output, "pooler_output", None)
    features = pooled if pooled is not None else getattr(output, "last_hidden_state", output)
    return sum(feature.reshape(-1, feature.shape[-1]).shape[0] for feature in features)


def _pixel_row(family: str, model) -> dict[str, torch.Tensor]:
    pixels, extra = _pixels(family, model.config)
    ids = torch.tensor([[6] + [_IMAGE_TOKEN] * _feature_count(model, pixels, extra) + [8, 9]])
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": ids.clone(),
        "pixel_values": pixels,
    } | extra


def _point_at_token(model, image_token_id: int) -> None:
    """Some composites read the id once, at construction, and some spell it as an index."""
    model.config.image_token_id = image_token_id
    for holder in (model, model.config):
        for attr in ("image_token_id", "image_token_index"):
            if hasattr(holder, attr):
                setattr(holder, attr, image_token_id)


def composite_model_and_rows(name: str) -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """``(model in training mode, image row, text row)``, the model seeded by the caller."""
    family = COMPOSITE_FAMILIES[name]
    if name in TINY_PROCESSOR_MODEL_TYPES:
        processor = tiny_processor(name)
        model = tiny_family_model(family, processor.tokenizer).train()
        _point_at_token(model, processor.image_token_id)
        return model, tiny_image_batch(processor), text_row(processor.tokenizer(_TEXT_ROW)["input_ids"])
    model = tiny_family_model(family).train()
    _point_at_token(model, _IMAGE_TOKEN)
    if hasattr(model.config.vision_config, "spatial_merge_size") and hasattr(
        model.config.vision_config, "temporal_patch_size"
    ):
        model.config.vision_start_token_id = _VISION_START
        return model, _grid_row(model.config), text_row()
    return model, _pixel_row(name, model), text_row()
