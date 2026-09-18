#!/usr/bin/env python
"""A real ``fully_shard`` over the multimodal wrappers under LoRA: forward, backward and a step.

The structural roster test derives each parameter's owning unit from recorded ``fully_shard`` calls;
this one lets FSDP2 itself do the sharding. One rank is enough: every managed parameter becomes a
DTensor that only its unit's pre-forward unshards, so ``embed_tokens`` owned by the wrong unit raises
``aten.embedding.default got mixed torch.Tensor and DTensor`` on the first forward, exactly as a
multi-GPU run does. Tied and untied embeddings, attention LoRA and ``modules_to_save`` copies (which
untie a tied embedding), and an image batch through the Qwen-style towers, whose base parameters live
in the composite's unit (their projections spell `qkv`/`proj`, so an attention target list adapts none
of them).

Run: pytest tests/cpu/parallelism/test_fsdp2_real_shard_multimodal_lora.py
"""

import src.models.patches.kernel_dispatch  # noqa: F401  # isort: skip

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

import src.distributed.fsdp as fsdp
from src.distributed.fsdp import IdentityParamSet
from tests.cpu.parallelism.test_fsdp2_unit_active_at_param_use import _FAMILIES, _MODULES_TO_SAVE, _attention_lora

_COMPOSITES = sorted(family for family in _FAMILIES if family.endswith("composite"))
_QWEN_TOWERS = {"qwen3_vl_composite", "qwen3_5_composite", "qwen3_5_moe_composite"}
_IMAGE_TOKEN, _VISION_START = 5, 4


@pytest.fixture
def single_rank_mesh(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def _text_batch() -> dict[str, torch.Tensor]:
    ids = torch.tensor([[6, 7, 8, 9, 10, 11]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def _image_batch(model) -> dict[str, torch.Tensor]:
    """One image in the Qwen patch layout, its tokens spelled with ids inside the tiny vocab."""
    config = model.config
    config.image_token_id, config.vision_start_token_id = _IMAGE_TOKEN, _VISION_START
    vision = config.vision_config
    merge = vision.spatial_merge_size
    grid = (1, 2 * merge, 2 * merge)
    n_patches = grid[0] * grid[1] * grid[2]
    ids = [6, _VISION_START] + [_IMAGE_TOKEN] * (n_patches // (merge * merge)) + [8, 9]
    ids = torch.tensor([ids])
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": ids.clone(),
        "pixel_values": torch.randn(n_patches, patch_dim),
        "image_grid_thw": torch.tensor([grid]),
        "mm_token_type_ids": (ids == _IMAGE_TOKEN).long(),
    }


@pytest.mark.parametrize("images", [False, True], ids=["text", "images"])
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
@pytest.mark.parametrize("mode", ["peft", "peft_modules_to_save"])
@pytest.mark.parametrize("family", _COMPOSITES)
def test_lora_wrapper_trains_under_real_fsdp2(family, mode, tied, images, single_rank_mesh):
    if images and family not in _QWEN_TOWERS:
        pytest.skip("no tiny image batch for this family's tower")
    torch.manual_seed(0)
    model = _FAMILIES[family](tied)
    if tied and model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        pytest.skip("the family does not tie its embeddings")
    model = _attention_lora(model.train(), _MODULES_TO_SAVE if mode == "peft_modules_to_save" else None)
    batch = _image_batch(model) if images else _text_batch()

    fsdp.apply_fsdp2_per_layer(model, single_rank_mesh, MixedPrecisionPolicy(), False, IdentityParamSet())
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    for _ in range(2):
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

    assert torch.isfinite(loss.detach()).item()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
