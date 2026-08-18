#!/usr/bin/env python
"""Smoke test for the Mistral3 multimodal path (Pixtral vision + Mistral4 text).

Ensures the EP/CP/TP plumbing for Mistral4 does not break the vision side of
``Mistral3ForConditionalGeneration``. The test does three checks on a single GPU:

1. ``Mistral3ForConditionalGeneration`` instantiates from a tiny synthetic
   ``Mistral3Config`` (Pixtral vision tower + Mistral4 text backbone).
2. Text-only forward (no images) runs through both the inner ``Mistral4Model``
   and the LM head.
3. Text + ``pixel_values`` forward exercises the vision tower, multi-modal
   projector, and image-token injection without raising — the loss is finite.

Run::

    torchrun --nproc_per_node=1 \
        tests/gpu/parallelism/test_mistral3_vision_smoke.py

The single-process path keeps the test cheap; EP/CP/TP correctness of the
text backbone is exercised by ``test_mistral4_all_parallelism.py``.
"""

from __future__ import annotations

import torch
from transformers.models.mistral3 import Mistral3Config, Mistral3ForConditionalGeneration
from transformers.models.mistral4 import Mistral4Config
from transformers.models.pixtral import PixtralVisionConfig

from src.distributed.context_parallel.patching import WRAPPER_CLASS_MAP
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP
from src.distributed.tensor_parallel.module_types import TP_SHARDABLE_ATTENTION_CLASSES
from tests.common.harness import gpu_test_main
from tests.common.utils import log


def build_tiny_mistral3() -> Mistral3ForConditionalGeneration:
    """Build a tiny Mistral3 model (vision + text) that fits comfortably on one GPU."""
    text_config = Mistral4Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        qk_head_dim=16,
        v_head_dim=16,
        first_k_dense_replace=0,
        hidden_act="silu",
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 2.0,
            "original_max_position_embeddings": 64,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
        rope_interleave=True,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    vision_config = PixtralVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        head_dim=16,
        image_size=56,
        patch_size=14,
    )
    config = Mistral3Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=10,
        spatial_merge_size=2,
        multimodal_projector_bias=False,
        projector_hidden_act="gelu",
        vision_feature_layer=-1,
    )

    torch.manual_seed(0)
    model = Mistral3ForConditionalGeneration(config).to(torch.bfloat16).cuda()
    return model


def make_image_input(model: Mistral3ForConditionalGeneration) -> dict:
    """Build a minimal text + single-image input batch for Mistral3.

    The vision tower processes one 56×56 image into 16 patches (4×4 grid after
    a 14×14 patch size). With ``spatial_merge_size=2`` the multi-modal projector
    produces ``(4/2) * (4/2) = 4`` image tokens that replace the image_token_index
    placeholders in input_ids.
    """
    cfg = model.config
    image_token = int(cfg.image_token_index)
    patches_per_side = cfg.vision_config.image_size // cfg.vision_config.patch_size
    image_tokens = (patches_per_side // cfg.spatial_merge_size) ** 2

    input_ids = torch.tensor(
        [[1, 2, 3] + [image_token] * image_tokens + [4, 5, 6, 7]],
        dtype=torch.long,
        device="cuda",
    )
    labels = input_ids.clone()
    labels[input_ids == image_token] = -100  # don't backprop on image tokens

    pixel_values = torch.randn(
        1,
        3,
        cfg.vision_config.image_size,
        cfg.vision_config.image_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    image_sizes = torch.tensor(
        [[cfg.vision_config.image_size, cfg.vision_config.image_size]],
        dtype=torch.long,
        device="cuda",
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
    }


def check_registration() -> dict[str, bool]:
    """Ensure the EP/CP/TP class registries cover the Mistral4 modules used here."""
    return {
        "ep_mistral4_moe_registered": "Mistral4MoE" in MOE_LAYER_MAP,
        "cp_mistral4_attention_registered": "Mistral4Attention" in WRAPPER_CLASS_MAP,
        "tp_mistral4_attention_registered": "Mistral4Attention" in TP_SHARDABLE_ATTENTION_CLASSES,
    }


def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    log("Building tiny Mistral3 (Pixtral + Mistral4 MoE)...")

    model = build_tiny_mistral3()
    checks["model_instantiated"] = True

    registry_checks = check_registration()
    checks.update(registry_checks)
    log(f"  Registry checks: {registry_checks}")

    # Sanity: Mistral4MoE blocks live under model.model.language_model.layers
    moe_blocks = [name for name, mod in model.named_modules() if mod.__class__.__name__ == "Mistral4MoE"]
    attn_blocks = [name for name, mod in model.named_modules() if mod.__class__.__name__ == "Mistral4Attention"]
    checks["moe_blocks_in_language_model"] = (
        all("language_model.layers" in n for n in moe_blocks)
        and len(moe_blocks) == model.config.text_config.num_hidden_layers
    )
    checks["attn_blocks_in_language_model"] = (
        all("language_model.layers" in n for n in attn_blocks)
        and len(attn_blocks) == model.config.text_config.num_hidden_layers
    )
    log(f"  Found {len(moe_blocks)} Mistral4MoE and {len(attn_blocks)} Mistral4Attention blocks")

    # Step 1: Text-only forward (no pixel_values).
    text_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device="cuda")
    text_labels = text_ids.clone()
    model.eval()
    with torch.no_grad():
        text_out = model(input_ids=text_ids, labels=text_labels)
    text_loss = text_out.loss.item()
    log(f"  Text-only loss: {text_loss:.6f}")
    checks["text_only_forward"] = torch.isfinite(text_out.loss).item()

    # Step 2: Text + image forward.
    batch = make_image_input(model)
    model.train()
    out = model(**batch)
    log(f"  Multimodal loss: {out.loss.item():.6f}")
    checks["multimodal_forward"] = torch.isfinite(out.loss).item()

    # Step 3: Backward.
    out.loss.backward()
    # Verify gradients on both vision and text paths.
    vision_param = next(model.model.vision_tower.parameters())
    text_param = next(model.model.language_model.parameters())
    checks["vision_grad_present"] = vision_param.grad is not None and torch.isfinite(vision_param.grad).all().item()
    checks["text_grad_present"] = text_param.grad is not None and torch.isfinite(text_param.grad).all().item()
    log(f"  Vision grad finite: {checks['vision_grad_present']}")
    log(f"  Text grad finite:   {checks['text_grad_present']}")

    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="mistral3_vision_smoke")(run)

if __name__ == "__main__":
    main()
