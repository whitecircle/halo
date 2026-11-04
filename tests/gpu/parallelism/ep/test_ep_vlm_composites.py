#!/usr/bin/env python
"""Multimodal MoE composites at EP2 against the replicated composite, on their processor's image batch.

The batch with its pixels zeroed must score a different loss, so the images reach it at all.
"""

import argparse
from functools import partial

import numpy as np
import torch
from PIL import Image

from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import world_spread
from tests.common.ep_reference import compare_ep_grad, ep_layers
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import GEMMA4_26B_A4B, QWEN3_6_MOE_35B
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, padded_vocab_size, tiny_family_model
from tests.common.tiny_processors import tiny_processor
from tests.common.tokenizers import load_cached_processor
from tests.common.tolerances import TOL

SEED = 42
IMAGE_SIDE = 64
# Each processor, released or the family's own over a tiny vocabulary, and the text fields its tiny tower needs (Gemma
# 4's per-layer input table embeds the same ids as the main one).
FAMILIES = {
    "qwen3_5_moe": (partial(load_cached_processor, QWEN3_6_MOE_35B), lambda vocab: {}),
    "gemma4": (partial(load_cached_processor, GEMMA4_26B_A4B), lambda vocab: {"vocab_size_per_layer_input": vocab}),
    "nemotron_h_omni": (partial(tiny_processor, "nemotron_h_omni"), lambda vocab: {}),
}

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
ARGS = parser.parse_args()


def _image_batch(processor, device) -> tuple[dict[str, torch.Tensor], int]:
    """Labels masked at the image placeholders; returns the batch and the image token id."""
    image = Image.fromarray((np.random.RandomState(SEED).rand(IMAGE_SIDE, IMAGE_SIDE, 3) * 255).astype("uint8"))
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A square of coloured noise."}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False)
    batch = processor(text=[text], images=[image], return_tensors="pt")
    image_token_id = processor.image_token_id
    labels = batch["input_ids"].clone()
    labels[labels == image_token_id] = -100
    batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    return batch | {"labels": labels.to(device)}, image_token_id


def _build(processor, image_token_id: int, device):
    vocab = padded_vocab_size(processor.tokenizer)
    torch.manual_seed(SEED)
    model = tiny_family_model(
        TINY_MOE_VLM_FAMILIES[ARGS.family], processor.tokenizer, overrides=FAMILIES[ARGS.family][1](vocab)
    )
    model.config.image_token_id = image_token_id
    # The composites that read the id once, at construction.
    if hasattr(model, "image_token_id"):
        model.image_token_id = image_token_id
    model.config._attn_implementation = "eager"
    return model.to(device=device, dtype=torch.bfloat16)


def _vision_tower(model):
    """Under ``.model`` where the wrapper keeps one, else on the wrapper itself, in the spelling it uses."""
    backbone = getattr(model, "model", model)
    return next(
        getattr(backbone, attr) for attr in ("visual", "vision_tower", "vision_model") if hasattr(backbone, attr)
    )


def _vision_grads(model) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().float().clone()
        for name, param in _vision_tower(model).named_parameters()
        if param.grad is not None
    }


@gpu_test_main(exact_world_size=2, prefix=f"ep_vlm_composite_{ARGS.family}")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    torch.cuda.set_device(ctx.device)
    processor = FAMILIES[ARGS.family][0]()
    batch, image_token_id = _image_batch(processor, ctx.device)
    checks["batch_carries_image_placeholders"] = bool((batch["input_ids"] == image_token_id).any())

    reference = _build(processor, image_token_id, ctx.device)
    reference.train()
    reference_loss = reference(**batch).loss
    reference_loss.backward()
    reference_vision = _vision_grads(reference)
    del reference
    cleanup_memory()

    model = apply_ep_layers(_build(processor, image_token_id, ctx.device), ParallelismConfig(ep_size=2).ep_config())
    layers = ep_layers(model)
    checks["ep_layers_patched_under_the_wrapper"] = bool(layers) and all(layer.ep_size == 2 for layer in layers)
    model.train()
    loss = model(**batch).loss
    loss.backward()
    metrics["reference_loss"], metrics["ep_loss"] = float(reference_loss), float(loss)
    checks["ep_loss_matches_the_replicated_composite"] = (
        abs(float(loss) - float(reference_loss)) < TOL.parallel_vs_baseline_loss_abs
    )
    checks["ranks_agree_on_the_loss"] = TOL.identical_batch_ranks_agree(world_spread(float(loss)))
    vision = _vision_grads(model)
    checks["vision_grads_live"] = bool(vision) and sum(float(g.abs().sum()) for g in vision.values()) > 0
    names = sorted(reference_vision.keys() & vision.keys())
    ratio, cosine = compare_ep_grad(
        torch.cat([vision[n].flatten() for n in names]) if names else None,
        torch.cat([reference_vision[n].flatten() for n in names]) if names else torch.zeros(1),
        "vision tower",
    )
    low, high = TOL.ep_grad_norm_ratio_band
    checks["vision_grads_match_the_reference"] = bool(names) and cosine > TOL.ep_grad_cosine_min and low < ratio < high
    metrics["vision_grad_cos"], metrics["vision_grad_ratio"] = cosine, ratio

    with torch.no_grad():
        blank = model(**batch | {"pixel_values": torch.zeros_like(batch["pixel_values"])}).loss
    checks["zeroed_pixels_change_the_loss"] = not torch.equal(blank, loss.detach())
    log(
        f"  {ARGS.family}: loss {float(loss):.6f} vs reference {float(reference_loss):.6f}, zeroed pixels "
        f"{float(blank):.6f}; vision grad cos {cosine:.5f} ratio {ratio:.4f} over {len(names)} tensors"
    )
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
