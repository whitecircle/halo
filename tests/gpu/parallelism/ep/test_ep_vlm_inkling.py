#!/usr/bin/env python
"""Inkling multimodal (VLM) EP=2 vs replicated-reference on a tiny random-init composite model.

The production class is `InklingForConditionalGeneration` — vision/audio towers beside a
`language_model` text backbone. EP patching must find `InklingMoE` nested under the wrapper, the
towers must survive as replicated modules, and an image-carrying batch (pixel features
masked-scattered at `image_token_id` placeholders) must train: loss matches the undistributed
composite reference and the vision tower's gradients stay live alongside the expert shards.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_vlm_inkling.py
"""

import torch
from transformers.models.inkling.configuration_inkling import InklingConfig, InklingVisionConfig
from transformers.models.inkling.modeling_inkling import InklingForConditionalGeneration

from src.distributed.expert_parallel.layers.inkling import EPInklingMoELayer
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_INKLING_CONFIG
from tests.common.utils import cleanup_memory, log

SEED = 42
BATCH, SEQ = 2, 48
N_PATCHES = 8
IMAGE_TOKEN_ID = 500  # within the tiny text vocab (512); the real config uses a dedicated special
LOSS_TOL = 5e-2
RANK_LOSS_TOL = 1e-3

# The upstream scale planner (`plan_out_scales`) rejects most tiny shapes; this one builds a valid
# 4-layer pixel-shuffle stack. text_hidden_size must equal the text config's hidden_size.
TINY_VISION = {"temporal_patch_size": 2, "patch_size": 4, "n_layers": 4, "hidden_size": 32, "text_hidden_size": 256}


def _build_model(device):
    """Identical tiny composite model on every rank (same seed)."""
    torch.manual_seed(SEED)
    config = InklingConfig(
        text_config=dict(TINY_INKLING_CONFIG),
        vision_config=InklingVisionConfig(**TINY_VISION).to_dict(),
        image_token_id=IMAGE_TOKEN_ID,
    )
    config._attn_implementation = "eager"
    model = InklingForConditionalGeneration(config)
    return model.to(device=device, dtype=torch.bfloat16)


def _image_batch(device):
    """A fixed batch whose first row carries N_PATCHES image placeholders; pixel layout [N,T,H,W,C]."""
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, 400, (BATCH, SEQ), device=device)
    input_ids[0, 2 : 2 + N_PATCHES] = IMAGE_TOKEN_ID
    labels = input_ids.clone()
    labels[input_ids == IMAGE_TOKEN_ID] = -100
    pixel_values = torch.randn(
        N_PATCHES,
        TINY_VISION["temporal_patch_size"],
        TINY_VISION["patch_size"],
        TINY_VISION["patch_size"],
        3,
        device=device,
        dtype=torch.bfloat16,
    )
    return input_ids, pixel_values, labels


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)

    input_ids, pixel_values, labels = _image_batch(device)

    # ── Reference: undistributed composite forward+backward ───────────────────
    ref = _build_model(device)
    ref.train()
    ref_out = ref(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
    ref_out.loss.backward()
    ref_loss = ref_out.loss.item()
    log(f"reference loss: {ref_loss:.6f}")
    del ref, ref_out
    cleanup_memory()

    # ── EP=2 on the composite wrapper, same weights, same batch ───────────────
    model = _build_model(device)
    pc = ParallelismConfig(ep_size=ctx.world_size)
    model = patch_moe_model_for_ep(model, pc.create_ep_config())
    create_ep_buffers(model)

    ep_layers = [m for m in model.modules() if isinstance(m, EPInklingMoELayer)]
    checks["ep_layers_patched_under_wrapper"] = len(ep_layers) == TINY_INKLING_CONFIG["num_hidden_layers"]
    checks["vision_tower_preserved"] = hasattr(model.model, "vision_tower")

    model.train()
    out = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
    out.loss.backward()
    ep_loss = out.loss.item()
    log(f"EP VLM loss: {ep_loss:.6f}  |Δref| = {abs(ep_loss - ref_loss):.2e}")
    metrics["ref_loss"] = ref_loss
    metrics["ep_loss"] = ep_loss
    checks["ep_loss_finite"] = bool(torch.isfinite(out.loss))
    checks["ep_loss_matches_ref"] = abs(ep_loss - ref_loss) < LOSS_TOL

    loss_t = torch.tensor([ep_loss], device=device)
    gathered = [torch.zeros_like(loss_t) for _ in range(ctx.world_size)]
    torch.distributed.all_gather(gathered, loss_t)
    spread = max(abs(g.item() - gathered[0].item()) for g in gathered)
    metrics["rank_loss_spread"] = spread
    checks["losses_match_across_ranks"] = spread < RANK_LOSS_TOL

    vision_grad = sum(p.grad.abs().sum().item() for p in model.model.vision_tower.parameters() if p.grad is not None)
    metrics["vision_grad_mass"] = vision_grad
    checks["vision_grads_live"] = vision_grad > 0
    checks["expert_grads_live"] = all(
        ep.gate_up_proj.grad is not None and ep.gate_up_proj.grad.abs().sum().item() > 0 for ep in ep_layers
    )

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="ep_vlm_inkling")(run)

if __name__ == "__main__":
    main()
