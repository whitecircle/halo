#!/usr/bin/env python
"""
Gemma4 VLM smoke test: SFT forward + backward with real images, EP enabled.

Mirrors test_sft_gemma4_moe.py but feeds image+text inputs through the
processor instead of pure text. This exercises the vision encoder path
alongside the EPGemma4MoELayer expert wrapping to confirm the two
components coexist correctly under FSDP+EP.

Usage:
    torchrun --nproc_per_node=4 \\
        tests/gpu/trainers/sft/test_sft_gemma4_vlm.py

Requirements:
    - 2-8 B200/B300 GPUs (>=80GB)
    - DeepEP installed
    - Local checkpoint at /mnt/models/gemma-4-26B-A4B-it-patched
      (override via HALO_TEST_GEMMA4_MODEL env var)
"""

import os
import random
import sys
import traceback

import torch
from accelerate import PartialState
from PIL import Image
from transformers import AutoProcessor

from src.distributed.expert_parallel.layers.gemma4 import EPGemma4MoELayer
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_int, env_str
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import GEMMA4_26B_A4B_PATCHED
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

MODEL_NAME = env_str("HALO_TEST_GEMMA4_MODEL", GEMMA4_26B_A4B_PATCHED)
EP_SIZE_OVERRIDE = env_int("HALO_TEST_EP", None)
SEED = 42


def make_random_image(rank: int) -> Image.Image:
    random.seed(SEED + rank)
    arr = torch.randint(0, 255, (128, 128, 3), dtype=torch.uint8).numpy()
    return Image.fromarray(arr)


def main():
    rank, world_size, local_rank = init_distributed()
    ep_size = EP_SIZE_OVERRIDE if EP_SIZE_OVERRIDE is not None else world_size

    log(f"\n{'=' * 70}")
    log(f"Gemma4 VLM smoke test: EP={ep_size}, world_size={world_size}")
    log(f"Model: {MODEL_NAME}")
    log(f"{'=' * 70}")

    if not os.path.isdir(MODEL_NAME):
        log(f"SKIP: local model path missing: {MODEL_NAME} (set HALO_TEST_GEMMA4_MODEL to a present checkpoint)")
        teardown_distributed()
        sys.exit(0)  # skip, not fail — local-checkpoint-dependent test

    output_dir, cache_dir = setup_cache_dirs("gemma4_vlm_smoke", rank)

    success = False
    try:
        ensure_model_downloaded(MODEL_NAME, rank)
        PartialState()

        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        log(f"Processor: {processor.__class__.__name__}")

        log(f"Loading Gemma4 (EP={ep_size}, sdpa)...")
        log(f"GPU memory before: {gpu_mem_gb():.1f}GB")
        pc = ParallelismConfig(ep_size=ep_size)
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=pc,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",  # Gemma4 global_head_dim=512 > FA2's 256 limit
            use_liger_kernel=False,
        )
        log(f"GPU memory after load: {gpu_mem_gb():.1f}GB")

        ep_layers = sum(1 for m in model.modules() if isinstance(m, EPGemma4MoELayer))
        log(f"EPGemma4MoELayer instances: {ep_layers}")
        if ep_layers == 0:
            log("ERROR: no EPGemma4MoELayer modules found")
            return

        # Build a single-image multimodal sample using the processor's chat template.
        img = make_random_image(rank)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It's a synthetic test pattern of random colors."}],
            },
        ]
        try:
            templated = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception as e:
            log(f"WARN: apply_chat_template failed ({e}); falling back to manual prompt")
            templated = "<image>Describe this image in one sentence. It's a synthetic test pattern."

        # Processor handles image tokenisation + insertion of <image> placeholders.
        try:
            inputs = processor(
                text=[templated],
                images=[img],
                return_tensors="pt",
                padding=True,
            )
        except Exception as e:
            log(f"\nProcessor call failed: {e}")
            if rank == 0:
                traceback.print_exc()
            return

        device = f"cuda:{local_rank}"
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        # Mask image tokens out of the loss; only train on text generation.
        for special in ("image_token_id",):
            tok_id = getattr(model.config, special, None)
            if tok_id is not None:
                labels[labels == tok_id] = -100

        log(f"Input ids shape: {tuple(inputs['input_ids'].shape)}")
        if "pixel_values" in inputs:
            log(f"pixel_values shape: {tuple(inputs['pixel_values'].shape)}, dtype={inputs['pixel_values'].dtype}")

        model.train()

        log("Forward pass with image+text...")
        # pixel_values must match model dtype (bf16 here).
        if "pixel_values" in inputs and inputs["pixel_values"].dtype != torch.bfloat16:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        out = model(**inputs, labels=labels)
        log(f"Forward loss: {out.loss.item():.4f}, finite: {torch.isfinite(out.loss).item()}")
        log(
            f"Logits: shape={tuple(out.logits.shape)}, "
            f"finite_frac={torch.isfinite(out.logits).float().mean().item():.4f}, "
            f"min={out.logits.min().item():.3g}, max={out.logits.max().item():.3g}"
        )

        log("Backward...")
        out.loss.backward()

        # Validate gradient health.
        nan_grads = []
        finite_count = 0
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                nan_grads.append(name)
            else:
                finite_count += 1

        log(f"Finite-grad params: {finite_count}; NaN/Inf-grad params: {len(nan_grads)}")
        if nan_grads:
            for n in nan_grads[:5]:
                log(f"  bad-grad: {n}")

        success = torch.isfinite(out.loss).item() and len(nan_grads) == 0 and finite_count > 0

        log(f"\n{'=' * 70}")
        log(f"GEMMA4 VLM SMOKE TEST {'PASSED' if success else 'FAILED'}")
        log(f"{'=' * 70}")

    except Exception as e:
        log(f"\nFATAL: {e}")
        if rank == 0:
            traceback.print_exc()
    finally:
        barrier()
        cleanup_dirs(output_dir, cache_dir)
        cleanup_memory()
        teardown_distributed()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
