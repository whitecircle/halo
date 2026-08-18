#!/usr/bin/env python
"""
Test: Muon optimizer with FSDP2 on Qwen3.5-2B (real model).

Validates that the _to_local() DTensor fix works with a production model
under FSDP2, not just synthetic FFN models. Qwen3.5-2B has a mix of
2D (Linear.weight) and 1D (bias, LayerNorm) parameters that exercise
both Muon and scalar AdamW optimizer paths.

Run with 2 GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        tests/gpu/optimizers/test_muon_fsdp_qwen35.py
"""

import gc
import math
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from src.optimizers.muon import create_muon_optimizer

MODEL_NAME = "Qwen/Qwen3.5-2B"
BATCH = 2
SEQ = 256
MAX_STEPS = 10
LR = 2e-4


def main() -> int:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"\n{'=' * 70}")
        print("  Muon + FSDP2 on Qwen3.5-2B")
        print(f"  World size: {world_size}")
        print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"  Free memory: {free_gb:.1f} GB")
        print(f"{'=' * 70}")

    all_passed = True

    try:
        # ── Load model ──────────────────────────────────────────────
        if rank == 0:
            print("\n[1/4] Loading Qwen3.5-2B...")

        from transformers import AutoConfig, AutoModelForCausalLM

        # Download on rank 0 first
        if rank == 0:
            AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
        dist.barrier()

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": f"cuda:{local_rank}"},
            attn_implementation="eager",
        )
        model.train()

        total_params = sum(p.numel() for p in model.parameters())
        if rank == 0:
            print(f"  Model loaded: {total_params / 1e9:.2f}B params")
            print(f"  Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # ── Apply FSDP2 ─────────────────────────────────────────────
        if rank == 0:
            print("\n[2/4] Applying FSDP2...")

        # Shard each transformer layer, then the root model
        from torch.distributed.tensor import DTensor

        for module in model.modules():
            if hasattr(module, "self_attn") or (hasattr(module, "mlp") and hasattr(module, "input_layernorm")):
                fully_shard(module)
        fully_shard(model)

        has_dtensor = any(isinstance(p, DTensor) for p in model.parameters())
        if rank == 0:
            print(f"  Parameters wrapped as DTensors: {has_dtensor}")
            print(f"  Memory after FSDP2: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # ── Create Muon optimizer ───────────────────────────────────
        if rank == 0:
            print("\n[3/4] Creating Muon optimizer...")

        optimizer = create_muon_optimizer(model, lr=LR)

        muon_count = sum(p.numel() for p in model.parameters() if p.requires_grad and p.ndim >= 2)
        scalar_count = sum(p.numel() for p in model.parameters() if p.requires_grad and p.ndim < 2)
        if rank == 0:
            print(f"  Muon (2D+) params: {muon_count / 1e6:.1f}M")
            print(f"  Scalar (1D) params: {scalar_count / 1e6:.1f}M")

        # ── Train ───────────────────────────────────────────────────
        if rank == 0:
            print(f"\n[4/4] Training {MAX_STEPS} steps...")

        config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is None:
            vocab_size = config.text_config.vocab_size

        torch.manual_seed(42 + rank)
        input_ids = torch.randint(0, vocab_size, (BATCH, SEQ), device=f"cuda:{local_rank}")
        labels = input_ids.clone()

        losses = []
        torch.cuda.reset_peak_memory_stats()

        for step in range(MAX_STEPS):
            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()

            t0 = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            losses.append(loss.item())
            if rank == 0:
                opt_ms = (t1 - t0) * 1000
                print(f"  Step {step}: loss={loss.item():.4f}, opt_step={opt_ms:.1f}ms")

        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        # ── Assertions ──────────────────────────────────────────────
        if rank == 0:
            print(f"\n  Peak memory: {peak_gb:.2f} GB")
            print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

        assert not math.isnan(losses[-1]), "Loss is NaN"
        assert all(math.isfinite(l) for l in losses), "Non-finite loss detected"
        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        reduction = (losses[0] - losses[-1]) / losses[0]
        if rank == 0:
            print(f"  Loss reduction: {reduction * 100:.1f}%")
            assert reduction > 0.01, f"Expected >1% loss reduction, got {reduction * 100:.1f}%"

        if rank == 0:
            print(f"\n{'=' * 70}")
            print("  QWEN3.5-2B FSDP2 TEST PASSED")
            print(f"{'=' * 70}")

    except Exception as e:
        print(f"[Rank {rank}] TEST FAILED: {e}")
        traceback.print_exc()
        all_passed = False

    finally:
        gc.collect()
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
