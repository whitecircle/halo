#!/usr/bin/env python
"""
Test: Muon optimizer with FSDP2 (multi-GPU torchrun).

Reproduces and validates the fix for: Muon's fused Triton kernels fail with
FSDP2 because DTensor params cannot be passed directly to Triton kernels.

When FSDP2 wraps parameters, p.data and p.grad become DTensors. The to_local()
helper extracts the underlying local tensor shard before passing to Triton,
making Muon compatible with FSDP2.

Test Phases:
1. FSDP2 multi-GPU with FFN model (2D + 1D params): loss descends
2. FSDP2 multi-GPU with NoBias model (2D params only): loss descends
3. FSDP2 weight decay: parameter norms shrink
4. to_local DTensor unwrapping verification

Run with 2 GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        tests/gpu/optimizers/test_muon_fsdp.py
"""

import math
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from src.distributed.runtime import to_local
from src.optimizers.muon import create_muon_optimizer

# ── Configuration ────────────────────────────────────────────────────────────

HIDDEN = 256
BATCH = 4
SEQ = 32
NUM_STEPS = 30
LR = 2e-3


# ── Models ───────────────────────────────────────────────────────────────────


class FFNModel(nn.Module):
    """Small FFN with both 2D (Linear.weight) and 1D (bias, LayerNorm) params."""

    def __init__(self, hidden=HIDDEN, layers=3):
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks.extend([nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU()])
        blocks.append(nn.Linear(hidden, hidden))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class NoBiasModel(nn.Module):
    """Model with only 2D params (no bias, no layernorm)."""

    def __init__(self, hidden=HIDDEN, layers=3):
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks.extend([nn.Linear(hidden, hidden, bias=False), nn.GELU()])
        blocks.append(nn.Linear(hidden, hidden, bias=False))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def train_steps(model, optimizer, num_steps, hidden=HIDDEN):
    """Run training steps, return list of loss values."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    torch.manual_seed(42)
    x = torch.randn(BATCH, SEQ, hidden, device=device, dtype=dtype)
    y = torch.randn(BATCH, SEQ, hidden, device=device, dtype=dtype)
    losses = []
    for _ in range(num_steps):
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# ── Tests ────────────────────────────────────────────────────────────────────


def test_muon_fsdp_ffn():
    """Muon + FSDP2 with FFN model (2D weights + 1D bias/norm params)."""
    rank = dist.get_rank()
    print(f"[Rank {rank}] TEST 1: Muon + FSDP2 (FFN with bias + layernorm)")

    torch.manual_seed(42 + rank)
    model = FFNModel(hidden=HIDDEN, layers=3).cuda().to(torch.bfloat16)

    for module in model.net:
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            fully_shard(module)
    fully_shard(model)

    # Verify DTensor wrapping
    has_dtensor = any(isinstance(p, DTensor) for p in model.parameters())
    if rank == 0:
        print(f"  Parameters wrapped as DTensors: {has_dtensor}")

    optimizer = create_muon_optimizer(model, lr=LR)
    losses = train_steps(model, optimizer, NUM_STEPS)

    assert not math.isnan(losses[-1]), "Loss is NaN"
    assert all(math.isfinite(l) for l in losses), "Non-finite loss detected"
    assert losses[-1] < losses[0], f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    reduction = (losses[0] - losses[-1]) / losses[0]
    if rank == 0:
        print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({reduction * 100:.1f}% reduction)")
    assert reduction > 0.05, f"Expected >5% loss reduction, got {reduction * 100:.1f}%"
    if rank == 0:
        print("  PASSED")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def test_muon_fsdp_nobias():
    """Muon + FSDP2 with model that has only 2D params (no scalar optimizer)."""
    rank = dist.get_rank()
    print(f"[Rank {rank}] TEST 2: Muon + FSDP2 (no-bias model, Muon only)")

    torch.manual_seed(42 + rank)
    model = NoBiasModel(hidden=HIDDEN, layers=3).cuda().to(torch.bfloat16)

    for module in model.net:
        if isinstance(module, nn.Linear):
            fully_shard(module)
    fully_shard(model)

    optimizer = create_muon_optimizer(model, lr=LR)
    assert optimizer.scalar_optimizer is None, "No 1D params, scalar optimizer should be None"

    losses = train_steps(model, optimizer, NUM_STEPS)

    assert not math.isnan(losses[-1]), "Loss is NaN"
    assert all(math.isfinite(l) for l in losses), "Non-finite loss detected"
    assert losses[-1] < losses[0], f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    if rank == 0:
        reduction = (losses[0] - losses[-1]) / losses[0]
        print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({reduction * 100:.1f}% reduction)")
        print("  PASSED")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def test_muon_fsdp_weight_decay():
    """Weight decay should shrink parameter norms with FSDP2."""
    rank = dist.get_rank()
    print(f"[Rank {rank}] TEST 3: Muon + FSDP2 weight decay")

    torch.manual_seed(42 + rank)
    model = NoBiasModel(hidden=128, layers=2).cuda().to(torch.bfloat16)

    initial_norm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5

    for module in model.net:
        if isinstance(module, nn.Linear):
            fully_shard(module)
    fully_shard(model)

    optimizer = create_muon_optimizer(model, lr=1e-4, weight_decay=0.5)

    dtype = next(model.parameters()).dtype
    torch.manual_seed(42 + rank)
    x = torch.randn(BATCH, SEQ, 128, device="cuda", dtype=dtype)
    y = torch.randn(BATCH, SEQ, 128, device="cuda", dtype=dtype)

    for _ in range(30):
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()
        optimizer.step()

    final_norm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5
    if rank == 0:
        print(f"  Param norm: {initial_norm:.4f} -> {final_norm:.4f}")
    assert final_norm < initial_norm, f"Weight decay should reduce norm: {initial_norm:.4f} -> {final_norm:.4f}"
    if rank == 0:
        print("  PASSED")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def test_dtensor_unwrapping():
    """Verify to_local correctly unwraps DTensor params and grads."""
    rank = dist.get_rank()
    print(f"[Rank {rank}] TEST 4: to_local DTensor unwrapping")

    torch.manual_seed(42 + rank)
    model = FFNModel(hidden=128, layers=1).cuda().to(torch.bfloat16)

    for module in model.net:
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            fully_shard(module)
    fully_shard(model)

    dtype = next(model.parameters()).dtype
    x = torch.randn(2, 16, 128, device="cuda", dtype=dtype)
    out = model(x)
    loss = out.sum()
    loss.backward()

    dtensor_grad_count = 0
    for p in model.parameters():
        if p.grad is not None:
            if isinstance(p.grad, DTensor):
                dtensor_grad_count += 1
                local_grad = to_local(p.grad)
                assert isinstance(local_grad, torch.Tensor) and not isinstance(local_grad, DTensor), (
                    "to_local should return plain Tensor for DTensor input"
                )
                assert local_grad.device.type == "cuda", f"to_local result should be on CUDA, got {local_grad.device}"

            if isinstance(p.data, DTensor):
                local_data = to_local(p.data)
                assert isinstance(local_data, torch.Tensor) and not isinstance(local_data, DTensor), (
                    "to_local should return plain Tensor for DTensor p.data"
                )

    if rank == 0:
        print(f"  DTensor gradients found: {dtensor_grad_count}")
        assert dtensor_grad_count > 0, "Expected at least some DTensor gradients after FSDP2"
        print("  PASSED")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"\n{'=' * 70}")
        print("  Muon + FSDP2 Compatibility Test")
        print(f"  World size: {world_size}, Rank: {rank}")
        print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"  Free memory: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")
        print(f"{'=' * 70}")

    all_passed = True
    try:
        test_muon_fsdp_ffn()
        test_muon_fsdp_nobias()
        test_muon_fsdp_weight_decay()
        test_dtensor_unwrapping()

        if rank == 0:
            print(f"\n{'=' * 70}")
            print("  ALL FSDP2 TESTS PASSED")
            print(f"{'=' * 70}")

    except Exception as e:
        print(f"[Rank {rank}] TEST FAILED with exception: {e}")
        traceback.print_exc()
        all_passed = False

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    return 0 if all_passed else 1


import gc

if __name__ == "__main__":
    sys.exit(main())
