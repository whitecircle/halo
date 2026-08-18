#!/usr/bin/env python
"""
Test: FlashAdamW optimizer (quantized states + master weights).

Verifies that create_flash_adamw_optimizer:
1. Creates a valid FlashAdamW instance with correct param groups
2. Achieves loss descent on a simple regression task
3. Handles weight decay parameter filtering correctly
4. Uses less memory than standard AdamW

Usage (single GPU, no torchrun needed):
    python tests/gpu/optimizers/test_flash_adamw.py
"""

import copy
import gc
import math

import torch
import torch.nn as nn

from src.optimizers.flash_adamw import create_flash_adamw_optimizer
from tests.common.utils import assert_optimizer_state_bit_exact

# ─── Models ──────────────────────────────────────────────────────────────────

HIDDEN = 256
BATCH = 32
SEQ = 64
NUM_STEPS = 50


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


def run_training(model, optimizer, num_steps=NUM_STEPS, hidden=HIDDEN):
    device = next(model.parameters()).device
    x = torch.randn(BATCH, SEQ, hidden, device=device, dtype=torch.bfloat16)
    y = torch.randn(BATCH, SEQ, hidden, device=device, dtype=torch.bfloat16)
    losses = []

    for _ in range(num_steps):
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.mse_loss(out.float(), y.float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_creates_optimizer():
    """FlashAdamW should be created with correct param groups."""
    print("TEST 1: Optimizer creation")
    model = FFNModel().to(device="cuda", dtype=torch.bfloat16)
    optimizer = create_flash_adamw_optimizer(model, lr=1e-3)

    total_opt_params = sum(len(g["params"]) for g in optimizer.param_groups)
    total_model_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert total_opt_params == total_model_params, (
        f"Param count mismatch: optimizer has {total_opt_params}, model has {total_model_params}"
    )
    print(f"  Optimizer param groups: {len(optimizer.param_groups)}, total params: {total_opt_params}")
    print("  PASSED")


def test_loss_descends():
    """FlashAdamW should decrease loss on a regression task."""
    print("\nTEST 2: Loss descent")
    torch.manual_seed(42)
    model = FFNModel().to(device="cuda", dtype=torch.bfloat16)
    optimizer = create_flash_adamw_optimizer(model, lr=1e-3)

    losses = run_training(model, optimizer, num_steps=100)

    assert not math.isnan(losses[-1]), "Loss is NaN"
    assert losses[-1] < losses[0], f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    reduction = (losses[0] - losses[-1]) / losses[0]
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({reduction * 100:.1f}% reduction)")
    assert reduction > 0.1, f"Expected >10% loss reduction, got {reduction * 100:.1f}%"
    print("  PASSED")


def test_decay_parameter_names():
    """Only named decay params should receive weight decay."""
    print("\nTEST 3: Decay parameter name filtering")
    model = FFNModel(hidden=64, layers=1).to(device="cuda", dtype=torch.bfloat16)
    decay_names = {n for n, _ in model.named_parameters() if "weight" in n and "norm" not in n}

    optimizer = create_flash_adamw_optimizer(
        model,
        lr=1e-3,
        weight_decay=0.1,
        decay_parameters=decay_names,
    )

    assert len(optimizer.param_groups) == 2, (
        f"Expected 2 param groups (decay + no_decay), got {len(optimizer.param_groups)}"
    )

    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    # Run a few steps to make sure it doesn't crash
    run_training(model, optimizer, num_steps=5, hidden=64)
    print("  PASSED")


def test_memory_savings():
    """FlashAdamW should use less peak memory than standard AdamW on a realistic model."""
    print("\nTEST 4: Memory savings vs AdamW")
    hidden = 2048
    intermediate = 4 * hidden
    layers = 8

    class TransformerFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            for _ in range(layers):
                self.layers.append(
                    nn.ModuleDict(
                        {
                            "norm": nn.LayerNorm(hidden),
                            "gate": nn.Linear(hidden, intermediate, bias=False),
                            "up": nn.Linear(hidden, intermediate, bias=False),
                            "down": nn.Linear(intermediate, hidden, bias=False),
                        }
                    )
                )

        def forward(self, x):
            for layer in self.layers:
                h = layer["norm"](x)
                x = x + layer["down"](nn.functional.silu(layer["gate"](h)) * layer["up"](h))
            return x

    def measure_peak(opt_factory):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        torch.manual_seed(42)
        model = TransformerFFN().to(device="cuda", dtype=torch.bfloat16)
        opt = opt_factory(model)
        x = torch.randn(4, 64, hidden, device="cuda", dtype=torch.bfloat16)
        y = torch.randn_like(x)

        for _ in range(3):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x).float(), y.float())
            loss.backward()
            opt.step()

        peak = torch.cuda.max_memory_allocated() / 1e6
        del model, opt, x, y
        gc.collect()
        torch.cuda.empty_cache()
        return peak

    # Warmup FlashAdamW Triton kernels on a tiny model first
    m_warmup = nn.Linear(64, 64, bias=False).to(device="cuda", dtype=torch.bfloat16)
    opt_warmup = create_flash_adamw_optimizer(m_warmup, lr=1e-3)
    x_w = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    opt_warmup.zero_grad()
    m_warmup(x_w).sum().backward()
    opt_warmup.step()
    del m_warmup, opt_warmup, x_w
    gc.collect()
    torch.cuda.empty_cache()

    adamw_peak = measure_peak(lambda m: torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True))
    flash_peak = measure_peak(lambda m: create_flash_adamw_optimizer(m, lr=1e-3))

    savings = (adamw_peak - flash_peak) / adamw_peak * 100
    print(f"  AdamW peak: {adamw_peak:.0f} MB, FlashAdamW peak: {flash_peak:.0f} MB ({savings:+.1f}%)")
    assert flash_peak < adamw_peak, (
        f"FlashAdamW should use less memory than AdamW: {flash_peak:.0f} MB vs {adamw_peak:.0f} MB"
    )
    print("  PASSED")


def test_state_dict_roundtrip():
    """Train k steps -> deepcopy state_dict -> fresh model+optimizer -> load: every state tensor
    (quantized moments, scales, error_bits, step) bit-exact vs the snapshot, and a post-load step
    still moves weights. A load that drops, casts, or re-quantizes state warm-restarts silently —
    exactly what the bit-exact comparison exists to catch."""
    print("\nTEST 5: state_dict round-trip")
    torch.manual_seed(42)
    model = FFNModel().to(device="cuda", dtype=torch.bfloat16)
    optimizer = create_flash_adamw_optimizer(model, lr=1e-3)
    run_training(model, optimizer, num_steps=5)
    saved = copy.deepcopy(optimizer.state_dict())

    torch.manual_seed(7)
    model2 = FFNModel().to(device="cuda", dtype=torch.bfloat16)
    optimizer2 = create_flash_adamw_optimizer(model2, lr=1e-3)
    optimizer2.load_state_dict(copy.deepcopy(saved))
    assert_optimizer_state_bit_exact(saved, optimizer2.state_dict())

    before = [p.detach().clone() for p in model2.parameters()]
    run_training(model2, optimizer2, num_steps=1)
    moved = any(not torch.equal(b, p.detach()) for b, p in zip(before, model2.parameters(), strict=False))
    assert moved, "post-load step did not move any weight"
    print("  PASSED")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, cannot run test")
        exit(1)

    print(f"PyTorch: {torch.__version__}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print()

    test_creates_optimizer()
    test_loss_descends()
    test_decay_parameter_names()
    test_memory_savings()
    test_state_dict_roundtrip()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
