#!/usr/bin/env python
"""
Test: Muon optimizer (Newton-Schulz orthogonalization).

Verifies that create_muon_optimizer:
1. Correctly splits 2D+ params (Muon) vs 1D params (AdamW scalar)
2. Achieves loss descent on a simple regression task
3. Handles models with only 2D params (no scalar optimizer)
4. Applies weight decay correctly

Usage (single GPU, no torchrun needed):
    python tests/gpu/optimizers/test_muon.py
"""

import copy
import math

import torch
import torch.nn as nn

from src.optimizers.muon import create_muon_optimizer
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


def run_training(model, optimizer, num_steps=NUM_STEPS, hidden=HIDDEN):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
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


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_param_split():
    """Muon should put 2D+ params in Muon groups and 1D params in scalar AdamW."""
    print("TEST 1: Parameter splitting")
    model = FFNModel().cuda().to(torch.bfloat16)
    optimizer = create_muon_optimizer(model, lr=3e-4)

    muon_params = []
    for g in optimizer._muon_param_groups:
        muon_params.extend(g["params"])

    scalar_params = []
    assert optimizer.scalar_optimizer is not None, "FFN with bias/LN should have scalar optimizer"
    for g in optimizer.scalar_optimizer.param_groups:
        scalar_params.extend(g["params"])

    for name, p in model.named_parameters():
        if p.ndim >= 2:
            assert any(p is mp for mp in muon_params), f"{name} (ndim={p.ndim}) missing from Muon groups"
        else:
            assert any(p is sp for sp in scalar_params), f"{name} (ndim={p.ndim}) missing from scalar groups"

    muon_count = sum(p.numel() for p in muon_params)
    scalar_count = sum(p.numel() for p in scalar_params)
    print(f"  Muon params: {muon_count / 1e3:.1f}K, Scalar params: {scalar_count / 1e3:.1f}K")
    print("  PASSED")


def test_loss_descends():
    """Muon should decrease loss on a regression task."""
    print("\nTEST 2: Loss descent (with bias + layernorm)")
    torch.manual_seed(42)
    model = FFNModel().cuda().to(torch.bfloat16)
    optimizer = create_muon_optimizer(model, lr=2e-3)

    losses = run_training(model, optimizer, num_steps=100)

    assert not math.isnan(losses[-1]), "Loss is NaN"
    assert losses[-1] < losses[0], f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    reduction = (losses[0] - losses[-1]) / losses[0]
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({reduction * 100:.1f}% reduction)")
    assert reduction > 0.1, f"Expected >10% loss reduction, got {reduction * 100:.1f}%"
    print("  PASSED")


def test_loss_descends_no_bias():
    """Muon should work on a model with only 2D params (no scalar optimizer)."""
    print("\nTEST 3: Loss descent (no-bias model, Muon only)")
    torch.manual_seed(42)
    model = NoBiasModel().cuda().to(torch.bfloat16)
    optimizer = create_muon_optimizer(model, lr=2e-3)

    assert optimizer.scalar_optimizer is None, "No 1D params, scalar optimizer should be None"

    losses = run_training(model, optimizer, num_steps=100)

    assert not math.isnan(losses[-1]), "Loss is NaN"
    assert losses[-1] < losses[0], f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    reduction = (losses[0] - losses[-1]) / losses[0]
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} ({reduction * 100:.1f}% reduction)")
    print("  PASSED")


def test_weight_decay():
    """Weight decay should shrink parameter norms."""
    print("\nTEST 4: Weight decay")
    torch.manual_seed(42)
    model = NoBiasModel(hidden=128, layers=2).cuda().to(torch.bfloat16)
    optimizer = create_muon_optimizer(model, lr=1e-4, weight_decay=0.5)

    initial_norm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5
    run_training(model, optimizer, num_steps=30, hidden=128)
    final_norm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5

    print(f"  Param norm: {initial_norm:.4f} -> {final_norm:.4f}")
    assert final_norm < initial_norm, f"Weight decay should reduce norm: {initial_norm:.4f} -> {final_norm:.4f}"
    print("  PASSED")


def test_decay_parameter_names():
    """Only named decay params should receive weight decay in Muon groups."""
    print("\nTEST 5: Decay parameter name filtering")
    model = FFNModel(hidden=64, layers=1).cuda().to(torch.bfloat16)
    decay_names = {n for n, _ in model.named_parameters() if "weight" in n and "norm" not in n}

    optimizer = create_muon_optimizer(
        model,
        lr=3e-4,
        weight_decay=0.1,
        decay_parameters=decay_names,
    )

    # Verify no-decay groups have weight_decay=0
    for g in optimizer._muon_param_groups:
        for p in g["params"]:
            name = next(n for n, param in model.named_parameters() if param is p)
            if name not in decay_names:
                assert g["weight_decay"] == 0.0, f"{name} should have wd=0, got {g['weight_decay']}"

    # Run a few steps to make sure it doesn't crash
    run_training(model, optimizer, num_steps=5, hidden=64)
    print("  PASSED")


def test_state_dict_roundtrip():
    """Train k steps -> deepcopy state_dict -> fresh optimizer built with DIFFERENT hyperparams ->
    load: state tensors bit-exact (Muon momentum AND the scalar AdamW moments), restored group
    hyperparams follow the CHECKPOINT rather than the fresh constructor (``Muon.load_state_dict``
    copies them onto the live group dicts, which upstream's ``param_groups`` property shadows from
    the base restore), and a post-load step still moves weights."""
    print("\nTEST 6: state_dict round-trip")
    torch.manual_seed(42)
    model = FFNModel().cuda().to(torch.bfloat16)
    optimizer = create_muon_optimizer(model, lr=2e-3, weight_decay=0.1, momentum=0.95)
    run_training(model, optimizer, num_steps=5)
    saved = copy.deepcopy(optimizer.state_dict())

    torch.manual_seed(7)
    model2 = FFNModel().cuda().to(torch.bfloat16)
    optimizer2 = create_muon_optimizer(model2, lr=5e-1, weight_decay=0.0, momentum=0.5)
    # Premise: the fresh constructor's hyperparams differ from the checkpoint's, otherwise the
    # follows-the-checkpoint assertion below could not tell the two apart.
    for live, saved_group in zip(optimizer2.param_groups, saved["param_groups"], strict=True):
        assert live["lr"] != saved_group["lr"] and live["weight_decay"] != saved_group["weight_decay"]

    optimizer2.load_state_dict(copy.deepcopy(saved))
    assert_optimizer_state_bit_exact(saved, optimizer2.state_dict())

    for i, (live, saved_group) in enumerate(zip(optimizer2.param_groups, saved["param_groups"], strict=True)):
        for key in ("lr", "momentum", "weight_decay"):
            if key in saved_group:
                assert live[key] == saved_group[key], (
                    f"group {i} '{key}': {live[key]} != saved {saved_group[key]} — restored "
                    f"hyperparams must follow the checkpoint, not the fresh constructor"
                )

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

    test_param_split()
    test_loss_descends()
    test_loss_descends_no_bias()
    test_weight_decay()
    test_decay_parameter_names()
    test_state_dict_roundtrip()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
