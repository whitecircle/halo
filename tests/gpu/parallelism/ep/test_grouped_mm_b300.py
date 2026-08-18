#!/usr/bin/env python
"""
Grouped MM backward correctness on B300 (SM100+).

``src.kernels.grouped_mm_autograd.grouped_mm`` must carry the broadcast-gradient case,
where the bare Blackwell path fails ``out.sum().backward()`` with
``"Invalid strides/sizes, got [0, 0, 0]"``.

Also validates that training converges, using dimensions
matching the ``nikita-savelyev-cerebras/tiny-random-kimi-k2.5`` MoE model
(DeepSeek V3-style: 32 routed experts, 8 per token, hidden=8, intermediate=64).

Run on single GPU:
    python tests/gpu/parallelism/ep/test_grouped_mm_b300.py

Requirements:
    - 1x GPU (B300/B200 or any SM90+)
    - No DeepEP required (tests the grouped_mm wrapper directly)
"""

import sys
import traceback

import torch
import torch.nn as nn

from src.kernels.grouped_mm_autograd import grouped_mm

# Model dimensions from tiny-random-kimi-k2.5

N_ROUTED_EXPERTS = 32
HIDDEN_SIZE = 8
MOE_INTERMEDIATE_SIZE = 64

if not torch.cuda.is_available():
    # The sentinel the launcher skips on. A CPU fallback would run every case below and report this
    # GPU-kernel gate green without ever touching the kernel it exists to pin.
    print("SKIP: no CUDA")
    sys.exit(0)

device = "cuda:0"
G, M, K, N = N_ROUTED_EXPERTS, 16, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE

PASS = 0
FAIL = 0


def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS  {name}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        FAIL += 1


# 1. Fixed grouped_mm: 3D uniform layout (G, M, K) @ (G, K, N)

print()
print("=" * 70)
print("1. FIXED GROUPED_MM: 3D UNIFORM LAYOUT (model MoE dims)")
print("=" * 70)


def test_3d_forward_correctness():
    A = torch.randn(G, M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(G, K, N, dtype=torch.bfloat16, device=device)
    out = grouped_mm(A, B)
    ref = torch.stack([A[g] @ B[g] for g in range(G)])
    diff = (ref.float() - out.float()).abs().max().item()
    assert out.shape == (G, M, N), f"shape mismatch: {out.shape}"
    assert diff < 0.1, f"max diff too large: {diff}"
    print(f"    shape={out.shape}, max_diff={diff:.6f}")


test("3D forward correctness", test_3d_forward_correctness)


def test_3d_backward_sum():
    A = torch.randn(G, M, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    B = torch.randn(G, K, N, dtype=torch.bfloat16, device=device, requires_grad=True)
    out = grouped_mm(A, B)
    out.sum().backward()
    assert A.grad is not None and A.grad.shape == A.shape
    assert B.grad is not None and B.grad.shape == B.shape
    print(f"    A.grad={A.grad.shape}, B.grad={B.grad.shape}")


test("3D backward .sum() (zero-stride grad)", test_3d_backward_sum)


def test_3d_backward_correctness():
    A = torch.randn(G, M, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    B = torch.randn(G, K, N, dtype=torch.bfloat16, device=device, requires_grad=True)
    out = grouped_mm(A, B)
    out.sum().backward()
    A2 = A.detach().clone().requires_grad_(True)
    B2 = B.detach().clone().requires_grad_(True)
    ref = torch.stack([A2[g] @ B2[g] for g in range(G)])
    ref.sum().backward()
    ga_diff = (A2.grad.float() - A.grad.float()).abs().max().item()
    gb_diff = (B2.grad.float() - B.grad.float()).abs().max().item()
    assert ga_diff < 1.0, f"grad_a diff too large: {ga_diff}"
    assert gb_diff < 1.0, f"grad_b diff too large: {gb_diff}"
    print(f"    grad_a max_diff={ga_diff:.6f}, grad_b max_diff={gb_diff:.6f}")


test("3D backward correctness vs loop", test_3d_backward_correctness)


# 2. 2D jagged layout with offsets (MoE-style variable tokens per expert)

print()
print("=" * 70)
print("2. 2D JAGGED LAYOUT (MoE-style)")
print("=" * 70)


def test_jagged_forward_correctness():
    tokens_per_expert = [3, 5, 2, 4, 6, 1, 8, 3]
    Gj = len(tokens_per_expert)
    offs = torch.tensor(tokens_per_expert, dtype=torch.int32, device=device).cumsum(0)
    total_M = offs[-1].item()
    A = torch.randn(total_M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(Gj, K, N, dtype=torch.bfloat16, device=device)
    out = grouped_mm(A, B, offs=offs)
    parts = []
    for g in range(Gj):
        s = sum(tokens_per_expert[:g])
        e = s + tokens_per_expert[g]
        parts.append(A[s:e] @ B[g])
    ref = torch.cat(parts, dim=0)
    diff = (ref.float() - out.float()).abs().max().item()
    assert diff < 0.1, f"max diff too large: {diff}"
    print(f"    shape={out.shape}, max_diff={diff:.6f}")


test("jagged forward correctness", test_jagged_forward_correctness)


def test_jagged_backward_sum():
    tokens_per_expert = [3, 5, 2, 4, 6, 1, 8, 3]
    Gj = len(tokens_per_expert)
    offs = torch.tensor(tokens_per_expert, dtype=torch.int32, device=device).cumsum(0)
    total_M = offs[-1].item()
    A = torch.randn(total_M, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    B = torch.randn(Gj, K, N, dtype=torch.bfloat16, device=device, requires_grad=True)
    out = grouped_mm(A, B, offs=offs)
    out.sum().backward()
    assert A.grad is not None and A.grad.shape == A.shape
    assert B.grad is not None and B.grad.shape == B.shape
    print(f"    A.grad={A.grad.shape}, B.grad={B.grad.shape}")


test("jagged backward .sum() (zero-stride grad)", test_jagged_backward_sum)


def test_jagged_backward_correctness():
    tokens_per_expert = [3, 5, 2, 4, 6, 1, 8, 3]
    Gj = len(tokens_per_expert)
    offs = torch.tensor(tokens_per_expert, dtype=torch.int32, device=device).cumsum(0)
    total_M = offs[-1].item()
    A = torch.randn(total_M, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    B = torch.randn(Gj, K, N, dtype=torch.bfloat16, device=device, requires_grad=True)
    out = grouped_mm(A, B, offs=offs)
    out.sum().backward()
    A2 = A.detach().clone().requires_grad_(True)
    B2 = B.detach().clone().requires_grad_(True)
    parts = []
    for g in range(Gj):
        s = sum(tokens_per_expert[:g])
        e = s + tokens_per_expert[g]
        parts.append(A2[s:e] @ B2[g])
    ref = torch.cat(parts, dim=0)
    ref.sum().backward()
    ga_diff = (A2.grad.float() - A.grad.float()).abs().max().item()
    gb_diff = (B2.grad.float() - B.grad.float()).abs().max().item()
    assert ga_diff < 1.0, f"grad_A diff too large: {ga_diff}"
    assert gb_diff < 1.0, f"grad_B diff too large: {gb_diff}"
    print(f"    grad_A max_diff={ga_diff:.6f}, grad_B max_diff={gb_diff:.6f}")


test("jagged backward correctness vs loop", test_jagged_backward_correctness)


# 3. Full 32-expert SwiGLU MoE FFN (matching ep_moe_layer compute path)

print()
print("=" * 70)
print("3. FULL 32-EXPERT SwiGLU MoE FFN (ep_moe_layer compute path)")
print("=" * 70)


def test_swiglu_moe_forward_backward():
    """Test the full MoE FFN: gate_proj, up_proj, down_proj with grouped_mm,
    matching the compute path in ep_moe_layer.py (_fused_glu_experts_gmm)."""
    G = N_ROUTED_EXPERTS
    M = 16
    # Weights in matmul convention [E, K, N] as used in ep_moe_layer.py
    gate_proj = torch.randn(G, K, N, dtype=torch.bfloat16, device=device)
    up_proj = torch.randn(G, K, N, dtype=torch.bfloat16, device=device)
    down_proj = torch.randn(G, N, K, dtype=torch.bfloat16, device=device)

    x = torch.randn(G, M, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    gate = grouped_mm(x, gate_proj, offs=None)
    up = grouped_mm(x, up_proj, offs=None)
    activated = torch.nn.functional.silu(gate) * up
    output = grouped_mm(activated, down_proj, offs=None)

    assert output.shape == (G, M, K), f"shape mismatch: {output.shape}"
    output.sum().backward()
    assert x.grad is not None
    print(f"    output shape={output.shape}, x.grad shape={x.grad.shape}")


test("32-expert SwiGLU MoE FFN forward+backward", test_swiglu_moe_forward_backward)


# 4. Offset dtype normalization (cumsum upcast fix)

print()
print("=" * 70)
print("4. OFFSET DTYPE NORMALIZATION")
print("=" * 70)


def test_offset_int64_auto_cast():
    """Verify int64 offsets from cumsum are auto-cast to int32."""
    tokens_per_expert = [4, 8, 2, 6]
    offs = torch.tensor(tokens_per_expert, dtype=torch.int32, device=device).cumsum(0)
    assert offs.dtype == torch.int64, f"expected int64, got {offs.dtype}"
    total_M = offs[-1].item()
    A = torch.randn(total_M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(len(tokens_per_expert), K, N, dtype=torch.bfloat16, device=device)
    out = grouped_mm(A, B, offs=offs)
    assert out.shape == (total_M, N), f"shape mismatch: {out.shape}"
    print(f"    offset dtype={offs.dtype}, output shape={out.shape}")


test("int64 offsets auto-cast to int32", test_offset_int64_auto_cast)


# 5. Training convergence with grouped_mm fix

print()
print("=" * 70)
print("5. TRAINING CONVERGENCE")
print("=" * 70)


class MoEFFN(nn.Module):
    """Mini MoE FFN using grouped_mm, matching ep_moe_layer compute path."""

    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        # Weights in matmul convention [E, K, N] like ep_moe_layer
        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16) * 0.02
        )

    def forward(self, x):
        gate = grouped_mm(x, self.gate_proj.transpose(-2, -1))
        up = grouped_mm(x, self.up_proj.transpose(-2, -1))
        hidden = torch.nn.functional.silu(gate) * up
        return grouped_mm(hidden, self.down_proj.transpose(-2, -1))


def test_training_convergence():
    """Train a MoE FFN with grouped_mm and verify loss decreases."""
    num_experts, hidden, intermediate = 8, 64, 256
    model = MoEFFN(num_experts, hidden, intermediate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Fixed target
    torch.manual_seed(99)
    with torch.no_grad():
        x_target = torch.randn(num_experts, 16, hidden, dtype=torch.bfloat16, device=device)
        y_target = model(x_target)

    torch.manual_seed(42)
    x_train = torch.randn(num_experts, 16, hidden, dtype=torch.bfloat16, device=device)

    losses = []
    for step in range(50):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = torch.nn.functional.mse_loss(y_pred, y_target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    print(f"    loss: {losses[0]:.6f} -> {losses[-1]:.6f} ({(1 - losses[-1] / losses[0]) * 100:.1f}% drop)")


test("MoE FFN training convergence", test_training_convergence)


def test_training_with_sum_loss():
    """Train with .sum() loss (the exact zero-stride backward that crashes natively)."""
    num_experts, hidden, intermediate = 8, 64, 256
    model = MoEFFN(num_experts, hidden, intermediate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(20):
        optimizer.zero_grad()
        x = torch.randn(num_experts, 16, hidden, dtype=torch.bfloat16, device=device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

    print("    .sum() backward completed 20 steps without crash")


test("Training with .sum() loss (zero-stride backward)", test_training_with_sum_loss)


# Summary

print()
print("=" * 70)
print(f"RESULTS: {PASS} PASSED, {FAIL} FAILED out of {PASS + FAIL} tests")
print("=" * 70)

if FAIL > 0:
    sys.exit(1)
