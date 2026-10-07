#!/usr/bin/env python
"""
Grouped MM backward correctness on B300 (SM100+).

``src.kernels.grouped_mm_autograd.grouped_mm`` must carry the broadcast-gradient case,
where the bare Blackwell path fails ``out.sum().backward()`` with
``"Invalid strides/sizes, got [0, 0, 0]"``.

Also validates that training converges, using dimensions
matching the ``nikita-savelyev-cerebras/tiny-random-kimi-k2.5`` MoE model
(DeepSeek V3-style: 32 routed experts, 8 per token, hidden=8, intermediate=64).

The jagged forward against the per-expert loop and the int64 offset cast are pinned exactly in
``tests/gpu/kernels/test_grouped_mm_empty_groups.py``.

Run on single GPU:
    torchrun --nproc_per_node=1 tests/gpu/parallelism/ep/test_grouped_mm_b300.py

Requirements:
    - 1x GPU (B300/B200 or any SM90+)
    - No DeepEP required (tests the grouped_mm wrapper directly)
"""

import torch
import torch.nn as nn

from src.kernels.grouped_mm_autograd import grouped_mm
from tests.common.harness import gpu_test_main, record_check
from tests.common.utils import log

# Model dimensions from tiny-random-kimi-k2.5

N_ROUTED_EXPERTS = 32
HIDDEN_SIZE = 8
MOE_INTERMEDIATE_SIZE = 64

device = "cuda"
G, M, K, N = N_ROUTED_EXPERTS, 16, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE
TOKENS_PER_EXPERT = [3, 5, 2, 4, 6, 1, 8, 3]

# bf16 grouped kernel vs a per-group matmul loop: accumulation order differs, magnitudes are O(1).
FORWARD_MAX_ABS_DIFF = 0.1
GRAD_MAX_ABS_DIFF = 1.0


def _jagged_operands(requires_grad: bool):
    """MoE-style 2D jagged layout: variable tokens per expert, cumulative end offsets."""
    offs = torch.tensor(TOKENS_PER_EXPERT, dtype=torch.int32, device=device).cumsum(0)
    total_m = offs[-1].item()
    A = torch.randn(total_m, K, dtype=torch.bfloat16, device=device, requires_grad=requires_grad)
    B = torch.randn(len(TOKENS_PER_EXPERT), K, N, dtype=torch.bfloat16, device=device, requires_grad=requires_grad)
    return A, B, offs


def test_3d_forward_correctness():
    A = torch.randn(G, M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(G, K, N, dtype=torch.bfloat16, device=device)
    out = grouped_mm(A, B)
    ref = torch.stack([A[g] @ B[g] for g in range(G)])
    diff = (ref.float() - out.float()).abs().max().item()
    assert out.shape == (G, M, N), f"shape mismatch: {out.shape}"
    assert diff < FORWARD_MAX_ABS_DIFF, f"max diff too large: {diff}"
    log(f"    shape={out.shape}, max_diff={diff:.6f}")


def test_3d_backward_sum():
    A = torch.randn(G, M, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    B = torch.randn(G, K, N, dtype=torch.bfloat16, device=device, requires_grad=True)
    out = grouped_mm(A, B)
    out.sum().backward()
    assert A.grad is not None and A.grad.shape == A.shape
    assert B.grad is not None and B.grad.shape == B.shape
    log(f"    A.grad={A.grad.shape}, B.grad={B.grad.shape}")


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
    assert ga_diff < GRAD_MAX_ABS_DIFF, f"grad_a diff too large: {ga_diff}"
    assert gb_diff < GRAD_MAX_ABS_DIFF, f"grad_b diff too large: {gb_diff}"
    log(f"    grad_a max_diff={ga_diff:.6f}, grad_b max_diff={gb_diff:.6f}")


def test_jagged_backward_sum():
    A, B, offs = _jagged_operands(requires_grad=True)
    out = grouped_mm(A, B, offs=offs)
    out.sum().backward()
    assert A.grad is not None and A.grad.shape == A.shape
    assert B.grad is not None and B.grad.shape == B.shape
    log(f"    A.grad={A.grad.shape}, B.grad={B.grad.shape}")


def test_jagged_backward_correctness():
    A, B, offs = _jagged_operands(requires_grad=True)
    out = grouped_mm(A, B, offs=offs)
    out.sum().backward()
    A2 = A.detach().clone().requires_grad_(True)
    B2 = B.detach().clone().requires_grad_(True)
    parts = []
    for g in range(len(TOKENS_PER_EXPERT)):
        s = sum(TOKENS_PER_EXPERT[:g])
        e = s + TOKENS_PER_EXPERT[g]
        parts.append(A2[s:e] @ B2[g])
    ref = torch.cat(parts, dim=0)
    ref.sum().backward()
    ga_diff = (A2.grad.float() - A.grad.float()).abs().max().item()
    gb_diff = (B2.grad.float() - B.grad.float()).abs().max().item()
    assert ga_diff < GRAD_MAX_ABS_DIFF, f"grad_A diff too large: {ga_diff}"
    assert gb_diff < GRAD_MAX_ABS_DIFF, f"grad_B diff too large: {gb_diff}"
    log(f"    grad_A max_diff={ga_diff:.6f}, grad_B max_diff={gb_diff:.6f}")


def test_swiglu_moe_forward_backward():
    """Test the full MoE FFN: gate_proj, up_proj, down_proj with grouped_mm,
    matching the compute path of EPMoELayerBase._separate_glu_experts_gmm."""
    G = N_ROUTED_EXPERTS
    M = 16
    # Weights in matmul convention [E, K, N], as the EP MoE layers store them
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
    log(f"    output shape={output.shape}, x.grad shape={x.grad.shape}")


class MoEFFN(nn.Module):
    """Mini MoE FFN using grouped_mm, matching the EP MoE layers' compute path."""

    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        # Weights stored [E, out, in]; forward transposes them to the [E, K, N] matmul convention
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
    """Train a MoE FFN with grouped_mm on one fixed batch and verify loss decreases."""
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
    for _ in range(50):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = torch.nn.functional.mse_loss(y_pred, y_target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    log(f"    loss: {losses[0]:.6f} -> {losses[-1]:.6f} ({(1 - losses[-1] / losses[0]) * 100:.1f}% drop)")


def test_training_with_sum_loss():
    """Train with .sum() loss (the exact zero-stride backward that crashes natively)."""
    num_experts, hidden, intermediate = 8, 64, 256
    model = MoEFFN(num_experts, hidden, intermediate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for _ in range(20):
        optimizer.zero_grad()
        x = torch.randn(num_experts, 16, hidden, dtype=torch.bfloat16, device=device)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

    log("    .sum() backward completed 20 steps without crash")


@gpu_test_main(exact_world_size=1, prefix="grouped_mm_b300", partial_state=False)
def run(ctx) -> dict:
    log(f"Grouped MM backward tests on {torch.cuda.get_device_name()}")
    checks: dict[str, bool] = {}
    record_check(checks, "3d_forward_correctness", test_3d_forward_correctness)
    record_check(checks, "3d_backward_sum_zero_stride_grad", test_3d_backward_sum)
    record_check(checks, "3d_backward_correctness_vs_loop", test_3d_backward_correctness)
    record_check(checks, "jagged_backward_sum_zero_stride_grad", test_jagged_backward_sum)
    record_check(checks, "jagged_backward_correctness_vs_loop", test_jagged_backward_correctness)
    record_check(checks, "swiglu_moe_ffn_forward_backward", test_swiglu_moe_forward_backward)
    record_check(checks, "moe_ffn_training_convergence", test_training_convergence)
    record_check(checks, "training_with_sum_loss_zero_stride_backward", test_training_with_sum_loss)
    return {"checks": checks}


if __name__ == "__main__":
    run()
