#!/usr/bin/env python
"""Production-readiness: apply_mixed_precision_compute on a representative model, verifying that
- only the dense MLP projections (gate/up/down) are converted to low precision; attention q/k/v/o stay bf16,
- a forward->backward->optimizer loop converges with BOTH bf16 and fp32 master weights (the low-precision
  fake-quant path keeps the master unchanged — bf16 master, low-precision compute).

Run: python tests/gpu/kernels/test_lowp_production.py
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kernels.lowp.linear import LowPrecisionLinear
from src.kernels.lowp.mixed_precision import apply_mixed_precision_compute

DEV = "cuda"


class Attn(nn.Module):  # attention projections — must stay bf16
    def __init__(self, h):
        super().__init__()
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class MLP(nn.Module):  # SwiGLU dense MLP — gate/up/down should be converted
    def __init__(self, h, inter):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.self_attn = Attn(h)
        self.mlp = MLP(h, inter)

    def forward(self, x):
        return x + self.mlp(x + self.self_attn(x))


class Model(nn.Module):
    def __init__(self, h, inter, n=2):
        super().__init__()
        self.layers = nn.ModuleList([Block(h, inter) for _ in range(n)])

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


def test_conversion_scope():
    """Only MLP gate/up/down are converted; attention q/k/v/o stay bf16."""
    m = Model(2048, 8192).to(DEV, torch.bfloat16)
    summary = apply_mixed_precision_compute(m, precision="fp8", apply_dense_mlp=True, apply_moe_experts=False)
    for blk in m.layers:
        for p in ("gate_proj", "up_proj", "down_proj"):
            assert isinstance(getattr(blk.mlp, p), LowPrecisionLinear), f"mlp.{p} not converted"
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert not isinstance(getattr(blk.self_attn, p), LowPrecisionLinear), f"attn.{p} WAS converted"
    assert summary["dense_linears"] == 6, summary  # 2 blocks * 3 MLP projections
    print(f"  conversion scope: MLP gate/up/down converted ({summary['dense_linears']}), attention bf16 — OK")
    x = torch.randn(8192, 2048, device=DEV, dtype=torch.bfloat16) * 0.3
    out = m(x)
    assert out.isfinite().all(), "forward produced non-finite output"
    print("  dense fp8 fake-quant forward finite — OK")


def test_converges_both_masters():
    """A train loop converges with bf16 AND fp32 master weights; params keep their master dtype."""
    torch.manual_seed(0)
    M, H, I = 8192, 2048, 8192
    x = torch.randn(M, H, device=DEV, dtype=torch.bfloat16) * 0.3
    tgt = torch.randn(M, H, device=DEV, dtype=torch.bfloat16) * 0.3
    for master in (torch.bfloat16, torch.float32):
        torch.manual_seed(1)
        m = Model(H, I, n=1).to(DEV, master)
        apply_mixed_precision_compute(m, precision="fp8", apply_dense_mlp=True, apply_moe_experts=False)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        first = last = None
        for i in range(20):
            opt.zero_grad(set_to_none=True)
            # bf16-autocast compute (as the trainer does) — for an fp32 master (fp32_non_ep_params) the
            # bf16 activations meet the fp32 weights under autocast, exactly the real-training path.
            with torch.autocast(DEV, dtype=torch.bfloat16):
                out = m(x)
            loss = (out.float() - tgt.float()).pow(2).mean()
            loss.backward()
            opt.step()
            if i == 0:
                first = loss.item()
            last = loss.item()
        assert last < first, f"{master}: did not converge {first:.3f}->{last:.3f}"
        assert all(p.dtype == master for p in m.parameters()), f"{master}: a param changed dtype"
        print(
            f"  master={str(master).replace('torch.', ''):8}: loss {first:.3f}->{last:.3f} (converges), "
            f"params stay {str(master).replace('torch.', '')} — OK"
        )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        sys.exit(0)
    print("test_conversion_scope")
    test_conversion_scope()
    print("test_converges_both_masters")
    test_converges_both_masters()
    print("ALL PASS")
