"""Mean step of AdamWBF16's stochastically rounded weight update against the exact AdamW step.

One AdamW step (lr 1e-5, first step, so the exact step is -lr * sign(grad)) on 2^20 bf16 weights of a given
magnitude, repeated over 64 seeds; prints the projection of the mean rounded step onto the exact step (1.000 =
unbiased). Run in the Halo image from a Halo checkout: python scripts/benchmarks/gemma4_sft/analysis/sr_rounding_bias.py
"""

import torch
from src.optimizers.adamw_bf16 import AdamWBF16

LR = 1e-5
TRIALS = 64
N = 1 << 20

torch.manual_seed(0)
for scale in (0.002, 0.02, 0.2):
    w0 = (torch.randn(N, device="cuda") * scale).to(torch.bfloat16)
    grad = torch.randn(N, device="cuda").to(torch.bfloat16)
    total = torch.zeros(N, device="cuda")
    for _ in range(TRIALS):
        p = torch.nn.Parameter(w0.clone())
        p.grad = grad.clone()
        AdamWBF16([p], lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0).step()
        total += p.detach().float() - w0.float()
    exact = -LR * torch.sign(grad.float())
    ratio = ((total / TRIALS) * exact).sum() / (exact * exact).sum()
    print(f"|w| ~ {scale}: mean rounded step / exact step = {ratio.item():.3f}")
