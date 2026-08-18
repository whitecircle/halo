#!/usr/bin/env python
"""Single-EP-group hook regime: the /world divide must survive a zero-token sync microbatch.

``create_expert_grad_hook`` divides the ACCUMULATED expert gradient on the sync microbatch, but a
post-accumulate hook fires only for params in that backward's graph. A rank whose expert bank
receives tokens in an early accumulation microbatch and ZERO dispatched tokens in the final one
must still end at accumulated/world — not at the raw sum (``_expert_hook_grad_edge`` keeps the
hooks firing). Routing is forced by a fixed-selection router stub so which rank idles is exact.

Schedule (ep2, top_k=1, all tokens to ONE expert):
  control: one microbatch to rank 1's expert, sync=True            → grad_ctl = g / world
  main:    µb1 to rank 1's expert (sync=False), µb2 to rank 0's expert (sync=True)
           → rank 1 idles in the sync microbatch; its grad must STILL equal grad_ctl
             (the missed-divide bug leaves it at world × grad_ctl).

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_hook_divide_zero_token.py
"""

import torch
import torch.nn as nn
from accelerate.state import GradientState
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe import Qwen3MoeConfig

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_MOE_CONFIG
from tests.common.utils import cleanup_memory, log, log_all

SEED = 42
BATCH, SEQ = 2, 64
# The bug signal is exactly ×world (2.0 here); kernel reruns of the identical microbatch only carry
# bf16 accumulation jitter, so a ±10% band is decisive.
GRAD_NORM_RATIO = (0.9, 1.11)


class _FixedRouter(nn.Module):
    """Router stub with x-independent selection: every token to ``target``, weight 1."""

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.target = 0

    def forward(self, x: torch.Tensor):
        tokens = x.shape[0]
        logits = torch.full((tokens, self.num_experts), -10.0, device=x.device, dtype=x.dtype)
        logits[:, self.target] = 10.0
        experts = torch.full((tokens, self.top_k), self.target, dtype=torch.long, device=x.device)
        weights = torch.ones((tokens, self.top_k), dtype=x.dtype, device=x.device)
        return logits, weights, experts


def _expert_grad_norms(ep_layers) -> list[float | None]:
    """Per-layer L2 norm over the hook-synced expert params' grads (None = no grad anywhere)."""
    norms: list[float | None] = []
    for layer in ep_layers:
        grads = [p.grad for p in layer._hook_synced_expert_params if p.grad is not None]
        norms.append(
            torch.linalg.vector_norm(torch.stack([g.float().norm() for g in grads])).item() if grads else None
        )
    return norms


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)

    torch.manual_seed(SEED)
    config = Qwen3MoeConfig(**{**TINY_QWEN3_MOE_CONFIG, "num_experts_per_tok": 1})
    model = AutoModelForCausalLM.from_config(config).to(device=device, dtype=torch.bfloat16)

    pc = ParallelismConfig(ep_size=ctx.world_size)
    model = patch_moe_model_for_ep(model, pc.create_ep_config())
    create_ep_buffers(model)

    ep_layers = [m for m in model.modules() if isinstance(m, EPMoELayerBase)]
    ep_cfg = ep_layers[0].ep_config

    # Anti-vacuity: this file certifies the HOOK regime — if defer_grad_sync ever covers the
    # single-group topology too, the premise (and this test's subject) is gone and must fail loudly.
    checks["hook_regime_active"] = (
        not ep_cfg.defer_grad_sync
        and not ep_cfg.experts_fsdp_managed
        and all(layer._hook_synced_expert_params for layer in ep_layers)
    )

    router = _FixedRouter(config.num_experts, top_k=1)
    for layer in ep_layers:
        layer.gate = router

    experts_per_rank = config.num_experts // ctx.world_size
    idle_rank_expert = experts_per_rank * (ctx.world_size - 1)  # rank world-1's first expert
    owns_idle_expert = ctx.rank == ctx.world_size - 1

    torch.manual_seed(SEED + ctx.rank)  # distinct per-rank batches, as in real DP
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=device)
    labels = input_ids.clone()
    model.train()

    grad_state = GradientState()

    # ── Control: one microbatch to the last rank's expert, sync=True → divided grad ──
    grad_state._set_sync_gradients(True)
    router.target = idle_rank_expert
    model(input_ids=input_ids, labels=labels).loss.backward()
    ctl_norms = _expert_grad_norms(ep_layers)
    if not owns_idle_expert:
        # This rank's bank never saw a token: the hook edge must NOT have materialized a grad.
        checks["untouched_bank_grad_stays_none"] = all(n is None for n in ctl_norms)
    log_all(f"  control expert-grad norms: {[f'{n:.4f}' if n is not None else 'None' for n in ctl_norms]}")

    model.zero_grad(set_to_none=True)

    # ── Main: accumulate on the last rank's expert, then a sync microbatch it idles in ──
    grad_state._set_sync_gradients(False)
    router.target = idle_rank_expert
    model(input_ids=input_ids, labels=labels).loss.backward()
    grad_state._set_sync_gradients(True)
    router.target = 0
    model(input_ids=input_ids, labels=labels).loss.backward()
    main_norms = _expert_grad_norms(ep_layers)
    log_all(f"  main expert-grad norms:    {[f'{n:.4f}' if n is not None else 'None' for n in main_norms]}")

    if owns_idle_expert:
        ratios = []
        for layer_idx, (main, ctl) in enumerate(zip(main_norms, ctl_norms, strict=True)):
            ok = main is not None and ctl is not None and ctl > 0.0
            ratio = (main / ctl) if ok else float("inf")
            ratios.append(ratio)
            checks[f"l{layer_idx}_idle_sync_microbatch_grad_divided"] = (
                ok and GRAD_NORM_RATIO[0] < ratio < GRAD_NORM_RATIO[1]
            )
        metrics["max_norm_ratio"] = max(ratios)
        log(f"  idle-rank grad norm ratios (bug = ~{ctx.world_size}.0): {[f'{r:.3f}' for r in ratios]}")

    del model
    cleanup_memory()
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, prefix="ep_hook_divide_zero_token")(run)

if __name__ == "__main__":
    main()
