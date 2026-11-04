#!/usr/bin/env python
"""Single EP group: the /world expert divide survives a zero-token sync microbatch, on a fixed-selection router stub.

A rank idling in the final microbatch must still end at accumulated/world (a missed divide leaves it at world x the
control), and a bank that never got a token keeps ``grad is None``.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe import Qwen3MoeConfig

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common.ep_reference import sweep_ep_grads
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import TINY_QWEN3_MOE_CONFIG
from tests.common.numerics import max_or_nan
from tests.common.rank_log import log, log_all

SEED = 42
BATCH, SEQ = 2, 64
# The bug is exactly x world (2.0 here); reruns carry only bf16 accumulation jitter.
GRAD_NORM_RATIO = (0.9, 1.11)


class _FixedRouter(nn.Module):
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


def _expert_params(ep_layers) -> list[list[nn.Parameter]]:
    return [[p for _, p in layer.expert_named_params() if p.requires_grad] for layer in ep_layers]


def _expert_grad_norms(ep_layers) -> list[float | None]:
    norms: list[float | None] = []
    for params in _expert_params(ep_layers):
        grads = [p.grad for p in params if p.grad is not None]
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
    model = apply_ep_layers(model, pc.ep_config())

    ep_layers = [m for m in model.modules() if isinstance(m, EPMoELayerBase)]
    ep_cfg = ep_layers[0].ep_config

    # Anti-vacuity: certifies the single-group sweep regime, router on its hook, experts swept.
    checks["single_group_sweep_regime"] = (
        ep_cfg.runs_post_backward_sweep
        and not ep_cfg.sweep_reduces_across_replicas
        and not ep_cfg.experts_fsdp_managed
    )

    router = _FixedRouter(config.num_experts, top_k=1)
    for layer in ep_layers:
        layer.gate = router

    experts_per_rank = config.num_experts // ctx.world_size
    idle_rank_expert = experts_per_rank * (ctx.world_size - 1)  # Rank world-1's first expert.
    owns_idle_expert = ctx.rank == ctx.world_size - 1

    torch.manual_seed(SEED + ctx.rank)  # Distinct per-rank batches, as in real DP.
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ), device=device)
    labels = input_ids.clone()
    model.train()

    router.target = idle_rank_expert
    model(input_ids=input_ids, labels=labels).loss.backward()
    sweep_ep_grads(model)
    ctl_norms = _expert_grad_norms(ep_layers)
    if not owns_idle_expert:
        # The sweep must not have materialized a grad on this never-routed bank.
        checks["untouched_bank_grad_stays_none"] = all(n is None for n in ctl_norms)
    log_all(f"  control expert-grad norms: {[f'{n:.4f}' if n is not None else 'None' for n in ctl_norms]}")

    model.zero_grad(set_to_none=True)

    router.target = idle_rank_expert
    model(input_ids=input_ids, labels=labels).loss.backward()
    router.target = 0
    model(input_ids=input_ids, labels=labels).loss.backward()
    sweep_ep_grads(model)
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
        metrics["max_norm_ratio"] = max_or_nan(ratios)
        log(f"  idle-rank grad norm ratios (bug = ~{ctx.world_size}.0): {[f'{r:.3f}' for r in ratios]}")

    del model
    cleanup_memory()
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, prefix="ep_expert_divide_zero_token")(run)

if __name__ == "__main__":
    main()
