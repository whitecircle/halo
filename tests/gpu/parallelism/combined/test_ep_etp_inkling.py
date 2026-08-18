#!/usr/bin/env python
"""Inkling ETP correctness vs an undistributed reference on a tiny random-init model (no download).

`EPInklingMoELayer` inherits the whole ETP path from `EPMoELayerBase` (contiguous-halves split into
separate ``gate_proj``/``up_proj`` shards, token-space partial-sum reduce), so this gate proves the
inherited path holds for Inkling's layout — the joint routed+shared normalisation runs in token
space on every ETP partner, and the replicated shared-experts leg must not be double-counted by the
partial-sum reduce. The world size picks the shape:

  * 2 GPUs — pure ETP (``ep_size=1, expert_tp_size=2``), experts replicated, FFNs 2-way sharded.
  * 4 GPUs — EP+ETP (``ep_size=2, expert_tp_size=2``), the production combination.

Checks: the ETP layers store split (not fused) GLU shards; per-rank loss matches the undistributed
reference; losses agree across ranks (every rank sees the full batch); router-gate, expert-shard,
and shared-experts gradients are live and finite.

Run with 2 or 4 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/combined/test_ep_etp_inkling.py
"""

import torch
from transformers import AutoModelForCausalLM
from transformers.models.inkling.configuration_inkling import InklingTextConfig

from src.distributed.expert_parallel.layers.inkling import EPInklingMoELayer
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_INKLING_CONFIG
from tests.common.utils import cleanup_memory, log

SEED = 42
BATCH, SEQ = 2, 64
LOSS_TOL = 5e-2  # bf16 noise + ETP's token-space reduce reordering near-tied top-k picks
RANK_LOSS_TOL = 1e-3  # every rank sees the full batch, so per-rank losses must agree


def _build_model(device):
    """Identical tiny model on every rank (same seed)."""
    torch.manual_seed(SEED)
    config = InklingTextConfig(**{**TINY_INKLING_CONFIG, "attn_implementation": "eager"})
    model = AutoModelForCausalLM.from_config(config)
    return model.to(device=device, dtype=torch.bfloat16)


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, TINY_INKLING_CONFIG["vocab_size"], (BATCH, SEQ), device=device)
    labels = input_ids.clone()

    # ── Reference: undistributed forward on the same weights and batch ────────
    ref = _build_model(device)
    ref.train()
    ref_loss = ref(input_ids=input_ids, labels=labels).loss.item()
    log(f"reference loss: {ref_loss:.6f}")
    del ref
    cleanup_memory()

    # ── ETP: world 2 → pure ETP (ep1×etp2); world 4 → EP+ETP (ep2×etp2) ──────
    model = _build_model(device)
    ep_size = ctx.world_size // 2
    pc = ParallelismConfig(ep_size=ep_size, expert_tp_size=2)
    log(f"patching for {pc.mode_string}")
    model = patch_moe_model_for_ep(model, pc.create_ep_config())
    create_ep_buffers(model)

    ep_layers = [m for m in model.modules() if isinstance(m, EPInklingMoELayer)]
    checks["ep_layers_patched"] = len(ep_layers) == TINY_INKLING_CONFIG["num_hidden_layers"]
    checks["shared_experts_preserved"] = all(ep.shared_experts is not None for ep in ep_layers)
    # ETP must store the GLU halves as separate shards — slicing the fused tensor along dim 2
    # would hand each rank only gate or only up.
    checks["etp_split_glu_storage"] = all(
        hasattr(ep, "gate_proj") and hasattr(ep, "up_proj") and not hasattr(ep, "gate_up_proj") for ep in ep_layers
    )

    model.train()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    etp_loss = out.loss.item()
    log(f"ETP loss: {etp_loss:.6f}  |Δref| = {abs(etp_loss - ref_loss):.2e}")
    metrics["ref_loss"] = ref_loss
    metrics["etp_loss"] = etp_loss
    checks["etp_loss_finite"] = bool(torch.isfinite(out.loss))
    checks["etp_loss_matches_ref"] = abs(etp_loss - ref_loss) < LOSS_TOL

    loss_t = torch.tensor([etp_loss], device=device)
    gathered = [torch.zeros_like(loss_t) for _ in range(ctx.world_size)]
    torch.distributed.all_gather(gathered, loss_t)
    spread = max(abs(g.item() - gathered[0].item()) for g in gathered)
    metrics["rank_loss_spread"] = spread
    checks["losses_match_across_ranks"] = spread < RANK_LOSS_TOL

    def _grad_live(param) -> bool:
        return param.grad is not None and bool(torch.isfinite(param.grad).all()) and param.grad.abs().sum().item() > 0

    checks["router_grad_live"] = all(_grad_live(ep.gate.weight) for ep in ep_layers)
    checks["expert_shard_grads_live"] = all(
        _grad_live(ep.gate_proj) and _grad_live(ep.up_proj) and _grad_live(ep.down_proj) for ep in ep_layers
    )
    checks["shared_grads_live"] = all(_grad_live(ep.shared_experts.gate_proj) for ep in ep_layers)

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, prefix="ep_etp_inkling")(run)

if __name__ == "__main__":
    main()
