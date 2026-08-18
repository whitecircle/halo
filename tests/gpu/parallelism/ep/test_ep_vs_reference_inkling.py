#!/usr/bin/env python
"""Inkling EP=2 vs replicated-reference correctness on a tiny random-init model (no download).

The real checkpoint is 532 GB, so this builds the architecture from config and runs the same batch
through (a) the stock HF model and (b) the EP=2-patched model. Verifies:

  1. EP patching installs EPInklingMoELayer on every MoE block, shared experts preserved.
  2. EP loss == reference loss (per rank) and losses agree across ranks — this is what proves the
     joint routed+shared normalisation was reproduced: splitting it into two softmaxes changes
     every weight and the loss diverges.
  3. Expert-shard gradients equal the reference expert gradients' local slice.
  4. Router-gate and shared-expert gradients are live.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_vs_reference_inkling.py
"""

import torch
from transformers import AutoModelForCausalLM
from transformers.models.inkling.configuration_inkling import InklingTextConfig

from src.distributed.expert_parallel.layers.inkling import EPInklingMoELayer
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_INKLING_CONFIG
from tests.common.utils import cos_sim, log

SEED = 42
BATCH, SEQ = 2, 64
LOSS_TOL = 5e-2  # bf16 dispatch/accumulation-order noise on a tiny model
RANK_LOSS_TOL = 1e-3  # EP is orthogonal to DP: identical input → identical loss
GRAD_COS_TOL = 0.97  # bf16 grads on a 128-token tiny model; fp32 EP routing vs bf16 HF scoring
# flips occasional near-ties, so direction is checked loosely (matches the DeepSeek-V4 test).


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

    # ── Reference: stock HF model, forward + backward ─────────────────────────
    ref = _build_model(device)
    ref.train()
    ref_out = ref(input_ids=input_ids, labels=labels)
    ref_out.loss.backward()
    ref_loss = ref_out.loss.item()
    log(f"reference loss: {ref_loss:.6f}")

    ref_grads = []
    for layer in ref.model.layers:
        ref_grads.append(
            {
                "gate_up": layer.mlp.experts.gate_up_proj.grad.detach().clone(),  # [E, 2M, H]
                "down": layer.mlp.experts.down_proj.grad.detach().clone(),  # [E, H, M]
                "gate": layer.mlp.gate.weight.grad.detach().clone(),
            }
        )
    del ref, ref_out
    torch.cuda.empty_cache()

    # ── EP=2: same weights, same batch ────────────────────────────────────────
    model = _build_model(device)
    pc = ParallelismConfig(ep_size=ctx.world_size)
    model = patch_moe_model_for_ep(model, pc.create_ep_config())
    create_ep_buffers(model)

    ep_layers = [m for m in model.modules() if isinstance(m, EPInklingMoELayer)]
    checks["ep_layers_patched"] = len(ep_layers) == TINY_INKLING_CONFIG["num_hidden_layers"]
    checks["shared_experts_preserved"] = all(ep.shared_experts is not None for ep in ep_layers)
    # gate.weight is [n_routed + n_shared, hidden]; the wrapper must own only the routed experts.
    checks["routed_expert_count"] = all(ep.num_experts == TINY_INKLING_CONFIG["n_routed_experts"] for ep in ep_layers)

    model.train()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    ep_loss = out.loss.item()
    log(f"EP loss: {ep_loss:.6f}  |Δref| = {abs(ep_loss - ref_loss):.2e}")
    metrics["ref_loss"] = ref_loss
    metrics["ep_loss"] = ep_loss
    checks["ep_loss_finite"] = bool(torch.isfinite(out.loss))
    checks["ep_loss_matches_ref"] = abs(ep_loss - ref_loss) < LOSS_TOL

    loss_t = torch.tensor([ep_loss], device=device)
    gathered = [torch.zeros_like(loss_t) for _ in range(ctx.world_size)]
    torch.distributed.all_gather(gathered, loss_t)
    spread = max(abs(g.item() - gathered[0].item()) for g in gathered)
    metrics["rank_loss_spread"] = spread
    checks["losses_match_across_ranks"] = spread < RANK_LOSS_TOL

    # ── Gradient equivalence vs reference ─────────────────────────────────────
    for i, (ep, refs) in enumerate(zip(ep_layers, ref_grads, strict=True)):
        s, e = ep.expert_start, ep.expert_end
        # EP stores matmul convention [E_local, H, 2M] / [E_local, M, H]; ref is nn.Linear layout.
        pairs = {
            f"l{i}_gate_up_grad": (ep.gate_up_proj.grad, refs["gate_up"][s:e].transpose(1, 2)),
            f"l{i}_down_grad": (ep.down_proj.grad, refs["down"][s:e].transpose(1, 2)),
            f"l{i}_gate_grad": (ep.gate.weight.grad, refs["gate"]),
        }
        for name, (got, want) in pairs.items():
            cos = cos_sim(got, want)
            metrics[f"{name}_cos"] = cos
            checks[name] = cos > GRAD_COS_TOL

    checks["shared_grads_nonzero"] = all(
        ep.shared_experts.gate_proj.grad is not None and ep.shared_experts.gate_proj.grad.abs().sum().item() > 0
        for ep in ep_layers
    )
    checks["layer_accepts_bias"] = ep_layers[0].enable_bias_balancing() is True

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="ep_vs_ref_inkling")(run)

if __name__ == "__main__":
    main()
