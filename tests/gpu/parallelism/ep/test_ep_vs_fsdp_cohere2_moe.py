#!/usr/bin/env python
"""Cohere2 MoE EP=2 vs replicated-reference correctness on a tiny random-init model (no download).

Both ranks build the identical tiny Cohere2 MoE (fixed seed) and run the same batch through (a) the
stock HF model and (b) the EP=2-patched model. Verifies:

  1. EP patching installs EPCohere2MoELayer on every sparse MoE block, preserving the shared
     experts and the "average" combination (``_output_scale == 0.5``).
  2. EP loss == reference loss (per rank) and losses agree across ranks.
  3. Expert-shard gradients equal the reference expert gradients' local slice; router-gate and
     shared-expert gradients match the reference (DP-average hook path).
  4. Transient bias balancing: the layer accepts the side-buffer, a large bias forces the target
     expert into every selection, and the natural selection does not already do so.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_vs_fsdp_cohere2_moe.py
"""

import torch
from transformers import AutoModelForCausalLM
from transformers.models.cohere2_moe import Cohere2MoeConfig

from src.distributed.expert_parallel.layers.cohere2_moe import EPCohere2MoELayer
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_COHERE2_MOE_CONFIG
from tests.common.utils import cos_sim, log

SEED = 42
BATCH, SEQ = 2, 64
LOSS_TOL = 5e-2  # bf16 dispatch/accumulation-order noise on a tiny model
RANK_LOSS_TOL = 1e-3  # EP is orthogonal to DP: identical input → identical loss
GRAD_COS_MIN = 0.9
GRAD_NORM_RATIO = (0.67, 1.5)


def _build_model(device):
    torch.manual_seed(SEED)
    config = Cohere2MoeConfig(**{**TINY_COHERE2_MOE_CONFIG, "attn_implementation": "eager"})
    model = AutoModelForCausalLM.from_config(config)
    return model.to(device=device, dtype=torch.bfloat16)


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, TINY_COHERE2_MOE_CONFIG["vocab_size"], (BATCH, SEQ), device=device)
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
                "shared_gate": layer.mlp.shared_experts.gate_proj.weight.grad.detach().clone(),
            }
        )
    del ref, ref_out
    torch.cuda.empty_cache()

    # ── EP=2: same weights, same batch ────────────────────────────────────────
    model = _build_model(device)
    pc = ParallelismConfig(ep_size=ctx.world_size)
    model = patch_moe_model_for_ep(model, pc.create_ep_config())
    create_ep_buffers(model)

    ep_layers = [m for m in model.modules() if isinstance(m, EPCohere2MoELayer)]
    checks["ep_layers_patched"] = len(ep_layers) == TINY_COHERE2_MOE_CONFIG["num_hidden_layers"]
    checks["shared_experts_preserved"] = all(ep.shared_experts is not None for ep in ep_layers)
    checks["average_combination_scaled"] = all(ep._output_scale == 0.5 for ep in ep_layers)
    checks["sigmoid_selection_adopted"] = all(ep.expert_selection_fn == "sigmoid" for ep in ep_layers)

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
            f"l{i}_shared_grad": (ep.shared_experts.gate_proj.weight.grad, refs["shared_gate"]),
        }
        for name, (got, want) in pairs.items():
            ok = got is not None and got.shape == want.shape
            cos = cos_sim(got, want) if ok else -1.0
            ratio = (got.float().norm() / want.float().norm().clamp_min(1e-12)).item() if ok else -1.0
            metrics[f"{name}_cos"] = cos
            metrics[f"{name}_norm_ratio"] = ratio
            checks[f"{name}_matches"] = ok and cos > GRAD_COS_MIN and GRAD_NORM_RATIO[0] < ratio < GRAD_NORM_RATIO[1]
            if not checks[f"{name}_matches"]:
                log(
                    f"  GRAD MISMATCH {name}: cos={cos:.5f} norm_ratio={ratio:.4f} "
                    f"shape={None if got is None else tuple(got.shape)}"
                )
    checks["shared_grads_nonzero"] = all(
        ep.shared_experts.gate_proj.weight.grad.abs().sum().item() > 0 for ep in ep_layers
    )

    # ── Transient bias balancing shifts selection ──────────────────────────────
    layer = ep_layers[0]
    checks["layer_accepts_transient_bias"] = layer.enable_bias_balancing() is True

    model.eval()
    with torch.no_grad():
        num_experts = TINY_COHERE2_MOE_CONFIG["num_experts"]
        logits = torch.randn(BATCH * SEQ, num_experts, device=device)
        target = 3
        logits[:, target] = logits.min() - 10.0  # never picked naturally — the check cannot be vacuous
        natural, _ = layer.route_tokens_to_experts(logits)
        layer.balancing_biases[target] = 1e3
        forced, _ = layer.route_tokens_to_experts(logits)
        checks["bias_shift_forces_target_expert"] = bool((forced == target).any(-1).all()) and not bool(
            (natural == target).any(-1).any()
        )
        layer.balancing_biases.zero_()

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="ep_vs_fsdp_cohere2_moe")(run)

if __name__ == "__main__":
    main()
