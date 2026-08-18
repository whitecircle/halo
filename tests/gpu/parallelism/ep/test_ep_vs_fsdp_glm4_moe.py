#!/usr/bin/env python
"""GLM-4 MoE Lite EP=2 vs replicated-reference correctness on a tiny random-init model (no download).

Both ranks build the identical tiny GLM-4 MoE Lite (fixed seed) and run the same batch through
(a) the stock HF model and (b) the EP=2-patched model. Verifies:

  1. EP patching installs EPGlm4MoELayer on every sparse MoE block and leaves the
     ``first_k_dense_replace`` leading dense layers' plain MLP untouched, preserving the shared
     experts and the group-limited routing knobs, with the experts split evenly across ranks.
  2. EP loss == reference loss (per rank) and losses agree across ranks.
  3. Expert-shard gradients equal the reference expert gradients' local slice; router-gate and
     shared-expert gradients match the reference (DP-average hook path).

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_vs_fsdp_glm4_moe.py
"""

import torch
from transformers import AutoModelForCausalLM
from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteMLP

from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_GLM4_MOE_LITE_CONFIG
from tests.common.utils import cos_sim, log

SEED = 42
BATCH, SEQ = 2, 64
LOSS_TOL = 5e-2  # bf16 dispatch/accumulation-order noise on a tiny model
RANK_LOSS_TOL = 1e-3  # EP is orthogonal to DP: identical input → identical loss
# Grad equivalence: bf16 grads on a 128-token tiny model carry real rounding noise and the two
# paths accumulate in different orders — so direction is checked loosely (cos) while MAGNITUDE is
# checked tightly enough that a missing or doubled /world_size grad-sync divide (ratio 2.0 / 0.5)
# fails.
GRAD_COS_MIN = 0.9
GRAD_NORM_RATIO = (0.67, 1.5)

_DENSE_LAYERS = list(range(TINY_GLM4_MOE_LITE_CONFIG["first_k_dense_replace"]))
_SPARSE_LAYERS = list(
    range(TINY_GLM4_MOE_LITE_CONFIG["first_k_dense_replace"], TINY_GLM4_MOE_LITE_CONFIG["num_hidden_layers"])
)


def _build_model(device):
    """Identical tiny model on every rank (same seed)."""
    torch.manual_seed(SEED)
    config = Glm4MoeLiteConfig(**{**TINY_GLM4_MOE_LITE_CONFIG, "attn_implementation": "eager"})
    model = AutoModelForCausalLM.from_config(config)
    return model.to(device=device, dtype=torch.bfloat16)


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, TINY_GLM4_MOE_LITE_CONFIG["vocab_size"], (BATCH, SEQ), device=device)
    labels = input_ids.clone()

    # ── Reference: stock HF model, forward + backward ─────────────────────────
    ref = _build_model(device)
    ref.train()
    ref_out = ref(input_ids=input_ids, labels=labels)
    ref_out.loss.backward()
    ref_loss = ref_out.loss.item()
    log(f"reference loss: {ref_loss:.6f}")

    ref_grads = []
    for i in _SPARSE_LAYERS:
        mlp = ref.model.layers[i].mlp
        ref_grads.append(
            {
                "gate_up": mlp.experts.gate_up_proj.grad.detach().clone(),  # [E, 2M, H]
                "down": mlp.experts.down_proj.grad.detach().clone(),  # [E, H, M]
                "gate": mlp.gate.weight.grad.detach().clone(),
                "shared_gate": mlp.shared_experts.gate_proj.weight.grad.detach().clone(),
            }
        )
    del ref, ref_out
    torch.cuda.empty_cache()

    # ── EP=2: same weights, same batch ────────────────────────────────────────
    model = _build_model(device)
    pc = ParallelismConfig(ep_size=ctx.world_size)
    model = patch_moe_model_for_ep(model, pc.create_ep_config())
    create_ep_buffers(model)

    ep_layers = [m for m in model.modules() if isinstance(m, EPGlm4MoELayer)]
    checks["ep_layers_found"] = len(ep_layers) == len(_SPARSE_LAYERS)
    checks["dense_layers_untouched"] = all(
        isinstance(model.model.layers[i].mlp, Glm4MoeLiteMLP) for i in _DENSE_LAYERS
    )
    checks["correct_ep_type"] = all(type(ep) is EPGlm4MoELayer for ep in ep_layers)
    checks["ep_config_present"] = all(hasattr(ep, "ep_config") for ep in ep_layers)
    checks["dispatcher_present"] = all(hasattr(ep, "dispatcher") for ep in ep_layers)
    checks["ep_rank_present"] = all(hasattr(ep, "ep_rank") for ep in ep_layers)
    checks["expert_distribution"] = all(
        ep.experts_per_rank == ep.num_experts // ctx.world_size
        and ep.expert_end - ep.expert_start == ep.experts_per_rank
        for ep in ep_layers
    )
    checks["shared_experts_preserved"] = all(ep.shared_experts is not None for ep in ep_layers)
    checks["routing_params_present"] = all(
        hasattr(ep, "n_group") and hasattr(ep, "topk_group") and hasattr(ep, "top_k") for ep in ep_layers
    )

    model.train()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    ep_loss = out.loss.item()
    log(f"EP loss: {ep_loss:.6f}  |Δref| = {abs(ep_loss - ref_loss):.2e}")
    metrics["ref_loss"] = ref_loss
    metrics["ep_loss"] = ep_loss
    checks["loss_finite"] = bool(torch.isfinite(out.loss))
    checks["ep_loss_matches_ref"] = abs(ep_loss - ref_loss) < LOSS_TOL

    # Losses must agree across ranks (identical input; EP orthogonal to DP).
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

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="ep_vs_fsdp_glm4_moe")(run)

if __name__ == "__main__":
    main()
