#!/usr/bin/env python
"""Step-3.7 EP=2 vs replicated-reference correctness on a tiny random-init composite model.

Both ranks build the identical tiny ``Step3p7ForConditionalGeneration`` (fixed seed — the family
ships no text-only CausalLM sibling) and run the same text-only batch through (a) the stock HF
model and (b) the EP=2-patched model. The tiny config is HETEROGENEOUS (per-layer 4-vs-2 attention
heads), so a global ``config.num_attention_heads`` read anywhere in the load/patch path raises
instead of silently mis-sizing. Verifies:

  1. EP patching installs EPStep3p7MoELayer on every SPARSE MoE block and leaves the leading dense
     layer's plain MLP untouched (``mlp_layer_types`` span), across the full/sliding interleave.
  2. EP loss == reference loss (per rank) and losses agree across ranks.
  3. Expert-shard gradients equal the reference expert gradients' local slice, and router-gate /
     shared-expert gradients match the reference (DP-average hook path).
  4. The PER-LAYER clamp bounds (inf on the unclamped MoE layer, the active bound on the clamped
     ones) and the block-level routed scaling are adopted from each block (not defaulted).
  5. ``moe_balancing: auto`` resolves to ``bias_update`` (no aux machinery in the modeling, native
     exported slot), bias balancing adopts the native ``gate.e_score_correction_bias`` buffer, and
     a large bias forces the target expert while the natural selection does not already pick it.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_vs_fsdp_step3p7.py
"""

import math

import torch
import torch.nn.functional as F
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config
from transformers.models.step3p7.modeling_step3p7 import Step3p7ForConditionalGeneration, Step3p7MLP

from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from src.models.moe_balancing import resolve_balancing_mode
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_STEP3P7_CONFIG, TINY_STEP3P7_VISION_CONFIG
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

_SPARSE_LAYERS = [i for i, kind in enumerate(TINY_STEP3P7_CONFIG["mlp_layer_types"]) if kind == "sparse"]
_DENSE_LAYERS = [i for i, kind in enumerate(TINY_STEP3P7_CONFIG["mlp_layer_types"]) if kind == "dense"]
# The per-layer contract: 0 in ``swiglu_limits`` means unclamped (the wrapper stores inf).
_EXPECTED_LIMITS = [TINY_STEP3P7_CONFIG["swiglu_limits"][i] or math.inf for i in _SPARSE_LAYERS]


def _build_model(device):
    """Identical tiny composite model on every rank (same seed); ``image_token_id`` sits inside the
    tiny vocab (the family default 151679 does not)."""
    torch.manual_seed(SEED)
    config = Step3p7Config(
        text_config=dict(TINY_STEP3P7_CONFIG),
        vision_config=dict(TINY_STEP3P7_VISION_CONFIG),
        image_token_id=2000,
    )
    config._attn_implementation = "eager"
    model = Step3p7ForConditionalGeneration(config)
    return model.to(device=device, dtype=torch.bfloat16)


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, TINY_STEP3P7_CONFIG["vocab_size"], (BATCH, SEQ), device=device)
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
        mlp = ref.model.language_model.layers[i].mlp
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

    ep_layers = [m for m in model.modules() if isinstance(m, EPStep3p7MoELayer)]
    text_layers = model.model.language_model.layers
    checks["ep_layers_patched"] = len(ep_layers) == len(_SPARSE_LAYERS)
    checks["dense_layers_untouched"] = all(isinstance(text_layers[i].mlp, Step3p7MLP) for i in _DENSE_LAYERS)
    checks["shared_experts_preserved"] = all(ep.shared_experts is not None for ep in ep_layers)
    checks["per_layer_swiglu_limits_adopted"] = [ep.swiglu_limit for ep in ep_layers] == _EXPECTED_LIMITS
    checks["routed_scaling_adopted"] = all(
        ep.routed_scaling_factor == TINY_STEP3P7_CONFIG["moe_router_scaling_factor"] for ep in ep_layers
    )
    # No aux machinery in the modeling + native exported slot → auto must land on bias_update.
    checks["auto_balancing_is_bias_update"] = resolve_balancing_mode("auto", model, is_moe=True) == "bias_update"

    model.train()
    out = model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    ep_loss = out.loss.item()
    log(f"EP loss: {ep_loss:.6f}  |Δref| = {abs(ep_loss - ref_loss):.2e}")
    metrics["ref_loss"] = ref_loss
    metrics["ep_loss"] = ep_loss
    checks["ep_loss_finite"] = bool(torch.isfinite(out.loss))
    checks["ep_loss_matches_ref"] = abs(ep_loss - ref_loss) < LOSS_TOL

    # Losses must agree across ranks (identical input; EP orthogonal to DP).
    loss_t = torch.tensor([ep_loss], device=device)
    gathered = [torch.zeros_like(loss_t) for _ in range(ctx.world_size)]
    torch.distributed.all_gather(gathered, loss_t)
    spread = max(abs(g.item() - gathered[0].item()) for g in gathered)
    metrics["rank_loss_spread"] = spread
    checks["losses_match_across_ranks"] = spread < RANK_LOSS_TOL

    # ── 3. Gradient equivalence vs reference ──────────────────────────────────
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

    # ── 5. Bias balancing: native buffer adopted + bias shifts selection ──────
    layer = ep_layers[0]
    checks["can_adopt_native"] = layer.can_adopt_native_balancing() is True
    checks["accepts_bias"] = layer.enable_bias_balancing() is True
    checks["native_slot_adopted"] = layer.balancing_biases is layer.gate.e_score_correction_bias

    model.eval()
    with torch.no_grad():
        flat = torch.randn(BATCH * SEQ, layer.hidden_dim, device=device, dtype=torch.bfloat16)
        logits = F.linear(flat.float(), layer.gate.weight.float())
        natural = layer.route_tokens_to_experts(logits)[0]  # zero bias == natural selection
        target = 0
        layer.balancing_biases[target] = 1e3
        forced = layer.route_tokens_to_experts(logits)[0]
        # The huge bias must pull the target expert into EVERY selection — and the natural
        # selection must not already do that (else the check is vacuous).
        checks["bias_shift_forces_target_expert"] = bool((forced == target).any(-1).all()) and not bool(
            (natural == target).any(-1).all()
        )
        layer.balancing_biases.zero_()

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="ep_vs_fsdp_step3p7")(run)

if __name__ == "__main__":
    main()
