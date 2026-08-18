#!/usr/bin/env python
"""Bailing V3 (Ling 3.0) EP=2 vs replicated-reference correctness on a tiny random block.

``EPBailingMoELayer`` claims ``BailingMoeV3SparseMoeBlock`` on the strength of V3's MoE block being
V2's: the same per-expert ``gate_proj``/``up_proj``/``down_proj`` modules and the same gate
arithmetic (sigmoid, ``expert_bias`` on selection only, group-limited top-k, renormalisation,
``routed_scaling_factor``). That claim is what this test defends — if inclusionAI changes either, or
the registration is dropped, the numbers move.

Only the MoE block is built, never the model: Ling 3.0's decoder is 3/4 KDA linear attention behind
``fla`` Triton kernels whose first-use JIT dominates the runtime and tests nothing about experts.

Verifies:

  1. Both registries claim V3 — the live-module map keyed on class name, and the config-keyed map
     used off-line by the shard merge and the hub renames.
  2. EP output == the stock block's output, and the block's own routing decisions are reproduced
     (a wrapper that renormalised differently, or dropped ``routed_scaling_factor``, still returns
     plausible values — comparing against the stock block is what separates those from noise).
  3. Expert-shard gradients equal the reference gradients' local slice, and the router gate stays
     live.

The remote code is downloaded (config + modeling module only, no weights).

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_vs_reference_bailing_v3.py
"""

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type
from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP, create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from src.models.patches.remote_code_compat import apply_remote_code_compat_shims
from tests.common.harness import gpu_test_main
from tests.common.models import BAILING_LING_3_TINY
from tests.common.utils import cos_sim, log

SEED = 42
BATCH, SEQ = 2, 16
HIDDEN = 256  # DeepEP pads the token dim to a multiple of 256; keep the block at the boundary
NUM_EXPERTS = 8
OUT_TOL = 2e-2  # bf16 grouped-GEMM vs the reference's per-expert loop, different accumulation order
GRAD_COS_TOL = 0.97  # bf16 grads on a tiny block; fp32 EP routing vs the block's own scoring


def _tiny_config():
    """Ling-3.0-tiny's config shrunk to a single small MoE block's worth of dimensions."""
    config = AutoConfig.from_pretrained(BAILING_LING_3_TINY, trust_remote_code=True)
    config.hidden_size = HIDDEN
    config.moe_intermediate_size = 64
    config.moe_shared_expert_intermediate_size = 64
    config.num_experts = NUM_EXPERTS
    config.num_experts_per_tok = 2
    config.n_group = 2
    config.topk_group = 1
    config.hidden_act = "silu"
    return config


def _build_block(config, device):
    """Identical tiny MoE block on every rank (same seed)."""
    block_cls = get_class_from_dynamic_module(
        "modeling_bailing_moe_v3.BailingMoeV3SparseMoeBlock", BAILING_LING_3_TINY
    )
    torch.manual_seed(SEED)
    return block_cls(config).to(device=device, dtype=torch.bfloat16)


def _hidden(block_output):
    """The V3 block returns ``(hidden, router_logits)``; the EP wrapper returns the tensor alone."""
    return block_output[0] if isinstance(block_output, tuple) else block_output


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)
    apply_remote_code_compat_shims()

    # ── 1. Both registries claim V3 ───────────────────────────────────────────
    checks["module_registry_claims_v3"] = MOE_LAYER_MAP.get("BailingMoeV3SparseMoeBlock") is EPBailingMoELayer
    checks["model_type_registry_claims_v3"] = ep_layer_class_by_model_type().get("bailing_hybrid") is EPBailingMoELayer

    config = _tiny_config()
    torch.manual_seed(SEED)
    hidden_states = torch.randn(BATCH, SEQ, HIDDEN, device=device, dtype=torch.bfloat16)

    # ── 2. Reference: the stock block, forward + backward ─────────────────────
    reference = _build_block(config, device)
    reference.train()
    ref_out = _hidden(reference(hidden_states))
    ref_out.sum().backward()
    ref_grads = {
        "gate_proj": torch.stack([e.gate_proj.weight.grad for e in reference.experts]),  # [E, M, H]
        "up_proj": torch.stack([e.up_proj.weight.grad for e in reference.experts]),  # [E, M, H]
        "down_proj": torch.stack([e.down_proj.weight.grad for e in reference.experts]),  # [E, H, M]
        "gate": reference.gate.weight.grad.detach().clone(),
    }
    ref_out = ref_out.detach().clone()

    # ── 3. EP=2 over the same weights and the same batch ──────────────────────
    container = nn.Module()  # patching walks a module tree; the block alone is the tree here
    container.mlp = _build_block(config, device)
    pc = ParallelismConfig(ep_size=ctx.world_size)
    patch_moe_model_for_ep(container, pc.create_ep_config())
    create_ep_buffers(container)

    ep_layer = container.mlp
    checks["block_patched"] = isinstance(ep_layer, EPBailingMoELayer)
    checks["shared_experts_preserved"] = ep_layer.shared_experts is not None
    checks["expert_count"] = ep_layer.num_experts == NUM_EXPERTS
    checks["experts_sharded"] = (ep_layer.expert_end - ep_layer.expert_start) == NUM_EXPERTS // ctx.world_size

    ep_layer.train()
    ep_out = _hidden(ep_layer(hidden_states))
    ep_out.sum().backward()

    max_abs = (ep_out.detach().float() - ref_out.float()).abs().max().item()
    cosine = cos_sim(ep_out.detach(), ref_out)
    metrics["out_max_abs_diff"] = max_abs
    metrics["out_cosine"] = cosine
    checks["ep_output_finite"] = bool(torch.isfinite(ep_out).all())
    checks["ep_output_matches_reference"] = max_abs < OUT_TOL
    log(f"output: max|Δ|={max_abs:.3e} (tol {OUT_TOL:.1e})  cosine={cosine:.6f}")

    # ── 4. Gradient equivalence vs the reference's local expert slice ─────────
    s, e = ep_layer.expert_start, ep_layer.expert_end
    # EP stores matmul convention: gate/up [E_local, H, M], down [E_local, M, H]; ref is nn.Linear.
    pairs = {
        "gate_proj_grad": (ep_layer.gate_proj.grad, ref_grads["gate_proj"][s:e].transpose(1, 2)),
        "up_proj_grad": (ep_layer.up_proj.grad, ref_grads["up_proj"][s:e].transpose(1, 2)),
        "down_proj_grad": (ep_layer.down_proj.grad, ref_grads["down_proj"][s:e].transpose(1, 2)),
        "gate_grad": (ep_layer.gate.weight.grad, ref_grads["gate"]),
    }
    for name, (got, want) in pairs.items():
        cos = cos_sim(got, want)
        metrics[f"{name}_cos"] = cos
        checks[name] = cos > GRAD_COS_TOL

    checks["shared_expert_grad_live"] = (
        ep_layer.shared_experts.gate_proj.weight.grad is not None
        and ep_layer.shared_experts.gate_proj.weight.grad.abs().sum().item() > 0
    )
    # Ling 3.0 balancing: the wrapper ADOPTS the gate's native ``expert_bias`` for bias_update, so
    # the trained bias exports with every checkpoint. Enabling must succeed and must hand back the
    # gate's own buffer — a private copy would train a bias no checkpoint carries.
    checks["layer_adopts_native_bias"] = (
        ep_layer.enable_bias_balancing() is True and ep_layer.balancing_biases is ep_layer.gate.expert_bias
    )

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="ep_vs_ref_bailing_v3")(run)

if __name__ == "__main__":
    main()
