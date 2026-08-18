#!/usr/bin/env python
"""DeepSeek-V4 EP=2 vs replicated-reference correctness on a tiny random-init model (no download).

Both ranks build the identical tiny DeepSeek-V4 (fixed seed, randomized tid2eid) and run the same
batch through (a) the stock HF model and (b) the EP=2-patched model. Verifies:

  1. EP patching installs EPDeepseekV4MoELayer on every MoE block (hash + top-k variants).
  2. EP loss == reference loss (per rank) and losses agree across ranks.
  3. Expert-shard gradients equal the reference expert gradients' local slice (the EP grad-sync
     hook's /world divide must reproduce the DP average).
  4. Router-gate and shared-expert gradients match the reference (DP-average hook path).
  5. Hash-layer routing: the captured EP selection is exactly tid2eid[input_ids].
  6. Bias balancing: hash layers refuse the bias (frozen selection), top-k layers accept it, and a
     large balancing bias forces the target expert while a zero bias changes nothing.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_vs_fsdp_deepseek_v4.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.deepseek_v4 import DeepseekV4Config

from src.distributed.expert_parallel.layers.deepseek_v4 import EPDeepseekV4MoELayer
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_DSV4_CONFIG
from tests.common.utils import cos_sim, log

SEED = 42
BATCH, SEQ = 2, 64
LOSS_TOL = 5e-2  # bf16 dispatch/accumulation-order noise on a tiny model
RANK_LOSS_TOL = 1e-3  # EP is orthogonal to DP: identical input → identical loss
# Grad equivalence: bf16 grads on a 128-token tiny model carry real rounding noise, and the EP
# wrapper routes in fp32 while stock HF scores in bf16 (occasional near-tie selection flips) — so
# direction is checked loosely (cos) while MAGNITUDE is checked tightly enough that a missing or
# doubled /world_size grad-sync divide (ratio 2.0 / 0.5) fails.
GRAD_COS_MIN = 0.9
GRAD_NORM_RATIO = (0.67, 1.5)


def _randomize_tid2eid(model) -> None:
    """DISTINCT experts per token id (random-init leaves the table all-zero; DeepEP dispatch and
    the wrapper's init guard both require distinct top-k experts per token)."""
    gen = torch.Generator().manual_seed(SEED)
    for layer in model.model.layers:
        if layer.mlp.is_hash:
            table = layer.mlp.gate.tid2eid
            perm = torch.rand(table.shape[0], model.config.n_routed_experts, generator=gen).argsort(dim=-1)
            table.copy_(perm[:, : table.shape[1]])


def _build_model(device):
    """Identical tiny model on every rank (same seed; tid2eid spread over experts)."""
    torch.manual_seed(SEED)
    config = DeepseekV4Config(**{**TINY_DSV4_CONFIG, "attn_implementation": "eager"})
    model = AutoModelForCausalLM.from_config(config)
    _randomize_tid2eid(model)
    return model.to(device=device, dtype=torch.bfloat16)


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, TINY_DSV4_CONFIG["vocab_size"], (BATCH, SEQ), device=device)
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

    ep_layers = [m for m in model.modules() if isinstance(m, EPDeepseekV4MoELayer)]
    checks["ep_layers_patched"] = len(ep_layers) == len(TINY_DSV4_CONFIG["layer_types"])
    checks["hash_flag_preserved"] = [ep.is_hash for ep in ep_layers] == [True, False, False]
    checks["shared_experts_preserved"] = all(ep.shared_experts is not None for ep in ep_layers)

    # Capture the hash layer's routing during the training forward.
    hash_layer = ep_layers[0]
    hash_layer._capture_routing = True

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

    # ── 5. Hash-layer routing == tid2eid lookup (exact) ───────────────────────
    hash_layer._capture_routing = False
    captured = torch.cat(hash_layer._captured_routing_chunks, dim=0).long()
    expected = hash_layer.gate.tid2eid[input_ids.reshape(-1)].long()
    checks["hash_routing_is_tid2eid"] = torch.equal(captured, expected)
    hash_layer._captured_routing_chunks.clear()

    # ── 3+4. Gradient equivalence vs reference ────────────────────────────────
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

    # ── 6. Bias balancing: hash refuses, top-k accepts + bias shifts selection ─
    checks["hash_layer_refuses_bias"] = hash_layer.enable_bias_balancing() is False
    topk_layer = ep_layers[1]
    checks["topk_layer_accepts_bias"] = topk_layer.enable_bias_balancing() is True

    model.eval()
    with torch.no_grad():
        flat = torch.randn(BATCH * SEQ, topk_layer.hidden_dim, device=device, dtype=torch.bfloat16)
        logits = F.linear(flat, topk_layer.gate.weight)
        scores = topk_layer.score_fn(logits.float())
        natural = topk_layer._select_experts(scores, None)  # zero bias == natural selection
        target = 0
        topk_layer.balancing_biases[target] = 1e3
        forced = topk_layer._select_experts(scores, None)
        # The huge bias must pull the target expert into EVERY selection — and the natural
        # selection must not already do that (else the check is vacuous).
        checks["bias_shift_forces_target_expert"] = bool((forced == target).any(-1).all()) and not bool(
            (natural == target).any(-1).all()
        )
        topk_layer.balancing_biases.zero_()

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="ep_vs_fsdp_dsv4")(run)

if __name__ == "__main__":
    main()
