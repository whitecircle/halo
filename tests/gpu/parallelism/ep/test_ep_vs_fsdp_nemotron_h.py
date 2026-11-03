#!/usr/bin/env python
"""NemotronH EP=2 vs replicated-reference correctness on a tiny random-init model (no download).

Both ranks build the identical tiny hybrid Mamba-2 / MoE / attention stack (fixed seed), once plain and
once with a latent projection narrower than the hidden size in front of the experts. Each rank trains
the EP=2-patched model on its OWN batch; the reference is the stock HF model backpropagating the mean
of every rank's loss, which is what the EP hooks' averaging must reproduce. Verifies:

  1. EP patching installs EPNemotronHMoELayer on every MoE block, keeping the shared expert, and a
     latent block dispatches at the latent width.
  2. Each rank's EP loss == the reference loss on that rank's batch.
  3. Expert-shard gradients equal the reference expert gradients' local slice; router, shared-expert
     and both latent-projection gradients match the reference — distinct batches per rank, so a
     missing DP average leaves them off by the other rank's contribution.
  4. Native bias balancing: the correction bias the layer adopts forces a target expert into every
     selection that the natural selection never picks.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_vs_fsdp_nemotron_h.py
"""

import torch
from transformers import AutoModelForCausalLM
from transformers.models.nemotron_h import NemotronHConfig

from src.distributed.expert_parallel.layers.nemotron_h import EPNemotronHMoELayer
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.ep_reference import score_ep_grad_pairs
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_NEMOTRON_H_CONFIG
from tests.common.tolerances import TOL
from tests.common.utils import log

SEED = 42
BATCH, SEQ = 2, 64
NUM_MOE_LAYERS = TINY_NEMOTRON_H_CONFIG["layers_block_type"].count("moe")
LATENT = 128


def _build_model(device, moe_latent_size):
    torch.manual_seed(SEED)
    config = NemotronHConfig(
        **{**TINY_NEMOTRON_H_CONFIG, "moe_latent_size": moe_latent_size, "attn_implementation": "eager"}
    )
    return AutoModelForCausalLM.from_config(config).to(device=device, dtype=torch.bfloat16)


def _moe_blocks(model):
    return [layer.mixer for layer in model.model.layers if layer.block_type == "moe"]


def _reference_grads(model):
    grads = []
    for block in _moe_blocks(model):
        entry = {
            "up": block.experts.up_proj.grad.detach().clone(),  # [E, M, H_in]
            "down": block.experts.down_proj.grad.detach().clone(),  # [E, H_in, M]
            "gate": block.gate.weight.grad.detach().clone(),
            "shared_up": block.shared_experts.up_proj.weight.grad.detach().clone(),
        }
        if isinstance(block.fc1_latent_proj, torch.nn.Linear):
            entry["latent_in"] = block.fc1_latent_proj.weight.grad.detach().clone()
            entry["latent_out"] = block.fc2_latent_proj.weight.grad.detach().clone()
        grads.append(entry)
    return grads


def _batch(rank, device):
    generator = torch.Generator(device=device).manual_seed(SEED + rank)
    return torch.randint(0, TINY_NEMOTRON_H_CONFIG["vocab_size"], (BATCH, SEQ), device=device, generator=generator)


def _compare(ctx, moe_latent_size, checks, metrics):
    tag = f"latent{moe_latent_size}" if moe_latent_size else "plain"
    device = ctx.device

    ref = _build_model(device, moe_latent_size)
    ref.train()
    for rank in range(ctx.world_size):
        ids = _batch(rank, device)
        ref_out = ref(input_ids=ids, labels=ids)
        (ref_out.loss / ctx.world_size).backward()
        if rank == ctx.rank:
            ref_loss = ref_out.loss.item()
    ref_grads = _reference_grads(ref)
    del ref, ref_out
    torch.cuda.empty_cache()

    model = _build_model(device, moe_latent_size)
    model = patch_moe_model_for_ep(model, ParallelismConfig(ep_size=ctx.world_size).create_ep_config())
    create_ep_buffers(model)
    ep_layers = [m for m in model.modules() if isinstance(m, EPNemotronHMoELayer)]
    checks[f"{tag}_ep_layers_patched"] = len(ep_layers) == NUM_MOE_LAYERS
    checks[f"{tag}_shared_experts_preserved"] = all(ep.shared_experts is not None for ep in ep_layers)
    expected_width = moe_latent_size or TINY_NEMOTRON_H_CONFIG["hidden_size"]
    checks[f"{tag}_dispatch_width"] = all(ep.hidden_dim == expected_width for ep in ep_layers)

    model.train()
    input_ids = _batch(ctx.rank, device)
    out = model(input_ids=input_ids, labels=input_ids)
    out.loss.backward()
    ep_loss = out.loss.item()
    log(f"{tag}: reference loss {ref_loss:.6f}, EP loss {ep_loss:.6f} (|Δ|={abs(ep_loss - ref_loss):.2e})")
    metrics[f"{tag}_ref_loss"] = ref_loss
    metrics[f"{tag}_ep_loss"] = ep_loss
    checks[f"{tag}_ep_loss_finite"] = bool(torch.isfinite(out.loss))
    checks[f"{tag}_ep_loss_matches_ref"] = abs(ep_loss - ref_loss) < TOL.parallel_vs_baseline_loss_abs

    for i, (ep, refs) in enumerate(zip(ep_layers, ref_grads, strict=True)):
        s, e = ep.expert_start, ep.expert_end
        # EP stores matmul convention [E_local, H_in, M] / [E_local, M, H_in]; ref is nn.Linear layout.
        pairs = {
            f"{tag}_l{i}_up_grad": (ep.up_proj.grad, refs["up"][s:e].transpose(1, 2)),
            f"{tag}_l{i}_down_grad": (ep.down_proj.grad, refs["down"][s:e].transpose(1, 2)),
            f"{tag}_l{i}_gate_grad": (ep.gate.weight.grad, refs["gate"]),
            f"{tag}_l{i}_shared_grad": (ep.shared_experts.up_proj.weight.grad, refs["shared_up"]),
        }
        if "latent_in" in refs:
            pairs[f"{tag}_l{i}_latent_in_grad"] = (ep.fc1_latent_proj.weight.grad, refs["latent_in"])
            pairs[f"{tag}_l{i}_latent_out_grad"] = (ep.fc2_latent_proj.weight.grad, refs["latent_out"])
        score_ep_grad_pairs(pairs, checks, metrics, cos_min=TOL.ep_grad_cosine_min)
    return model, ep_layers


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    torch.cuda.set_device(ctx.device)

    _compare(ctx, None, checks, metrics)
    model, ep_layers = _compare(ctx, LATENT, checks, metrics)

    layer = ep_layers[0]
    checks["native_slot_adoptable"] = layer.can_adopt_native_balancing()
    model.eval()
    with torch.no_grad():
        num_experts = TINY_NEMOTRON_H_CONFIG["n_routed_experts"]
        logits = torch.randn(BATCH * SEQ, num_experts, device=ctx.device)
        target = 3
        logits[:, target] = logits.min() - 10.0  # never picked naturally — the check cannot be vacuous
        bias = layer.gate.e_score_correction_bias
        saved = bias.clone()
        natural, _ = layer.route_tokens_to_experts(logits)
        bias[target] = 1e3
        forced, _ = layer.route_tokens_to_experts(logits)
        checks["correction_bias_forces_target_expert"] = bool((forced == target).any(-1).all()) and not bool(
            (natural == target).any(-1).any()
        )
        bias.copy_(saved)

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="ep_vs_fsdp_nemotron_h")(run)

if __name__ == "__main__":
    main()
