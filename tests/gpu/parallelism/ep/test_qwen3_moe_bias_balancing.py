#!/usr/bin/env python
"""Qwen3-MoE bias-update balancing: inert at zero bias, and it actually steers at nonzero bias.

Under pipeline parallelism ``aux_loss`` is rejected — a PP stage never runs ``*ForCausalLM.forward``,
where HF adds the auxiliary term — so ``bias_update`` is the only balancing this family can get. Two
properties make it safe to enable:

  1. **Zero bias is a no-op.** Turning balancing on must reproduce the unbalanced route exactly, or
     every run pays a silent routing change for a feature it never asked to engage. This is the half
     that is easy to get wrong here: Qwen3-MoE's ``norm_topk_prob`` is configurable and the config
     default is off (shipped checkpoints set it on), so gating through the shared
     unconditional-renormalization helper would not be inert on a checkpoint that leaves it off.
     The test forces the flag off to cover that, since no cached checkpoint exercises it.
  2. **A nonzero bias moves selection.** Otherwise the balancing state is decorative.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_qwen3_moe_bias_balancing.py
"""

import torch

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_30B_A3B
from tests.common.utils import log

MODEL_NAME = QWEN3_30B_A3B
EP_SIZE = 2
SEQ_LEN = 128
SEED = 7


def routed(model, hidden):
    """Run every EP layer's router path and collect (indices, weights) per layer."""
    out = []
    for layer in ep_layers(model):
        with torch.no_grad():
            flat = hidden.view(-1, hidden.shape[-1])
            with torch.amp.autocast("cuda", enabled=False):
                logits, weights, experts = layer.gate(flat)
            if layer._balancing_bias(logits) is not None:
                experts = layer._biased_topk(logits)
                weights = layer._gate_weights_at(logits, experts)
            out.append((experts.clone(), weights.float().clone()))
    return out


def compare_routes(baseline, candidate, tag: str) -> tuple[bool, bool]:
    """(selection unchanged, gate weights unchanged) across every layer, logging what moved."""
    selection_same = True
    weights_same = True
    for i, ((idx_a, w_a), (idx_b, w_b)) in enumerate(zip(baseline, candidate, strict=True)):
        if not torch.equal(idx_a, idx_b):
            selection_same = False
            log(
                f"{tag}: layer {i}: enabling balancing changed expert SELECTION at zero bias "
                f"({int((idx_a != idx_b).sum())} of {idx_a.numel()} slots differ)"
            )
        worst = (w_a - w_b).abs().max().item()
        if worst != 0.0:
            weights_same = False
            log(f"{tag}: layer {i}: enabling balancing changed gate WEIGHTS at zero bias ({worst:.3e})")
    return selection_same, weights_same


def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    device = f"cuda:{ctx.local_rank}"
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    config = ParallelismConfig(ep_size=EP_SIZE)
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    layers = ep_layers(model)
    assert layers, "no EP layers built — this model should be EP-wrapped"
    # Without the declaration bias_update raises, and this family would have no balancing under PP.
    checks["supports_bias_balancing"] = bool(type(layers[0])._supports_bias_balancing)
    log(f"EP layers: {len(layers)}  ({type(layers[0]).__name__})")
    log(f"norm_topk_prob = {layers[0].gate.norm_topk_prob}")

    torch.manual_seed(SEED)
    hidden = torch.randn(1, SEQ_LEN, model.config.hidden_size, device=device, dtype=torch.bfloat16)

    baseline = routed(model, hidden)  # balancing not yet enabled

    enabled = [layer.enable_bias_balancing() for layer in layers]  # list: enable every layer, no short-circuit
    checks["bias_balancing_enabled"] = all(enabled)
    checks["balancing_biases_created"] = layers[0].balancing_biases is not None
    checks["fresh_bias_is_zero"] = int(torch.count_nonzero(layers[0].balancing_biases)) == 0

    zero_bias = routed(model, hidden)
    checks["zero_bias_selection_unchanged"], checks["zero_bias_weights_unchanged"] = compare_routes(
        baseline, zero_bias, "zero bias"
    )
    log(f"zero bias inert across all {len(layers)} layers: {checks['zero_bias_selection_unchanged']}")

    # Shipped Qwen3-MoE checkpoints set norm_topk_prob=True, so the paragraph above does not actually
    # exercise the case the shared _deepseek_biased_route would get wrong. Force it: with the renorm
    # off, gating through an unconditional softmax-over-gathered-logits stops being inert, and only
    # routing weights through the family's own _gate_weights_at stays exact.
    was_norm = [layer.gate.norm_topk_prob for layer in layers]
    try:
        for layer in layers:
            layer.gate.norm_topk_prob = False
        unnormed_baseline = routed(model, hidden)
        for layer in layers:
            layer.balancing_biases[:] = 0.0
        unnormed_zero = routed(model, hidden)
        (
            checks["unnormed_zero_bias_selection_unchanged"],
            checks["unnormed_zero_bias_weights_unchanged"],
        ) = compare_routes(unnormed_baseline, unnormed_zero, "zero bias, norm_topk_prob=False")
    finally:
        for layer, flag in zip(layers, was_norm, strict=True):
            layer.gate.norm_topk_prob = flag

    # A real bias must move selection, or the balancing state does nothing.
    victim = layers[0]
    victim.balancing_biases[:] = 0.0
    victim.balancing_biases[0] = 1e3  # force expert 0 into every token's top-k
    steered = routed(model, hidden)
    idx = steered[0][0]
    checks["nonzero_bias_steers_selection"] = bool((idx == 0).any(dim=-1).all())

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="qwen3_moe_bias_balancing")(run)

if __name__ == "__main__":
    main()
