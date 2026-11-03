#!/usr/bin/env python
"""Qwen3-MoE ``bias_update_transient`` balancing: inert at zero bias, steering at nonzero bias.

``norm_topk_prob`` defaults off while shipped checkpoints set it on, so gating through the shared
unconditional-renormalization helper is not inert on a checkpoint leaving it off. The test forces it off, since no
cached checkpoint exercises that.
"""

import torch

from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_30B_A3B
from tests.common.rank_log import log

MODEL_NAME = QWEN3_30B_A3B
EP_SIZE = 2
SEQ_LEN = 128
SEED = 7


def routed(model, hidden):
    out = []
    for layer in ep_layers(model):
        with torch.no_grad():
            flat = hidden.view(-1, hidden.shape[-1])
            with torch.amp.autocast("cuda", enabled=False):
                logits, weights, experts = layer.gate(flat)
            if layer._transient_balancing_bias(logits) is not None:
                experts = layer._biased_topk(logits)
                weights = layer._gate_weights_at(logits, experts)
            out.append((experts.clone(), weights.float().clone()))
    return out


def compare_routes(baseline, candidate, tag: str) -> tuple[bool, bool]:
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
    # Without the declaration both bias modes raise.
    checks["supports_bias_balancing"] = bool(type(layers[0]).SUPPORTS_BIAS_BALANCING)
    log(f"EP layers: {len(layers)}  ({type(layers[0]).__name__})")
    log(f"norm_topk_prob = {layers[0].gate.norm_topk_prob}")

    torch.manual_seed(SEED)
    hidden = torch.randn(1, SEQ_LEN, model.config.hidden_size, device=device, dtype=torch.bfloat16)

    baseline = routed(model, hidden)  # Balancing not yet enabled.

    enabled = [layer.enable_bias_balancing() for layer in layers]  # List: enable every layer, no short-circuit.
    checks["bias_balancing_enabled"] = all(enabled)
    checks["balancing_biases_created"] = layers[0].balancing_biases is not None
    checks["fresh_bias_is_zero"] = int(torch.count_nonzero(layers[0].balancing_biases)) == 0

    zero_bias = routed(model, hidden)
    checks["zero_bias_selection_unchanged"], checks["zero_bias_weights_unchanged"] = compare_routes(
        baseline, zero_bias, "zero bias"
    )
    log(f"zero bias inert across all {len(layers)} layers: {checks['zero_bias_selection_unchanged']}")

    # With the renorm off, an unconditional softmax over gathered logits stops being inert; only the family's
    # _gate_weights_at stays exact.
    was_norm = [layer.gate.norm_topk_prob for layer in layers]
    biases = [layer.balancing_biases for layer in layers]
    try:
        for layer in layers:
            layer.gate.norm_topk_prob = False
            # The baseline has balancing off: a None side-buffer routes through the gate's own weights.
            layer.balancing_biases = None
        unnormed_baseline = routed(model, hidden)
        for layer, bias in zip(layers, biases, strict=True):
            bias.zero_()
            layer.balancing_biases = bias
        unnormed_zero = routed(model, hidden)
        (
            checks["unnormed_zero_bias_selection_unchanged"],
            checks["unnormed_zero_bias_weights_unchanged"],
        ) = compare_routes(unnormed_baseline, unnormed_zero, "zero bias, norm_topk_prob=False")
    finally:
        for layer, flag, bias in zip(layers, was_norm, biases, strict=True):
            layer.gate.norm_topk_prob = flag
            layer.balancing_biases = bias

    victim = layers[0]
    victim.balancing_biases[:] = 0.0
    victim.balancing_biases[0] = 1e3  # Forces expert 0 into every token's top-k.
    steered = routed(model, hidden)
    idx = steered[0][0]
    checks["nonzero_bias_steers_selection"] = bool((idx == 0).any(dim=-1).all())

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="qwen3_moe_bias_balancing")(run)

if __name__ == "__main__":
    main()
