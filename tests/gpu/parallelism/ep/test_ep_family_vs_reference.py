#!/usr/bin/env python
"""A tiny ``TINY_MOE_FAMILIES`` model under EP=2 against the stock model it was copied from: loss and every MoE block
gradient.

``--expert-tp-size 2`` also splits each expert FFN two ways. A ``PINNED_ROUTING_FAMILIES`` member runs its stock model
on the EP model's top-k picks, which must be the stock router's own up to bf16 near-ties.
"""

import argparse
import contextlib
import copy
import math
from operator import attrgetter

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers, ep_claimed_blocks
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.moe.balancing import resolve_balancing_mode
from halo.models.moe.expert_layout import SEPARATE_GLU_PROJECTIONS
from tests.common.distributed import world_spread
from tests.common.ep_reference import gathered_expert_grads, random_token_batch, score_ep_grad_pairs, sweep_ep_grads
from tests.common.fla_autotune import pin_fla_autotune_first_fit
from tests.common.harness import gpu_test_main
from tests.common.models import (
    BAILING_LING_3_TINY,
    TINY_GLM4_MOE_LITE_CONFIG,
    TINY_GLM5_CONFIG,
    TINY_STEP3P7_CONFIG,
)
from tests.common.numerics import cos_sim
from tests.common.rank_log import log
from tests.common.rosters import ep_layer_class_by_model_type
from tests.common.routed_first_step import RoutedStep, capturing_routed_forwards, pinned_reference_routing
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.tolerances import TOL

SEED = 42
BATCH, SEQ = 2, 64
# bf16 gradients of a tiny model, where fp32 EP routing against bf16 stock scoring flips occasional near-ties.
TIGHT_GRAD_COS_MIN = 0.97
GRAD_COS_MIN = {"inkling_text": TIGHT_GRAD_COS_MIN}
# Knobs each EP layer must adopt from its block; Cohere2's "average" shared-expert combination halves the sum.
ADOPTED_KNOBS = {
    "cohere2_moe": {"_output_scale": 0.5, "expert_selection_fn": "sigmoid"},
    "glm4_moe_lite": {
        "n_group": TINY_GLM4_MOE_LITE_CONFIG["n_group"],
        "topk_group": TINY_GLM4_MOE_LITE_CONFIG["topk_group"],
        "top_k": TINY_GLM4_MOE_LITE_CONFIG["num_experts_per_tok"],
    },
    "glm5_next": {
        "swiglu_limit": TINY_GLM5_CONFIG["swiglu_limit"],
        "routed_scaling_factor": TINY_GLM5_CONFIG["routed_scaling_factor"],
    },
    "step3p7": {"routed_scaling_factor": TINY_STEP3P7_CONFIG["moe_router_scaling_factor"]},
}
# Ling-3.0-tiny shrunk to one small MoE block; hidden 256 sits at DeepEP's token-dim padding boundary.
BAILING_V3_BLOCK = {
    "hidden_size": 256,
    "moe_intermediate_size": 64,
    "moe_shared_expert_intermediate_size": 64,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "n_group": 2,
    "topk_group": 1,
    "hidden_act": "silu",
}
# bf16 grouped GEMM against the stock per-expert loop, which accumulates in another order.
BAILING_V3_OUTPUT_TOL = 2e-2


def _stock_expert_grads(experts: nn.Module) -> dict[str, torch.Tensor | None]:
    """As fused banks, or GPT-OSS's interleaved banks and biases; Bailing's per-expert modules are stacked, an
    unreached expert at zero."""
    if not isinstance(experts, nn.ModuleList):
        return {name: param.grad for name, param in experts.named_parameters()}
    banks = {
        proj: torch.stack(
            [
                torch.zeros_like(weight) if weight.grad is None else weight.grad
                for weight in (getattr(expert, proj).weight for expert in experts)
            ]
        )
        for proj in SEPARATE_GLU_PROJECTIONS
    }
    return {"gate_up_proj": torch.cat([banks["gate_proj"], banks["up_proj"]], dim=1), "down_proj": banks["down_proj"]}


def _compare_blocks(model, reference, ep_cls: type, ep_size: int, metrics: dict, *, cos_min: float) -> dict[str, bool]:
    """COLLECTIVE (the expert gradient gather). EP layers against the stock blocks they replaced, then bias balancing
    adopts the family's exported slot."""
    blocks = dict(ep_claimed_blocks(reference))
    layers = dict(named_ep_layers(model))
    unclaimed = [
        (name, module)
        for name, module in reference.named_modules()
        if not any(name == path or name.startswith(f"{path}.") for path in blocks)
    ]
    checks = {
        "ep_layers_patched": bool(blocks) and layers.keys() == blocks.keys(),
        "ep_layers_are_the_family_class": all(type(layer) is ep_cls for layer in layers.values()),
        "unclaimed_modules_untouched": all(type(model.get_submodule(name)) is type(m) for name, m in unclaimed),
    }
    pairs, split, etp_split, etp_unsharded_on_rank0 = {}, [], [], []
    for path, layer in layers.items():
        block = blocks[path]
        experts = type(layer)._find_experts_container(block)
        prefix = next(name for name, module in block.named_modules() if module is experts)
        bank = len(experts) if isinstance(experts, nn.ModuleList) else experts.down_proj.shape[0]
        split.append(
            layer.num_experts == bank
            and layer.experts_per_rank * ep_size == bank
            and layer.expert_end - layer.expert_start == layer.experts_per_rank
        )
        if layer.expert_tp_size > 1:
            etp_split.append(
                hasattr(layer, "gate_proj") and hasattr(layer, "up_proj") and not hasattr(layer, "gate_up_proj")
            )
        if layer.expert_tp_rank > 0:
            # A tensor expert TP can't shard (GptOss's down bias) stays whole on expert-TP rank 0; a copy here would be
            # added once per partner by the token-space reduce.
            etp_unsharded_on_rank0.append(
                all(name in layer.EXPERT_TP_SHARD_DIMS for name, _ in layer.expert_named_params())
            )
        gathered = gathered_expert_grads(layer)
        for key, grad in _stock_expert_grads(experts).items():
            if grad is not None and (key in gathered or layer.expert_tp_rank == 0):
                pairs[".".join(filter(None, (path, prefix, key)))] = (gathered.get(key), grad)
        ep_params = dict(layer.named_parameters())
        for name, param in block.named_parameters():
            if param.grad is not None and prefix and not name.startswith(f"{prefix}."):
                pairs[f"{path}.{name}"] = (getattr(ep_params.get(name), "grad", None), param.grad)
    checks["expert_bank_split_ep_way"] = bool(split) and all(split)
    if etp_split:
        # Slicing the fused GLU bank along its output would hand each rank only gate or only up.
        checks["etp_split_glu_storage"] = all(etp_split)
    if etp_unsharded_on_rank0:
        checks["etp_unsharded_experts_on_rank0_only"] = all(etp_unsharded_on_rank0)
    score_ep_grad_pairs(pairs, checks, metrics, cos_min=cos_min)
    if (slot := ep_cls.FAMILY.balancing_slot) is not None:
        # The trained bias must be the tensor the family's routing reads and every checkpoint carries.
        adopter = next((layer for layer in layers.values() if layer.enable_bias_balancing()), None)
        checks["native_slot_adopted"] = adopter is not None and (
            adopter.balancing_biases is attrgetter(slot.attrs[0])(adopter)
        )
    return checks


def _bailing_v3_block_checks(model, layers, input_ids) -> dict[str, bool]:
    """EP claims Ling 3.0's block on the strength of its being V2's, so the stock V3 block is compared alone: the
    decoder's KDA layers JIT ``fla`` kernels and test nothing about experts."""
    config = AutoConfig.from_pretrained(BAILING_LING_3_TINY, trust_remote_code=True)
    config.update(BAILING_V3_BLOCK)
    block_cls = get_class_from_dynamic_module(
        "modeling_bailing_moe_v3.BailingMoeV3SparseMoeBlock", BAILING_LING_3_TINY
    )
    torch.manual_seed(SEED)
    stock = nn.Module()
    stock.mlp = block_cls(config).to(input_ids.device, torch.bfloat16)
    wrapped = apply_ep_layers(copy.deepcopy(stock), ParallelismConfig(ep_size=layers[0].ep_size).ep_config())
    hidden = torch.randn(BATCH, 16, config.hidden_size, device=input_ids.device, dtype=torch.bfloat16)
    outputs = []
    for container in (stock, wrapped):
        out = container.mlp(hidden)
        out = out[0] if isinstance(out, tuple) else out  # The stock block also returns its router logits.
        out.sum().backward()
        outputs.append(out.detach().float())
    sweep_ep_grads(wrapped)
    max_abs = (outputs[1] - outputs[0]).abs().max().item()
    log(f"bailing v3 block: max|Δ|={max_abs:.3e} cosine={cos_sim(outputs[1], outputs[0], label='v3 block'):.6f}")
    ep_cls = ep_layer_class_by_model_type()["bailing_hybrid"]
    checks = {"output_matches_reference": max_abs < BAILING_V3_OUTPUT_TOL}
    checks |= _compare_blocks(wrapped, stock, ep_cls, layers[0].ep_size, {}, cos_min=TIGHT_GRAD_COS_MIN)
    return {f"v3_{name}": ok for name, ok in checks.items()}


def _deepseek_v4_checks(model, layers, input_ids) -> dict[str, bool]:
    """Hash layers select by the frozen ``tid2eid[input_ids]`` lookup and refuse a balancing bias."""
    hash_layer = next(layer for layer in layers if layer.is_hash)
    hash_layer._capture_routing = True
    with torch.no_grad():
        model(input_ids=input_ids)
    hash_layer._capture_routing = False
    captured = torch.cat(hash_layer._captured_routing_chunks).long()
    hash_layer._captured_routing_chunks.clear()
    return {
        "hash_flag_preserved": [layer.is_hash for layer in layers]
        == [kind == "hash_moe" for kind in model.config.mlp_layer_types],
        "hash_routing_is_tid2eid": torch.equal(captured, hash_layer.gate.tid2eid[input_ids.reshape(-1)].long()),
        "hash_layer_refuses_bias": hash_layer.enable_bias_balancing() is False,
    }


def _step3p7_checks(model, layers, input_ids) -> dict[str, bool]:
    """Per-layer clamp bounds (0 = unclamped, stored as inf); a native exported slot makes ``auto`` balancing
    ``bias_update``."""
    config = model.config.get_text_config()
    sparse = [index for index, kind in enumerate(config.mlp_layer_types) if kind == "sparse"]
    return {
        "per_layer_swiglu_limits_adopted": [layer.swiglu_limit for layer in layers]
        == [config.swiglu_limits[index] or math.inf for index in sparse],
        "auto_balancing_is_bias_update": resolve_balancing_mode("auto", model, is_moe=True) == "bias_update",
    }


# ``(model, EP layers, input_ids) -> checks``, after the shared comparison.
FAMILY_CHECKS = {
    "bailing_moe": _bailing_v3_block_checks,
    "deepseek_v4": _deepseek_v4_checks,
    "step3p7": _step3p7_checks,
}


# Each router's gate is in tests.common.routed_first_step.PINNED_GATE_WEIGHTS. The tiny GptOss's random routers hold
# exact bf16 logit ties (top-2 of 8), so reduction order flips picks and moves whole tokens' gradients (router bias
# cosine down to 0.70 at ep2 x etp2) with no parallelism bug.
PINNED_ROUTING_FAMILIES = frozenset({"gpt_oss"})


def run_family(ctx, family: str, cos_min: float, expert_tp_size: int = 1) -> dict:
    """COLLECTIVE. ``cos_min`` floors every gradient's direction."""
    pc = ParallelismConfig(ep_size=ctx.world_size // expert_tp_size, expert_tp_size=expert_tp_size)
    log(f"{family}: {pc.mode_string}")
    torch.manual_seed(SEED)
    reference = tiny_family_model(TINY_MOE_FAMILIES[family]).to(ctx.device, torch.bfloat16)
    pin_fla_autotune_first_fit(ctx)
    model = apply_ep_layers(copy.deepcopy(reference), pc.ep_config())
    vocab = reference.get_input_embeddings().num_embeddings
    input_ids, labels = random_token_batch(vocab, BATCH, SEQ, ctx.device, seed=SEED)
    layers = [layer for _, layer in named_ep_layers(model)]
    pinned_routing = family in PINNED_ROUTING_FAMILIES

    def train_step(module) -> float:
        module.train()
        out = module(input_ids=input_ids, labels=labels)
        out.loss.backward()
        return out.loss.item()

    # The EP model first, so a pinned reference can route on the picks it captured.
    routed = RoutedStep()
    capturing = capturing_routed_forwards(model, routed) if pinned_routing else contextlib.nullcontext()
    with capturing:
        ep_loss = train_step(model)
    pinning = pinned_reference_routing(reference, routed) if pinned_routing else contextlib.nullcontext()
    with pinning as pinned:
        ref_loss = train_step(reference)
    # Every rank ran the same batch: the trainer's sweep divides the combine's world-wide expert sum back down.
    sweep_ep_grads(model)
    log(f"reference loss {ref_loss:.6f}, EP loss {ep_loss:.6f} (|Δ| {abs(ep_loss - ref_loss):.2e})")
    metrics = {"ref_loss": ref_loss, "ep_loss": ep_loss}
    checks = {
        "ep_loss_matches_reference": abs(ep_loss - ref_loss) < TOL.parallel_vs_baseline_loss_abs,
        # Every rank ran the same batch, so a spread is a desync.
        "losses_agree_across_ranks": TOL.identical_batch_ranks_agree(world_spread(ep_loss, ctx.device)),
        **{
            f"{knob.lstrip('_')}_adopted": all(getattr(layer, knob) == value for layer in layers)
            for knob, value in ADOPTED_KNOBS.get(family, {}).items()
        },
    }
    if pinned_routing:
        regret = metrics["pinned_pick_regret_max"] = max(pinned.regrets, default=math.nan)
        log(f"pinned picks: worst shortfall below the stock router's k-th logit {regret:.3e}")
        # A wrong selection falls short by a whole top-k gap.
        every_block_pinned = len(pinned.regrets) == len(layers) and not any(pinned.unconsumed.values())
        checks["ep_picks_are_reference_near_ties"] = every_block_pinned and regret <= TOL.router_near_tie_logit_abs
        # A token naming one expert twice falls short by nothing, so duplicates are counted apart.
        duplicated = metrics["pinned_duplicate_pick_tokens"] = routed.duplicate_pick_tokens()
        checks["ep_picks_are_distinct_experts"] = duplicated == 0
    ep_cls = ep_layer_class_by_model_type()[family]
    checks |= _compare_blocks(model, reference, ep_cls, pc.ep_size, metrics, cos_min=cos_min)
    if family in FAMILY_CHECKS:
        checks |= FAMILY_CHECKS[family](model, layers, input_ids)
    return {"checks": checks, "metrics": metrics}


parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(TINY_MOE_FAMILIES), required=True)
parser.add_argument("--expert-tp-size", type=int, choices=(1, 2), default=1)
ARGS = parser.parse_args()


@gpu_test_main(
    min_world_size=2, exact_world_size=2 if ARGS.expert_tp_size == 1 else None, prefix="ep_family_vs_reference"
)
def run(ctx):
    default = TOL.ep_grad_cosine_min
    cos_min = GRAD_COS_MIN.get(ARGS.family, default) if ARGS.expert_tp_size == 1 else default
    return run_family(ctx, ARGS.family, cos_min, expert_tp_size=ARGS.expert_tp_size)


if __name__ == "__main__":
    run()
