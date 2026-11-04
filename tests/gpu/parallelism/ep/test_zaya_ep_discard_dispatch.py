#!/usr/bin/env python
"""Zaya's discarded picks never reach expert 0's rank under EP, and the MoE output is unchanged.

Dispatched as is, each discard loads expert 0's rank with rows that contribute nothing, so the wrapper sends ``-1``. A
fresh gate routes near-uniformly with its discard slot at a -1 bias, so gates are re-initialized at unit gain.
"""

import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.models.zaya.modeling_zaya import ZayaRouter

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log, log_all
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

EP_SIZE = 2
BATCH, SEQ = 4, 64
SEED = 7
# Relative spacing of bf16 at 1.0: the most a changed GEMM tiling may move one output element.
BF16_ULP = 2**-7

_COMPUTE = EPMoELayerBase._compute_experts


@contextlib.contextmanager
def count_received_rows(received: list[int]):
    def compute(self, tokens, experts, weights, output_dtype, *, recv_counts):
        received.append(int(((experts >= 0) & (experts < self.experts_per_rank)).sum()))
        return _COMPUTE(self, tokens, experts, weights, output_dtype, recv_counts=recv_counts)

    EPMoELayerBase._compute_experts = compute
    try:
        yield
    finally:
        EPMoELayerBase._compute_experts = _COMPUTE


def compare_layer(layer, hidden: torch.Tensor) -> dict:
    """Run ``layer`` and the unmasked call on ``hidden``; return outputs, row counts and discards."""
    fixed_rows, unmasked_rows = [], []
    with count_received_rows(fixed_rows):
        fixed, _ = layer(hidden)
    _, probs, indices, _ = layer.gate(hidden)
    flat = hidden.reshape(-1, hidden.shape[-1])
    with count_received_rows(unmasked_rows):
        unmasked = layer._dispatch_compute_combine(flat, indices.long(), probs.float(), hidden.dtype)
    unmasked = unmasked.view_as(fixed)
    return {
        "equal": torch.equal(fixed, unmasked),
        "within_ulp": torch.allclose(fixed, unmasked, rtol=BF16_ULP, atol=0.0),
        "finite": bool(fixed.isfinite().all()),
        "fixed_rows": sum(fixed_rows),
        "unmasked_rows": sum(unmasked_rows),
        "discards": int((probs == 0).sum()),
        "expert0_picks": int(((indices == 0) & (probs > 0)).sum()),
    }


def spread_routing(model) -> None:
    """Unit-gain gate projections and no selection bias, so every class wins a share of the tokens."""
    generator = torch.Generator().manual_seed(SEED)
    for router in (module for module in model.modules() if isinstance(module, ZayaRouter)):
        for linear in (module for module in router.modules() if isinstance(module, nn.Linear)):
            linear.weight.data.copy_(torch.randn(linear.weight.shape, generator=generator) * linear.in_features**-0.5)
        router.balancing_biases.zero_()


def build_model(device, ep_size: int):
    torch.manual_seed(SEED)
    model = tiny_family_model(TINY_MOE_FAMILIES["zaya"])
    spread_routing(model)
    model = apply_ep_layers(model.to(device, torch.bfloat16), ParallelismConfig(ep_size=ep_size).ep_config())
    return model


@gpu_test_main(exact_world_size=EP_SIZE, prefix="zaya_ep_discard_dispatch")
def run(ctx):
    checks, metrics = {}, {}
    torch.cuda.set_device(ctx.device)
    ep2 = build_model(ctx.device, EP_SIZE)
    ep1 = build_model(ctx.device, 1)
    hidden_size = ep2.config.hidden_size
    generator = torch.Generator().manual_seed(SEED + ctx.rank)

    with torch.no_grad():
        for index, (layer, local) in enumerate(zip(ep_layers(ep2), ep_layers(ep1), strict=True)):
            hidden = torch.randn(BATCH, SEQ, hidden_size, generator=generator).to(ctx.device, torch.bfloat16)
            sharded = compare_layer(layer, hidden)
            counts = torch.tensor(
                [sharded["discards"], sharded["expert0_picks"], sharded["unmasked_rows"] - sharded["fixed_rows"]],
                device=ctx.device,
            )
            dropped = [torch.empty_like(counts) for _ in range(ctx.world_size)]
            dist.all_gather(dropped, counts)
            total_discards = sum(int(c[0]) for c in dropped)
            owns_expert0 = layer.expert_start == 0
            expected_drop = total_discards if owns_expert0 else 0

            checks[f"l{index}_ep2_output_unchanged"] = sharded["within_ulp"] and sharded["finite"]
            metrics[f"l{index}_ep2_bit_identical"] = sharded["equal"]
            checks[f"l{index}_ep2_expert0_rank_rows_drop_by_discards"] = int(counts[2]) == expected_drop
            checks[f"l{index}_discards_and_expert0_picks_present"] = (
                total_discards > 0 and sum(int(c[1]) for c in dropped) > 0
            )
            replicated = compare_layer(local, hidden)
            checks[f"l{index}_ep1_output_unchanged"] = replicated["equal"] and replicated["finite"]
            checks[f"l{index}_ep1_expert0_computes_the_discards"] = (
                replicated["fixed_rows"] == replicated["unmasked_rows"]
            )
            metrics[f"l{index}_discards"] = total_discards
            log_all(
                f"  layer {index} rank {ctx.rank}: received {sharded['fixed_rows']} rows "
                f"(unmasked {sharded['unmasked_rows']}), {sharded['discards']} local discards, "
                f"{total_discards} total"
            )
    log(f"  discards per layer: {[metrics[f'l{i}_discards'] for i in range(len(ep_layers(ep2)))]}")
    del ep1
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
