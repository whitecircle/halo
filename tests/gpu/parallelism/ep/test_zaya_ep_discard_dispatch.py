#!/usr/bin/env python
"""Zaya's discarded picks never reach expert 0's rank, and the MoE output is unchanged.

The Zaya gate masks a pick of its learned discard slot to expert 0 at weight 0. Dispatched as is, every
discard rides the all-to-all to expert 0's rank in every layer, loading it with rows that contribute
nothing; under EP the wrapper sends them as ``-1`` (no expert) instead. Per MoE layer of a tiny Zaya
(two experts, top-1), against the unmasked call ``_dispatch_compute_combine(gate indices, gate probs)``:

  ep2 — the layer output matches to one bf16 ulp (expert 0's grouped GEMM runs fewer rows, which a
        kernel may tile differently), and the rows expert 0's rank receives drop by exactly the
        discards summed over both ranks while the other rank's receive count is unchanged.
  ep1 — nothing is dispatched and the grouped path takes real local ids only, so the output stays
        bit-identical to the unmasked call.

A fresh tiny gate routes near-uniformly with almost no spread across tokens, and its discard slot starts
at a -1 selection bias no token overcomes; the gates are re-initialized at unit gain with the bias
cleared, so each layer's tokens split between the discard slot and both experts. The run fails if a
layer sees no discard or no real expert-0 pick.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_zaya_ep_discard_dispatch.py
"""

import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.models.zaya.configuration_zaya import ZayaConfig
from transformers.models.zaya.modeling_zaya import ZayaForCausalLM, ZayaRouter

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_ZAYA_CONFIG
from tests.common.utils import log, log_all

EP_SIZE = 2
BATCH, SEQ = 4, 64
SEED = 7
# Relative spacing of bf16 at 1.0: the most a changed GEMM tiling may move one output element.
BF16_ULP = 2**-7

_COMPUTE = EPMoELayerBase._compute_experts


@contextlib.contextmanager
def count_received_rows(received: list[int]):
    """Append the number of rows each ``_compute_experts`` call assigns to a local expert."""

    def compute(self, tokens, experts, weights, output_dtype):
        received.append(int(((experts >= 0) & (experts < self.experts_per_rank)).sum()))
        return _COMPUTE(self, tokens, experts, weights, output_dtype)

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
    model = ZayaForCausalLM(ZayaConfig(**TINY_ZAYA_CONFIG))
    spread_routing(model)
    model = patch_moe_model_for_ep(
        model.to(device, torch.bfloat16), ParallelismConfig(ep_size=ep_size).create_ep_config()
    )
    create_ep_buffers(model)
    return model


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
            metrics[f"l{index}_discards"] = total_discards
            log_all(
                f"  layer {index} rank {ctx.rank}: received {sharded['fixed_rows']} rows "
                f"(unmasked {sharded['unmasked_rows']}), {sharded['discards']} local discards, "
                f"{total_discards} total"
            )
    log(f"  discards per layer: {[metrics[f'l{i}_discards'] for i in range(len(ep_layers(ep2)))]}")
    del ep1
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(exact_world_size=EP_SIZE, prefix="zaya_ep_discard_dispatch")(run)

if __name__ == "__main__":
    main()
