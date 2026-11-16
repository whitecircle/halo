#!/usr/bin/env python
"""Native DeepGEMM expert compute with each expert's group sorted straight into whole 128-row segments against the
dense sort the adapter pads per projection, on one MoE block at as many EP ranks as the launcher gives (1 or 4): the
same output and token and router gradients bit for bit, and the expert-weight gradients too at EP 1. Past it DeepEP's
receive order varies run to run, so the dense path's own weight gradients differ between two runs (about 2e-5); there
they must stay within that spread. ``--family qwen3`` stores separate GLU banks, ``qwen3_5`` a fused ``gate_up_proj``,
``gpt_oss`` expert biases, whose padding rows must take a zero bias. An expert bias's gradient is one GEMM over every
row, which the zero rows reassociate: it matches to its dtype's rounding. The padded layer's GEMMs must take the sorted
tokens as their operand (an aligned layout, with no gather back), the dense layer's never, and a second receive of
another size must compile no Triton kernel in either.

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/ep/test_deepgemm_padded_groups.py --family qwen3
"""

import argparse
import copy
import sys

import torch
import torch.nn as nn
import triton
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from halo.distributed.expert_parallel.moe_layer import compute
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.deepgemm import deepgemm_available
from halo.kernels.lowp.spec import LowpSpec
from tests.common.harness import gpu_test_main
from tests.common.numerics import fro_rel_err
from tests.common.rank_log import log

SEED = 0
HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 512, 256, 32, 4
TOKENS = 1000  # uneven per-expert counts: every group pads, some experts on some ranks get none
SECOND_TOKENS = 777  # another receive size, past the warm-up
BIAS_REL_MAX = 1e-2  # a bf16 bias gradient one rounding apart
EXPERT_GRAD_REL_MAX = 1e-4  # 5x the dense path's own run-to-run spread at EP 4
FAMILIES = {
    "qwen3": lambda: Qwen3MoeSparseMoeBlock(
        Qwen3MoeConfig(
            hidden_size=HIDDEN, moe_intermediate_size=INTERMEDIATE, num_experts=EXPERTS, num_experts_per_tok=TOP_K
        )
    ),
    "qwen3_5": lambda: Qwen3_5MoeSparseMoeBlock(
        Qwen3_5MoeTextConfig(
            hidden_size=HIDDEN,
            moe_intermediate_size=INTERMEDIATE,
            num_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
            shared_expert_intermediate_size=INTERMEDIATE,
        )
    ),
    "gpt_oss": lambda: GptOssMLP(
        GptOssConfig(
            hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_local_experts=EXPERTS, num_experts_per_tok=TOP_K
        )
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=tuple(FAMILIES), default="qwen3")
parser.add_argument("--precision", choices=("fp8", "mxfp4"), default="fp8")
parser.add_argument("--backward", choices=("fp8", "bf16"), default="fp8")
parser.add_argument("--ep-size", type=int, help="The EP ranks this row launches at, checked against the launcher's")
ARGS = parser.parse_args()
LOWP = LowpSpec(precision=ARGS.precision, backend="native", backward_precision=ARGS.backward, apply_dense_mlp=False)


def _ep_layer(block: nn.Module, ep_size: int, padded: bool) -> nn.Module:
    pc = ParallelismConfig(ep_size=ep_size, ep_rows_per_device=1, ep_declared_max_length=TOKENS, lowp=LOWP)
    holder = nn.Module()
    holder.mlp = copy.deepcopy(block)
    layer = apply_ep_layers(holder, pc.ep_config()).mlp
    assert layer.set_lowp_compute(LOWP, weight_cacheable=True)
    assert layer._pads_expert_groups, "native DeepGEMM compute sorts into padded groups"
    layer._pads_expert_groups = padded
    return layer


sorts = []  # (pad_align, receive rows) of every routing sort the layers run
_sort_routing_by_expert = compute.sort_routing_by_expert


def _recorded_sort(experts, weights, num_local_experts, counts, pad_align=None):
    sorts.append((pad_align, experts.shape[0]))
    return _sort_routing_by_expert(experts, weights, num_local_experts, counts, pad_align)


compute.sort_routing_by_expert = _recorded_sort

aligned_layouts = []  # of every native GEMM output: whether its layout was the sorted tokens' own
_token_rows = deepgemm._token_rows


def _recorded_token_rows(d, layout):
    aligned_layouts.append(layout.aligned)
    return _token_rows(d, layout)


deepgemm._token_rows = _recorded_token_rows


def _step(layer: nn.Module, x: torch.Tensor, grad: torch.Tensor) -> dict[str, torch.Tensor]:
    x = x.clone().requires_grad_(True)
    out = layer(x)
    out = out[0] if isinstance(out, tuple) else out  # GPT-OSS's block also returns its router logits
    out.backward(grad)
    return {
        "output": out.detach(),
        "grad_tokens": x.grad,
        **{f"grad_{name}": param.grad for name, param in layer.named_parameters() if param.grad is not None},
    }


@gpu_test_main(prefix=f"deepgemm_padded_groups_{ARGS.family}_{ARGS.precision}_{ARGS.backward}", partial_state=False)
def run(ctx) -> dict:
    torch.manual_seed(SEED)
    block = FAMILIES[ARGS.family]().to(ctx.device, torch.bfloat16)
    for param in block.parameters():
        nn.init.normal_(param, std=0.02)
    torch.manual_seed(SEED + 1 + ctx.rank)  # each rank routes its own tokens
    x = torch.randn(1, TOKENS, HIDDEN, device=ctx.device, dtype=torch.bfloat16)
    grad = torch.randn_like(x)

    dense_layer, padded_layer = (_ep_layer(block, ctx.world_size, padded=padded) for padded in (False, True))
    dense = _step(dense_layer, x, grad)
    dense_sorts = list(sorts)
    padded = _step(padded_layer, x, grad)
    padded_sorts = sorts[len(dense_sorts) :]
    expert_grads = {f"grad_{name}" for name, _ in dense_layer.expert_named_params()}
    expected = {"output", "grad_tokens", *expert_grads}
    router_grads = dense.keys() - expected

    compiled = []

    def hook(*, fn, **_):
        compiled.append(fn.name)
        return False  # a truthy return would skip the compile

    # Past each layer's first forward, whose warm-up runs the FFN over a few rows per expert.
    second_x, second_grad = x[:, :SECOND_TOKENS].contiguous(), grad[:, :SECOND_TOKENS].contiguous()
    previous, triton.knobs.runtime.jit_cache_hook = triton.knobs.runtime.jit_cache_hook, hook
    try:
        aligned_layouts.clear()
        _step(dense_layer, second_x, second_grad)
        dense_layouts = list(aligned_layouts)
        _step(padded_layer, second_x, second_grad)
        padded_layouts = aligned_layouts[len(dense_layouts) :]
    finally:
        triton.knobs.runtime.jit_cache_hook = previous

    checks = {
        "launched_at_the_rows_ep_size": ARGS.ep_size in (None, ctx.world_size),
        "every_expert_and_router_gradient_reached": expected <= dense.keys() == padded.keys() and bool(router_grads),
        "the_dense_layer_never_pads": bool(dense_sorts) and all(align is None for align, _ in dense_sorts),
        "the_padded_layer_pads_its_receive": bool(padded_sorts) and all(align == 128 for align, _ in padded_sorts),
        "the_dense_layers_gemms_gather_their_rows_back": bool(dense_layouts) and not any(dense_layouts),
        "the_padded_layers_gemms_run_on_the_sorted_tokens": bool(padded_layouts) and all(padded_layouts),
        "a_second_receive_size_compiles_no_triton_kernel": not compiled,
    }
    if compiled:
        log(f"  compiled past the warm-up: {compiled}")
    metrics = {}
    for name, expected in dense.items():
        got = padded[name]
        rel = fro_rel_err(got.float(), expected.float())
        metrics[f"{name}_rel"] = rel
        if name.endswith("_bias"):
            checks[f"{name}_matches_to_rounding"] = rel < BIAS_REL_MAX
            log(f"  {name:34s} rel {rel:.2e}")
        elif ctx.world_size > 1 and name in expert_grads:
            checks[f"{name}_within_the_dense_runs_spread"] = rel < EXPERT_GRAD_REL_MAX
            log(f"  {name:34s} rel {rel:.2e} (DeepEP receive order varies past EP 1)")
        else:
            checks[f"{name}_bitwise"] = torch.equal(got, expected)
            log(f"  {name:34s} {'bitwise equal' if checks[f'{name}_bitwise'] else 'DIFFERS'}")
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    if not deepgemm_available():
        log("SKIP: deep_gemm absent (Blackwell image only)")
        sys.exit(0)
    run()
