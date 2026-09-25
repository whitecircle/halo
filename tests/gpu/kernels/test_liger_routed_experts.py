#!/usr/bin/env python
"""Routed experts Halo leaves unwrapped compute the family's own function, forward and backward.

At ``ep_size: 1`` with ``use_grouped_gemm: false`` Halo installs no EP wrapper, so after Liger is
applied the routed experts run whatever the model builds for them. Liger must not be that:
``LigerExperts``, upstream liger-kernel's fused MoE kernel, computes the input gradient wrong on
Blackwell in the pinned release while its forward and weight gradients stay close — a silently wrong
gradient into every layer below. This drives a Qwen3-MoE experts block through the path Halo
selects (``ParallelismConfig`` → ``apply_liger_kernel`` → model build) and checks the forward, the
input gradient, the routing-weight gradient and both weight gradients against the family's eager
experts in fp32, with routing held identical so a near-tied top-k pick cannot move the result.

    torchrun --nproc_per_node=1 tests/gpu/kernels/test_liger_routed_experts.py
"""

import copy

import torch
from transformers import AutoModelForCausalLM, Qwen3MoeConfig
from transformers.models.qwen3_moe import modeling_qwen3_moe

from src.distributed.parallelism_config import ParallelismConfig
from src.kernels.liger.orchestrator import apply_liger_kernel
from src.models.loading.dtype import configure_float32_matmul_precision
from tests.common.harness import gpu_test_main, record_check
from tests.common.utils import log

SEED = 0
# Qwen3-30B-A3B's MoE geometry: 128 experts of width 768 over a 2048 hidden, top-8.
HIDDEN, EXPERT_WIDTH, NUM_EXPERTS, TOP_K, TOKENS = 2048, 768, 128, 8, 8192
INIT_STD = 0.02
# Mean relative error against the fp32 oracle. bf16 storage puts every tensor near 6e-3 on the
# path Halo selects; LigerExperts' input gradient on a B300 sits near 0.6.
TOL = 2e-2
GRADIENTS = ("fwd", "dx", "d_routing_weights", "d_gate_up_proj", "d_down_proj")


def _config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hidden_size=HIDDEN,
        intermediate_size=EXPERT_WIDTH,
        moe_intermediate_size=EXPERT_WIDTH,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=128,
        attn_implementation="eager",
    )


def _relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual.float() - reference.float()).abs().mean() / reference.float().abs().mean()).item()


def _forward_backward(experts, hidden, top_k_index, top_k_weights, grad_output) -> tuple[torch.Tensor, ...]:
    """The block's output and the four gradients it feeds back: input, routing weights, both weights."""
    hidden = hidden.clone().requires_grad_(True)
    top_k_weights = top_k_weights.clone().requires_grad_(True)
    output = experts(hidden, top_k_index, top_k_weights)
    output.backward(grad_output)
    return output.detach(), hidden.grad, top_k_weights.grad, experts.gate_up_proj.grad, experts.down_proj.grad


def _check_the_experts_are_the_familys_own(selected, stock) -> None:
    assert type(selected) is stock, (
        f"the routed experts are {type(selected).__module__}.{type(selected).__qualname__}, not the "
        f"family's {stock.__qualname__} — Liger replaced them"
    )


def _check_numerics(selected, stock, config, device) -> None:
    """The selected bf16 path against the family's eager experts in fp32, under identical routing."""
    oracle_config = copy.deepcopy(config)
    oracle_config._experts_implementation = "eager"
    torch.manual_seed(SEED)
    for parameter in selected.parameters():
        torch.nn.init.normal_(parameter, std=INIT_STD)
    oracle = stock(oracle_config).to(device=device, dtype=torch.float32)
    oracle.load_state_dict({name: tensor.float() for name, tensor in selected.state_dict().items()})

    hidden = torch.randn(TOKENS, HIDDEN, device=device)
    top_k_weights, top_k_index = torch.randn(TOKENS, NUM_EXPERTS, device=device).softmax(-1).topk(TOP_K, dim=-1)
    top_k_weights = top_k_weights / top_k_weights.sum(-1, keepdim=True)
    grad_output = torch.randn(TOKENS, HIDDEN, device=device)

    reference = _forward_backward(oracle, hidden, top_k_index, top_k_weights, grad_output)
    selected_path = _forward_backward(
        selected, hidden.bfloat16(), top_k_index, top_k_weights.bfloat16(), grad_output.bfloat16()
    )
    residuals = {
        name: _relative_error(actual, expected)
        for name, actual, expected in zip(GRADIENTS, selected_path, reference, strict=True)
    }
    log("  relative error vs fp32 eager: " + " ".join(f"{name}={value:.4f}" for name, value in residuals.items()))
    worst = max(residuals, key=residuals.get)
    assert residuals[worst] < TOL, f"{worst} relative error {residuals[worst]:.4f} exceeds {TOL}"


@gpu_test_main(min_world_size=1, prefix="test_liger_routed_experts")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    torch.cuda.set_device(ctx.device)
    configure_float32_matmul_precision()

    # Captured before Liger runs: an applier's class swap is process-global.
    stock = modeling_qwen3_moe.Qwen3MoeExperts
    parallelism = ParallelismConfig(use_grouped_gemm=False)
    assert not parallelism.needs_ep_wrappers, "ep_size 1 without grouped GEMM must leave the experts unwrapped"

    config = _config()
    applied = apply_liger_kernel(config, None, needs_ep_wrappers=parallelism.needs_ep_wrappers)
    log(f"  Liger applied: {applied}")
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_config(config)
    selected = model.model.layers[0].mlp.experts.to(device=ctx.device, dtype=torch.bfloat16)
    log(f"  routed experts: {type(selected).__qualname__} ({model.config._experts_implementation})")

    record_check(
        checks, "experts_are_the_familys_own", lambda: _check_the_experts_are_the_familys_own(selected, stock)
    )
    record_check(checks, "matches_fp32_eager_experts", lambda: _check_numerics(selected, stock, config, ctx.device))
    return {"checks": checks}


if __name__ == "__main__":
    run()
