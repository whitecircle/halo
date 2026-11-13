#!/usr/bin/env python
"""``moe_backend: mok`` against the DeepEP path on one MoE block, at as many EP ranks as the launcher gives (1 or 4):
the same weights and tokens give the same output and the same token, router and expert-weight gradients within bf16
tolerance. ``--family qwen3`` stores separate GLU banks, ``qwen3_5`` a fused ``gate_up_proj`` beside a gated shared
expert, which stays on the layer's own path. ``--mxfp8`` runs MoK's MXFP8 forward and backward, which must land in fp8's
band of the bf16 reference instead. The token count is off MoK's 256 multiple, so the padded tail is checked.
"""

import argparse
import copy
import sys

import torch
import torch.nn as nn
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from halo.distributed.expert_parallel.mok import mok_available
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.lowp.spec import LowpSpec
from tests.common.harness import gpu_test_main
from tests.common.numerics import cos_sim, fro_rel_err
from tests.common.rank_log import log

SEED = 0
HIDDEN, INTERMEDIATE, EXPERTS, TOP_K = 512, 256, 32, 4
TOKENS = 1000  # padded to 1024
# Both paths accumulate in fp32 and round to bf16, in different orders and tilings.
REL_MAX, COS_MIN = 2e-2, 0.9995
# MXFP8 against bf16; the lower bound fails a run that silently stayed bf16. Weight gradients sum many fp8 products
# and land well inside it; the shared expert's stay bf16 on both paths.
MXFP8_BAND = (5e-3, 0.1)
MXFP8 = LowpSpec(precision="fp8", backend="native", backward_precision="fp8", apply_dense_mlp=False)
SIZES = {"hidden_size": HIDDEN, "moe_intermediate_size": INTERMEDIATE, "num_experts": EXPERTS}
FAMILIES = {
    "qwen3": lambda: Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(**SIZES, num_experts_per_tok=TOP_K, norm_topk_prob=True)),
    "qwen3_5": lambda: Qwen3_5MoeSparseMoeBlock(
        Qwen3_5MoeTextConfig(**SIZES, num_experts_per_tok=TOP_K, shared_expert_intermediate_size=INTERMEDIATE)
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=tuple(FAMILIES), default="qwen3")
parser.add_argument("--mxfp8", action="store_true", help="Run MoK's MXFP8 forward and backward")
ARGS = parser.parse_args()


def _ep_layer(block: nn.Module, backend: str, ep_size: int, lowp: LowpSpec | None = None) -> nn.Module:
    pc = ParallelismConfig(
        ep_size=ep_size,
        moe_backend=backend,
        ep_rows_per_device=1,
        ep_declared_max_length=TOKENS,
        lowp=lowp or LowpSpec(),
    )
    holder = nn.Module()
    holder.mlp = copy.deepcopy(block)
    layer = apply_ep_layers(holder, pc.ep_config()).mlp
    if lowp is not None:
        layer.set_lowp_compute(lowp, weight_cacheable=True)
    return layer


def _step(layer: nn.Module, x: torch.Tensor, grad: torch.Tensor) -> dict[str, torch.Tensor]:
    x = x.clone().requires_grad_(True)
    out = layer(x)
    out.backward(grad)
    return {
        "output": out.detach(),
        "grad_tokens": x.grad,
        **{f"grad_{name}": param.grad for name, param in layer.named_parameters() if param.grad is not None},
    }


@gpu_test_main(prefix=f"mok_backend_{ARGS.family}{'_mxfp8' if ARGS.mxfp8 else ''}", partial_state=False)
def run(ctx) -> dict:
    torch.manual_seed(SEED)
    block = FAMILIES[ARGS.family]().to(ctx.device, torch.bfloat16)
    for param in block.parameters():
        nn.init.normal_(param, std=0.02)
    torch.manual_seed(SEED + 1 + ctx.rank)  # each rank routes its own tokens
    x = torch.randn(1, TOKENS, HIDDEN, device=ctx.device, dtype=torch.bfloat16)
    grad = torch.randn_like(x)

    reference = _step(_ep_layer(block, "deepep", ctx.world_size), x, grad)
    mok = _step(_ep_layer(block, "mok", ctx.world_size, MXFP8 if ARGS.mxfp8 else None), x, grad)

    checks = {"every_gradient_reached": reference.keys() == mok.keys()}
    metrics = {}
    for name, expected in reference.items():
        rel = fro_rel_err(mok[name].float(), expected.float())
        cos = cos_sim(mok[name].float(), expected.float(), label=name)
        metrics[f"{name}_rel"] = rel
        if ARGS.mxfp8 and "shared_expert" not in name:
            checks[f"{name}_in_mxfp8_band"] = MXFP8_BAND[0] < rel < MXFP8_BAND[1]
        else:
            checks[f"{name}_matches_deepep"] = rel < REL_MAX and cos > COS_MIN
        log(f"  {name:34s} rel {rel:.2e} cos {cos:.6f}")
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    if not mok_available():
        log("SKIP: mok absent (Blackwell image only)")
        sys.exit(0)
    run()
