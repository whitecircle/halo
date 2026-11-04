#!/usr/bin/env python
"""fp32-pinned families train in the run dtype on every loader: one GPU, ep1, plain FSDP2 and EP.

DeepSeek-V4, GLM-5 Next and Inkling pin parameters in fp32 (``_keep_in_fp32_modules_strict``). A loader
that keeps the pins hands FSDP2 a mixed-dtype shard group it refuses ("expects uniform original parameter
dtype"), or, with the pinned modules left outside FSDP2 under LoRA, feeds DeepSeek-V4's fp32 norm output
into its bf16 projections. Each row loads a tiny checkpoint of one family, requires every floating base
parameter in bf16 after the load, and trains two SFT steps (see ``tests/common/pinned_params.py``):

  * ``--world 1``: the single-GPU loader, where no FSDP2 wrap casts anything; DeepSeek-V4's norm returns
    ``weight * x`` in the weight's dtype, so a kept pin feeds fp32 activations into the bf16 ``q_a_proj``
    and the first forward raises.
  * ``--mode full --ep 1``: the ep1 grouped-GEMM loader, full fine-tuning under FSDP2.
  * ``--mode full --ep 1 --no-grouped-gemm``: the plain FSDP2 data-parallel loader.
  * ``--mode expert_lora --ep 1``: native expert LoRA over a frozen base whose pins stay out of FSDP2.
  * ``--mode mixed --ep 1``: attention + expert adapters; the adapters must train identically on both
    ranks, which an adapter left out of FSDP2's shard groups does not.
  * ``--ep 2``: the EP loaders, which already cast, as the control. Under an adapter mode the
    cross-rank check leaves out the expert adapters, each rank's own for its own experts.
  * ``--fp32-masters`` (``fp32_non_ep_params``): the pins keep the checkpoint's stored fp32 values
    instead of a bf16 round trip before the trainer's upcast, at ep1 and through the EP lazy loader.
  * ``--dsa-kernel`` (``glm5_next``): on SDPA under reentrant checkpointing, where the DSA kernel must
    serve its layer on both passes of every step. The numerics hold on either path, so only a count
    shows a call that fell to stock.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_fp32_pinned_params.py \
        --family deepseek_v4 --mode full --ep 1
"""

import argparse
from unittest.mock import patch

from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.attention.masked_matmul import _MaskedMatmulAttention
from tests.common.harness import gpu_test_main
from tests.common.pinned_params import MODES, NUM_STEPS, run_pinned_family_row
from tests.common.tiny_models import PINNED_FP32_FAMILIES

DSA_KERNEL_FAMILY = "glm5_next"
# The tiny config's one indexed-attention layer, on a reentrant checkpoint's two passes.
DSA_KERNEL_CALLS_PER_STEP = 2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--family", choices=sorted(PINNED_FP32_FAMILIES), required=True)
parser.add_argument("--mode", choices=MODES, required=True)
parser.add_argument("--ep", type=int, choices=(1, 2), required=True)
parser.add_argument("--no-grouped-gemm", action="store_true", help="plain FSDP2 DP")
parser.add_argument("--fp32-masters", action="store_true", help="fp32_non_ep_params")
parser.add_argument("--world", type=int, choices=(1, 2), default=2)
parser.add_argument("--dsa-kernel", action="store_true", help=f"{DSA_KERNEL_FAMILY} counting the DSA kernel's calls")
ARGS = parser.parse_args()
if ARGS.dsa_kernel and ARGS.family != DSA_KERNEL_FAMILY:
    parser.error(f"--dsa-kernel takes --family {DSA_KERNEL_FAMILY}")


def _dsa_kernel_row(ctx, pc: ParallelismConfig) -> dict:
    calls = []
    apply = _MaskedMatmulAttention.apply
    with patch.object(_MaskedMatmulAttention, "apply", lambda *args: calls.append(1) or apply(*args)):
        result = run_pinned_family_row(
            ctx, ARGS.family, ARGS.mode, pc, attn_implementation="sdpa", gradient_checkpointing=True
        )
    result["checks"]["dsa_kernel_served_every_pass"] = len(calls) >= DSA_KERNEL_CALLS_PER_STEP * NUM_STEPS
    return result


@gpu_test_main(exact_world_size=ARGS.world, prefix="sft_fp32_pinned_params")
def run(ctx):
    # fp32 masters beside FSDP-managed ep1 experts is refused at config time; the documented remedy.
    pc = ParallelismConfig(
        ep_size=ARGS.ep,
        use_grouped_gemm=not ARGS.no_grouped_gemm,
        fp32_non_ep_params=ARGS.fp32_masters,
        fsdp_shard_ep1_experts=not ARGS.fp32_masters,
    )
    return _dsa_kernel_row(ctx, pc) if ARGS.dsa_kernel else run_pinned_family_row(ctx, ARGS.family, ARGS.mode, pc)


if __name__ == "__main__":
    run()
