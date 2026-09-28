#!/usr/bin/env python
"""fp32-pinned families train in the run dtype on every 2-rank loader: ep1, plain FSDP2 and EP.

DeepSeek-V4, GLM-5 Next and Inkling pin parameters in fp32 (``_keep_in_fp32_modules_strict``). A loader
that keeps the pins hands FSDP2 a mixed-dtype shard group it refuses ("expects uniform original parameter
dtype"), or, with the pinned modules left outside FSDP2 under LoRA, feeds DeepSeek-V4's fp32 norm output
into its bf16 projections. Each row loads a tiny checkpoint of one family, requires every floating base
parameter in bf16 after the load, and trains two SFT steps (see ``tests/common/pinned_params.py``):

  * ``--mode full --ep 1``: the ep1 grouped-GEMM loader, full fine-tuning under FSDP2.
  * ``--mode full --ep 1 --no-grouped-gemm``: the plain FSDP2 data-parallel loader.
  * ``--mode expert_lora --ep 1``: native expert LoRA over a frozen base whose pins stay out of FSDP2.
  * ``--mode mixed --ep 1``: attention + expert adapters; the adapters must train identically on both
    ranks, which an adapter left out of FSDP2's shard groups does not.
  * ``--ep 2``: the EP loaders, which already cast, as the control. Under an adapter mode the
    cross-rank check leaves out the expert adapters, each rank's own for its own experts.
  * ``--fp32-masters`` (``fp32_non_ep_params``): the pins keep the checkpoint's stored fp32 values
    instead of a bf16 round trip before the trainer's upcast, at ep1 and through the EP lazy loader.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_fp32_pinned_params.py \
        --family deepseek_v4 --mode full --ep 1
"""

import argparse

from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.pinned_params import MODES, run_pinned_family_row
from tests.common.tiny_models import PINNED_FP32_FAMILIES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(PINNED_FP32_FAMILIES), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--ep", type=int, choices=(1, 2), required=True)
    parser.add_argument("--no-grouped-gemm", action="store_true", help="plain FSDP2 DP")
    parser.add_argument("--fp32-masters", action="store_true", help="fp32_non_ep_params")
    return parser.parse_args()


@gpu_test_main(exact_world_size=2, prefix="sft_fp32_pinned_params")
def run(ctx):
    args = _parse_args()
    # fp32 masters beside FSDP-managed ep1 experts is refused at config time; the documented remedy.
    pc = ParallelismConfig(
        ep_size=args.ep,
        use_grouped_gemm=not args.no_grouped_gemm,
        fp32_non_ep_params=args.fp32_masters,
        fsdp_shard_ep1_experts=not args.fp32_masters,
    )
    return run_pinned_family_row(ctx, args.family, args.mode, pc)


if __name__ == "__main__":
    run()
