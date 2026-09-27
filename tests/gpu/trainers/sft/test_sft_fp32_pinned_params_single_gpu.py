#!/usr/bin/env python
"""A fp32-pinned family trains in the run dtype on one GPU, where no FSDP2 wrap casts anything.

DeepSeek-V4 pins its RMSNorm weights in fp32 (``_keep_in_fp32_modules_strict``) and its norm returns
``weight * x`` in the weight's dtype, so a single-GPU loader that keeps the pin feeds fp32 activations
into the bf16 ``q_a_proj`` and the first forward raises. The row requires every floating parameter in
bf16 after the load and two finite SFT steps (see ``tests/common/pinned_params.py``).

Run with 1 GPU:
    torchrun --nproc_per_node=1 tests/gpu/trainers/sft/test_sft_fp32_pinned_params_single_gpu.py \
        --family deepseek_v4
"""

import argparse

from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.pinned_params import run_pinned_family_row
from tests.common.tiny_models import PINNED_FP32_FAMILIES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(PINNED_FP32_FAMILIES), required=True)
    return parser.parse_args()


@gpu_test_main(exact_world_size=1, prefix="sft_fp32_pinned_params_single_gpu")
def run(ctx):
    return run_pinned_family_row(ctx, _parse_args().family, "full", ParallelismConfig())


if __name__ == "__main__":
    run()
