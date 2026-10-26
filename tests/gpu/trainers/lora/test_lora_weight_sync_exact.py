#!/usr/bin/env python
"""LoRA weight syncs leave the frozen base bit-identical, on every family the sync serves under every mode.

The rows of :mod:`tests.common.lora_sync_exactness`, which holds the checks. The core rows: dense Qwen3
under FSDP2, Qwen3-MoE at ep1 (DTensor experts), ep2 (plain experts) and pure ETP, each MoE mode with
attention PEFT alone and mixed with native expert LoRA where allowed. The full rows: the dense Qwen3.5 and
every other MoE family of ``tests/common/tiny_models.py`` some engine takes an online update for, which
``tests/cpu/conventions/test_tiny_family_roster.py`` holds the rows to.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_weight_sync_exact.py --family qwen3_moe --mode ep2 --adapters mixed
"""

from tests.common.harness import gpu_test_main
from tests.common.lora_sync_exactness import WORLD_SIZE, parse_row, row_families, run_lora_sync_exactness


def run(ctx) -> dict:
    row = parse_row(row_families())
    return run_lora_sync_exactness(ctx, family=row.family, mode=row.mode, adapters=row.adapters)


main = gpu_test_main(exact_world_size=WORLD_SIZE, prefix="lora_weight_sync_exact")(run)

if __name__ == "__main__":
    main()
