#!/usr/bin/env python
"""LoRA weight syncs leave the frozen base bit-identical: a dense and a MoE family under every mode.

The representative rows of :mod:`tests.common.lora_sync_exactness`, which holds the checks: dense
Qwen3 under FSDP2, Qwen3-MoE at ep1 (DTensor experts), ep2 (plain experts) and pure ETP, each MoE
mode with attention PEFT alone and mixed with native expert LoRA where allowed. The other families
the sync serves are swept by ``test_lora_weight_sync_exact_families.py``.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_weight_sync_exact.py --family qwen3_moe --mode ep2 --adapters mixed
"""

from tests.common.harness import gpu_test_main
from tests.common.lora_sync_exactness import REPRESENTATIVE_FAMILIES, WORLD_SIZE, parse_row, run_lora_sync_exactness


def run(ctx) -> dict:
    row = parse_row(REPRESENTATIVE_FAMILIES)
    return run_lora_sync_exactness(ctx, family=row.family, mode=row.mode, adapters=row.adapters)


main = gpu_test_main(exact_world_size=WORLD_SIZE, prefix="lora_weight_sync_exact")(run)

if __name__ == "__main__":
    main()
