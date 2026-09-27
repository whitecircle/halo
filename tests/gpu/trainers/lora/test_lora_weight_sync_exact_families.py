#!/usr/bin/env python
"""LoRA weight syncs leave the frozen base bit-identical: every other family the sync serves.

The per-family sweep of :mod:`tests.common.lora_sync_exactness`, which holds the checks: the dense
Qwen3.5 and every MoE family of ``tests/common/tiny_models.py`` some engine takes an online update for,
beyond the representative rows of ``test_lora_weight_sync_exact.py``.
``tests/cpu/conventions/test_tiny_family_roster.py`` holds both suites' rows to that roster.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_weight_sync_exact_families.py --family step3p7 --mode ep2 --adapters mixed
"""

from tests.common.harness import gpu_test_main
from tests.common.lora_sync_exactness import (
    REPRESENTATIVE_FAMILIES,
    WORLD_SIZE,
    parse_row,
    row_families,
    run_lora_sync_exactness,
)


def run(ctx) -> dict:
    row = parse_row(family for family in row_families() if family not in REPRESENTATIVE_FAMILIES)
    return run_lora_sync_exactness(ctx, family=row.family, mode=row.mode, adapters=row.adapters)


main = gpu_test_main(exact_world_size=WORLD_SIZE, prefix="lora_weight_sync_exact_families")(run)

if __name__ == "__main__":
    main()
