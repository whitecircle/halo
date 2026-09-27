#!/usr/bin/env python
"""LoRA weight syncs leave the frozen base bit-identical: every other family the sync serves.

The per-family sweep of :mod:`tests.common.lora_sync_exactness`, which holds the checks and the
family table; the representative dense and MoE rows are ``test_lora_weight_sync_exact.py``. Each row
checks that the table still covers every EP family some engine takes an online update for.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_weight_sync_exact_families.py --family step3p7 --mode ep2 --adapters mixed
"""

from tests.common.harness import gpu_test_main
from tests.common.lora_sync_exactness import (
    FAMILIES,
    REPRESENTATIVE_FAMILIES,
    WORLD_SIZE,
    parse_row,
    run_lora_sync_exactness,
)


def run(ctx) -> dict:
    row = parse_row(tuple(family for family in FAMILIES if family not in REPRESENTATIVE_FAMILIES))
    return run_lora_sync_exactness(ctx, family=row.family, mode=row.mode, adapters=row.adapters)


main = gpu_test_main(exact_world_size=WORLD_SIZE, prefix="lora_weight_sync_exact_families")(run)

if __name__ == "__main__":
    main()
