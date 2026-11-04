#!/usr/bin/env python
"""LoRA weight syncs leave the frozen base bit-identical, on every family the sync serves under every mode; checks in
:mod:`tests.common.lora_sync_exactness`."""

from tests.common.harness import gpu_test_main
from tests.common.lora_sync_exactness import WORLD_SIZE, parse_row, row_families, run_lora_sync_exactness


def run(ctx) -> dict:
    row = parse_row(row_families())
    return run_lora_sync_exactness(ctx, family=row.family, mode=row.mode, adapters=row.adapters)


main = gpu_test_main(exact_world_size=WORLD_SIZE, prefix="lora_weight_sync_exact")(run)

if __name__ == "__main__":
    main()
