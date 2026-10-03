#!/usr/bin/env python
"""Native TP=4 LoRA SFT: save, resume, best reload, stock PEFT load and CLI merge.

Usage:
    torchrun --nproc_per_node=4 tests/gpu/trainers/lora/test_lora_tp4_save_load.py
"""

from tests.common.harness import gpu_test_main
from tests.common.tp_lora_lifecycle import run_tp_lora_lifecycle


def run(ctx):
    return {"checks": {"qwen3_tp4": run_tp_lora_lifecycle(ctx, tp_size=4)}}


main = gpu_test_main(exact_world_size=4, prefix="test_lora_tp4_save_load")(run)

if __name__ == "__main__":
    main()
