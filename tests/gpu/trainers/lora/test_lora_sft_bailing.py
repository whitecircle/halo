#!/usr/bin/env python
"""Attention LoRA SFT on Bailing Ring-mini under EP or pure ETP, saved and reloaded on a fresh unsharded base.

The core tier's rows of :func:`tests.common.lora_sft_modes.run_lora_sft`; ``test_lora_sft_modes.py`` runs the
others.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/lora/test_lora_sft_bailing.py --mode etp
"""

import argparse

from tests.common.harness import gpu_test_main
from tests.common.lora_sft_modes import WORLD_SIZE, run_lora_sft

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["ep", "etp"], required=True)
MODE = parser.parse_args().mode


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"lora_sft_bailing_{MODE}")
def run(ctx):
    return run_lora_sft(ctx, model="bailing", mode=MODE, save_reload=True)


if __name__ == "__main__":
    run()
