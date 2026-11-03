#!/usr/bin/env python
"""LoRA SFT on one model under one two-rank parallel shape, through the production adapter path; rows of
:func:`tests.common.lora_sft_modes.run_lora_sft`.

``dp_stock_experts`` is a MoE's QLoRA, whose base keeps its Params4bit and whose grouped-GEMM load is refused. LoRA
under TP and QLoRA under EP or TP are refused at construction, pinned on CPU.
"""

import argparse

from tests.common.harness import gpu_test_main
from tests.common.lora_sft_modes import MODELS, MODES, WORLD_SIZE, run_lora_sft

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=sorted(MODELS), required=True)
parser.add_argument("--mode", choices=sorted(MODES), required=True)
parser.add_argument("--attn", choices=["flash_attention_2", "flex_attention", "sdpa"])
parser.add_argument("--qlora", action="store_true")
parser.add_argument("--convergence", action="store_true")
parser.add_argument("--save-reload", action="store_true")
parser.add_argument("--resume", action="store_true")
ARGS = parser.parse_args()
if ARGS.resume and not ARGS.save_reload:
    parser.error("--resume restores the --save-reload save")


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"lora_sft_{ARGS.model}_{ARGS.mode}")
def run(ctx):
    return run_lora_sft(ctx, **vars(ARGS))


if __name__ == "__main__":
    run()
