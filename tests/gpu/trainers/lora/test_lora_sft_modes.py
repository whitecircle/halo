#!/usr/bin/env python
"""LoRA SFT on one model under one two-rank parallel shape, through the production adapter path.

The rows of :func:`tests.common.lora_sft_modes.run_lora_sft`: ``--model`` (gpt-oss-20b, Bailing Ring-mini,
Qwen3-0.6B) under ``--mode`` (``ep``, ``ep_cp``, ``etp``, ``cp``, ``fsdp``, and ``dp_stock_experts`` for a MoE's
QLoRA, whose base keeps its Params4bit and whose grouped-GEMM load is refused), with ``--qlora`` on a 4-bit base,
``--attn`` on another attention kernel, ``--convergence`` for 25 steps judged by 5-step window means,
``--save-reload`` to reload the saved adapters on a fresh unsharded base, and ``--resume`` to restore that save
onto the run. The core tier runs the Bailing rows from ``test_lora_sft_bailing.py``; LoRA under TP and QLoRA
under EP or TP are refused at construction and load, pinned on CPU.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/lora/test_lora_sft_modes.py --model gpt_oss_20b --mode ep
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
