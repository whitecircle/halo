#!/usr/bin/env python
"""An embedding checkpoint, LoRA folded or fully fine-tuned, serves and resumes exactly.

EP shapes only check LoRA's refusal. ``--head`` checks construction with a projection head after the pooling: refused
under FSDP2, TP and EP, accepted under DDP.
"""

import argparse

from tests.common.embedding_lora_resume import FAMILIES, LORA_TARGETS, MODES, WORLD_SIZES, run_embedding_lora_resume
from tests.common.harness import gpu_test_main

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
parser.add_argument("--mode", choices=MODES, required=True)
parser.add_argument("--lora", choices=LORA_TARGETS, default="attention")
parser.add_argument("--head", action="store_true")
parser.add_argument("--gradient-checkpointing", action="store_true")
ARGS = parser.parse_args()
SUFFIX = ("_head" if ARGS.head else "") + ("_gc" if ARGS.gradient_checkpointing else "")


@gpu_test_main(
    exact_world_size=WORLD_SIZES[ARGS.mode],
    prefix=f"embedding_lora_resume_{ARGS.family}_{ARGS.mode}_{ARGS.lora}{SUFFIX}",
)
def run(ctx):
    return run_embedding_lora_resume(
        ctx, ARGS.family, ARGS.mode, ARGS.lora, head=ARGS.head, gradient_checkpointing=ARGS.gradient_checkpointing
    )


if __name__ == "__main__":
    run()
