#!/usr/bin/env python
"""Test: an embedding checkpoint, LoRA folded or fully fine-tuned, serves and resumes exactly (one GPU, full tier).

The single-process row of :func:`tests.common.embedding_lora_resume.run_embedding_lora_resume`: plain tensors,
the base Trainer's ``optimizer.pt``. ``--lora`` picks the adapted modules (``attention``, ``mixed``,
``embedding``), or ``off`` for a full fine-tune. The full tier's families, beyond the core one of
``test_embedding_lora_resume_1gpu.py``.

Usage:
    torchrun --nproc_per_node=1 tests/gpu/trainers/lora/test_embedding_lora_resume_roster_1gpu.py --family gemma4
"""

import argparse

from tests.common.embedding_lora_resume import FAMILIES, LORA_TARGETS, run_embedding_lora_resume
from tests.common.harness import gpu_test_main

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
parser.add_argument("--lora", choices=LORA_TARGETS, default="attention")
ARGS, _ = parser.parse_known_args()


@gpu_test_main(exact_world_size=1, prefix=f"embedding_lora_resume_roster_{ARGS.family}_single_{ARGS.lora}")
def run(ctx):
    return run_embedding_lora_resume(ctx, ARGS.family, "single", ARGS.lora)


if __name__ == "__main__":
    run()
