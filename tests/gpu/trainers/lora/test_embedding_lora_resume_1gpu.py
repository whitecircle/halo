#!/usr/bin/env python
"""Test: an injected-LoRA embedding checkpoint serves from its fold and resumes exactly (one GPU, core).

The single-process row of :func:`tests.common.embedding_lora_resume.run_embedding_lora_resume`: plain
tensors, the base Trainer's ``optimizer.pt``. The core tier's family; the rest of the roster is
``test_embedding_lora_resume_roster_1gpu.py``, the 2-GPU rows ``test_embedding_lora_resume.py``.

Usage:
    torchrun --nproc_per_node=1 tests/gpu/trainers/lora/test_embedding_lora_resume_1gpu.py --family bert
"""

import argparse

from tests.common.embedding_lora_resume import FAMILIES, run_embedding_lora_resume
from tests.common.harness import gpu_test_main

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
ARGS, _ = parser.parse_known_args()


@gpu_test_main(exact_world_size=1, prefix=f"embedding_lora_resume_{ARGS.family}_single")
def run(ctx):
    return run_embedding_lora_resume(ctx, ARGS.family, "single")


if __name__ == "__main__":
    run()
