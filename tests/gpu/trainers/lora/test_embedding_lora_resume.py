#!/usr/bin/env python
"""Test: an injected-LoRA embedding checkpoint serves from its fold and resumes exactly (2 GPUs, core).

The body is :func:`tests.common.embedding_lora_resume.run_embedding_lora_resume`. ``--mode`` picks the
run shape: ``fsdp`` (torchrun: mixin FSDP2, DTensor adapters), ``ddp`` (what ``accelerate launch`` with
a MULTI_GPU config runs), ``presharded`` (FSDP2 over per-rank dataset slices), or ``tp`` / ``ep``,
which only check the refusal. The core tier's rows; the rest of the family roster is
``test_embedding_lora_resume_roster.py``, the one-GPU rows ``test_embedding_lora_resume_1gpu.py``.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/lora/test_embedding_lora_resume.py --family bert --mode fsdp
"""

import argparse

from tests.common.embedding_lora_resume import FAMILIES, MODES, run_embedding_lora_resume
from tests.common.harness import gpu_test_main

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
parser.add_argument("--mode", choices=[mode for mode in MODES if mode != "single"], required=True)
ARGS, _ = parser.parse_known_args()


@gpu_test_main(exact_world_size=2, prefix=f"embedding_lora_resume_{ARGS.family}_{ARGS.mode}")
def run(ctx):
    return run_embedding_lora_resume(ctx, ARGS.family, ARGS.mode)


if __name__ == "__main__":
    run()
