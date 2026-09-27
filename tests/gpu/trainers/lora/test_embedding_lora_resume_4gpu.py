#!/usr/bin/env python
"""Test: a full fine-tune embedding checkpoint resumes exactly under TP over FSDP2 (4 GPUs, full tier).

The body is :func:`tests.common.embedding_lora_resume.run_embedding_lora_resume` in its ``tpdp`` shape:
tp 2 by dp 2, the resume built from the checkpoint, and the best-model load refused rather than run
over FSDP2's stacked TP shards. The 2-GPU TP rows are ``test_embedding_lora_resume.py`` and
``test_embedding_lora_resume_roster.py``.

Usage:
    torchrun --nproc_per_node=4 tests/gpu/trainers/lora/test_embedding_lora_resume_4gpu.py --family qwen3
"""

import argparse

from tests.common.embedding_lora_resume import FAMILIES, run_embedding_lora_resume
from tests.common.harness import gpu_test_main

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
ARGS, _ = parser.parse_known_args()


@gpu_test_main(exact_world_size=4, prefix=f"embedding_lora_resume_{ARGS.family}_tpdp_off")
def run(ctx):
    return run_embedding_lora_resume(ctx, ARGS.family, "tpdp", "off")


if __name__ == "__main__":
    run()
