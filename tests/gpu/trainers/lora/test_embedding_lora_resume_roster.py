#!/usr/bin/env python
"""Test: an embedding checkpoint, LoRA folded or fully fine-tuned, serves and resumes exactly (2 GPUs, full tier).

The body is :func:`tests.common.embedding_lora_resume.run_embedding_lora_resume`. ``--mode`` picks the run
shape: ``fsdp`` (torchrun: mixin FSDP2, DTensor adapters), ``ddp`` (what ``accelerate launch`` with a
MULTI_GPU config runs), ``presharded`` (FSDP2 over per-rank dataset slices), ``tp`` or ``ep``, where LoRA only
checks the refusal. ``--lora`` picks the adapted modules (``attention``, the input embedding beside them as
``mixed``, ``embedding`` alone, or DoRA on the attention projections as ``dora``), or ``off`` for a full fine-
tune. The full tier's rows, the family roster beyond the core rows of ``test_embedding_lora_resume.py``.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/lora/test_embedding_lora_resume_roster.py --family gpt_oss --mode ddp
"""

import argparse

from tests.common.embedding_lora_resume import FAMILIES, LORA_TARGETS, MODES, run_embedding_lora_resume
from tests.common.harness import gpu_test_main

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
# One process and TP over FSDP2 take their own world sizes (the _1gpu and _4gpu scripts).
parser.add_argument("--mode", choices=[mode for mode in MODES if mode not in ("single", "tpdp")], required=True)
parser.add_argument("--lora", choices=LORA_TARGETS, default="attention")
ARGS, _ = parser.parse_known_args()


@gpu_test_main(exact_world_size=2, prefix=f"embedding_lora_resume_roster_{ARGS.family}_{ARGS.mode}_{ARGS.lora}")
def run(ctx):
    return run_embedding_lora_resume(ctx, ARGS.family, ARGS.mode, ARGS.lora)


if __name__ == "__main__":
    run()
