#!/usr/bin/env python
"""Test: an embedding checkpoint, LoRA folded or fully fine-tuned, serves and resumes exactly.

The body is :func:`tests.common.embedding_lora_resume.run_embedding_lora_resume`. ``--mode`` picks the run
shape: ``single`` (one process: plain tensors, the base Trainer's ``optimizer.pt``), ``fsdp`` (torchrun:
mixin FSDP2, DTensor adapters), ``ddp`` (what ``accelerate launch`` with a MULTI_GPU config runs),
``presharded`` (FSDP2 over per-rank dataset slices), the FSDP2 knobs (``zero3``, ``hsdp`` on four ranks, and
``ep1`` / ``ep1_unsharded``, a MoE on the grouped-GEMM EP layers), ``tp``, ``ep``, ``etp``, ``eptp`` and on four
ranks ``epetp``, ``ep2dp2`` and ``ep2x2dom``, where LoRA only checks the refusal, and ``tpdp`` (TP over FSDP2 on
four ranks: the resume built from the checkpoint, the best-model load refused rather than run over FSDP2's
stacked TP shards). ``--lora`` picks the adapted modules
(``attention``, the input embedding beside them as ``mixed``, ``embedding`` alone, or DoRA on the attention
projections as ``dora``), or ``off`` for a full fine-tune. ``--head`` instead checks construction over a
pipeline with a projection head after the pooling: refused under FSDP2, TP and EP, accepted under DDP.
The core rows run the ST encoder and a decoder; the full rows the rest of the family roster.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/lora/test_embedding_lora_resume.py --family bert --mode fsdp
"""

import argparse

from tests.common.embedding_lora_resume import FAMILIES, LORA_TARGETS, MODES, WORLD_SIZES, run_embedding_lora_resume
from tests.common.harness import gpu_test_main

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
parser.add_argument("--mode", choices=MODES, required=True)
parser.add_argument("--lora", choices=LORA_TARGETS, default="attention")
parser.add_argument("--head", action="store_true")
ARGS = parser.parse_args()
HEAD = "_head" if ARGS.head else ""


@gpu_test_main(
    exact_world_size=WORLD_SIZES[ARGS.mode],
    prefix=f"embedding_lora_resume_{ARGS.family}_{ARGS.mode}_{ARGS.lora}{HEAD}",
)
def run(ctx):
    return run_embedding_lora_resume(ctx, ARGS.family, ARGS.mode, ARGS.lora, head=ARGS.head)


if __name__ == "__main__":
    run()
