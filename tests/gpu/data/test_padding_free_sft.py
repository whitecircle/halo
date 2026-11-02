#!/usr/bin/env python
"""``padding_free`` SFT trains exactly the padded batch's objective, under FSDP2, EP and TP.

The padding-free collator (``DataCollatorWithFlattening``) concatenates a microbatch into one row whose
``position_ids`` restart at every document, from which the varlen flash kernel takes each document's bounds; the
padded collator keeps one row per document behind padding. Both score the same next tokens, so a run's first step
must agree between them: each microbatch loss, the logged loss and the gradient of every FSDP2-sharded parameter,
by direction and norm (:mod:`tests.common.first_step`). A document whose positions ran on from the previous one
would attend across the boundary and move the loss and gradients off the padded run's.

Modes: ``dense`` (Qwen3-0.6B, FSDP2), ``tp2`` (Qwen3-0.6B, TP2), ``ep2`` (a tiny Qwen3-MoE, EP2), on two ranks, each
on the attention backend the loader resolves for its host.

Run: torchrun --nproc_per_node=2 tests/gpu/data/test_padding_free_sft.py --mode dense
"""

import argparse
import os

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.data.collators.factory import select_data_collator
from halo.data.collators.packing import DataCollatorWithFlattening
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.first_step import first_step_checks, first_step_gradient_checks, train_recording_first_step
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import QWEN3_0_6B
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint

SEED = 42
STEPS = 2
BATCH_SIZE = 4
MAX_LENGTH = 512
# The two collations differ only in how the kernel bounds each document; bf16 reduction order and the varlen
# against the padded kernel path remain. A document attending across a boundary moves the loss far past this.
LOSS_RTOL = 0.02
# Two-way axes: a loss counted once per rank of one is 100% off.
MISCOUNT_FACTOR = 2
MODES = {"dense": {}, "tp2": {"tp_size": 2}, "ep2": {"ep_size": 2}}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=sorted(MODES), required=True)
ARGS = parser.parse_args()


def _train(source: str, tokenizer, output_dir: str, *, padding_free: bool):
    """One run's trainer over ``source`` with the padded or the padding-free collator; returns its first step."""
    pc = ParallelismConfig(**MODES[ARGS.mode])
    model, _ = load_distributed_model(source, pc, torch.bfloat16, trust_remote_code=False, use_liger_kernel=False)
    collator = select_data_collator(
        tokenizer, train_on_completions_only=False, padding_free=padding_free, model_config=model.config
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir,
            max_steps=STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=1e-5,
            # Off, so the recorded gradient is the backward's own.
            max_grad_norm=0.0,
            bf16=True,
            gradient_checkpointing=False,
            use_liger_kernel=False,
            padding_free=padding_free,
            max_length=MAX_LENGTH,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            logging_nan_inf_filter=False,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            seed=SEED,
        ),
        train_dataset=create_sft_dataset(64, tokenizer, seed=SEED),
        processing_class=tokenizer,
        data_collator=collator,
        parallelism_config=pc,
    )
    checks = parallel_shape_checks(trainer.model, pc)
    checks["collator_is_the_arms_own"] = isinstance(trainer.data_collator, DataCollatorWithFlattening) == padding_free
    _, record = train_recording_first_step(trainer)
    finish_phase(trainer)
    return record, checks


@gpu_test_main(exact_world_size=2, prefix=f"padding_free_sft_{ARGS.mode}")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    source = (
        shared_tiny_family_checkpoint(ctx, TINY_MOE_FAMILIES["qwen3_moe"], "padding_free_sft", tokenizer, SEED)
        if ARGS.mode == "ep2"
        else QWEN3_0_6B
    )
    padded, padded_checks = _train(source, tokenizer, os.path.join(ctx.output_dir, "padded"), padding_free=False)
    flat, flat_checks = _train(source, tokenizer, os.path.join(ctx.output_dir, "flat"), padding_free=True)
    checks = {f"padded_{k}": v for k, v in padded_checks.items()} | {f"flat_{k}": v for k, v in flat_checks.items()}
    loss_checks, metrics = first_step_checks(flat, padded, miscount_factor=MISCOUNT_FACTOR, loss_rtol=LOSS_RTOL)
    grad_checks, grad_metrics = first_step_gradient_checks(flat, padded)
    flat_rows = [batch["input_ids"].shape[0] for batch in flat.batches]
    checks["flat_batches_are_one_row"] = bool(flat_rows) and set(flat_rows) == {1}
    log(f"  {ARGS.mode}: padded step-1 loss {padded.logged_loss:.5f}, padding-free {flat.logged_loss:.5f}")
    return {"checks": checks | loss_checks | grad_checks, "metrics": metrics | grad_metrics}


if __name__ == "__main__":
    run()
