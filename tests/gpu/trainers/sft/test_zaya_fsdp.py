#!/usr/bin/env python
"""FSDP2 smoke for ZAYA1-8B: full copies load, the trainer shards them, and a few steps stay finite. Compares no loss
against a reference.

On 2x B300 FSDP halves the ~17 GB of bf16 weights; with adamw fp32 master + state the total stays well below 60 GB per
rank.
"""

import torch
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_int, env_str
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import ZAYA_8B
from tests.common.rank_log import log
from tests.common.training_checks import training_run_checks

MODEL = env_str("HALO_TEST_ZAYA_MODEL", ZAYA_8B)
MAX_STEPS = env_int("HALO_TEST_ZAYA_FSDP_STEPS", 4)
BATCH = 1
SEQ = 512  # Short: this validates distributed mechanics, not throughput.
LR = 5e-6
NUM_SAMPLES = 16


@gpu_test_main(min_world_size=1, prefix="test_zaya_fsdp")
def run(ctx):
    log("=" * 70)
    log("  ZAYA1-8B FSDP2 smoke test (multi-GPU)")
    log(f"  Model: {MODEL}")
    log(f"  World: {ctx.world_size}, GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Steps: {MAX_STEPS}, Batch: {BATCH}, Seq: {SEQ}")
    log("=" * 70)

    log("\n[1/4] Loading model via load_distributed_model...")
    pc = ParallelismConfig()
    model, tokenizer = load_distributed_model(
        model_name_or_path=MODEL,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  ✓ {n_params / 1e9:.2f}B params loaded")
    log(f"  ✓ HBM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    log("\n[2/4] Creating synthetic SFT dataset...")
    train_ds = create_sft_dataset(NUM_SAMPLES, tokenizer, seed=42)
    log(f"  ✓ {len(train_ds)} samples")

    log("\n[3/4] Configuring DistributedSFTTrainer (FSDP2)...")
    cfg = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        bf16=True,
        # GC recompute faults in cuDNN on the CCA Conv1d pair, so the patch refuses it.
        gradient_checkpointing=False,
        # FLCE: outputs.logits is a placeholder, so TRL must take its Liger path or its metrics crash on the empty
        # tensor.
        use_liger_kernel=True,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=SEQ,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=42,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        parallelism_config=pc,
    )
    log(f"  ✓ Trainer: {type(trainer).__name__}")
    log(f"  ✓ Parallelism: {pc.mode_string or 'Standard FSDP2'}")

    log(f"\n[4/4] Training {MAX_STEPS} steps...")
    result = trainer.train()
    log(f"  ✓ HBM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return {"checks": training_run_checks(result, trainer, MAX_STEPS)}


if __name__ == "__main__":
    run()
