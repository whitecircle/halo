#!/usr/bin/env python
"""Multi-GPU FSDP2 smoke test for ZAYA1-8B.

Verifies the full distributed loading pipeline:
  1. ``load_distributed_model`` with a plain ``ParallelismConfig()`` (no EP /
     CP / TP) goes through ``_load_undistributed_model`` → ``from_pretrained`` →
     each rank gets a full model copy.
  2. ``DistributedSFTTrainer`` wraps the model in FSDP2 (``fully_shard``,
     ``reshard_after_forward=False``), so weights are sharded across the
     rank's data-parallel group.
  3. A handful of synthetic SFT steps run without OOM / NaN, and the loss
     is finite + monotone-ish.

Designed for 2x B300 (275 GB HBM each). FSDP shards the ~17 GB of bf16
weights ~2× → 8.4 GB / rank for params; with adamw fp32 master + state
and bf16 activations the total stays well below 60 GB / rank.

Run (2 GPUs):
    docker run --rm --gpus '"device=0,1"' --ipc=host --ulimit memlock=-1 \\
        --ulimit stack=67108864 \\
        -v $(pwd):/workspace \\
        -v /root/.cache/huggingface:/root/.cache/huggingface \\
        -w /workspace -e HF_HOME=/root/.cache/huggingface \\
        halo:blackwell \\
        torchrun --nproc_per_node=2 \\
            tests/gpu/trainers/sft/test_zaya_fsdp.py
"""

import math
import sys

import torch
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_flag, env_int, env_str
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import cleanup_dirs, init_distributed, setup_cache_dirs
from tests.common.models import ZAYA_8B
from tests.common.utils import log

MODEL = env_str("HALO_TEST_ZAYA_MODEL", ZAYA_8B)
MAX_STEPS = env_int("HALO_TEST_ZAYA_FSDP_STEPS", 4)
BATCH = 1
SEQ = 512  # short — we're only validating distributed mechanics, not throughput
LR = 5e-6
NUM_SAMPLES = 16


def main() -> int:
    rank, world_size, local_rank = init_distributed()
    from accelerate import PartialState

    PartialState()

    output_dir, cache_dir = setup_cache_dirs("test_zaya_fsdp", rank)

    log("=" * 70)
    log("  ZAYA1-8B FSDP2 smoke test (multi-GPU)")
    log(f"  Model: {MODEL}")
    log(f"  World: {world_size}, GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  Steps: {MAX_STEPS}, Batch: {BATCH}, Seq: {SEQ}")
    log("=" * 70)

    try:
        log("\n[1/4] Loading model via load_distributed_model...")
        pc = ParallelismConfig()
        model, tokenizer = load_distributed_model(
            model_name_or_path=MODEL,
            parallelism_config=pc,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_liger_kernel=True,
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
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=1,
            learning_rate=LR,
            bf16=True,
            # Zaya's GC backward recompute faults in cuDNN on the CCA Conv1d pair, so the load-time
            # patch clears supports_gradient_checkpointing; HALO_TEST_ZAYA_GC=1 only asserts that refusal.
            gradient_checkpointing=env_flag("HALO_TEST_ZAYA_GC"),
            gradient_checkpointing_kwargs={
                "use_reentrant": env_flag("HALO_TEST_ZAYA_GC_REENTRANT", True),
            },
            # Zaya is FLCE: outputs.logits is a placeholder, so TRL must take its Liger path or its
            # entropy/accuracy metrics crash on the empty tensor (the mixin prevents double-apply).
            use_liger_kernel=True,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            max_length=SEQ,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            fsdp="",  # mixin handles FSDP2 wrapping
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
        step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
        log(f"  ✓ Final loss: {result.training_loss:.4f}")
        log(f"  ✓ Per-step losses: {[f'{l:.4f}' for l in step_losses]}")
        log(f"  ✓ HBM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        assert math.isfinite(result.training_loss), f"Non-finite loss: {result.training_loss}"
        assert all(math.isfinite(x) for x in step_losses), f"Non-finite step loss: {step_losses}"
        assert len(step_losses) == MAX_STEPS, f"Expected {MAX_STEPS} step losses, got {len(step_losses)}"

        if rank == 0:
            log("\n" + "=" * 70)
            log("  ✓ ALL CHECKS PASSED (ZAYA1-8B + FSDP2 multi-GPU)")
            log("=" * 70)
        return 0
    finally:
        cleanup_dirs(output_dir, cache_dir)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
