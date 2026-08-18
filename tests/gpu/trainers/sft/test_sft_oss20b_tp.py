#!/usr/bin/env python
"""
SFT training test with Tensor Parallelism (TP=2) on GptOss-20B.

Validates that DistributedSFTTrainer works correctly with TP=2 on a MoE model.
In TP-only mode, attention/embedding/lm_head weights are sharded via DTensor
while MoE experts remain replicated on each GPU.

Also validates sequential model loading (max_concurrent_loading=1).

Model: unsloth/gpt-oss-20b-BF16 (MoE, 32 experts)

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_oss20b_tp.py
"""

import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import GPT_OSS_20B
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log

MODEL_NAME = GPT_OSS_20B
TP_SIZE = 2
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
SEED = 42


def main():
    rank, world_size, local_rank = init_distributed()
    PartialState()

    log(f"\n{'#' * 70}")
    log(f"  SFT with Tensor Parallelism (TP={TP_SIZE}) Test on GptOss-20B")
    log(f"  World: {world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log("  max_concurrent_loading: 1 (sequential)")
    log(f"{'#' * 70}")

    if world_size < TP_SIZE:
        log(f"\nERROR: Need at least {TP_SIZE} GPUs, got {world_size}")
        teardown_distributed()
        return 1

    output_dir, cache_dir = setup_cache_dirs("sft_oss20b_tp", rank)

    try:
        log("\nEnsuring model is downloaded...")
        ensure_model_downloaded(MODEL_NAME, rank)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log("\n--- Creating synthetic datasets ---")
        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
        log(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        log(f"\n--- Loading model with TP={TP_SIZE} (sequential loading) ---")
        parallelism_config = ParallelismConfig(tp_size=TP_SIZE, max_concurrent_loading=1)
        log(f"Config: {parallelism_config.summary()}")

        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=True,
        )
        log(f"GPU mem after load: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        log(f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            max_grad_norm=1.0,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,  # already applied at load
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            fsdp="",  # Mixin handles FSDP wrapping
        )

        log("\n--- Creating DistributedSFTTrainer ---")
        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        assert trainer.is_tp_mode, "Trainer should be in TP mode"
        assert not trainer.is_ep_mode, "Should NOT be in EP mode"
        log("Confirmed: TP mode active, EP mode inactive")

        log(f"\n--- Training ({NUM_TRAIN_STEPS} steps) ---")
        train_result = trainer.train()

        training_loss = train_result.training_loss
        log_history = trainer.state.log_history
        step_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
        grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]

        log("\n--- Metrics ---")
        log(f"Final loss: {training_loss:.6f}")
        log(f"Step losses: {[f'{l:.4f}' for l in step_losses]}")
        if grad_norms:
            log(f"Grad norms: {[f'{g:.2f}' for g in grad_norms]}")

        log("\n--- Checks ---")
        checks = {}

        loss_finite = all(
            not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in step_losses + [training_loss]
        )
        checks["loss_finite"] = loss_finite
        log(f"Loss finite: {'PASS' if loss_finite else 'FAIL'}")

        if len(step_losses) >= 2:
            first_loss, last_loss = step_losses[0], step_losses[-1]
            loss_decreased = last_loss < first_loss
            checks["loss_decreased"] = loss_decreased
            log(f"Loss decreased: {'PASS' if loss_decreased else 'FAIL'} ({first_loss:.4f} -> {last_loss:.4f})")
        else:
            checks["loss_decreased"] = False
            log("Loss decreased: FAIL (not enough steps logged)")

        checks["tp_mode"] = trainer.is_tp_mode
        log(f"TP mode active: {'PASS' if checks['tp_mode'] else 'FAIL'}")

        loss_tensor = torch.tensor([training_loss], device=f"cuda:{local_rank}")
        all_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
        dist.all_gather(all_losses, loss_tensor)
        if rank == 0:
            losses_list = [l.item() for l in all_losses]
            spread = max(losses_list) - min(losses_list)
            checks["loss_consistent"] = spread < TOL.rank_loss_consistency_abs
            log(f"Loss consistent (spread={spread:.6f}): {'PASS' if checks['loss_consistent'] else 'FAIL'}")
        else:
            checks["loss_consistent"] = True

        if grad_norms:
            grad_ok = all(not (torch.isnan(torch.tensor(g)) or torch.isinf(torch.tensor(g))) for g in grad_norms)
            checks["grad_finite"] = grad_ok
            log(f"Grad norms finite: {'PASS' if grad_ok else 'FAIL'}")

        all_passed = all(checks.values())
        log(f"\n{'#' * 70}")
        log(f"  SFT TP TEST (GptOss-20B) {'PASSED' if all_passed else 'FAILED'}")
        if not all_passed:
            log(f"  Failed: {[k for k, v in checks.items() if not v]}")
        log(f"{'#' * 70}\n")

        del trainer, model
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)
        barrier()
        teardown_distributed()
        return 0 if all_passed else 1

    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        if rank == 0:
            traceback.print_exc()
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)
        teardown_distributed()
        return 1


if __name__ == "__main__":
    sys.exit(main())
