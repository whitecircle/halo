#!/usr/bin/env python
"""
SFT training test with plain FSDP2 (no EP/TP/CP) on GptOss-20B.

Validates DistributedSFTTrainer on a MoE model under pure data-parallel FSDP2 — experts
stay local (grouped-GEMM expert compute, no DeepEP all-to-all), params/grads/optstates
sharded across the DP mesh. Complements the EP/ETP/TP/CP gpt-oss SFT tests.

Model: unsloth/gpt-oss-20b-BF16 (MoE, 32 experts)

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_oss20b_fsdp.py
"""

import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.base_layer import has_grouped_mm
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
    log("  SFT with plain FSDP2 (no EP/TP/CP) Test on GptOss-20B")
    log(f"  World: {world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"{'#' * 70}")

    output_dir, cache_dir = setup_cache_dirs("sft_oss20b_fsdp_test", rank)

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

        log("\n--- Loading model with plain FSDP2 ---")
        parallelism_config = ParallelismConfig(use_grouped_gemm=has_grouped_mm())
        log(f"Config: {parallelism_config.summary()}")

        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flex_attention",
            use_liger_kernel=True,
        )
        log(f"GPU mem after load: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            fsdp="",  # Mixin handles FSDP2 wrapping
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

        # Plain FSDP2: not EP/TP/CP.
        assert not trainer.is_ep_mode, "FSDP test must not be in EP mode"
        assert not trainer.is_tp_mode, "FSDP test must not be in TP mode"
        assert not trainer.is_cp_mode, "FSDP test must not be in CP mode"
        log("Confirmed: plain FSDP2 (no EP/TP/CP)")

        log(f"\n--- Training ({NUM_TRAIN_STEPS} steps) ---")
        train_result = trainer.train()

        training_loss = train_result.training_loss
        log_history = trainer.state.log_history
        step_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
        grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]

        log("\n--- Metrics ---")
        log(f"Final loss: {training_loss:.6f}")
        log(f"Step losses: {[f'{lv:.4f}' for lv in step_losses]}")

        checks = {}
        loss_finite = all(
            not (torch.isnan(torch.tensor(lv)) or torch.isinf(torch.tensor(lv)))
            for lv in step_losses + [training_loss]
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

        if grad_norms:
            grad_ok = all(not (torch.isnan(torch.tensor(g)) or torch.isinf(torch.tensor(g))) for g in grad_norms)
            checks["grad_finite"] = grad_ok
            log(f"Grad norms finite: {'PASS' if grad_ok else 'FAIL'}")

        # Loss consistency across DP ranks (FSDP2 replicates the loss reduction).
        loss_tensor = torch.tensor([training_loss], device=f"cuda:{local_rank}")
        all_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
        dist.all_gather(all_losses, loss_tensor)
        if rank == 0:
            spread = max(lv.item() for lv in all_losses) - min(lv.item() for lv in all_losses)
            checks["loss_consistent"] = spread < TOL.rank_loss_consistency_abs
            log(f"Loss consistent (spread={spread:.6f}): {'PASS' if checks['loss_consistent'] else 'FAIL'}")

        all_passed = all(checks.values())
        log(f"\n{'#' * 70}")
        log(f"  SFT FSDP (GptOss-20B) TEST {'PASSED' if all_passed else 'FAILED'}")
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
