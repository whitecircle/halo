#!/usr/bin/env python
"""
SFT training test with Expert Parallelism (EP=2) on GptOss-20B.

Validates that DistributedSFTTrainer works correctly with EP=2 on a MoE model
(32 experts distributed across 2 GPUs via DeepEP all-to-all routing). Each GPU
holds 16 experts and receives all data (EP is orthogonal to data parallelism).

Model: unsloth/gpt-oss-20b-BF16 (MoE, 32 experts)

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_ep.py

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
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
from tests.common.ep_reference import ep_layers
from tests.common.models import GPT_OSS_20B
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
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
    log(f"  SFT with Expert Parallelism (EP={EP_SIZE}) Test")
    log(f"  World: {world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"{'#' * 70}")

    if world_size < EP_SIZE:
        log(f"\nERROR: Need at least {EP_SIZE} GPUs, got {world_size}")
        teardown_distributed()
        return 1

    output_dir, cache_dir = setup_cache_dirs("sft_ep_test", rank)

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
        if rank == 0:
            log(f"Sample (first 200 chars): {train_dataset[0]['text'][:200]}...")

        log(f"\n--- Loading model with EP={EP_SIZE} ---")
        parallelism_config = ParallelismConfig(ep_size=EP_SIZE, use_grouped_gemm=has_grouped_mm())
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
        log(f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,  # already applied at load
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,  # Required for EP (inactive experts)
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

        assert trainer.is_ep_mode, "Trainer should be in EP mode"
        log("Confirmed: trainer.is_ep_mode is True")

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

        checks["ep_mode"] = trainer.is_ep_mode
        log(f"EP mode active: {'PASS' if checks['ep_mode'] else 'FAIL'}")

        # A check, not a log line: with no EP wrappers this file trains a plain dense model and
        # reports every other check green.
        moe_layers = ep_layers(model)
        checks["ep_layers_wrapped"] = bool(moe_layers)
        log(f"EP layers wrapped ({len(moe_layers)}): {'PASS' if checks['ep_layers_wrapped'] else 'FAIL'}")

        # Read off the layer's own ep_size rather than the module constant, so a config that
        # silently resolved to a different group size fails here instead of agreeing with itself.
        first = moe_layers[0] if moe_layers else None
        checks["expert_bank_split_ep_way"] = first is not None and (
            first.experts_per_rank == first.num_experts // first.ep_size
            and first.expert_end - first.expert_start == first.experts_per_rank
            and first.experts_per_rank < first.num_experts
        )
        log(f"Expert bank split EP-way: {'PASS' if checks['expert_bank_split_ep_way'] else 'FAIL'}")

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
        log(f"  SFT EP TEST {'PASSED' if all_passed else 'FAILED'}")
        if not all_passed:
            log(f"  Failed: {[k for k, v in checks.items() if not v]}")
        log(f"{'#' * 70}\n")

        trainer.cleanup_ep()
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
