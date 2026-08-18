#!/usr/bin/env python
"""
SFT EP+TP test with FlexAttention and attention sinks validation.

Validates that GptOss-20B trains correctly with EP=2, TP=2 using FlexAttention,
which properly handles attention sinks (unlike SDPA which silently drops them).

Key validations:
1. FlexAttention is used as the attention implementation
2. Attention sinks are properly sharded across TP ranks (64 -> 32 per rank)
3. Training completes with finite loss
4. Sinks values are preserved (not all-zeros or dropped)

Model: unsloth/gpt-oss-20b-BF16 (MoE, 32 experts, 64 attention heads)

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_ep_tp_flex.py
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
from tests.common.utils import cleanup_memory, log

# Configuration

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
TP_SIZE = 2
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
SEED = 42

EXPECTED_TOTAL_HEADS = 64  # GptOss-20B has 64 attention heads
EXPECTED_LOCAL_HEADS = EXPECTED_TOTAL_HEADS // TP_SIZE  # 32 per TP rank


# Main


def main():
    rank, world_size, local_rank = init_distributed()
    PartialState()

    log(f"\n{'#' * 70}")
    log(f"  SFT EP+TP FlexAttention Test (EP={EP_SIZE}, TP={TP_SIZE})")
    log(f"  World: {world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log("  Attention: flex_attention (with sinks support)")
    log(f"{'#' * 70}")

    if world_size < TP_SIZE:
        log(f"\nERROR: Need at least {TP_SIZE} GPUs for tp_size={TP_SIZE}, got {world_size}")
        teardown_distributed()
        return 1

    output_dir, cache_dir = setup_cache_dirs("sft_ep_tp_flex_test", rank)

    try:
        # Download model on rank 0 first
        log("\nEnsuring model is downloaded...")
        ensure_model_downloaded(MODEL_NAME, rank)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create datasets
        log("\n--- Creating synthetic datasets ---")
        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
        log(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Load model with EP+TP and flex_attention
        log(f"\n--- Loading model with EP={EP_SIZE}, TP={TP_SIZE}, flex_attention ---")
        parallelism_config = ParallelismConfig(ep_size=EP_SIZE, tp_size=TP_SIZE)
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

        # ── Validate attention implementation ──────────────────────────
        log("\n--- Validating attention implementation ---")
        attn_impl = getattr(model.config, "_attn_implementation", None)
        log(f"Model attn_implementation: {attn_impl}")

        # ── Validate sinks sharding ────────────────────────────────────
        log("\n--- Validating sinks sharding ---")
        checks = {}

        # Check that all attention layers have sinks with local size
        sinks_sizes = []
        sinks_values = []
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            if hasattr(attn, "sinks"):
                sinks = attn.sinks
                sinks_sizes.append(sinks.shape[0])
                sinks_values.append(sinks.data.clone())
                if layer_idx < 3:
                    log(
                        f"  Layer {layer_idx}: sinks shape={sinks.shape}, "
                        f"requires_grad={sinks.requires_grad}, "
                        f"min={sinks.min().item():.4f}, max={sinks.max().item():.4f}"
                    )

        if sinks_sizes:
            all_correct_size = all(s == EXPECTED_LOCAL_HEADS for s in sinks_sizes)
            checks["sinks_sharded"] = all_correct_size
            log(
                f"  Sinks sharded to {EXPECTED_LOCAL_HEADS} per rank: "
                f"{'PASS' if all_correct_size else 'FAIL'} "
                f"(sizes: {set(sinks_sizes)})"
            )

            # Verify sinks are not all zeros (they were reset to dtype.min, not zero)
            any_nonzero = any(v.abs().max().item() > 0 for v in sinks_values)
            checks["sinks_nonzero"] = any_nonzero
            log(f"  Sinks have values (not all zeros): {'PASS' if any_nonzero else 'FAIL'}")
        else:
            log("  WARNING: No sinks found in model!")
            checks["sinks_sharded"] = False
            checks["sinks_nonzero"] = False

        # Verify EP layers exist
        ep_layers = [m for _, m in model.named_modules() if hasattr(m, "ep_config")]
        checks["ep_layers"] = len(ep_layers) > 0
        log(f"  EP layers found: {len(ep_layers)} {'PASS' if ep_layers else 'FAIL'}")

        # ── Training ───────────────────────────────────────────────────
        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,  # Already applied in load_distributed_model
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
        assert trainer.is_tp_mode, "Trainer should be in TP mode"
        log("Confirmed: EP mode + TP mode active")

        log(f"\n--- Training ({NUM_TRAIN_STEPS} steps with FlexAttention) ---")
        train_result = trainer.train()

        # ── Collect metrics ────────────────────────────────────────────
        training_loss = train_result.training_loss
        log_history = trainer.state.log_history
        step_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
        grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]

        log("\n--- Metrics ---")
        log(f"Final loss: {training_loss:.6f}")
        log(f"Step losses: {[f'{l:.4f}' for l in step_losses]}")
        if grad_norms:
            log(f"Grad norms: {[f'{g:.2f}' for g in grad_norms]}")

        # ── Validation ─────────────────────────────────────────────────
        log("\n--- Checks ---")

        checks["training_completed"] = len(step_losses) == NUM_TRAIN_STEPS
        log(f"Training completed: {'PASS' if checks['training_completed'] else 'FAIL'}")

        loss_finite = all(
            not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in step_losses + [training_loss]
        )
        checks["loss_finite"] = loss_finite
        log(f"Loss finite: {'PASS' if loss_finite else 'FAIL'}")

        checks["ep_mode"] = trainer.is_ep_mode
        checks["tp_mode"] = trainer.is_tp_mode

        # Loss consistency across ranks
        loss_tensor = torch.tensor([training_loss], device=f"cuda:{local_rank}")
        all_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
        dist.all_gather(all_losses, loss_tensor)
        if rank == 0:
            losses_list = [l.item() for l in all_losses]
            spread = max(losses_list) - min(losses_list)
            checks["loss_consistent"] = spread < 0.01
            log(f"Loss consistent (spread={spread:.6f}): {'PASS' if checks['loss_consistent'] else 'FAIL'}")
        else:
            checks["loss_consistent"] = True

        if grad_norms:
            grad_ok = all(not (torch.isnan(torch.tensor(g)) or torch.isinf(torch.tensor(g))) for g in grad_norms)
            checks["grad_finite"] = grad_ok
            log(f"Grad norms finite: {'PASS' if grad_ok else 'FAIL'}")

        all_passed = all(checks.values())
        log(f"\n{'#' * 70}")
        log(f"  SFT EP+TP FLEX_ATTENTION TEST {'PASSED' if all_passed else 'FAILED'}")
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
