#!/usr/bin/env python
"""
SFT EP=2 test comparing flex_attention vs flash_attention_2 on GptOss-20B.

Validates that both attention implementations work correctly with EP=2,
compares training metrics, and checks whether attention sinks get updated.

Run each mode separately (DeepEP buffer cleanup requires separate processes):

    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_ep_flex_vs_fa2.py --mode=flex

    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_ep_flex_vs_fa2.py --mode=fa2

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed
    - flash-attn installed (for FA2 mode)
"""

import argparse
import math
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
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
SEED = 42


# Helpers


def _get_sink_tensor(sinks: torch.Tensor) -> torch.Tensor:
    """Extract raw tensor from a sink parameter, handling DTensor (FSDP2)."""
    data = sinks.data
    if hasattr(data, "full_tensor"):
        data = data.full_tensor()
    return data.clone().cpu().float()


def _capture_sinks(model) -> dict:
    """Capture per-layer sink state; ``None`` for a layer whose sinks were dropped.

    ``reset_sinks`` under flash_attention_2 sets ``self_attn.sinks = None`` (FA2's s_aux path rejects
    a tensor on Blackwell) while every other backend fills it with ``dtype.min``. Both are live
    contracts, so the entry stays in the dict and the caller asserts which one it got.
    """
    sinks = {}
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if not hasattr(attn, "sinks"):
            continue
        if attn.sinks is None:
            sinks[layer_idx] = None
            continue
        sinks[layer_idx] = {
            "tensor": _get_sink_tensor(attn.sinks),
            "requires_grad": attn.sinks.requires_grad,
            "is_parameter": isinstance(attn.sinks, torch.nn.Parameter),
            "shape": list(attn.sinks.shape),
        }
    return sinks


# Main


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["flex", "fa2"], required=True, help="flex=flex_attention, fa2=flash_attention_2"
    )
    parser.add_argument(
        "--reset_sinks",
        action="store_true",
        default=False,
        help="Reset sinks to dtype min (default: preserve pretrained values)",
    )
    args, _ = parser.parse_known_args()

    attn_impl = "flex_attention" if args.mode == "flex" else "flash_attention_2"

    rank, world_size, local_rank = init_distributed()
    PartialState()

    log(f"\n{'#' * 70}")
    log(f"  SFT EP={EP_SIZE} + {attn_impl} on GptOss-20B")
    log(f"  World: {world_size}, GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"{'#' * 70}")

    if world_size < EP_SIZE:
        log(f"\nERROR: Need at least {EP_SIZE} GPUs, got {world_size}")
        teardown_distributed()
        return 1

    output_dir, cache_dir = setup_cache_dirs(f"sft_ep_{args.mode}", rank)

    try:
        # Download model
        ensure_model_downloaded(MODEL_NAME, rank)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create datasets
        log("\n--- Creating synthetic datasets ---")
        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
        log(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Load model
        log(f"\n--- Loading model with EP={EP_SIZE}, attn={attn_impl} ---")
        parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
        log(f"Config: {parallelism_config.summary()}")

        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            use_liger_kernel=True,
            reset_sinks=args.reset_sinks,
        )
        log(f"GPU mem after load: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        log(f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

        actual_attn = getattr(model.config, "_attn_implementation", "unknown")
        log(f"Actual attn_implementation: {actual_attn}")

        # Capture sinks BEFORE training
        log("\n--- Sink state BEFORE training ---")
        sinks_before = _capture_sinks(model)
        total_sink_layers = len(sinks_before)
        dropped_sink_layers = sum(1 for s in sinks_before.values() if s is None)
        trainable_sinks = sum(1 for s in sinks_before.values() if s is not None and s["requires_grad"])
        for idx in sorted(sinks_before)[:3]:
            s = sinks_before[idx]
            if s is None:
                log(f"  Layer {idx}: sinks dropped (None)")
                continue
            log(
                f"  Layer {idx}: shape={s['shape']}, requires_grad={s['requires_grad']}, "
                f"min={s['tensor'].min().item():.4e}, max={s['tensor'].max().item():.4e}"
            )
        log(f"  Total: {total_sink_layers} layers, Dropped: {dropped_sink_layers}, Trainable: {trainable_sinks}")

        # Training
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
        log(f"\n--- Training ({NUM_TRAIN_STEPS} steps, {attn_impl}) ---")
        train_result = trainer.train()

        # Collect metrics
        training_loss = train_result.training_loss
        log_history = trainer.state.log_history
        step_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
        grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]

        log("\n--- Metrics ---")
        log(f"Final loss: {training_loss:.6f}")
        log(f"Step losses: {[f'{l:.4f}' for l in step_losses]}")
        log(f"Grad norms: {[f'{g:.4f}' for g in grad_norms]}")

        # Check sinks AFTER training
        log("\n--- Sink state AFTER training ---")
        sinks_after = _capture_sinks(model)
        sinks_updated = 0
        max_delta = 0.0

        for layer_idx in sorted(sinks_before):
            if sinks_before[layer_idx] is None or sinks_after[layer_idx] is None:
                continue
            before_t = sinks_before[layer_idx]["tensor"]
            after_t = sinks_after[layer_idx]["tensor"]
            delta = (after_t - before_t).abs().max().item()

            if delta > 0:
                sinks_updated += 1
                max_delta = max(max_delta, delta)
                if layer_idx < 5:
                    log(f"  Layer {layer_idx}: UPDATED (delta={delta:.6e})")
            elif layer_idx < 3:
                log(f"  Layer {layer_idx}: unchanged (delta=0)")

        log(f"  Sinks updated: {sinks_updated}/{total_sink_layers}")
        log(f"  Max sink delta: {max_delta:.6e}")

        if sinks_updated > 0:
            log(f"  -> Sinks ARE being updated by {attn_impl}")
        else:
            log("  -> Sinks NOT updated (reset to dtype.min = inactive, gradient is zero)")

        # Validation checks
        log("\n--- Checks ---")
        checks = {}

        checks["training_completed"] = len(step_losses) == NUM_TRAIN_STEPS
        log(f"Training completed: {'PASS' if checks['training_completed'] else 'FAIL'}")

        loss_finite = all(math.isfinite(l) for l in step_losses + [training_loss])
        checks["loss_finite"] = loss_finite
        log(f"Loss finite: {'PASS' if loss_finite else 'FAIL'}")

        checks["ep_mode"] = trainer.is_ep_mode
        log(f"EP mode active: {'PASS' if checks['ep_mode'] else 'FAIL'}")

        # The sink reset is backend-specific: flash_attention_2 drops the tensor outright (its s_aux
        # path rejects one on Blackwell), every other backend keeps a dtype.min-filled parameter.
        # Pinning which shape arrived is what makes the delta comparison above mean anything.
        expect_dropped = args.reset_sinks and attn_impl == "flash_attention_2"
        checks["sink_layers_found"] = total_sink_layers > 0
        checks["sink_reset_shape_matches_backend"] = dropped_sink_layers == (
            total_sink_layers if expect_dropped else 0
        )
        checks["sink_shape_stable_across_training"] = all(
            (sinks_before[i] is None) == (sinks_after[i] is None) for i in sinks_before
        )
        log(
            f"Sinks {'dropped' if expect_dropped else 'live'} on {attn_impl} "
            f"({dropped_sink_layers}/{total_sink_layers} dropped): "
            f"{'PASS' if checks['sink_reset_shape_matches_backend'] else 'FAIL'}"
        )

        if grad_norms:
            grad_finite = all(math.isfinite(g) for g in grad_norms)
            checks["grad_finite"] = grad_finite
            log(f"Grad norms finite: {'PASS' if grad_finite else 'FAIL'}")

            grad_reasonable = all(g < 1e10 for g in grad_norms)
            checks["grad_reasonable"] = grad_reasonable
            log(f"Grad norms reasonable (<1e10): {'PASS' if grad_reasonable else 'FAIL'}")

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

        all_passed = all(checks.values())
        log(f"\n{'#' * 70}")
        log(f"  SFT EP + {attn_impl}: {'PASSED' if all_passed else 'FAILED'}")
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
