#!/usr/bin/env python
"""
SFT EP=2 test comparing flex_attention vs flash_attention_2 on GptOss-20B.

Smoke test, one attention implementation per launch: training with EP=2 completes every step with
finite losses and grad norms, and the sink reset takes the shape the backend requires (dropped under
FA2 with --reset_sinks, a live parameter otherwise) and keeps it through training. Whether the sinks
moved is logged, not checked, and no loss is compared across the two modes or against a reference.

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

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import log, max_or_nan, step_losses

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
ARGS, _ = parser.parse_known_args()


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


@gpu_test_main(min_world_size=EP_SIZE, prefix=f"sft_ep_{ARGS.mode}")
def run(ctx):
    attn_impl = "flex_attention" if ARGS.mode == "flex" else "flash_attention_2"

    log(f"\n{'#' * 70}")
    log(f"  SFT EP={EP_SIZE} + {attn_impl} on GptOss-20B")
    log(f"  World: {ctx.world_size}, GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'#' * 70}")

    # Download model
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

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
        reset_sinks=ARGS.reset_sinks,
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
        output_dir=ctx.output_dir,
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
        logging_nan_inf_filter=False,
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
    ctx.on_teardown(trainer.cleanup_ep)

    assert trainer.is_ep_mode, "Trainer should be in EP mode"
    log(f"\n--- Training ({NUM_TRAIN_STEPS} steps, {attn_impl}) ---")
    train_result = trainer.train()

    # Collect metrics
    training_loss = train_result.training_loss
    log_history = trainer.state.log_history
    losses = step_losses(trainer)
    grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]

    log("\n--- Metrics ---")
    log(f"Final loss: {training_loss:.6f}")
    log(f"Step losses: {[f'{l:.4f}' for l in losses]}")
    log(f"Grad norms: {[f'{g:.4f}' for g in grad_norms]}")

    # Check sinks AFTER training
    log("\n--- Sink state AFTER training ---")
    sinks_after = _capture_sinks(model)
    sinks_updated = 0
    deltas = []

    for layer_idx in sorted(sinks_before):
        if sinks_before[layer_idx] is None or sinks_after[layer_idx] is None:
            continue
        before_t = sinks_before[layer_idx]["tensor"]
        after_t = sinks_after[layer_idx]["tensor"]
        delta = (after_t - before_t).abs().max().item()
        deltas.append(delta)

        if delta > 0:
            sinks_updated += 1
            if layer_idx < 5:
                log(f"  Layer {layer_idx}: UPDATED (delta={delta:.6e})")
        elif layer_idx < 3:
            log(f"  Layer {layer_idx}: unchanged (delta=0)")

    log(f"  Sinks updated: {sinks_updated}/{total_sink_layers}")
    log(f"  Max sink delta: {max_or_nan(deltas, default=0.0):.6e}")

    if sinks_updated > 0:
        log(f"  -> Sinks ARE being updated by {attn_impl}")
    else:
        log("  -> Sinks NOT updated (reset to dtype.min = inactive, gradient is zero)")

    # Validation checks
    log("\n--- Checks ---")
    checks = {}

    checks["training_completed"] = len(losses) == NUM_TRAIN_STEPS
    log(f"Training completed: {'PASS' if checks['training_completed'] else 'FAIL'}")

    loss_finite = all(math.isfinite(l) for l in losses + [training_loss])
    checks["loss_finite"] = loss_finite
    log(f"Loss finite: {'PASS' if loss_finite else 'FAIL'}")

    checks["ep_mode"] = trainer.is_ep_mode
    log(f"EP mode active: {'PASS' if checks['ep_mode'] else 'FAIL'}")

    # The sink reset is backend-specific: flash_attention_2 drops the tensor outright (its s_aux
    # path rejects one on Blackwell), every other backend keeps a dtype.min-filled parameter.
    # Pinning which shape arrived is what makes the delta comparison above mean anything.
    expect_dropped = ARGS.reset_sinks and attn_impl == "flash_attention_2"
    checks["sink_layers_found"] = total_sink_layers > 0
    checks["sink_reset_shape_matches_backend"] = dropped_sink_layers == (total_sink_layers if expect_dropped else 0)
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

    return {"checks": checks}


if __name__ == "__main__":
    run()
