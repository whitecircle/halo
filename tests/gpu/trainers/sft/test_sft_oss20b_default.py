#!/usr/bin/env python
"""
SFT training test with default mode (FSDP2, no EP/CP/TP) on GptOss-20B.

Validates that DistributedSFTTrainer works correctly with FSDP2 (fully_shard)
data parallelism on a MoE model. Tests per-layer FSDP2 wrapping with
SHARD_GRAD_OP behavior for MoE models (no expert parallelism — all experts
on every GPU). Also validates checkpoint saving with FSDP2.

Model: unsloth/gpt-oss-20b-BF16 (MoE, 32 experts)

Note: Uses flash_attention_2. GptOss attention sinks require the
flex_attention torch.compile patch (patch_flex_attention_compile) for
FSDP2 compatibility — without it, the compiled backward produces NaN
gradients. The patch is auto-applied in load_distributed_model().

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_oss20b_default.py
"""

import math
import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.tolerances import TOL
from tests.common.utils import log

MODEL_NAME = GPT_OSS_20B
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
SEED = 42


def run(ctx) -> dict:
    log(f"\n{'#' * 70}")
    log("  SFT Default Mode (FSDP) Test on GptOss-20B")
    log(f"  World: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'#' * 70}")

    log("\nEnsuring model is downloaded...")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("\n--- Creating synthetic datasets ---")
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
    log(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    log("\n--- Loading model (default mode, no EP/CP/TP) ---")
    parallelism_config = ParallelismConfig()
    log(f"Config: {parallelism_config.mode_string or 'standard'}")

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

    save_dir = os.path.join(ctx.output_dir, "saved_model")
    sft_config = SFTConfig(
        output_dir=ctx.output_dir,
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

    assert not trainer.is_ep_mode, "Should NOT be in EP mode"
    assert not trainer.is_cp_mode, "Should NOT be in CP mode"
    assert not trainer.is_tp_mode, "Should NOT be in TP mode"
    log("Confirmed: default mode (no EP/CP/TP)")

    log(f"\n--- Training ({NUM_TRAIN_STEPS} steps) ---")
    train_result = trainer.train()

    log(f"\n--- Saving model to {save_dir} ---")
    trainer.save_model(save_dir)
    barrier()

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

    loss_finite = math.isfinite(training_loss) and all(math.isfinite(l) for l in step_losses)
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

    loss_reasonable = training_loss < 100
    checks["loss_reasonable"] = loss_reasonable
    log(f"Loss reasonable (<100): {'PASS' if loss_reasonable else 'FAIL'}")

    loss_tensor = torch.tensor([training_loss], device=ctx.device)
    all_losses = [torch.zeros_like(loss_tensor) for _ in range(ctx.world_size)]
    dist.all_gather(all_losses, loss_tensor)
    if ctx.rank == 0:
        spread = max(lv.item() for lv in all_losses) - min(lv.item() for lv in all_losses)
        checks["loss_consistent"] = spread < TOL.rank_loss_consistency_abs
        log(f"Loss consistent (spread={spread:.6f}): {'PASS' if checks['loss_consistent'] else 'FAIL'}")
    else:
        checks["loss_consistent"] = True

    if grad_norms:
        grad_ok = all(math.isfinite(g) for g in grad_norms)
        checks["grad_finite"] = grad_ok
        log(f"Grad norms finite: {'PASS' if grad_ok else 'FAIL'}")

    if ctx.rank == 0:
        dir_exists = os.path.isdir(save_dir)
        checks["save_dir_exists"] = dir_exists
        log(f"Save dir exists: {'PASS' if dir_exists else 'FAIL'}")
        if dir_exists:
            contents = os.listdir(save_dir)
            log(f"Save contents: {sorted(contents)}")
            has_model = any(f.startswith("model") and f.endswith(".safetensors") for f in contents) or any(
                f.startswith("pytorch_model") and f.endswith(".bin") for f in contents
            )
            checks["has_model_weights"] = has_model
            log(f"Has model weights: {'PASS' if has_model else 'FAIL'}")

    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="sft_oss20b_default")(run)

if __name__ == "__main__":
    main()
