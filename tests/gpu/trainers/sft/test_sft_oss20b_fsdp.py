#!/usr/bin/env python
"""
SFT training test with plain FSDP2 (no EP/TP/CP) on GptOss-20B.

Smoke test: DistributedSFTTrainer trains a MoE model under pure data-parallel FSDP2 (experts
stay local with grouped-GEMM expert compute and no DeepEP all-to-all; params, grads and optimizer
states are sharded across the DP mesh). Checks that no EP/TP/CP mode engages, that every logged loss
and grad norm is finite, and that the last-step loss is below the first. No loss is compared against
a reference, so a wrong-but-finite loss passes. Complements the EP/ETP/TP/CP gpt-oss SFT tests.

Model: unsloth/gpt-oss-20b-BF16 (MoE, 32 experts)

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_oss20b_fsdp.py
"""

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.base_layer import has_grouped_mm
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import log

MODEL_NAME = GPT_OSS_20B
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
SEED = 42


@gpu_test_main(min_world_size=1, prefix="sft_oss20b_fsdp_test")
def run(ctx):
    log(f"\n{'#' * 70}")
    log("  SFT with plain FSDP2 (no EP/TP/CP) Test on GptOss-20B")
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
        output_dir=ctx.output_dir,
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
        not (torch.isnan(torch.tensor(lv)) or torch.isinf(torch.tensor(lv))) for lv in step_losses + [training_loss]
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

    return {"checks": checks}


if __name__ == "__main__":
    run()
