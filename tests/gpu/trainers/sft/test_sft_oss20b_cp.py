#!/usr/bin/env python
"""
SFT training test with pure Context Parallelism (CP=2) on GptOss-20B.

Smoke test: DistributedSFTTrainer trains a MoE model under CP only (Ulysses sequence sharding,
experts local via grouped-GEMM). Checks that CP (and not EP) engages, that every logged loss and grad
norm is finite, and that the last-step loss is below the first. No loss is compared against a
reference, so a wrong-but-finite loss passes. Complements the EP+CP gpt-oss test (this isolates CP).

Model: unsloth/gpt-oss-20b-BF16 (MoE, 32 experts). CP requires flash/flex attention and
seq_length divisible by cp_size.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_oss20b_cp.py
"""

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
from tests.common.utils import log

MODEL_NAME = GPT_OSS_20B
CP_SIZE = 2
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096  # must be divisible by cp_size
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
SEED = 42


@gpu_test_main(min_world_size=CP_SIZE, prefix="sft_oss20b_cp_test")
def run(ctx):
    log(f"\n{'#' * 70}")
    log(f"  SFT with Context Parallelism (CP={CP_SIZE}) Test on GptOss-20B")
    log(f"  World: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'#' * 70}")
    assert MAX_SEQ_LENGTH % CP_SIZE == 0, f"max_seq_length ({MAX_SEQ_LENGTH}) must be divisible by cp_size ({CP_SIZE})"

    log("\nEnsuring model is downloaded...")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("\n--- Creating synthetic datasets ---")
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
    log(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    log(f"\n--- Loading model with CP={CP_SIZE} ---")
    parallelism_config = ParallelismConfig(cp_size=CP_SIZE)
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
        fsdp="",
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

    assert trainer.is_cp_mode, "Trainer should be in CP mode"
    assert not trainer.is_ep_mode, "CP test must not be in EP mode"
    log("Confirmed: trainer.is_cp_mode is True (pure CP)")

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

    checks["cp_mode"] = trainer.is_cp_mode
    log(f"CP mode active: {'PASS' if checks['cp_mode'] else 'FAIL'}")

    if grad_norms:
        grad_ok = all(not (torch.isnan(torch.tensor(g)) or torch.isinf(torch.tensor(g))) for g in grad_norms)
        checks["grad_finite"] = grad_ok
        log(f"Grad norms finite: {'PASS' if grad_ok else 'FAIL'}")

    return {"checks": checks}


if __name__ == "__main__":
    run()
