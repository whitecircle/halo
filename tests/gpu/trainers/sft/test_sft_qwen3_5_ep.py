#!/usr/bin/env python
"""
Focused SFT test for Qwen3.5/Qwen3.6 MoE with Expert Parallelism.

Single mode (EP only), env-configurable model path and EP size, 4-GPU friendly.

Note: Qwen3.5/3.6 attention uses M-RoPE whose varlen path crashes Flash
Attention 2 (cudaErrorIllegalAddress). Use attn_implementation=sdpa.

Usage:
    torchrun --nproc_per_node=4 \\
        tests/gpu/trainers/sft/test_sft_qwen3_5_ep.py

Requirements:
    - 2-8 B200/B300 GPUs
    - DeepEP installed
    - Model: Qwen/Qwen3.5-35B-A3B from the Hub; HALO_TEST_QWEN3_5_MODEL points it at another
      Qwen3.5/3.6 MoE checkpoint (e.g. a local $HALO_DATA_ROOT/models/Qwen3.6-35B-A3B-patched)
"""

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_int, env_str
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_single_turn_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_5_MOE_35B
from tests.common.utils import gpu_mem_gb, log, step_losses

# Test Configuration

MODEL_NAME = env_str("HALO_TEST_QWEN3_5_MODEL", QWEN3_5_MOE_35B)
EP_SIZE_OVERRIDE = env_int("HALO_TEST_EP", None)
NUM_TRAIN_SAMPLES = 16
NUM_EVAL_SAMPLES = 4
MAX_SEQ_LENGTH = 1024
NUM_TRAIN_STEPS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1e-5
SEED = 42


@gpu_test_main(min_world_size=2, prefix="sft_qwen3_5_ep_test")
def run(ctx):
    ep_size = EP_SIZE_OVERRIDE if EP_SIZE_OVERRIDE is not None else ctx.world_size

    log(f"\n{'=' * 70}")
    log(f"SFT QWEN3.5 MoE TEST: EP={ep_size} with {MODEL_NAME} (world_size={ctx.world_size})")
    log(f"{'=' * 70}")
    log(f"GPU: {torch.cuda.get_device_name(ctx.local_rank)}")

    if ctx.world_size % ep_size != 0:
        raise ValueError(f"world_size={ctx.world_size} must be divisible by ep_size={ep_size}")

    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer: {tokenizer.__class__.__name__}")

    train_dataset = create_single_turn_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_single_turn_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
    log(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    log(f"\n--- Loading Qwen3.5 with EP={ep_size} ---")
    log(f"GPU memory before load: {gpu_mem_gb():.1f}GB")

    parallelism_config = ParallelismConfig(ep_size=ep_size)

    # FA2 crashes on Qwen3.5 M-RoPE varlen path; use sdpa.
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        use_liger_kernel=True,
    )
    log(f"Model loaded: {model.config.model_type} ({type(model).__name__})")
    log(f"GPU memory after load: {gpu_mem_gb():.1f}GB")

    ep_layers = sum(1 for m in model.modules() if hasattr(m, "ep_config"))
    log(f"EP MoE layers: {ep_layers}")

    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=1,
        max_length=MAX_SEQ_LENGTH,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=NUM_TRAIN_STEPS,
        save_strategy="no",
        report_to=[],
        logging_nan_inf_filter=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        fsdp="",
    )

    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)

    log("\n--- Initial evaluation ---")
    barrier()
    eval_results = trainer.evaluate()
    initial_loss = eval_results.get("eval_loss", float("inf"))
    log(f"Initial eval loss: {initial_loss:.4f}")

    log("\n--- Training ---")
    barrier()
    train_result = trainer.train()
    log(f"Training loss: {train_result.training_loss:.6f}")
    log(f"Steps: {train_result.global_step}")
    log(f"GPU memory after training: {gpu_mem_gb():.1f}GB")

    log("\n--- Final evaluation ---")
    barrier()
    final_eval = trainer.evaluate()
    final_loss = final_eval.get("eval_loss", float("inf"))
    log(f"Loss: {initial_loss:.4f} -> {final_loss:.4f}")

    losses = step_losses(trainer)
    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    log(f"Per-step losses: {[f'{l:.4f}' for l in losses]}")
    if grad_norms:
        log(f"Per-step grad norms: {[f'{g:.2f}' for g in grad_norms]}")

    checks = {
        "train_loss_finite": bool(torch.isfinite(torch.tensor(train_result.training_loss))),
        "step_losses_finite": all(torch.isfinite(torch.tensor(l)) for l in losses),
        "trained_all_steps": train_result.global_step == NUM_TRAIN_STEPS,
        "final_eval_loss_finite": bool(torch.isfinite(torch.tensor(final_loss))),
    }
    if grad_norms:
        checks["grad_norms_finite"] = all(torch.isfinite(torch.tensor(g)) for g in grad_norms)
    return {"checks": checks}


if __name__ == "__main__":
    run()
