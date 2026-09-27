#!/usr/bin/env python
"""
Test script for SFT training with LiquidAI/LFM2-24B-A2B MoE model.

Validates two parallelism modes:
1. fsdp  — Default FSDP data parallelism (no EP), all experts on every GPU
2. ep    — EP=2, experts distributed across GPUs via DeepEP

LFM2-24B-A2B architecture:
  - 40 hidden layers (hybrid: conv + full attention)
  - MoE: 64 experts, top-k=4, sigmoid routing with expert bias
  - 2 dense layers (non-routed)
  - 2048 hidden size, 1536 MoE intermediate size
  - ~24B total params, ~2B active per token

Usage:
    # FSDP-only (no EP)
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_lfm2_moe.py --mode fsdp

    # EP=2
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_lfm2_moe.py --mode ep

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed (for EP mode)
    - Model: LiquidAI/LFM2-24B-A2B (auto-downloaded)
"""

import argparse
import os

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.dispatcher import destroy_all_dispatchers
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import LFM2_24B_A2B
from tests.common.utils import cleanup_memory, gpu_mem_gb, log, training_run_checks

MODEL_NAME = LFM2_24B_A2B
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 1e-5
SEED = 42

MODES = {
    "fsdp": {
        "name": "FSDP (no EP)",
        "parallelism": {},
        "sft_extra": {},
    },
    "ep": {
        "name": "EP=2",
        "parallelism": {"ep_size": 2},
        "sft_extra": {"ddp_find_unused_parameters": True},
    },
}


def run_mode(ctx, tokenizer, mode_key: str) -> dict[str, bool]:
    """Run SFT test for one parallelism mode; its checks are keyed ``<mode>_<check>``."""
    mode_config = MODES[mode_key]
    mode_name = mode_config["name"]
    log(f"\n{'=' * 70}")
    log(f"SFT LFM2 MoE TEST: {mode_name}")
    log(f"{'=' * 70}")

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
    log(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    log(f"\n--- Loading model ({mode_name}) ---")
    log(f"GPU memory before load: {gpu_mem_gb():.1f}GB")

    parallelism_config = ParallelismConfig(**mode_config["parallelism"])

    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )

    log(f"Model loaded: {model.config.model_type}")
    log(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    log(f"GPU memory after load: {gpu_mem_gb():.1f}GB")

    checks: dict[str, bool] = {}
    if parallelism_config.needs_ep_wrappers:
        wrapped = len(ep_layers(model))
        log(f"EP MoE layers detected: {wrapped}")
        checks["ep_layers_wrapped"] = wrapped > 0

    sft_kwargs = {
        "output_dir": os.path.join(ctx.output_dir, mode_key),
        "max_steps": NUM_TRAIN_STEPS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "warmup_steps": 1,
        "max_length": MAX_SEQ_LENGTH,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_liger_kernel": False,  # already applied by load_distributed_model
        "logging_steps": 1,
        "eval_strategy": "steps",
        "eval_steps": NUM_TRAIN_STEPS,
        "save_strategy": "no",
        "report_to": [],
        "logging_nan_inf_filter": False,
        "dataloader_num_workers": 0,
        "dataloader_drop_last": True,
        "remove_unused_columns": False,
        "fsdp": "",  # Mixin handles FSDP wrapping
    }
    sft_kwargs.update(mode_config["sft_extra"])
    config = SFTConfig(**sft_kwargs)

    log(f"\n--- Creating DistributedSFTTrainer ({mode_name}) ---")
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    log("Trainer created successfully")

    log("\n--- Running initial evaluation ---")
    barrier()
    eval_results = trainer.evaluate()
    initial_loss = eval_results.get("eval_loss", float("inf"))
    log(f"Initial eval loss: {initial_loss:.4f}")

    log(f"\n--- Starting training ({mode_name}) ---")
    barrier()
    train_result = trainer.train()

    log("\n--- Training completed ---")
    log(f"Training loss: {train_result.training_loss:.6f}")
    log(f"Steps completed: {train_result.global_step}")
    log(f"GPU memory after training: {gpu_mem_gb():.1f}GB")

    log("\n--- Running final evaluation ---")
    barrier()
    final_eval = trainer.evaluate()
    final_loss = final_eval.get("eval_loss", float("inf"))
    log(f"Loss: {initial_loss:.4f} -> {final_loss:.4f}")

    checks.update(training_run_checks(train_result, trainer, NUM_TRAIN_STEPS, grad_norms=True))
    checks["final_eval_loss_finite"] = bool(torch.isfinite(torch.tensor(final_loss)))

    trainer.cleanup_ep()
    return {f"{mode_key}_{name}": ok for name, ok in checks.items()}


def _parse_mode() -> list[str]:
    """Parse --mode argument. Returns the keys of the modes to run."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()) + ["all"],
        default="fsdp",
        help="Parallelism mode to test (default: fsdp). Use 'all' to run all modes "
        "sequentially (may fail for consecutive EP modes due to DeepEP buffer cleanup).",
    )
    args, _ = parser.parse_known_args()

    if args.mode == "all":
        return list(MODES)
    return [args.mode]


@gpu_test_main(exact_world_size=2, prefix="sft_lfm2")
def run(ctx):
    modes_to_run = _parse_mode()

    log(f"\n{'=' * 70}")
    log("SFT LFM2 MoE TEST SUITE")
    log(f"{'=' * 70}")
    log(f"World size: {ctx.world_size}")
    log(f"Model: {MODEL_NAME}")
    log(f"GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"GPU memory: {torch.cuda.get_device_properties(ctx.local_rank).total_memory / 1e9:.1f}GB")
    log(f"Modes: {[MODES[m]['name'] for m in modes_to_run]}")

    # Registry-wide rather than per trainer: a finalizer bound to a trainer would keep each mode's
    # model alive into the next mode.
    ctx.on_teardown(destroy_all_dispatchers)

    log("\n--- Ensuring model is downloaded ---")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    log("\n--- Loading tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    checks: dict[str, bool] = {}
    for mode_key in modes_to_run:
        checks.update(run_mode(ctx, tokenizer, mode_key))
        cleanup_memory()

    return {"checks": checks}


if __name__ == "__main__":
    run()
