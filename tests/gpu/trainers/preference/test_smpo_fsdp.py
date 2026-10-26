#!/usr/bin/env python
"""SMPO smoke test on plain FSDP2: Qwen3-0.6B, 10 steps, GC + Liger + bf16, padded or ``--mode padding_free``.

Every configured step runs with finite losses, and the padding-free row trains padding-free. Neither row compares
the objective against a reference; test_smpo_padding_free_segments.py pins padding-free per-document isolation.
Run: torchrun --nproc_per_node=2 tests/gpu/trainers/preference/test_smpo_fsdp.py [--mode padding_free]
"""

import argparse

import torch

from src.configs.preference import SmoothMarginPOConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.model_loading.parallel.loader import load_distributed_model
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.datasets import create_preference_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import log, training_run_checks

MAX_STEPS = 10
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("fsdp", "padding_free"), default="fsdp")
ARGS = parser.parse_args()
PADDING_FREE = ARGS.mode == "padding_free"


@gpu_test_main(min_world_size=2, prefix=f"smpo_{ARGS.mode}")
def run(ctx):
    log(f"SMPO {ARGS.mode} test: world size {ctx.world_size}, GPU {torch.cuda.get_device_name(ctx.local_rank)}")
    parallelism_config = ParallelismConfig()
    model, tokenizer = load_distributed_model(
        model_name_or_path=QWEN3_0_6B,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = SmoothMarginPOConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,  # Already applied in load_distributed_model
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=4096,
        max_prompt_length=2048,
        dataloader_drop_last=True,
        padding_free=PADDING_FREE,
    )
    trainer = SmoothMarginPOTrainer(
        model=model,
        args=config,
        train_dataset=create_preference_dataset(128, tokenizer, seed=SEED),
        eval_dataset=create_preference_dataset(16, tokenizer, seed=SEED + 1),
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    log(f"  Loss type: {config.loss_type}, beta: {config.beta}, target margin: {config.target_margin}")
    train_result = trainer.train()
    margins = [entry["margin"] for entry in trainer.state.log_history if "margin" in entry]
    log(f"  Margins: {[f'{margin:.4f}' for margin in margins]}")

    checks = training_run_checks(train_result, trainer, MAX_STEPS)
    if PADDING_FREE:
        checks["padding_free_active"] = trainer.args.padding_free is True
    return {"checks": checks, "metrics": {"final_train_loss": train_result.training_loss}}


if __name__ == "__main__":
    run()
