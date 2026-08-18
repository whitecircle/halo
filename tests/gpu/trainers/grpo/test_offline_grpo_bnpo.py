#!/usr/bin/env python
"""
Offline GRPO test with BNPO loss, production-like hyperparameters, and GenerateExamplesCallback.

Runs two modes sequentially:
1. FSDP (standard data parallelism, no TP)
2. TP=2 (tensor parallelism)

Both modes use the same config matching production training:
- loss_type: bnpo
- per_device_train_batch_size: 2
- gradient_accumulation_steps: 8
- gradient_checkpointing with non-reentrant
- cosine LR schedule with warmup
- best_completion_emphasis: 1.5 (a real boost; the consumer ignores factors <= 1.0)
- initial_min_log_prob scheduling: -0.2 -> -3.0
- GenerateExamplesCallback for text generation during eval

Model: Qwen/Qwen3-0.6B (dense, supports tp_plan="auto")

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_offline_grpo_bnpo.py
"""

import argparse
import math
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.callbacks.generate_examples import GenerateExamplesCallback
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.datasets import create_offline_grpo_dataset
from tests.common.distributed import cleanup_dirs, shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, log

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 20
EVAL_STEPS = 10
BATCH_SIZE = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-8
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 512
NUM_TRAIN_SAMPLES = 256
NUM_EVAL_SAMPLES = 32
NUM_GEN_EXAMPLES = 3
MAX_GEN_TOKENS = 64
SEED = 42
# 30% of prompts carry a prior exchange, so the collator sees multi-turn prompts too.
MULTI_TURN_RATIO = 0.3


def create_generate_dataset(tokenizer, eval_dataset: Dataset) -> Dataset:
    """Tokenize eval prompts for GenerateExamplesCallback."""
    records = []
    for sample in eval_dataset:
        encoded = tokenizer(
            sample["prompt"],
            truncation=True,
            padding=True,
            max_length=MAX_PROMPT_LENGTH,
            add_special_tokens=False,
        )
        records.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    return Dataset.from_list(records)


def merge_checks(checks: dict[str, bool], mode_checks: dict[str, bool]) -> None:
    """AND one mode's results into the run's checks — both modes report the same check names."""
    for name, ok in mode_checks.items():
        checks[name] = ok and checks.get(name, True)


def run_training(
    mode: str,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    generate_dataset: Dataset,
    output_dir: str,
) -> tuple[dict[str, bool], dict]:
    """Run a single training pass (FSDP or TP). Returns (checks, metrics)."""
    is_tp = mode == "tp"
    mode_label = "TP=2" if is_tp else "FSDP"

    log(f"\n{'=' * 70}")
    log(f"  Offline GRPO BNPO — {mode_label}")
    log(f"{'=' * 70}")

    if is_tp:
        parallelism_config = ParallelismConfig(tp_size=2)
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            use_liger_kernel=True,
            # FA4 re-JITs every step in the GRPO eager loop (compile storm), so pin FA2.
            attn_implementation="flash_attention_2",
        )
    else:
        parallelism_config = ParallelismConfig()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

    log(f"Model loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    run_output_dir = os.path.join(output_dir, f"run_{mode}")

    config = OfflineGRPOConfig(
        output_dir=run_output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=15,
        seed=SEED,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        use_liger_kernel=not is_tp,
        loss_type="bnpo",
        kl_beta=0.0,
        initial_min_log_prob=-0.2,
        best_completion_emphasis=1.5,
        disable_dropout=True,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        # The eval dataloader is not distributed: under TP the DTensor collective forward hangs.
        eval_strategy="steps" if not is_tp else "no",
        eval_steps=EVAL_STEPS,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        dataloader_drop_last=True,
        fsdp="",
    )

    callbacks = []
    if not is_tp:
        generate_callback = GenerateExamplesCallback(
            preprocessed_dataset=generate_dataset,
            tokenizer=tokenizer,
            num_examples=NUM_GEN_EXAMPLES,
            max_new_tokens=MAX_GEN_TOKENS,
            logger_backend="none",
        )
        callbacks.append(generate_callback)

    trainer = OfflineGRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if not is_tp else None,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
        callbacks=callbacks,
    )

    if is_tp:
        assert trainer.is_tp_mode, "Expected is_tp_mode=True"
    log(
        f"Trainer ready: loss_type={config.loss_type}, "
        f"best_completion_emphasis={config.best_completion_emphasis}, "
        f"initial_min_log_prob={config.initial_min_log_prob}"
    )

    train_result = trainer.train()

    training_loss = train_result.training_loss
    step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
    eval_losses = [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e]

    metrics = {
        "training_loss": training_loss,
        "step_losses": step_losses,
        "eval_losses": eval_losses,
    }

    log(f"\n  --- {mode_label} Results ---")
    log(f"  Training loss: {training_loss:.6f}")
    log(f"  Step losses: {[f'{l:.4f}' for l in step_losses]}")
    if eval_losses:
        log(f"  Eval losses: {[f'{l:.4f}' for l in eval_losses]}")

    checks = {}
    checks["loss_finite"] = math.isfinite(training_loss)
    checks["all_steps_finite"] = all(math.isfinite(l) for l in step_losses)
    checks["loss_reasonable"] = training_loss < 100.0
    if not is_tp:
        checks["has_eval"] = len(eval_losses) > 0
        checks["eval_finite"] = all(math.isfinite(l) for l in eval_losses) if eval_losses else False

    log("\n  --- Assertions ---")
    for name, passed in checks.items():
        log(f"  {name}: {'PASS' if passed else 'FAIL'}")

    del trainer, model
    cleanup_memory()
    barrier()

    return checks, metrics


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fsdp", "tp", "all"], default="all")
    args, _ = parser.parse_known_args()

    log(f"\n{'#' * 70}")
    log("  Offline GRPO BNPO Test (Production Hyperparameters)")
    log(f"  Model: {MODEL_NAME}")
    log(f"  World: {ctx.world_size}, GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Mode: {args.mode}")
    log(f"  Config: bs={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, lr={LEARNING_RATE}")
    log("  GRPO: loss=bnpo, bce=1.0, init_min_log_prob=-0.2")
    log(f"{'#' * 70}")

    # One output dir for the whole world (ctx.output_dir is per-rank).
    output_dir = shared_scratch_dir("offline_grpo_bnpo")
    ctx.on_teardown(lambda: cleanup_dirs(output_dir))

    log("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("Creating datasets...")
    train_dataset = create_offline_grpo_dataset(
        tokenizer, NUM_TRAIN_SAMPLES, seed=SEED, multi_turn_ratio=MULTI_TURN_RATIO
    )
    eval_dataset = create_offline_grpo_dataset(
        tokenizer, NUM_EVAL_SAMPLES, seed=SEED + 1, multi_turn_ratio=MULTI_TURN_RATIO
    )
    generate_dataset = create_generate_dataset(tokenizer, eval_dataset)
    log(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}, Generate: {len(generate_dataset)}")

    modes = []
    if args.mode in ("fsdp", "all"):
        modes.append("fsdp")
    if args.mode in ("tp", "all"):
        modes.append("tp")

    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    for mode in modes:
        mode_checks, mode_metrics = run_training(
            mode,
            tokenizer,
            train_dataset,
            eval_dataset,
            generate_dataset,
            output_dir,
        )
        merge_checks(checks, mode_checks)
        metrics[f"{mode}_training_loss"] = mode_metrics["training_loss"]

    log(f"\n{'#' * 70}")
    for mode, loss in metrics.items():
        log(f"  {mode}: {loss:.6f}")
    log(f"{'#' * 70}")

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, prefix="offline_grpo_bnpo")(run)

if __name__ == "__main__":
    main()
