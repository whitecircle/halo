#!/usr/bin/env python
"""
Distillation training test with EP/TP parallelism on GptOss-20B.

Validates that DistributedDistillationTrainer works with different parallelism
modes using GptOss-20B as both teacher and student (self-distillation for
testing). The teacher is always loaded without parallelism; only the student
gets EP/TP.

Modes:
    fsdp   — Standard FSDP2 (no EP/TP), 2 GPUs
    ep     — Expert Parallelism (EP=2), 2 GPUs
    tp     — Tensor Parallelism (TP=2), 2 GPUs
    ep_tp  — Combined EP+TP (EP=2, TP=2), 2 GPUs
    all    — Run all modes sequentially

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/other/test_distillation_oss20b.py --mode ep

    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/other/test_distillation_oss20b.py --mode all

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed (for EP and EP+TP modes)
"""

import argparse
import math
import random
import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from src.configs.distillation_config import DistillationConfig
from src.distributed.expert_parallel.base_layer import has_grouped_mm
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.distillation.teacher_distillation import DistributedDistillationTrainer
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, log

MODEL_NAME = GPT_OSS_20B
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
SEED = 42


def create_distillation_dataset(num_samples: int, tokenizer, seed: int = SEED) -> Dataset:
    """Create synthetic math SFT data for distillation testing."""
    random.seed(seed)

    templates = [
        ("What is {a} + {b}?", lambda a, b: a + b),
        ("Calculate {a} * {b}.", lambda a, b: a * b),
        ("What is {a} - {b}?", lambda a, b: a - b),
    ]
    answers = [
        "The answer is {result}. I calculated this with {a} and {b}.",
        "That equals {result}.",
        "{result} is the answer.",
    ]

    data = []
    for _ in range(num_samples):
        tmpl, op = random.choice(templates)
        a, b = random.randint(1, 100), random.randint(1, 100)
        result = op(a, b)
        messages = [
            {"role": "user", "content": tmpl.format(a=a, b=b)},
            {"role": "assistant", "content": random.choice(answers).format(a=a, b=b, result=result)},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        data.append({"text": text})

    return Dataset.from_list(data)


def run_distillation_test(mode: str, rank: int, world_size: int, local_rank: int) -> bool:
    """Run a single distillation test with the given parallelism mode."""
    use_gmm = has_grouped_mm()
    if mode == "fsdp":
        pc = ParallelismConfig()
        attn_impl = "flash_attention_2"
        mode_label = "FSDP2 (no EP/TP)"
    elif mode == "ep":
        pc = ParallelismConfig(ep_size=2, use_grouped_gemm=use_gmm)
        attn_impl = "flex_attention"
        mode_label = "EP=2"
    elif mode == "tp":
        pc = ParallelismConfig(tp_size=2, max_concurrent_loading=1)
        attn_impl = "flash_attention_2"
        mode_label = "TP=2"
    elif mode == "ep_tp":
        pc = ParallelismConfig(ep_size=2, tp_size=2)
        attn_impl = "flex_attention"
        mode_label = "EP=2+TP=2"
    else:
        log(f"Unknown mode: {mode}")
        return False

    log(f"\n{'#' * 70}")
    log(f"  Distillation Test — {mode_label}")
    log(f"  Model: {MODEL_NAME} (self-distillation)")
    log(f"  World: {world_size}, GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"{'#' * 70}")

    output_dir, cache_dir = setup_cache_dirs(f"distill_oss20b_{mode}", rank)

    try:
        log("\nEnsuring model is downloaded...")
        ensure_model_downloaded(MODEL_NAME, rank)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log("Creating synthetic datasets...")
        train_dataset = create_distillation_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_distillation_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)

        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH)

        train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        log(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        log(f"\n--- Loading student model ({mode_label}) ---")
        student_model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=pc,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            use_liger_kernel=True,
        )
        student_params = sum(p.numel() for p in student_model.parameters())
        log(f"Student params: {student_params / 1e9:.2f}B")
        log(f"GPU mem after student: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

        log("\n--- Loading teacher model (no parallelism) ---")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        log(f"Teacher params: {teacher_params / 1e9:.2f}B")

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        config = DistillationConfig(
            output_dir=output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,  # already applied by load_distributed_model
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            dataset_num_proc=1,
            fsdp="",  # Mixin handles wrapping
            ddp_find_unused_parameters=pc.is_ep_mode,
            distill_loss="kl_divergence",
            distill_temperature=1.0,
            distill_alpha=0.5,
        )

        log("\n--- Creating DistributedDistillationTrainer ---")
        trainer = DistributedDistillationTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            parallelism_config=pc,
        )

        if pc.is_ep_mode:
            assert trainer.is_ep_mode, "Expected EP mode"
            log("Confirmed: EP mode active")
        if pc.is_tp_mode:
            assert trainer.is_tp_mode, "Expected TP mode"
            log("Confirmed: TP mode active")

        teacher_trainable = sum(1 for p in trainer.teacher_model.parameters() if p.requires_grad)
        assert teacher_trainable == 0, f"Teacher has {teacher_trainable} trainable params"
        log("Confirmed: teacher frozen")

        log(f"GPU mem before train: {torch.cuda.memory_allocated() / 1e9:.1f}GB")

        log(f"\n--- Training ({NUM_TRAIN_STEPS} steps) ---")
        train_result = trainer.train()

        training_loss = train_result.training_loss
        log_history = trainer.state.log_history
        step_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
        grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]
        distill_losses = [e["distillation_loss"] for e in log_history if "distillation_loss" in e]
        sft_losses = [e["sft_loss"] for e in log_history if "sft_loss" in e]

        log("\n--- Metrics ---")
        log(f"Final loss: {training_loss:.6f}")
        log(f"Step losses: {[f'{l:.4f}' for l in step_losses]}")
        if grad_norms:
            log(f"Grad norms: {[f'{g:.2f}' for g in grad_norms]}")
        if distill_losses:
            log(f"Distillation losses: {[f'{l:.4f}' for l in distill_losses]}")
        if sft_losses:
            log(f"SFT losses: {[f'{l:.4f}' for l in sft_losses]}")

        log("\n--- Checks ---")
        checks = {}

        checks["training_completed"] = len(step_losses) == NUM_TRAIN_STEPS
        log(f"Training completed: {'PASS' if checks['training_completed'] else 'FAIL'}")

        loss_finite = math.isfinite(training_loss) and all(math.isfinite(l) for l in step_losses)
        checks["loss_finite"] = loss_finite
        log(f"Loss finite: {'PASS' if loss_finite else 'FAIL'}")

        if grad_norms:
            grad_ok = all(math.isfinite(g) for g in grad_norms)
            checks["grad_finite"] = grad_ok
            log(f"Grad norms finite: {'PASS' if grad_ok else 'FAIL'}")

        if distill_losses:
            distill_ok = all(math.isfinite(l) for l in distill_losses)
            checks["distill_loss_finite"] = distill_ok
            log(f"Distillation losses finite: {'PASS' if distill_ok else 'FAIL'}")

        teacher_trainable_after = sum(1 for p in trainer.teacher_model.parameters() if p.requires_grad)
        checks["teacher_frozen"] = teacher_trainable_after == 0
        log(f"Teacher frozen after training: {'PASS' if checks['teacher_frozen'] else 'FAIL'}")

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
        log(f"  DISTILLATION {mode_label} TEST {'PASSED' if all_passed else 'FAILED'}")
        if not all_passed:
            log(f"  Failed: {[k for k, v in checks.items() if not v]}")
        log(f"{'#' * 70}\n")

        if hasattr(trainer, "cleanup_ep"):
            trainer.cleanup_ep()
        del trainer, student_model, teacher_model
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)
        barrier()
        return all_passed

    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        if rank == 0:
            traceback.print_exc()
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["fsdp", "ep", "tp", "ep_tp", "all"])
    args, _ = parser.parse_known_args()  # tolerate torchrun's own argv

    rank, world_size, local_rank = init_distributed()
    PartialState()

    if world_size < 2:
        log("ERROR: Need at least 2 GPUs")
        teardown_distributed()
        return 1

    modes = ["fsdp", "ep", "tp", "ep_tp"] if args.mode == "all" else [args.mode]
    results = {}

    for mode in modes:
        passed = run_distillation_test(mode, rank, world_size, local_rank)
        results[mode] = passed

        cleanup_memory()
        barrier()

    log(f"\n{'=' * 70}")
    log("  DISTILLATION TEST SUMMARY (GptOss-20B)")
    log(f"{'=' * 70}")
    for mode, passed in results.items():
        log(f"  {mode:8s}: {'PASSED' if passed else 'FAILED'}")
    all_passed = all(results.values())
    log(f"{'=' * 70}")
    log(f"  OVERALL: {'PASSED' if all_passed else 'FAILED'}")
    log(f"{'=' * 70}\n")

    teardown_distributed()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
