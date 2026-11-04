#!/usr/bin/env python
"""SMPO with EP=2, CP=2 on GptOss-20B: step 1's loss equals an EP-only trainer's, on the initial weights.

An in-place ``dist.all_reduce`` reaches autograd only through c10d's fallback. The gradient isn't compared: CP needs
the sinks neutralized, and without them it doesn't survive a layout change (median cosine 0.26, against 0.99 with sinks
live).
"""

import torch
from transformers import AutoTokenizer

from halo.configs.preference import SMPOConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.preference.smpo import SMPOTrainer
from tests.common.datasets import create_preference_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.first_step import FirstStep, first_step_checks, score_first_step, train_recording_first_step
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory, gpu_mem_gb
from tests.common.models import GPT_OSS_20B
from tests.common.rank_log import log
from tests.common.training_checks import training_run_checks

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
CP_SIZE = 2
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
# Both lengths must be divisible by cp_size.
MAX_LENGTH = 4096
MAX_PROMPT_LENGTH = 2048
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 5e-6
SEED = 42
# With the sinks neutralized the layouts' losses drift: measured over two seeds, a microbatch by up to 16% and the
# logged step by up to 8%. A loss counted once per CP rank is 100% off.
LOSS_RTOL = 0.4


def _smpo_config(output_dir: str) -> SMPOConfig:
    return SMPOConfig(
        output_dir=output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,  # Already applied by load_distributed_model.
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )


def _load_model(parallelism_config: ParallelismConfig, attn_implementation: str):
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        use_liger_kernel=True,
    )
    return model


def _train_ep_cp(output_dir: str, tokenizer) -> tuple[dict[str, bool], FirstStep, str]:
    train_dataset = create_preference_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_preference_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
    log(f"Train dataset: {len(train_dataset)} samples, eval dataset: {len(eval_dataset)} samples")

    log(f"\n--- Loading model with EP+CP (GPU memory {gpu_mem_gb():.1f}GB) ---")
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, cp_size=CP_SIZE)
    # CP swaps flex_attention for its flash kernel; the reference scores with the one it resolved.
    model = _load_model(parallelism_config, "flex_attention")
    attn_implementation = model.config._attn_implementation
    log(f"Model loaded: {model.config.model_type}, attention {attn_implementation}, GPU memory {gpu_mem_gb():.1f}GB")

    config = _smpo_config(output_dir)
    log(f"Config: beta={config.beta}, target_margin={config.target_margin}, loss_type={config.loss_type}")
    trainer = SMPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )

    log(f"\n--- Starting training (GPU memory {gpu_mem_gb():.1f}GB) ---")
    train_result, first_step = train_recording_first_step(trainer, gradients=False)
    log(f"\n--- Training completed (GPU memory {gpu_mem_gb():.1f}GB) ---")
    checks = training_run_checks(train_result, trainer, NUM_TRAIN_STEPS, grad_norms=True)

    trainer.cleanup_ep()
    del trainer, model
    cleanup_memory()
    return checks, first_step, attn_implementation


def _score_reference(output_dir: str, tokenizer, first_step: FirstStep, attn_implementation: str) -> FirstStep:
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
    trainer = SMPOTrainer(
        model=_load_model(parallelism_config, attn_implementation),
        args=_smpo_config(output_dir),
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    reference = score_first_step(trainer, first_step)
    trainer.cleanup_ep()
    return reference


def run(ctx) -> dict:
    log(f"\n{'=' * 70}")
    log("SMPO EP+CP TEST: EP=2 + CP=2 with GptOss-20B")
    log(f"{'=' * 70}")
    log(f"World size: {ctx.world_size}, EP size: {EP_SIZE}, CP size: {CP_SIZE}, model: {MODEL_NAME}")
    log(f"Max length: {MAX_LENGTH}, max prompt length: {MAX_PROMPT_LENGTH} (divisible by cp_size={CP_SIZE})")
    log(f"Batch size: {BATCH_SIZE}, grad accum: {GRADIENT_ACCUMULATION}, steps: {NUM_TRAIN_STEPS}")
    log(f"GPU: {torch.cuda.get_device_name(ctx.local_rank)}")

    assert MAX_LENGTH % CP_SIZE == 0, f"max_length={MAX_LENGTH} must be divisible by cp_size={CP_SIZE}"
    assert MAX_PROMPT_LENGTH % CP_SIZE == 0, (
        f"max_prompt_length={MAX_PROMPT_LENGTH} must be divisible by cp_size={CP_SIZE}"
    )

    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    checks, first_step, attn_implementation = _train_ep_cp(ctx.output_dir, tokenizer)
    log(f"\n--- Scoring step 1 with an EP-only trainer (GPU memory {gpu_mem_gb():.1f}GB) ---")
    reference = _score_reference(ctx.output_dir, tokenizer, first_step, attn_implementation)
    loss_checks, metrics = first_step_checks(first_step, reference, miscount_factor=CP_SIZE, loss_rtol=LOSS_RTOL)
    return {"checks": checks | loss_checks, "metrics": metrics}


main = gpu_test_main(exact_world_size=EP_SIZE, prefix="smpo_ep_cp")(run)

if __name__ == "__main__":
    main()
