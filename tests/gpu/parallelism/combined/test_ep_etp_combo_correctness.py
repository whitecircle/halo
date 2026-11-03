#!/usr/bin/env python
"""EP+ETP training deadlock regression, past step 3 through the real trainer (FSDP2 + gradient checkpointing).

An expert-TP all-reduce between DeepEP dispatch and combine times out the combine barrier ("CUDA error: unspecified
launch failure", around step 3), so it runs in token space. A single forward/backward doesn't trigger it. The math is
``test_combined_ref_correctness.py --mode ep_etp``'s.
"""

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log
from tests.common.training_checks import step_losses, training_run_checks

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
EXPERT_TP_SIZE = 2
NUM_TRAIN_SAMPLES = 32
MAX_SEQ_LENGTH = 2048
NUM_TRAIN_STEPS = 6  # Past 3, so the combine deadlock, which surfaces around step 3, would show.
BATCH_SIZE = 1
SEED = 42


@gpu_test_main(min_world_size=EP_SIZE * EXPERT_TP_SIZE, prefix="ep_etp_combo_test")
def run(ctx):
    log(f"\n{'#' * 70}")
    log(
        f"  EP+ETP COMBO correctness (ep_size={EP_SIZE}, expert_tp_size={EXPERT_TP_SIZE}, "
        f"ep_group_size={EP_SIZE * EXPERT_TP_SIZE})"
    )
    log(f"  World: {ctx.world_size}, Model: {MODEL_NAME}, Steps: {NUM_TRAIN_STEPS}")
    log(f"{'#' * 70}")

    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)

    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, expert_tp_size=EXPERT_TP_SIZE)
    log(f"Config: {parallelism_config.summary()}")
    log(
        f"ep_group_size={parallelism_config.ep_group_size} dp_size={parallelism_config.data_parallel_size} "
        f"is_expert_tp_mode={parallelism_config.is_expert_tp_mode}"
    )
    assert parallelism_config.ep_size > 1 and parallelism_config.expert_tp_size > 1, (
        "this test must exercise the ep>1 AND expert_tp>1 combo"
    )

    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )
    sft_config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        bf16=True,
        gradient_checkpointing=True,  # Required: GC + FSDP2 is the deadlock condition.
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
    )

    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    checks = parallel_shape_checks(model, parallelism_config)
    log(f"_fsdp_wrapped={getattr(trainer, '_fsdp_wrapped', '?')} (FSDP2 must be active for the deadlock path)")

    log(f"\n--- Training {NUM_TRAIN_STEPS} steps (deadlock would surface ~step 3) ---")
    train_result = trainer.train()  # A reduction inside dispatch->combine deadlocks here.

    # SFT on this tiny set drops the loss several-fold, far past batch noise.
    checks |= training_run_checks(train_result, trainer, NUM_TRAIN_STEPS, grad_norms=True, loss_decreased=True)
    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    checks["every_step_logged"] = len(step_losses(trainer)) == len(grad_norms) == NUM_TRAIN_STEPS
    return {"checks": checks}


if __name__ == "__main__":
    run()
