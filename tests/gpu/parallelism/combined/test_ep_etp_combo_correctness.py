#!/usr/bin/env python
"""EP+ETP COMBO (ep_size>1 AND expert_tp_size>1) training deadlock regression.

Exercises the combination where the EP group splits into ``expert_tp_size`` DeepEP dispatch
groups coupled by the strided expert-TP all-reduce. That coupling deadlocks if the expert-TP
all-reduce sits BETWEEN DeepEP dispatch and combine: under FSDP2's multi-stream / multi-layer
execution it times out DeepEP's intranode combine barrier across the coupled dispatch groups
("CUDA error: unspecified launch failure", typically ~step 3). The reduction therefore runs in TOKEN
space, outside the dispatch->combine span (EPMoELayerBase._dispatch_compute_combine).

The deadlock needs several steps to surface, so this runs >3 steps through the REAL trainer
(FSDP2 + gradient checkpointing) — a single forward/backward does not trigger it. It checks that
every step completes and logs a finite loss and grad norm; the EP+ETP math against a reference is
covered by ``test_combined_ref_correctness.py --mode ep_etp`` and ``test_ep_etp_inkling.py``.

Config: ep_size=2, expert_tp_size=2 (ep_group_size=4) on GptOss-20B → 4 GPUs.

Run:
    torchrun --nproc_per_node=4 \
        tests/gpu/parallelism/combined/test_ep_etp_combo_correctness.py
"""

import math

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.utils import log, step_losses

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
EXPERT_TP_SIZE = 2
NUM_TRAIN_SAMPLES = 32
MAX_SEQ_LENGTH = 2048
NUM_TRAIN_STEPS = 6  # > 3 so the EP+ETP combine deadlock, which surfaces around step 3, would show
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
    etp_layers = [m for m in ep_layers(model) if getattr(m, "expert_tp_size", 1) > 1]
    log(f"ETP layers (expert_tp_size>1): {len(etp_layers)}")
    assert etp_layers, "no ETP-wrapped layers found"

    sft_config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        bf16=True,
        gradient_checkpointing=True,  # required: GC + FSDP2 is the deadlock condition
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,
        fsdp="",
    )

    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    assert trainer.is_ep_mode, "trainer should be in EP mode"
    log(f"_fsdp_wrapped={getattr(trainer, '_fsdp_wrapped', '?')} (FSDP2 must be active for the deadlock path)")

    log(f"\n--- Training {NUM_TRAIN_STEPS} steps (deadlock would surface ~step 3) ---")
    train_result = trainer.train()  # a reduction inside dispatch->combine deadlocks here

    training_loss = train_result.training_loss
    losses = step_losses(trainer)
    grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
    log(f"Final loss={training_loss:.6f}  step_losses={[f'{l:.4f}' for l in losses]}")
    log(f"grad_norms={[f'{g:.2f}' for g in grad_norms]}")

    # logging_steps=1: every completed step logs a loss and a grad norm.
    checks = {
        "training_completed": len(losses) == NUM_TRAIN_STEPS,
        "loss_finite": all(math.isfinite(l) for l in losses + [training_loss]),
        "grad_finite": len(grad_norms) == NUM_TRAIN_STEPS and all(math.isfinite(g) for g in grad_norms),
        # SFT on this tiny set drops the loss several-fold within the run, far past batch-to-batch noise.
        "loss_decreased": len(losses) >= 2 and losses[-1] < losses[0],
    }
    return {"checks": checks}


if __name__ == "__main__":
    run()
