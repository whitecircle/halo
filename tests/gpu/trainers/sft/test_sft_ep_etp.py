#!/usr/bin/env python
"""
SFT training test with Expert Tensor Parallelism (ep_size=1, expert_tp_size=2).

Validates that DistributedSFTTrainer works in ETP mode on a MoE model.
In this mode:
- Expert FFN weights are sharded across GPUs via expert_tp_size
- DeepEP handles dispatch/combine with sub-EP groups
- dp_size = world_size / expert_tp_size = 1

Model: unsloth/gpt-oss-20b-BF16 (MoE, 32 experts)

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_ep_etp.py

Requirements:
    - 2x GPUs with >=80GB memory each
    - DeepEP installed
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
"""

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
from tests.common.utils import log

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 1
EXPERT_TP_SIZE = 2
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
SEED = 42


@gpu_test_main(min_world_size=EXPERT_TP_SIZE, prefix="sft_ep_etp_test")
def run(ctx):
    log(f"\n{'#' * 70}")
    log(f"  SFT with EP+ETP Mode (ep_size={EP_SIZE}, expert_tp_size={EXPERT_TP_SIZE}) Test")
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
    if ctx.rank == 0:
        log(f"Sample (first 200 chars): {train_dataset[0]['text'][:200]}...")

    log(f"\n--- Loading model with ep_size={EP_SIZE}, expert_tp_size={EXPERT_TP_SIZE} ---")
    parallelism_config = ParallelismConfig(
        ep_size=EP_SIZE,
        expert_tp_size=EXPERT_TP_SIZE,
    )
    log(f"Config: {parallelism_config.summary()}")
    log(f"ep_group_size: {parallelism_config.ep_group_size}")
    log(f"is_expert_tp_mode: {parallelism_config.is_expert_tp_mode}")

    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )
    log(f"GPU mem after load: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    log(f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    moe_layers = ep_layers(model)
    etp_layers = [m for m in moe_layers if getattr(m, "expert_tp_size", 1) == EXPERT_TP_SIZE]
    log(f"EP layers found: {len(moe_layers)}, ETP-sharded (expert_tp_size={EXPERT_TP_SIZE}): {len(etp_layers)}")

    sft_config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,  # already applied at load
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,  # Required for EP (inactive experts)
        fsdp="",  # Mixin handles FSDP wrapping
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
    ctx.on_teardown(trainer.cleanup_ep)

    assert trainer.is_ep_mode, "Trainer should be in EP mode"
    log("Confirmed: trainer.is_ep_mode is True")
    log(f"Confirmed: is_expert_tp_mode = {parallelism_config.is_expert_tp_mode}")

    log(f"\n--- Training ({NUM_TRAIN_STEPS} steps) ---")
    train_result = trainer.train()

    training_loss = train_result.training_loss
    log_history = trainer.state.log_history
    step_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]
    grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]

    log("\n--- Metrics ---")
    log(f"Final loss: {training_loss:.6f}")
    log(f"Step losses: {[f'{l:.4f}' for l in step_losses]}")
    if grad_norms:
        log(f"Grad norms: {[f'{g:.2f}' for g in grad_norms]}")

    log("\n--- Checks ---")
    checks = {}

    loss_finite = all(
        not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in step_losses + [training_loss]
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

    checks["ep_mode"] = trainer.is_ep_mode
    log(f"EP mode active: {'PASS' if checks['ep_mode'] else 'FAIL'}")

    checks["etp_mode"] = parallelism_config.is_expert_tp_mode
    log(f"ETP mode active: {'PASS' if checks['etp_mode'] else 'FAIL'}")

    # A check, not a log line: with no EP wrappers this file trains a plain dense model and
    # reports every other check green.
    checks["ep_layers_wrapped"] = bool(moe_layers)
    log(f"EP layers wrapped ({len(moe_layers)}): {'PASS' if checks['ep_layers_wrapped'] else 'FAIL'}")

    # ep_size=1 leaves the expert bank whole, so the sharding claim rests entirely on the FFN
    # being split expert_tp_size-way — without this the file is an ep1 test wearing an ETP label.
    checks["expert_ffn_sharded_etp_way"] = bool(moe_layers) and len(etp_layers) == len(moe_layers)
    log(f"Expert FFN sharded ETP-way: {'PASS' if checks['expert_ffn_sharded_etp_way'] else 'FAIL'}")

    if grad_norms:
        grad_ok = all(not (torch.isnan(torch.tensor(g)) or torch.isinf(torch.tensor(g))) for g in grad_norms)
        checks["grad_finite"] = grad_ok
        log(f"Grad norms finite: {'PASS' if grad_ok else 'FAIL'}")

    return {"checks": checks}


if __name__ == "__main__":
    run()
