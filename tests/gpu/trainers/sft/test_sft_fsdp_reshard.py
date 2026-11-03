#!/usr/bin/env python
"""FULL_SHARD and SHARD_GRAD_OP are the same math, so their loss curves match on pure DP and CP=2.

TP+DP is excluded since FSDP2 has no all-gather strategy for TP-sharded DTensors (ParallelismConfig rejects it), and
pure TP has no DP axis to reshard. 4 GPUs, so CP=2 leaves DP=2: at DP=1 FSDP is skipped and the flag is a no-op.
"""

import math

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import barrier
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.training_checks import step_losses

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 8
MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 2e-5
SEED = 42
# Different re-gather and reduce order, so bf16 curves drift by rounding though step 0 is bit-identical. A real
# sharding bug (dropped or doubled grad sync) diverges monotonically by far more, or NaNs.
STEP0_ABS_TOL = 1e-3  # The first forward must be identical.
CURVE_REL_TOL = 0.03  # Mean |Δloss| / mean loss across the run.


def run_sft(mode_name, parallelism_kwargs, reshard, tokenizer, output_dir):
    reshard_label = "FULL_SHARD" if reshard else "SHARD_GRAD_OP"
    log(f"\n--- {mode_name} / {reshard_label} (reshard_after_forward={reshard}) ---")

    parallelism_config = ParallelismConfig(fsdp_reshard_after_forward=reshard, **parallelism_kwargs)
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )
    config = SFTConfig(
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
        data_seed=SEED,
    )
    train_dataset = create_sft_dataset(64, tokenizer, seed=SEED)
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    trainer.train()
    losses = step_losses(trainer)
    log(f"  losses: {[f'{x:.4f}' for x in losses]}")

    trainer.cleanup_ep()
    del trainer, model
    cleanup_memory()
    return losses


def run(ctx) -> dict:
    log(f"\n{'=' * 70}\n  FULL_SHARD vs SHARD_GRAD_OP equivalence (world_size={ctx.world_size})\n{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    modes = {
        "DP": {},
        "CP=2": {"cp_size": 2},
    }

    checks = {}
    for mode_name, kwargs in modes.items():
        off = run_sft(mode_name, kwargs, False, tokenizer, ctx.output_dir)
        barrier()
        on = run_sft(mode_name, kwargs, True, tokenizer, ctx.output_dir)
        barrier()

        both_finite = all(math.isfinite(x) for x in off + on)
        # Both modes must train, not just FULL_SHARD.
        both_decreased = len(on) >= 2 and on[-1] < on[0] and off[-1] < off[0]
        aligned = len(off) == len(on) and bool(off)
        step0_match = aligned and abs(off[0] - on[0]) <= STEP0_ABS_TOL
        mean_loss = (sum(off) / len(off)) if off else 0.0
        rel_dev = (
            (sum(abs(a - b) for a, b in zip(off, on, strict=True)) / len(off) / mean_loss)
            if aligned and mean_loss
            else float("inf")
        )
        curves_track = rel_dev <= CURVE_REL_TOL

        checks[f"{mode_name}_finite"] = both_finite
        checks[f"{mode_name}_decreased"] = both_decreased
        checks[f"{mode_name}_step0_identical"] = step0_match
        checks[f"{mode_name}_curves_track"] = curves_track
        log(
            f"\n  [{mode_name}] finite={both_finite} both_decreased={both_decreased} "
            f"step0_identical={step0_match} mean_rel_dev={rel_dev:.3%} (tol {CURVE_REL_TOL:.0%}) "
            f"track={curves_track}"
        )

    return {"checks": checks}


main = gpu_test_main(min_world_size=4, prefix="test_sft_fsdp_reshard")(run)

if __name__ == "__main__":
    main()
