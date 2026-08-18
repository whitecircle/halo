#!/usr/bin/env python
"""
FULL_SHARD (ZeRO-3) vs SHARD_GRAD_OP (ZeRO-2) equivalence across the EP=1 paths.

``fsdp_reshard_after_forward`` selects FSDP2's resharding mode: False (default) =
SHARD_GRAD_OP (params stay unsharded between forward and backward), True =
FULL_SHARD (params resharded after forward, re-gathered in backward — lower peak
memory). The two are mathematically identical; only memory/timing differ. So the
training loss curve must match between them on every supported (EP=1) path.

This runs a few SFT steps on Qwen3-0.6B in each parallelism mode with the flag OFF
then ON and checks, per mode:
  * both runs produce finite, decreasing loss;
  * the FULL_SHARD loss curve matches the SHARD_GRAD_OP curve within tolerance.

Modes (the EP=1 paths the config guard permits FULL_SHARD on). TP *with* DP is excluded:
FSDP2's backward re-gather of the TP-sharded DTensor params has no registered all-gather
strategy, so ParallelismConfig rejects TP+DP+FULL_SHARD (covered by
test_reshard_rejects_tp_with_dp in tests/cpu/parallelism/test_parallelism_config.py). Pure TP (dp=1) has
no DP axis to reshard, so there is nothing to equate against SHARD_GRAD_OP either.
  * pure DP   (FSDP shards over all ranks)
  * CP=2      (FSDP + Ulysses; the reshard all-gather interleaved with the CP all-to-all)

Needs 4 GPUs so CP=2 still leaves DP=2, where resharding actually engages
(at DP=1 FSDP is skipped and the flag is a no-op).

Run with 4 GPUs:
    torchrun --nproc_per_node=4 \
        tests/gpu/trainers/sft/test_sft_fsdp_reshard.py
"""

import math

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory, log

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 8
MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 2e-5
SEED = 42
# FULL_SHARD and SHARD_GRAD_OP are the same math but re-gather/reduce params in a different
# order, so over many bf16 optimizer steps the loss curves drift apart by rounding noise even
# though step 0 (a pure forward, no reduction yet) is bit-identical. We therefore check that the
# curves *track* — small deviation relative to the loss scale — not that they bit-match. A real
# sharding bug (dropped/doubled grad sync) diverges monotonically by >>this or NaNs.
STEP0_ABS_TOL = 1e-3  # the first forward must be identical (proves identical setup bar reshard mode)
CURVE_REL_TOL = 0.03  # mean |Δloss| / mean loss across the run


def run_sft(mode_name, parallelism_kwargs, reshard, tokenizer, output_dir):
    """Train a few SFT steps and return the per-step loss list."""
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
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
        data_seed=SEED,
        fsdp="",
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
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
    log(f"  losses: {[f'{x:.4f}' for x in losses]}")

    if hasattr(trainer, "cleanup_ep"):
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
        # both modes must train (decrease), not just the FULL_SHARD run
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
