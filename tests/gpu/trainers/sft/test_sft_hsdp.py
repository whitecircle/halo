#!/usr/bin/env python
"""
SFT Trainer test with HSDP (Hybrid Sharded Data Parallel).

Multi-node HSDP cannot be exercised on a single node directly, so this test SIMULATES two
NVLink domains on one box by telling ``ParallelismConfig`` that each domain is
``world_size // 2`` GPUs (``gpus_per_node = world_size // 2``). With ``use_hsdp=True`` that yields
a 2D ``(dp_replicate=2, dp_shard=world_size//2)`` FSDP mesh — the exact code path a real
multi-node job takes, only with the cross-domain replicate all-reduce running over NVLink
instead of RDMA. The math (shard placement, replica grad all-reduce, the 2D-mesh grad-norm
reduction) is identical.

Validates, on Qwen3-0.6B:
1. HSDP engages: non-expert params are DTensors on a 2D ``(dp_replicate, dp_shard)`` mesh with
   ``(Replicate, Shard)`` placements.
2. Replicas stay in sync: EVERY 2D-mesh DTensor param's local shard is bit-identical across
   the dp_replicate group after training (proves init replication + cross-domain gradient
   all-reduce). Checked over all such params, not just the first.
3. Training is healthy: losses and grad norms finite, the loss falling (exercises the 2D-mesh
   grad-norm path in DistributedTrainerMixin._fsdp_shard_group).

``--lora`` trains LoRA adapters on the attention projections instead, built here with each rank drawing its
adapters from its own RNG stream (the state a draw on some ranks only, before the build, leaves the streams in),
and checked apart across the ranks before the trainer wraps them: the replicas must start from the first
replica's adapters and keep them, and every adapter must move. ``--cp-size 2`` adds context parallelism inside
each simulated domain, whose wrap shards over the CP ranks too.

Run with 4 GPUs (simulates 2 domains of 2):
    torchrun --nproc_per_node=4 \
        tests/gpu/trainers/sft/test_sft_hsdp.py [--lora] [--cp-size 2]
"""

import argparse
import sys

import torch
from peft import LoraConfig, get_peft_model
from trl import SFTConfig

from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import assert_adapters_moved, snapshot_adapters
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import max_abs_diff_across
from tests.common.dp_layout import hsdp_checks
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.training_checks import training_run_checks

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 10
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-5
LORA_LEARNING_RATE = 2e-4
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--lora", action="store_true", help="LoRA adapters drawn per rank instead of a full fine-tune")
parser.add_argument("--cp-size", type=int, default=1, help="context parallelism inside each simulated domain")
ARGS = parser.parse_args()


def run(ctx) -> dict:
    """Run the SFT HSDP test on a simulated 2-domain topology."""
    if ctx.world_size % 2 != 0:
        # The SKIP: sentinel, not a bare exit 0 — the launcher counts an unmarked exit-0 as a PASS.
        log(f"SKIP: HSDP test needs an even world size >= 4 to simulate 2 domains; got {ctx.world_size}.")
        sys.exit(0)

    # Simulate two NVLink domains on one node: each "domain" is half the GPUs.
    simulated_gpus_per_node = ctx.world_size // 2

    log(f"\n{'=' * 70}")
    log("  SFT HSDP Training Test (simulated 2 NVLink domains)")
    log(f"  World size: {ctx.world_size}, simulated domain size: {simulated_gpus_per_node}")
    log(f"  → HSDP mesh: (dp_replicate=2, dp_shard={simulated_gpus_per_node})")
    log(f"  Model: {MODEL_NAME}, GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"{'=' * 70}")

    log("\n[1/4] Loading model with use_hsdp=True...")
    parallelism_config = ParallelismConfig(gpus_per_node=simulated_gpus_per_node, use_hsdp=True, cp_size=ARGS.cp_size)
    # The whole point of the test — confirm the topology resolved to real HSDP.
    assert parallelism_config.is_hsdp, "use_hsdp=True did not resolve to HSDP (check domain simulation)"
    assert parallelism_config.dp_replicate_size == 2
    assert parallelism_config.dp_shard_size == simulated_gpus_per_node

    model, tokenizer = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    log("\n[2/4] Creating synthetic datasets...")
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 1)

    log("\n[3/4] Configuring trainer...")
    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LORA_LEARNING_RATE if ARGS.lora else LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=5,
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
    )
    drawn_apart = None
    if ARGS.lora:
        torch.manual_seed(SEED + ctx.rank)
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=LORA_TARGETS))
        drawn_apart = max_abs_diff_across(param.data for name, param in model.named_parameters() if "lora_A" in name)
        log(f"  LoRA adapters drawn per rank, max|Δ| across ranks before the wrap: {drawn_apart:.2e}")
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    model = trainer.model
    adapters_before = snapshot_adapters(model, expert_lora=False) if ARGS.lora else {}

    log("\n[4/4] Training...")
    train_result = trainer.train()
    checks = training_run_checks(train_result, trainer, MAX_STEPS, grad_norms=True, loss_decreased=True)
    if ARGS.lora:
        reshard_fsdp2_modules(model)
        checks["premise_lora_adapters_trained"] = bool(adapters_before)
        checks["premise_adapters_apart_before_the_wrap"] = drawn_apart > 0
        moved, why = assert_adapters_moved(
            adapters_before, snapshot_adapters(model, expert_lora=False), require_all=True
        )
        checks["every_adapter_moved"] = moved
        log(f"  Every adapter moved: {'PASS' if moved else 'FAIL'} ({why})")

    checks |= hsdp_checks(model)
    return {"checks": checks}


main = gpu_test_main(
    min_world_size=4,
    prefix=f"test_sft_hsdp{'_lora' if ARGS.lora else ''}{f'_cp{ARGS.cp_size}' if ARGS.cp_size > 1 else ''}",
)(run)

if __name__ == "__main__":
    main()
