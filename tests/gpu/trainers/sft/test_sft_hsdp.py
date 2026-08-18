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
3. Training is healthy: loss is finite and decreases (exercises the 2D-mesh grad-norm path
   in DistributedTrainerMixin._fsdp_shard_group).

Run with 4 GPUs (simulates 2 domains of 2):
    torchrun --nproc_per_node=4 \
        tests/gpu/trainers/sft/test_sft_hsdp.py
"""

import math
import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.utils import log

MODEL_NAME = QWEN3_0_6B
MAX_STEPS = 10
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-5
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
SEED = 42


def _hsdp_params(model):
    """All (name, DTensor) params on a 2D HSDP mesh, in a deterministic order.

    ``named_parameters`` yields the same order on every rank, so iterating this
    list keeps the per-param all_gather collectives in lockstep across ranks.
    """
    return [
        (name, param.data)
        for name, param in model.named_parameters()
        if isinstance(param.data, DTensor) and param.data.device_mesh.ndim == 2
    ]


def _check_hsdp_placements(p) -> bool:
    """The param must be replicated on dim 0 (dp_replicate) and sharded on dim 1 (dp_shard)."""
    names = tuple(p.device_mesh.mesh_dim_names or ())
    placements = tuple(p.placements)
    ok = names == ("dp_replicate", "dp_shard")
    ok = ok and isinstance(placements[0], Replicate) and isinstance(placements[1], Shard)
    log(f"  HSDP mesh dims={names}, placements={placements}: {'PASS' if ok else 'FAIL'}")
    return ok


def _check_replica_consistency(p) -> tuple[bool, float]:
    """Each rank's local shard must be identical across the dp_replicate group.

    Returns ``(ok, max_diff)`` so the caller can aggregate over many params without
    one log line per param.
    """
    replicate_group = p.device_mesh["dp_replicate"].get_group()
    replicate_size = dist.get_world_size(replicate_group)
    local = p.to_local().detach().contiguous()
    gathered = [torch.empty_like(local) for _ in range(replicate_size)]
    dist.all_gather(gathered, local, group=replicate_group)
    max_diff = max((g - gathered[0]).abs().max().item() for g in gathered)
    return max_diff == 0.0, max_diff


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
    parallelism_config = ParallelismConfig(gpus_per_node=simulated_gpus_per_node, use_hsdp=True)
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
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=5,
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        fsdp="",
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )

    log("\n[4/4] Training...")
    train_result = trainer.train()

    training_loss = train_result.training_loss
    log_history = trainer.state.log_history
    step_losses = [e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e]

    log("\n  --- Results ---")
    log(f"  Final training loss: {training_loss:.6f}")
    log(f"  Per-step losses: {[f'{l:.4f}' for l in step_losses]}")

    checks = {}

    # HSDP structural checks
    hsdp_params = _hsdp_params(model)
    if not hsdp_params:
        checks["hsdp_mesh_present"] = False
        log("  HSDP 2D-mesh params found: FAIL (no 2D-mesh DTensor — HSDP did not engage)")
    else:
        checks["hsdp_mesh_present"] = True
        log(f"  HSDP 2D-mesh params found: PASS ({len(hsdp_params)} params)")
        # Placements: assert on the first (the mesh/placements are identical for
        # all 2D params); logging every one would be noise.
        checks["hsdp_placements"] = _check_hsdp_placements(hsdp_params[0][1])

        # Replica consistency over EVERY 2D-mesh param (not just the first): each
        # rank's local shard must be bit-identical across the dp_replicate group.
        # All ranks iterate the same param set in the same order, so the per-param
        # all_gather collectives stay in lockstep.
        replica_ok = True
        worst_name, worst_diff = None, 0.0
        for name, p in hsdp_params:
            ok, max_diff = _check_replica_consistency(p)
            replica_ok = replica_ok and ok
            if max_diff > worst_diff:
                worst_name, worst_diff = name, max_diff
        checks["replica_consistency"] = replica_ok
        log(
            f"  Replica consistency over all {len(hsdp_params)} params: "
            f"{'PASS' if replica_ok else 'FAIL'} (worst: {worst_name} max|Δ|={worst_diff:.2e})"
        )

    # Training health
    checks["loss_finite"] = math.isfinite(training_loss)
    checks["all_steps_finite"] = all(math.isfinite(l) for l in step_losses)
    if len(step_losses) >= 2:
        checks["loss_decreased"] = step_losses[-1] < step_losses[0]
        log(
            f"  Loss decreased: {'PASS' if checks['loss_decreased'] else 'FAIL'} "
            f"({step_losses[0]:.4f} -> {step_losses[-1]:.4f})"
        )
    else:
        checks["loss_decreased"] = False

    return {"checks": checks}


main = gpu_test_main(min_world_size=4, prefix="test_sft_hsdp")(run)

if __name__ == "__main__":
    main()
