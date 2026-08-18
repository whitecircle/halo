#!/usr/bin/env python
"""
SFT with node-local Expert Parallelism across SIMULATED multi-node (2 NVLink domains).

Real multi-node node-local EP cannot run on one box, so this SIMULATES two NVLink domains by
telling ``ParallelismConfig`` each domain is ``world_size // 2`` GPUs (``gpus_per_node =
world_size // 2``) and setting ``ep_size = gpus_per_node`` — one node-local EP group per domain.
That is the canonical production multi-node MoE shape: experts distributed node-locally (NVLink
all-to-all), DP across nodes, and expert gradients averaged across nodes via the
cross-domain expert-replica groups. It exercises the membership math in
``src.distributed.group_layout`` (``node_local_group_ranks`` / ``node_local_replica_ranks``) and the
``num_ep_groups > 1`` replica-grad-sync path — the parts that only run when there is more than one
EP group, i.e. only in multi-node.

Validates, on GptOss-20B (32 experts):
1. Topology: 2 node-local EP groups, ``needs_expert_grad_sync`` True (multi-group).
2. The cross-domain replica grad-sync actually runs and converges: expert weights are bit-identical
   across the two domains' replica ranks after training (a malformed replica/dispatch group from a
   rank-math bug would diverge or hang instead).
3. The sweep lands the DP average its contract names — replica ``SUM`` divided by
   ``world_size / expert_tp_size``. Bit-identity above cannot see a wrong divisor: it leaves every
   replica equally wrong, the loss finite and the curve decreasing.
4. Training is healthy: loss finite and decreasing.

Run with 4 or 8 GPUs (simulates 2 domains of world//2):
    torchrun --nproc_per_node=4 \
        tests/gpu/trainers/sft/test_sft_ep_multinode_sim.py
"""

import math
import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.base_layer import EPMoELayerBase, has_grouped_mm
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
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
MAX_SEQ_LENGTH = 2048
NUM_TRAIN_STEPS = 6
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
SEED = 42
# The sweep reduces through bucketed fp32-optional kernels while the probe re-reduces the whole
# tensor in one call, so only the reduction ORDER differs; a wrong divisor is a factor of 2.
SWEEP_DIVISOR_REL_TOL = 2e-2


def _install_sweep_divisor_probe(trainer, ep_config, verdict: dict) -> None:
    """Verify the deferred sweep lands expert grads at the DP average its contract names.

    Replica bit-identity (below) cannot see a WRONG DIVISOR: a sweep dividing by ``ep_group_size``
    instead of ``world_size``, or skipping the replica SUM entirely, leaves every replica equally
    wrong, the loss finite and the curve decreasing. This is the only check here that reads the
    magnitude.

    Recomputed from the CONTRACT rather than the implementation (agent-docs/parallelism/multi-node.md →
    Deferred cross-replica sync): the combine has already summed the dispatch group, so the sweep
    owes ``all_reduce(SUM)`` over the expert-replica group divided by ``world_size /
    expert_tp_size``. Wrapping the sweep is what makes the pre-image observable — after ``train()``
    the grads are gone, and the sweep is not idempotent so it cannot simply be re-run.

    Rank-uniform: the wrapper runs on every rank and its own all-reduce is over a group every rank
    is in, so the collective sequence stays identical across the world.
    """
    original = trainer._sync_deferred_expert_grads
    expert_ids = trainer._get_sharded_expert_param_ids()
    divisor = ep_config.world_size // ep_config.expert_tp_size

    def probed() -> None:
        if "sweep_divisor" in verdict:  # first optimizer step only; later ones add nothing
            original()
            return
        named = [(n, p) for n, p in trainer._top_level_model().named_parameters() if id(p) in expert_ids]
        before = [(n, p, p.grad.detach().clone()) for n, p in named if p.grad is not None]
        original()
        worst, scale = 0.0, 0.0
        for name, param, pre in before:
            expected = pre.float()
            if ep_config.expert_replica_group is not None:
                dist.all_reduce(expected, op=dist.ReduceOp.SUM, group=ep_config.expert_replica_group)
            expected /= divisor
            actual = param.grad.float()
            worst = max(worst, (actual - expected).abs().max().item())
            scale = max(scale, expected.abs().max().item())
        # Anti-vacuity: a zero pre-image would make any divisor look right.
        verdict["sweep_pre_image_nonzero"] = scale > 0.0
        verdict["sweep_divisor"] = scale > 0.0 and worst <= SWEEP_DIVISOR_REL_TOL * scale
        log(f"  Sweep divisor (/{divisor}): worst|Δ|={worst:.3e} vs scale={scale:.3e}")

    trainer._sync_deferred_expert_grads = probed


def _check_replica_consistency(model, ep_config) -> bool:
    """Expert weights must be bit-identical across the cross-domain expert-replica group.

    The replica group all-reduces expert gradients across domains every step; with matching init
    and optimizer, the expert weights stay identical across replicas. A wrong replica-group layout
    would let them diverge.

    Each precondition below is itself a failure this test exists to catch — a missing replica group,
    absent EP layers or unreachable expert params all mean the multi-group path never ran — so none
    of them may pass by default.
    """
    replica_group = getattr(ep_config, "expert_replica_group", None)
    if replica_group is None or dist.get_world_size(replica_group) <= 1:
        log(f"  Replica consistency: FAIL (no cross-domain replica group; got {replica_group!r})")
        return False

    ep_layers = [m for m in model.modules() if isinstance(m, EPMoELayerBase)]
    if not ep_layers:
        log("  Replica consistency: FAIL (no EP layers found — expert parallelism did not engage)")
        return False

    # Probe the first expert parameter of the first EP layer.
    named = list(ep_layers[0].expert_named_params())
    if not named:
        log("  Replica consistency: FAIL (EP layer exposes no expert params)")
        return False
    _, weight = named[0]
    local = weight.detach().contiguous()
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(replica_group))]
    dist.all_gather(gathered, local, group=replica_group)
    max_diff = max((g - gathered[0]).abs().max().item() for g in gathered)
    ok = max_diff == 0.0
    log(f"  Replica consistency: {'PASS' if ok else 'FAIL'} (max|Δ| across replicas = {max_diff:.2e})")
    return ok


def main() -> int:
    rank, world_size, local_rank = init_distributed()
    PartialState()

    if world_size < 4 or world_size % 2 != 0:
        log(f"Need an even world size >= 4 to simulate 2 domains; got {world_size}. Skipping.")
        teardown_distributed()
        return 0

    simulated_gpus_per_node = world_size // 2  # ep_size per domain

    log(f"\n{'#' * 70}")
    log("  SFT node-local EP across SIMULATED 2 NVLink domains")
    log(f"  World: {world_size}, simulated domain size: {simulated_gpus_per_node}")
    log(f"  → ep_size={simulated_gpus_per_node}, expect 2 node-local EP groups + cross-domain replica sync")
    log(f"{'#' * 70}")

    output_dir, cache_dir = setup_cache_dirs("sft_ep_multinode_sim", rank)
    success = False

    try:
        ensure_model_downloaded(MODEL_NAME, rank)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
        eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)

        # Node-local multi-group EP across the two simulated domains: FSDP shards the non-expert
        # params over each EP group, and the cross-domain replica average is DEFERRED to a
        # post-backward sweep (is_deferred_dp) — this is the production multi-node MoE shape. HSDP is
        # deliberately NOT used: for multi-group EP it is a no-op (the deferred sync already shards
        # within-domain + replicates cross-domain) and ParallelismConfig rejects hsdp+EP.
        parallelism_config = ParallelismConfig(
            ep_size=simulated_gpus_per_node,
            gpus_per_node=simulated_gpus_per_node,
            ep_scope="node",
            use_grouped_gemm=has_grouped_mm(),
        )
        log(f"Config: {parallelism_config.summary()}")
        # The whole point — confirm we got the multi-group (multi-node) deferred-DP topology.
        assert parallelism_config.num_ep_groups == 2, (
            f"expected 2 node-local EP groups, got {parallelism_config.num_ep_groups}"
        )
        assert parallelism_config.create_ep_config().is_deferred_dp, (
            "expected deferred cross-replica DP sync for multi-group node-local EP across 2 domains"
        )

        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flex_attention",
            use_liger_kernel=True,
        )

        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
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
            ddp_find_unused_parameters=True,
            fsdp="",
        )
        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        ep_config = parallelism_config.create_ep_config()
        log(f"EP groups={ep_config.num_ep_groups}, needs_expert_grad_sync={ep_config.needs_expert_grad_sync}")

        sweep_verdict: dict[str, bool] = {}
        _install_sweep_divisor_probe(trainer, ep_config, sweep_verdict)

        train_result = trainer.train()
        step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
        log(f"\nFinal loss: {train_result.training_loss:.6f}")
        log(f"Step losses: {[f'{l:.4f}' for l in step_losses]}")

        # Non-expert params are FSDP-sharded over the EP group (1D mesh of ep_group_size), NOT a 2D
        # HSDP mesh — the deferred sweep handles the cross-domain replica average. Confirm the
        # sharding mesh is the expected 1D EP-group shape.
        ep_group_size = parallelism_config.ep_group_size
        sharded_over_ep_group = any(
            isinstance(p.data, DTensor) and p.data.device_mesh.ndim == 1 and p.data.device_mesh.size() == ep_group_size
            for p in model.parameters()
        )
        log(
            f"  Non-expert params sharded over the EP group (1D mesh={ep_group_size}): "
            f"{'PASS' if sharded_over_ep_group else 'FAIL'}"
        )

        checks = {
            "two_ep_groups": ep_config.num_ep_groups == 2,
            "needs_expert_grad_sync": bool(ep_config.needs_expert_grad_sync),
            "is_deferred_dp": ep_config.is_deferred_dp,
            "sharded_over_ep_group": sharded_over_ep_group,
            "loss_finite": math.isfinite(train_result.training_loss),
            "loss_decreased": len(step_losses) >= 2 and step_losses[-1] < step_losses[0],
            "replica_consistency": _check_replica_consistency(model, ep_config),
            # Absent keys mean the probe never ran — the sweep did not fire, which is itself the
            # failure this file exists to catch, so they must not default to True.
            "sweep_pre_image_nonzero": sweep_verdict.get("sweep_pre_image_nonzero", False),
            "sweep_divisor": sweep_verdict.get("sweep_divisor", False),
        }
        for k, v in checks.items():
            log(f"  {k}: {'PASS' if v else 'FAIL'}")

        success = all(checks.values())
        log(f"\n{'#' * 70}")
        log("  EP MULTI-NODE-SIM TEST PASSED" if success else f"  FAILED: {[k for k, v in checks.items() if not v]}")
        log(f"{'#' * 70}")

    except Exception as e:
        log(f"\n  EP MULTI-NODE-SIM TEST FAILED with exception: {e}")
        traceback.print_exc()
        success = False
    finally:
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)
        teardown_distributed()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
