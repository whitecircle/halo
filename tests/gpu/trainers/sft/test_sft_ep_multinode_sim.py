#!/usr/bin/env python
"""SFT with node-local EP on two simulated NVLink domains: the ``num_ep_groups > 1`` replica-grad-sync path.

Replica bit-identity can't see a wrong sweep divisor, so it is probed separately. ``expert_lora_resume``: only each
replica group's lowest rank writes adapter moments, and the load draws lora_A per rank, so every rank is seeded first.
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer, set_seed
from trl import SFTConfig

import halo.checkpoint.parallel.optimizer as optimizer_store
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.distributed.parallelism_config import ParallelismConfig
from halo.hardware import supports_grouped_mm
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import check_only_adapters_trainable, load_peft_model, unwrap
from tests.common.checkpoint_io import RestorePointSnapshot
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    ensure_model_downloaded,
    group_max_abs_diff,
    pin_deterministic_ep_dispatch,
    shared_output_dir,
    world_all,
)
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import GPT_OSS_20B, QWEN3_0_6B
from tests.common.numerics import max_or_nan
from tests.common.optimizer_state import optimizer_state_matches
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log, logged_messages
from tests.common.resume_export import restore_log_checks
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint
from tests.common.tolerances import TOL
from tests.common.training_checks import resumed_loss_deltas, step_losses, training_run_checks

MODEL_NAME = GPT_OSS_20B
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 2048
NUM_TRAIN_STEPS = 6
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
SEED = 42
# The sweep reduces through bucketed kernels and the probe in one call, so only the order differs; a wrong divisor is
# 2x.
SWEEP_DIVISOR_REL_TOL = 2e-2
LORA_FAMILY = TINY_MOE_FAMILIES["qwen3_moe"]
LORA_TOTAL_STEPS = 6
LORA_SAVE_STEP = 3
LORA_BATCH_SIZE = 2
LORA_MAX_SEQ_LENGTH = 256
LORA_LEARNING_RATE = 2e-3
LORA_NUM_TRAIN_SAMPLES = 64

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("full_ft", "expert_lora_resume"), required=True)
ARGS = parser.parse_args()


def _simulated_two_domain_config(world_size: int) -> ParallelismConfig:
    """One EP group per simulated domain, the cross-domain average deferred to the post-backward sweep. Fresh per
    phase: ``ep_config`` caches the groups it builds on the config."""
    domain = world_size // 2
    return ParallelismConfig(
        ep_size=domain,
        gpus_per_node=domain,
        ep_scope="node",
        use_grouped_gemm=supports_grouped_mm(),
    )


def _install_sweep_divisor_probe(trainer, ep_config, verdict: dict) -> None:
    """The sweep lands expert grads at the DP average (agent-docs/parallelism/multi-node.md): replica SUM over
    ``world_size / expert_tp_size``. Wrapping exposes the pre-image, since the sweep isn't idempotent and ``train()``
    drops the grads."""
    original = trainer._run_deferred_grad_sweep
    expert_ids = trainer._get_sharded_expert_param_ids()
    divisor = ep_config.stage_world_size // ep_config.expert_tp_size

    def probed() -> None:
        if "sweep_divisor" in verdict:  # First optimizer step only.
            original()
            return
        named = [(n, p) for n, p in trainer._top_level_model().named_parameters() if id(p) in expert_ids]
        before = [(n, p, p.grad.detach().clone()) for n, p in named if p.grad is not None]
        original()
        diffs, scale = [], 0.0
        for name, param, pre in before:
            expected = pre.float()
            if ep_config.expert_replica_group is not None:
                dist.all_reduce(expected, op=dist.ReduceOp.SUM, group=ep_config.expert_replica_group)
            expected /= divisor
            actual = param.grad.float()
            diffs.append((actual - expected).abs().max().item())
            scale = max(scale, expected.abs().max().item())
        worst = max_or_nan(diffs, default=0.0)
        # A zero pre-image would make any divisor look right.
        verdict["sweep_pre_image_nonzero"] = scale > 0.0
        verdict["sweep_divisor"] = scale > 0.0 and worst <= SWEEP_DIVISOR_REL_TOL * scale
        log(f"  Sweep divisor (/{divisor}): worst|Δ|={worst:.3e} vs scale={scale:.3e}")

    trainer._run_deferred_grad_sweep = probed


def _check_replica_consistency(model, ep_config) -> bool:
    """Expert weights are bit-identical across the expert-replica group. A missing group, absent EP layers or
    unreachable expert params all mean the multi-group path never ran, so none may pass by default."""
    replica_group = getattr(ep_config, "expert_replica_group", None)
    if replica_group is None or dist.get_world_size(replica_group) <= 1:
        log(f"  Replica consistency: FAIL (no cross-domain replica group; got {replica_group!r})")
        return False

    layers = ep_layers(model)
    if not layers:
        log("  Replica consistency: FAIL (no EP layers found — expert parallelism did not engage)")
        return False

    named = list(layers[0].expert_named_params())
    if not named:
        log("  Replica consistency: FAIL (EP layer exposes no expert params)")
        return False
    _, weight = named[0]
    max_diff = group_max_abs_diff(weight, replica_group)
    ok = max_diff == 0.0
    log(f"  Replica consistency: {'PASS' if ok else 'FAIL'} (max|Δ| across replicas = {max_diff:.2e})")
    return ok


def _replica_adapter_max_diff(model, replica_group) -> float:
    """COLLECTIVE over the group. Largest ``|Δ|`` of a trainable EP-layer param; NaN when nothing on the layer
    trains."""
    params = [param for layer in ep_layers(model) for param in layer.parameters() if param.requires_grad]
    return max_or_nan((group_max_abs_diff(param, replica_group) for param in params), default=float("nan"))


def _expert_lora_sft_config(output_dir: str, *, save: bool) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_steps=LORA_TOTAL_STEPS,
        per_device_train_batch_size=LORA_BATCH_SIZE,
        learning_rate=LORA_LEARNING_RATE,
        lr_scheduler_type="constant",
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=1,
        save_strategy="steps" if save else "no",
        save_steps=LORA_SAVE_STEP,
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=LORA_MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        use_liger_kernel=False,
        seed=SEED,
    )


def _run_expert_lora_resume(ctx) -> dict:
    """Expert-only LoRA saved midway and resumed: each rank's optimizer state, the replicas' adapters and the
    losses."""
    # A replay compares two runs, which DeepEP's atomic receive order would part by rounding alone.
    pin_deterministic_ep_dispatch()
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    train_out = os.path.join(shared_output_dir(ctx), "train_out")
    checkpoint = os.path.join(train_out, f"checkpoint-{LORA_SAVE_STEP}")
    base_dir = shared_tiny_family_checkpoint(
        ctx, LORA_FAMILY, "sft_ep_multinode_sim_expert_lora_base", AutoTokenizer.from_pretrained(QWEN3_0_6B), SEED
    )

    def make_trainer(*, save: bool) -> tuple[DistributedSFTTrainer, ParallelismConfig]:
        # The load draws lora_A from each rank's RNG, so replicas start bit-identical only when every rank is seeded
        # first.
        set_seed(SEED)
        config = _simulated_two_domain_config(ctx.world_size)
        model, tokenizer, peft_config = load_peft_model(
            "expert_lora", config, model_name=base_dir, attn_implementation="eager", use_liger_kernel=False
        )
        trainer = DistributedSFTTrainer(
            model=model,
            args=_expert_lora_sft_config(train_out, save=save),
            train_dataset=create_sft_dataset(LORA_NUM_TRAIN_SAMPLES, tokenizer, seed=SEED),
            processing_class=tokenizer,
            parallelism_config=config,
            peft_config=peft_config,
        )
        ctx.on_teardown(trainer.cleanup_ep)
        return trainer, config

    log(f"\n[1/2] Unbroken {LORA_TOTAL_STEPS}-step expert-LoRA run, checkpoint at step {LORA_SAVE_STEP}...")
    trainer, config = make_trainer(save=True)
    ep_config = config.ep_config()
    checks["two_ep_groups_with_deferred_replica_sync"] = config.num_ep_groups == 2 and ep_config.is_deferred_dp
    checks["expert_lora_active"] = model_has_expert_lora(unwrap(trainer.model))
    adapters_only, detail = check_only_adapters_trainable(unwrap(trainer.model))
    log(f"  {detail}")
    checks["only_adapters_trainable"] = adapters_only
    build_diff = _replica_adapter_max_diff(trainer.model, ep_config.expert_replica_group)
    log(f"  replica adapters at build: max|Δ| = {build_diff:.3e}")
    checks["replica_adapters_bit_identical_at_build"] = world_all(build_diff == 0.0, ctx.device)
    saved = RestorePointSnapshot("save", trainer, capture_optimizer=True)
    trainer.add_callback(saved)
    trainer.train()
    unbroken_losses = step_losses(trainer)
    finish_phase(trainer)
    saved.trainer = None
    ctx.barrier()

    # The premise of this row: the dedup left the followers' shards no state.
    shard = torch.load(
        os.path.join(checkpoint, f"optimizer_shard_{ctx.rank:05d}.pt"), map_location="cpu", weights_only=False
    )
    writer = ctx.rank == min(ep_config.expert_replica_ranks)
    log(f"  rank {ctx.rank} ({'writer' if writer else 'follower'}): shard holds {len(shard['state'])} state entries")
    checks["only_replica_writers_shards_hold_state"] = world_all(bool(shard["state"]) == writer, ctx.device)

    log(f"\n[2/2] Resuming from {checkpoint}...")
    resumed, config = make_trainer(save=False)
    restored = RestorePointSnapshot("train_begin", resumed, capture_optimizer=True)
    resumed.add_callback(restored)
    with logged_messages(optimizer_store.logger) as messages:
        resumed.train(resume_from_checkpoint=checkpoint)
    resumed_losses = step_losses(resumed)
    replica_diff = _replica_adapter_max_diff(resumed.model, config.ep_config().expert_replica_group)
    finish_phase(resumed)
    restored.trainer = None

    saved_state, restored_state = saved.captured or {}, restored.captured or {}
    exact, why = optimizer_state_matches(
        saved_state.get("optimizer") or {"state": {}}, restored_state.get("optimizer") or {"state": {}}
    )
    if not exact:
        log(f"  rank {ctx.rank} optimizer restore: {why}")
    checks["optimizer_restored_bit_exact_on_every_rank"] = world_all(exact, ctx.device)
    checks["step_restored"] = restored_state.get("global_step") == LORA_SAVE_STEP
    log(f"  replica adapters after the resumed steps: max|Δ| = {replica_diff:.3e}")
    checks["replica_adapters_bit_identical_after_resume"] = world_all(replica_diff == 0.0, ctx.device)
    deltas = resumed_loss_deltas(
        unbroken_losses, resumed_losses, save_step=LORA_SAVE_STEP, total_steps=LORA_TOTAL_STEPS
    )
    checks["resumed_ran_remaining_steps"] = deltas is not None
    if deltas is not None:
        metrics["resume_loss_max_delta"] = max_or_nan(deltas)
        checks["resumed_losses_replay_unbroken"] = metrics["resume_loss_max_delta"] <= TOL.replayed_resume_loss_abs
    checks |= ctx.broadcast_checks(restore_log_checks(messages, checkpoint) if ctx.rank == 0 else {})
    return {"checks": checks, "metrics": metrics}


def _run_full_ft(ctx) -> dict:
    simulated_gpus_per_node = ctx.world_size // 2

    log(f"\n{'#' * 70}")
    log("  SFT node-local EP across SIMULATED 2 NVLink domains")
    log(f"  World: {ctx.world_size}, simulated domain size: {simulated_gpus_per_node}")
    log(f"  → ep_size={simulated_gpus_per_node}, expect 2 node-local EP groups + cross-domain replica sync")
    log(f"{'#' * 70}")

    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)

    # HSDP is deliberately unused: for multi-group EP it is a no-op, since the deferred sync already shards
    # within-domain and replicates cross-domain, and ParallelismConfig rejects hsdp+EP.
    parallelism_config = _simulated_two_domain_config(ctx.world_size)
    log(f"Config: {parallelism_config.summary()}")
    assert parallelism_config.num_ep_groups == 2, (
        f"expected 2 node-local EP groups, got {parallelism_config.num_ep_groups}"
    )
    assert parallelism_config.ep_config().is_deferred_dp, (
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
        output_dir=ctx.output_dir,
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
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )

    ep_config = parallelism_config.ep_config()
    log(f"EP groups={ep_config.num_ep_groups}, needs_expert_grad_sync={ep_config.needs_expert_grad_sync}")
    shape_checks = parallel_shape_checks(model, parallelism_config)

    sweep_verdict: dict[str, bool] = {}
    _install_sweep_divisor_probe(trainer, ep_config, sweep_verdict)

    train_result = trainer.train()

    # Non-expert params are FSDP-sharded over the EP group (1D mesh of ep_group_size), with no 2D HSDP mesh.
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
        **shape_checks,
        **training_run_checks(train_result, trainer, NUM_TRAIN_STEPS, grad_norms=True, loss_decreased=True),
        "sharded_over_ep_group": sharded_over_ep_group,
        "replica_consistency": _check_replica_consistency(model, ep_config),
        # An absent key means the sweep did not fire, the failure this file exists to catch, so none defaults to True.
        "sweep_pre_image_nonzero": sweep_verdict.get("sweep_pre_image_nonzero", False),
        "sweep_divisor": sweep_verdict.get("sweep_divisor", False),
    }
    return {"checks": checks}


@gpu_test_main(min_world_size=4, prefix=f"sft_ep_multinode_sim_{ARGS.mode}")
def run(ctx):
    if ctx.world_size % 2 != 0:
        raise ValueError(f"Need an even world size to simulate 2 domains; got {ctx.world_size}")
    return _run_full_ft(ctx) if ARGS.mode == "full_ft" else _run_expert_lora_resume(ctx)


if __name__ == "__main__":
    run()
