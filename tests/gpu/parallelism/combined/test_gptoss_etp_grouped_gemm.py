#!/usr/bin/env python
"""GptOss at ep4 x etp2 trains on grouped GEMM as it trained on the per-expert loop, and resumes a loop-path
ETP checkpoint exactly (8 GPUs).

The loop arm runs the per-expert loop under ``use_grouped_gemm: true`` (the loop rule patched onto the layer
class, the config unchanged), the configuration a loop-path ETP checkpoint carries, fingerprint included:

1. Loop, 20 steps: the reference trajectory, every EP layer on the loop.
2. Loop, 10 steps and a checkpoint, with this rank's optimizer state and trainable weights snapshotted.
3. Grouped GEMM, 20 steps from the same init: every EP layer on grouped GEMM, each step's loss and gradient
   norm on the loop's trajectory.
4. Grouped GEMM resuming step 2's checkpoint: optimizer state and weights restored exactly, no fingerprint
   warm restart, steps 11-20 on the loop's trajectory.

DeepEP dispatch is pinned deterministic, so the arms differ in the expert compute alone. A tiny random-init
GptOss (8 experts: two per dispatch rank, each FFN sharded two ways) keeps four runs cheap.

    torchrun --nproc_per_node=8 tests/gpu/parallelism/combined/test_gptoss_etp_grouped_gemm.py
"""

from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import SFTConfig

import halo.checkpoint.parallel.optimizer as optimizer_store_mod
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import pin_deterministic_ep_dispatch, world_all
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import GPT_OSS_20B
from tests.common.numerics import max_or_nan
from tests.common.optimizer_state import local_optimizer_state, optimizer_state_matches
from tests.common.rank_log import log, logged_messages
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint
from tests.common.tolerances import TOL
from tests.common.training_checks import snapshot_trainable, step_losses

EP_SIZE, EXPERT_TP_SIZE = 4, 2
WORLD_SIZE = EP_SIZE * EXPERT_TP_SIZE
SEED = 42
TOTAL_STEPS = 20
SAVE_AT_STEP = 10
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
MAX_SEQ_LENGTH = 256
NUM_SAMPLES = 256
# The step-1 forward differs by reduction order and the loop's bf16 index_add_ rounding only; later steps also
# carry the optimizer's response to those differences.
FIRST_STEP_LOSS_TOL = TOL.parallel_vs_baseline_loss_abs
TRAJECTORY_LOSS_TOL = TOL.parallel_vs_baseline_train_loss_abs


def _loop_rule(self) -> bool:
    """The per-expert loop at ``expert_tp_size > 1`` whatever the config asks."""
    return self._use_grouped_mm and self.expert_tp_size <= 1


def _per_expert_loop():
    return patch.object(EPGptOssMoELayer, "_grouped_mm_enabled", _loop_rule)


def _sft_config(output_dir: str, max_steps: int, save_at: int | None) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",  # phases with different max_steps stay comparable
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps" if save_at else "no",
        save_steps=save_at or 0,
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
    )


def _make_trainer(model_path, tokenizer, dataset, config, **load_kwargs) -> DistributedSFTTrainer:
    pc = ParallelismConfig(ep_size=EP_SIZE, expert_tp_size=EXPERT_TP_SIZE)
    model, _ = load_distributed_model(
        model_name_or_path=model_path,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        use_liger_kernel=False,
        **load_kwargs,
    )
    return DistributedSFTTrainer(
        model=model, args=config, train_dataset=dataset, processing_class=tokenizer, parallelism_config=pc
    )


def _layer_checks(model, *, grouped: bool, label: str) -> dict[str, bool]:
    """Every EP layer is GptOss's, split ep4 x etp2, and runs the compute path the phase is named for."""
    layers = [layer for _, layer in named_ep_layers(model)]
    return {
        f"{label}_layers_are_gptoss_etp": bool(layers)
        and all(
            isinstance(layer, EPGptOssMoELayer)
            and layer.expert_tp_size == EXPERT_TP_SIZE
            and layer.experts_per_rank == layer.num_experts // EP_SIZE
            for layer in layers
        ),
        f"{label}_compute_path": all(layer._grouped_mm_enabled() is grouped for layer in layers),
    }


def _grad_norms(trainer) -> list[float]:
    return [entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry]


def _trainable_weights(model) -> dict[str, torch.Tensor]:
    """Every trainable parameter, whole. Collective under FSDP2."""
    reshard_fsdp2_modules(model)
    return snapshot_trainable(model)


class _ResumeCapture(TrainerCallback):
    """The restored optimizer state and weights at ``on_train_begin``: after the restore, before the first step."""

    def __init__(self, holder: dict):
        self.holder = holder

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self.holder["trainer"]
        self.holder["capture"] = {
            "optimizer": local_optimizer_state(trainer.model, trainer.optimizer),
            "weights": _trainable_weights(trainer.model),
        }
        return control


def _trajectory_checks(reference: list[float], candidate: list[float], label: str, metrics: dict) -> dict[str, bool]:
    deltas = [abs(a - b) for a, b in zip(reference, candidate, strict=True)]
    metrics[f"{label}_loss_max_delta"] = max_or_nan(deltas)
    log(f"  {label}: reference {[f'{x:.4f}' for x in reference]}")
    worst = metrics[f"{label}_loss_max_delta"]
    log(f"  {label}: candidate {[f'{x:.4f}' for x in candidate]}  max |delta| {worst:.5f}")
    return {f"{label}_losses_track_the_loop": bool(deltas) and max(deltas) <= TRAJECTORY_LOSS_TOL}


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix="gptoss_etp_grouped_gemm")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    pin_deterministic_ep_dispatch()

    dirs = [ctx.output_dir]
    dist.broadcast_object_list(dirs, src=0)
    train_out = f"{dirs[0]}/train_out"
    ckpt_dir = f"{train_out}/checkpoint-{SAVE_AT_STEP}"

    release_tokenizer = AutoTokenizer.from_pretrained(GPT_OSS_20B)
    tiny_dir = shared_tiny_family_checkpoint(
        ctx, TINY_MOE_FAMILIES["gpt_oss"], "gptoss_etp_grouped_gemm_tiny", release_tokenizer, SEED
    )
    tokenizer = AutoTokenizer.from_pretrained(tiny_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = create_sft_dataset(NUM_SAMPLES, tokenizer, seed=SEED)

    log(f"\n--- 1. per-expert loop, {TOTAL_STEPS} steps ---")
    with _per_expert_loop():
        trainer = _make_trainer(tiny_dir, tokenizer, dataset, _sft_config(train_out, TOTAL_STEPS, None))
        ctx.on_teardown(trainer.cleanup_ep)
        checks |= _layer_checks(trainer.model, grouped=False, label="loop")
        trainer.train()
    loop_losses, loop_norms = step_losses(trainer), _grad_norms(trainer)
    checks["loop_ran_every_step"] = len(loop_losses) == len(loop_norms) == TOTAL_STEPS
    # Anti-vacuity: the trajectories carry signal well past the bound they are compared under.
    loop_drop = loop_losses[0] - min(loop_losses) if loop_losses else 0.0
    metrics["loop_loss_drop"] = loop_drop
    checks["loop_loss_moved_past_the_bound"] = loop_drop > TOL.control_min_loss_shift(TRAJECTORY_LOSS_TOL)
    del trainer
    cleanup_memory()
    ctx.barrier()

    log(f"\n--- 2. per-expert loop, {SAVE_AT_STEP} steps and a checkpoint ---")
    with _per_expert_loop():
        trainer = _make_trainer(tiny_dir, tokenizer, dataset, _sft_config(train_out, SAVE_AT_STEP, SAVE_AT_STEP))
        ctx.on_teardown(trainer.cleanup_ep)
        trainer.train()
    saved_optimizer = local_optimizer_state(trainer.model, trainer.optimizer)
    saved_weights = _trainable_weights(trainer.model)
    del trainer
    cleanup_memory()
    ctx.barrier()

    log(f"\n--- 3. grouped GEMM, {TOTAL_STEPS} steps ---")
    trainer = _make_trainer(tiny_dir, tokenizer, dataset, _sft_config(train_out, TOTAL_STEPS, None))
    ctx.on_teardown(trainer.cleanup_ep)
    checks |= _layer_checks(trainer.model, grouped=True, label="grouped")
    trainer.train()
    grouped_losses, grouped_norms = step_losses(trainer), _grad_norms(trainer)
    checks["grouped_ran_every_step"] = len(grouped_losses) == len(grouped_norms) == TOTAL_STEPS
    if checks["grouped_ran_every_step"] and checks["loop_ran_every_step"]:
        metrics["first_step_loss_delta"] = abs(grouped_losses[0] - loop_losses[0])
        checks["first_step_loss_matches_the_loop"] = metrics["first_step_loss_delta"] <= FIRST_STEP_LOSS_TOL
        checks |= _trajectory_checks(loop_losses, grouped_losses, "grouped", metrics)
        ratios = [max(g / l, l / g) for g, l in zip(grouped_norms, loop_norms, strict=True) if g > 0 and l > 0]
        metrics["grad_norm_max_ratio"] = max_or_nan(ratios)
        log(f"  grad norms: loop {[f'{x:.3f}' for x in loop_norms]}, grouped {[f'{x:.3f}' for x in grouped_norms]}")
        checks["grad_norms_track_the_loop"] = len(ratios) == TOTAL_STEPS and max(ratios) <= TOL.grad_norm_ratio_max
    del trainer
    cleanup_memory()
    ctx.barrier()

    log(f"\n--- 4. grouped GEMM resuming the loop's checkpoint-{SAVE_AT_STEP} ---")
    with logged_messages(optimizer_store_mod.logger) as store_log:
        trainer = _make_trainer(
            ckpt_dir,
            tokenizer,
            dataset,
            _sft_config(train_out, TOTAL_STEPS, None),
            preserve_checkpoint_precision=True,
        )
        ctx.on_teardown(trainer.cleanup_ep)
        checks |= _layer_checks(trainer.model, grouped=True, label="resumed")
        holder = {"trainer": trainer, "capture": None}
        trainer.add_callback(_ResumeCapture(holder))
        trainer.train(resume_from_checkpoint=ckpt_dir)
    warm_restarted = any("fingerprint mismatch" in message for message in store_log)
    checks["no_fingerprint_warm_restart"] = world_all(not warm_restarted, ctx.device)
    capture = holder["capture"]
    checks["resume_capture_fired"] = capture is not None
    if capture is not None:
        equal, why = optimizer_state_matches(saved_optimizer, capture["optimizer"])
        if not equal:
            log(f"optimizer state differs after the restore: {why}")
        checks["optimizer_state_restored_exactly"] = world_all(equal, ctx.device)
        unrestored = [
            name
            for name, value in saved_weights.items()
            if name not in capture["weights"] or not torch.equal(capture["weights"][name], value)
        ]
        checks["weights_restored_exactly"] = world_all(bool(saved_weights) and not unrestored, ctx.device)
    resumed_tail = step_losses(trainer)[-(TOTAL_STEPS - SAVE_AT_STEP) :]
    checks["resumed_ran_remaining_steps"] = len(resumed_tail) == TOTAL_STEPS - SAVE_AT_STEP
    if checks["resumed_ran_remaining_steps"] and checks["loop_ran_every_step"]:
        checks |= _trajectory_checks(loop_losses[SAVE_AT_STEP:], resumed_tail, "resumed", metrics)
    del trainer
    cleanup_memory()
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
