#!/usr/bin/env python
"""GptOss at ep4 x etp2 trains on grouped GEMM as it trained on the per-expert loop, and resumes a loop-path
ETP checkpoint exactly (8 GPUs).

The loop arm runs the per-expert loop under ``use_grouped_gemm: true`` (the loop rule patched onto the layer
class, the config unchanged), the configuration a loop-path ETP checkpoint carries, fingerprint included:

1. Loop, 20 steps: the reference trajectory, every EP layer on the loop.
2. Loop, 10 steps and a checkpoint, with this rank's optimizer state and trainable weights snapshotted.
3. The floor: the loop, 20 steps from the same init with every trainable weight perturbed by about one bf16 ULP.
4. Grouped GEMM, 20 steps from the same init: every EP layer on grouped GEMM. Its step-1 loss and gradient norm
   match the loop's (a forward or backward error shows there, before any drift), and over all 20 steps it strays
   from the loop's trajectory no further than the floor does, scaled by a margin.
5. The control: grouped GEMM with every rank's expert bank rotated by one expert, one step. Its step-1 loss must
   miss the loop's by over twice the step-1 bound, or that bound could not see a wrong expert.
6. Grouped GEMM resuming step 2's checkpoint: optimizer state and weights restored exactly, no fingerprint
   warm restart, steps 11-20 on the loop's trajectory.

From step 7 the loop's gradient norm surges (to about 12 at step 13): there any difference of start or reduction
order grows, and the perturbed loop strays from its own reference as far as grouped GEMM does. A fixed trajectory
bound would fail on that growth with no defect present, hence the measured floor.

DeepEP dispatch is pinned deterministic, so the arms differ in the expert compute alone. A tiny random-init
GptOss (8 experts: two per dispatch rank, each FFN sharded two ways) keeps the runs cheap.

    torchrun --nproc_per_node=8 tests/gpu/parallelism/combined/test_gptoss_etp_grouped_gemm.py
"""

import contextlib
import functools
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
from tests.common.ep_reference import roll_expert_banks
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import GPT_OSS_20B
from tests.common.numerics import max_or_nan, perturb_trainable
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
# The step-1 forward differs by reduction order and the loop's bf16 index_add_ rounding only (1.1e-3 measured); a
# rotated expert bank moves it by about 3.6e-2 (an fp32 CPU forward of this model and batch), past the control bound
# of twice this.
FIRST_STEP_LOSS_TOL = 1e-2
FIRST_STEP_GRAD_NORM_RATIO_MAX = TOL.grad_norm_ratio_max
# The floor arm's perturbation, about one bf16 ULP: it moves about half the weights by one ULP, where 2e-3 moves 0.15%
# and 5e-4 none.
FLOOR_PERTURBATION = 4e-3
# Grouped GEMM may stray from the loop's trajectory this many times as far as the floor arm does.
FLOOR_LOSS_DELTA_FACTOR = 2.0
FLOOR_GRAD_NORM_RATIO_FACTOR = 1.25
# The resumed arm starts from the loop's own step-10 state, so it stays inside a fixed bound.
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


def _ratio(a: float, b: float) -> float:
    return max(a / b, b / a) if a > 0 and b > 0 else float("inf")


def _deltas(reference: list[float], candidate: list[float]) -> list[float]:
    return [abs(a - b) for a, b in zip(reference, candidate, strict=True)]


def _trajectory_checks(reference: list[float], candidate: list[float], label: str, metrics: dict) -> dict[str, bool]:
    deltas = _deltas(reference, candidate)
    metrics[f"{label}_loss_max_delta"] = max_or_nan(deltas)
    log(f"  {label}: reference {[f'{x:.4f}' for x in reference]}")
    worst = metrics[f"{label}_loss_max_delta"]
    log(f"  {label}: candidate {[f'{x:.4f}' for x in candidate]}  max |delta| {worst:.5f}")
    return {f"{label}_losses_track_the_loop": bool(deltas) and max(deltas) <= TRAJECTORY_LOSS_TOL}


def _train_arm(ctx, tiny_dir, tokenizer, dataset, train_out, steps: int, *, grouped: bool, label: str, edit=None):
    """COLLECTIVE. One run from the tiny checkpoint: its layer checks, per-step losses and gradient norms, and what
    ``edit`` (applied to the built model before training) returned."""
    with contextlib.nullcontext() if grouped else _per_expert_loop():
        trainer = _make_trainer(tiny_dir, tokenizer, dataset, _sft_config(train_out, steps, None))
        ctx.on_teardown(trainer.cleanup_ep)
        checks = _layer_checks(trainer.model, grouped=grouped, label=label)
        edited = edit(trainer.model) if edit else None
        trainer.train()
    losses, norms = step_losses(trainer), _grad_norms(trainer)
    checks[f"{label}_ran_every_step"] = len(losses) == len(norms) == steps
    del trainer
    cleanup_memory()
    ctx.barrier()
    return checks, losses, norms, edited


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
    arm = functools.partial(_train_arm, ctx, tiny_dir, tokenizer, dataset, train_out)

    log(f"\n--- 1. per-expert loop, {TOTAL_STEPS} steps ---")
    arm_checks, loop_losses, loop_norms, _ = arm(TOTAL_STEPS, grouped=False, label="loop")
    checks |= arm_checks
    # Anti-vacuity: the trajectories carry signal well past the bound they are compared under.
    loop_drop = loop_losses[0] - min(loop_losses) if loop_losses else 0.0
    metrics["loop_loss_drop"] = loop_drop
    checks["loop_loss_moved_past_the_bound"] = loop_drop > TOL.control_min_loss_shift(TRAJECTORY_LOSS_TOL)

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

    log(f"\n--- 3. the floor: per-expert loop from weights perturbed by {FLOOR_PERTURBATION}, {TOTAL_STEPS} steps ---")
    arm_checks, floor_losses, floor_norms, changed = arm(
        TOTAL_STEPS, grouped=False, label="floor", edit=lambda model: perturb_trainable(model, FLOOR_PERTURBATION)
    )
    checks |= arm_checks
    checks["floor_perturbation_changed_weights"] = world_all(changed > 0, ctx.device)

    log(f"\n--- 4. grouped GEMM, {TOTAL_STEPS} steps ---")
    arm_checks, grouped_losses, grouped_norms, _ = arm(TOTAL_STEPS, grouped=True, label="grouped")
    checks |= arm_checks

    log("\n--- 5. control: grouped GEMM with every expert bank rotated by one expert, 1 step ---")
    arm_checks, control_losses, _, rotated = arm(1, grouped=True, label="control", edit=roll_expert_banks)
    checks |= arm_checks
    checks["control_rotated_every_bank"] = world_all(rotated > 0, ctx.device)

    full = all(checks[f"{label}_ran_every_step"] for label in ("loop", "floor", "grouped", "control"))
    if full:
        metrics["first_step_loss_delta"] = abs(grouped_losses[0] - loop_losses[0])
        metrics["first_step_grad_norm_ratio"] = _ratio(grouped_norms[0], loop_norms[0])
        checks["first_step_loss_matches_the_loop"] = metrics["first_step_loss_delta"] <= FIRST_STEP_LOSS_TOL
        checks["first_step_grad_norm_matches_the_loop"] = (
            metrics["first_step_grad_norm_ratio"] <= FIRST_STEP_GRAD_NORM_RATIO_MAX
        )
        metrics["control_first_step_loss_delta"] = abs(control_losses[0] - loop_losses[0])
        checks["control_misses_the_first_step_loss"] = metrics["control_first_step_loss_delta"] > (
            TOL.control_min_loss_shift(FIRST_STEP_LOSS_TOL)
        )

        metrics["floor_loss_max_delta"] = max_or_nan(_deltas(loop_losses, floor_losses))
        metrics["grouped_loss_max_delta"] = max_or_nan(_deltas(loop_losses, grouped_losses))
        metrics["floor_grad_norm_max_ratio"] = max_or_nan(map(_ratio, floor_norms, loop_norms))
        metrics["grouped_grad_norm_max_ratio"] = max_or_nan(map(_ratio, grouped_norms, loop_norms))
        log(f"  loop    losses {[f'{x:.4f}' for x in loop_losses]}")
        log(f"  floor   losses {[f'{x:.4f}' for x in floor_losses]}")
        log(f"  grouped losses {[f'{x:.4f}' for x in grouped_losses]}")
        log(f"  grad norms: loop {[f'{x:.3f}' for x in loop_norms]}")
        log(f"  grad norms: floor {[f'{x:.3f}' for x in floor_norms]}")
        log(f"  grad norms: grouped {[f'{x:.3f}' for x in grouped_norms]}")
        checks["floor_moved_the_trajectory"] = metrics["floor_loss_max_delta"] > 0
        checks["grouped_losses_inside_the_floor"] = (
            metrics["grouped_loss_max_delta"] <= FLOOR_LOSS_DELTA_FACTOR * metrics["floor_loss_max_delta"]
        )
        checks["grouped_grad_norms_inside_the_floor"] = (
            metrics["grouped_grad_norm_max_ratio"]
            <= FLOOR_GRAD_NORM_RATIO_FACTOR * metrics["floor_grad_norm_max_ratio"]
        )

    log(f"\n--- 6. grouped GEMM resuming the loop's checkpoint-{SAVE_AT_STEP} ---")
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
