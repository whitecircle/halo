#!/usr/bin/env python
"""Muon (``optim: muon``) through ``DistributedSFTTrainer`` under EP2, TP2 and expert TP2, on a tiny Qwen3-MoE.

Muon steps each rank's local shard: the Newton-Schulz leg on the 2D+ weights (3D expert stacks whole under EP,
split under expert TP) and its internal AdamWBF16 leg on the norms, embedding and head, a mix of DTensor shards and
plain replicated tensors. One row per layout:

  1. **Train.** ``TOTAL_STEPS`` steps, the last rank poisoning step ``POISON_STEP``: the loss descends over the
     finite steps; a clean step moves parameters of both legs, and the poisoned step moves none (the non-finite
     step guard skips it on both); every parameter the ranks replicate (TP's norms and embedding, EP's routers)
     ends the run bitwise equal across them, which the stochastic rounding keyed by step and name, not rank, has
     to give; every rank took the same Newton-Schulz kernel choice (the availability probe is an all-ranks
     agreement).
  2. **Save, exact resume, export** (:func:`tests.common.resume_export.run_resume_export`): Muon's state dict
     carries the scalar leg's moments, its restore primes the state with a zero-LR step, and the resumed run must
     replay the unbroken one bit for bit.

Run: torchrun --nproc_per_node=2 tests/gpu/optimizers/test_muon_parallel.py --mode tp2
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.runtime.dtensor import to_local
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import save_schedule
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import group_max_abs_diff, pin_deterministic_ep_dispatch, shared_output_dir
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import QWEN3_0_6B
from tests.common.numerics import max_or_nan
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.poisoned_step import AroundStep, PoisonSteps
from tests.common.rank_log import log
from tests.common.resume_export import ResumeExportRoute, load_route_model, run_resume_export
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint
from tests.common.training_checks import step_losses

SEED = 42
TOTAL_STEPS = 6
CONTROL_STEP = 2
POISON_STEP = 4
SAVE_STEP = 3
LEARNING_RATE = 2e-3
MAX_LENGTH = 128
MODES = {"ep2": {"ep_size": 2}, "tp2": {"tp_size": 2}, "etp2": {"expert_tp_size": 2}}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=sorted(MODES), required=True)
ARGS = parser.parse_args()
FIELDS = MODES[ARGS.mode]


def sft_args(output_dir: str, *, max_steps: int = TOTAL_STEPS, save_every: int | None = None) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        optim="muon",
        per_device_train_batch_size=2,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        max_length=MAX_LENGTH,
        packing=False,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        **save_schedule(save_every),
        report_to=[],
        logging_nan_inf_filter=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
    )


def leg_names(model, optimizer) -> tuple[set[str], set[str]]:
    """The parameters Muon's Newton-Schulz leg steps, and those its scalar AdamWBF16 leg steps."""
    names = {id(param): name for name, param in model.named_parameters()}
    muon = {names[id(p)] for group in optimizer.param_groups for p in group["params"] if id(p) in names}
    scalar = optimizer.scalar_optimizer
    scalar_names = (
        {names[id(p)] for group in scalar.param_groups for p in group["params"] if id(p) in names} if scalar else set()
    )
    return muon, scalar_names


def replicated_parameter_spread(model) -> tuple[int, float]:
    """``(count, largest difference)`` over the parameters every rank holds whole and alike: plain tensors that are
    neither FSDP2/TP shards nor an EP layer's (per-rank or expert-TP-split) experts. COLLECTIVE."""
    experts = {id(param) for layer in ep_layers(model) for _, param in layer.expert_named_params()}
    replicated = [
        param
        for _, param in model.named_parameters()
        if not isinstance(param.data, DTensor) and id(param) not in experts
    ]
    spread = max_or_nan((group_max_abs_diff(to_local(param)) for param in replicated), default=float("nan"))
    return len(replicated), spread


@gpu_test_main(exact_world_size=2, prefix=f"muon_parallel_{ARGS.mode}")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    if FIELDS.get("ep_size", 1) > 1:
        # The resume phase replays a run exactly, which DeepEP's atomic receive order would not.
        pin_deterministic_ep_dispatch()
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    base = shared_tiny_family_checkpoint(ctx, TINY_MOE_FAMILIES["qwen3_moe"], "muon_parallel", tokenizer, SEED)
    root = shared_output_dir(ctx)
    dataset = create_sft_dataset(64, tokenizer, seed=SEED)

    def route_trainer(model, output_dir, *, max_steps, save_every, resume_checkpoint):
        return DistributedSFTTrainer(
            model=model,
            args=sft_args(output_dir, max_steps=max_steps, save_every=save_every),
            train_dataset=dataset,
            processing_class=tokenizer,
            parallelism_config=ParallelismConfig(**FIELDS),
        )

    route = ResumeExportRoute(route_trainer, AutoModelForCausalLM, dtype=torch.bfloat16, attn_implementation="eager")
    pc = ParallelismConfig(**FIELDS)
    trainer = route_trainer(
        load_route_model(route, base, pc, resume=False),
        os.path.join(root, "poisoned"),
        max_steps=TOTAL_STEPS,
        save_every=None,
        resume_checkpoint=None,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    checks = parallel_shape_checks(trainer.model, pc)
    control, poisoned = AroundStep(trainer, CONTROL_STEP), AroundStep(trainer, POISON_STEP)
    for callback in (PoisonSteps(trainer, {POISON_STEP}), control, poisoned):
        trainer.add_callback(callback)
    trainer.train()

    optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    checks["optimizer_is_muon_with_a_scalar_leg"] = (
        type(optimizer).__name__ == "Muon" and optimizer.scalar_optimizer is not None
    )
    muon, scalar = leg_names(trainer.model, optimizer)
    checks["both_legs_hold_parameters"] = bool(muon) and bool(scalar)
    choices: list = [None] * ctx.world_size
    dist.all_gather_object(choices, bool(getattr(optimizer, "ns_use_kernels", None)))
    checks["every_rank_took_one_newton_schulz_kernel_choice"] = len(set(choices)) == 1
    moved = control.moved()
    checks["a_clean_step_moved_both_legs"] = bool(moved & muon) and bool(moved & scalar)
    skipped = poisoned.moved()
    checks["the_poisoned_step_moved_neither_leg"] = not skipped
    checks["skip_logged"] = any("nonfinite_steps_skipped" in entry for entry in trainer.state.log_history)
    losses = step_losses(trainer)
    finite = [loss for loss in losses if math.isfinite(loss)]
    checks["loss_descended_over_the_finite_steps"] = len(finite) >= TOTAL_STEPS - 1 and finite[-1] < finite[0]
    count, spread = replicated_parameter_spread(trainer.model)
    checks["replicated_parameters_bitwise_equal"] = count > 0 and spread == 0.0
    log(
        f"  {ARGS.mode}: Muon leg {len(muon)}, scalar leg {len(scalar)}; step {CONTROL_STEP} moved "
        f"{len(moved & muon)}/{len(moved & scalar)}, step {POISON_STEP} moved {sorted(skipped)[:3]}; losses {losses}; "
        f"{count} replicated params, spread {spread:.2e}; kernel choices {choices}"
    )
    finish_phase(trainer)

    phase_checks, phase_metrics = run_resume_export(
        ctx, route, pc, base=base, root=os.path.join(root, "resume"), save_step=SAVE_STEP, total_steps=TOTAL_STEPS
    )
    return {"checks": checks | phase_checks, "metrics": phase_metrics}


if __name__ == "__main__":
    run()
