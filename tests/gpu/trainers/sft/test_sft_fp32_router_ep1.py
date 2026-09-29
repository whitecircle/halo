#!/usr/bin/env python
"""``fp32_router`` at ep1 under the default ``fsdp_shard_ep1_experts``: the router trains as an fp32 master.

At ep1 FSDP2 manages the EP layers, so an fp32 router sits in its decoder layer's shard group beside
bf16 parameters, which FSDP2 refuses at the first forward ("expects uniform original parameter
dtype"). The per-layer wrap gives each router a nested group of its own. Each row trains a tiny family
on 2 ranks for ``TOTAL_STEPS`` steps with a checkpoint at ``SAVE_AT_STEP``, then resumes from it:

  1. Every router parameter is an fp32 DTensor master, has a finite nonzero gradient at every optimizer
     step and moves, while every other trainable parameter stays at the run dtype; the checkpoint holds
     each router at fp32 beside every file a resume reads.
  2. The resume restores the weights (a fixed batch's loss), the optimizer moments and the LR
     schedule, keeps the routers fp32 DTensors, and replays the uninterrupted run's later losses.

  * ``--family qwen3_moe``: the router module the EP forward calls.
  * ``--family gpt_oss``: a router named ``router``, with a bias.
  * ``--family zaya``: a router owning no parameter itself, its projections in child modules.
  * ``--family lfm2_moe``: the EP forward reads ``gate.weight`` without calling the router.
  * ``--lora``: attention LoRA with the router trained through ``modules_to_save``, whose copy stays
    fp32 beside the bf16 adapters. Its checkpoint holds adapters, so the whole-model file checks
    are left to the full fine-tunes.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_fp32_router_ep1.py --family qwen3_moe
"""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import TrainerCallback
from trl import SFTConfig

from src.checkpoint.format import load_full_state_dict
from src.distributed.expert_parallel.base_layer import find_ep_layers
from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from src.training.environment import resolve_resume_weights_source
from tests.common.checkpoint_io import (
    RestorePointSnapshot,
    ResumeCapture,
    fixed_batch_loss,
    fixed_text_batch,
    resume_checkpoint_checks,
    resume_continuity_checks,
)
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main
from tests.common.peft_helpers import load_peft_model_from_config, peft_model_config, unwrap
from tests.common.pinned_params import RUN_DTYPE, SEED, load_row_model, tiny_family_checkpoint
from tests.common.tolerances import TOL
from tests.common.utils import finish_phase, log, resumed_loss_deltas, step_losses, training_run_checks

FAMILIES = ("qwen3_moe", "gpt_oss", "zaya", "lfm2_moe")
TOTAL_STEPS = 4
SAVE_AT_STEP = 2
BATCH_SIZE = 2
MAX_LENGTH = 128
LEARNING_RATE = 2e-3
# An adapter resume restores the fp32 router copy exactly and replays the run. A full fine-tune resume
# builds the model from the checkpoint at the run dtype before the fp32 upcast, so its router masters
# restart rounded to bf16 (agent-docs/reference/checkpoints.md) and the later losses part by that.
RESUMED_LOSS_TOL = {True: TOL.replayed_resume_loss_abs, False: TOL.resume_loss_abs}
PROBE_TEXT = "User: Which dtype does the router train in?\nAssistant: fp32, as its own shard group."


def _parallelism_config() -> ParallelismConfig:
    """A fresh config per phase (``create_ep_config`` caches its ``EPConfig`` on the config); the defaults
    keep ``fsdp_shard_ep1_experts`` on, which is the point."""
    return ParallelismConfig(ep_size=1, ep_fp32_router=True)


def _load(source: str, family: str, lora: bool, pc: ParallelismConfig):
    """``(model, tokenizer, peft_config)`` from ``source``: a full fine-tune, or attention LoRA with the
    family's router (the EP layer's own ``_ROUTER_ATTR``) in ``modules_to_save``."""
    if not lora:
        return load_row_model(source, "full", pc)
    model_config = peft_model_config("lora", pc, model_name=source)
    model_config.lora_modules_to_save = [ep_layer_class_by_model_type()[family]._ROUTER_ATTR]
    return load_peft_model_from_config(model_config, pc, attn_implementation="eager", use_liger_kernel=False)


def _router_params(model) -> dict[str, torch.nn.Parameter]:
    """Every trainable router parameter by name, off each EP layer's live router (the ``modules_to_save``
    copy under LoRA)."""
    ids = {
        id(param)
        for _, layer in find_ep_layers(unwrap(model))
        for param in layer._live_router_module().parameters()
        if param.requires_grad
    }
    return {name: param for name, param in unwrap(model).named_parameters() if id(param) in ids}


def _whole(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` whole on every rank; collective for a DTensor."""
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().float().clone()


def _router_snapshot(model) -> dict[str, torch.Tensor]:
    """Every router parameter whole on every rank, read off the sharded masters. Collective."""
    reshard_fsdp2_modules(unwrap(model))
    return {name: _whole(param) for name, param in _router_params(model).items()}


def _fp32_dtensor_masters(params: dict[str, torch.nn.Parameter]) -> bool:
    return bool(params) and all(isinstance(p, DTensor) and p.dtype == torch.float32 for p in params.values())


class _RouterGrads(TrainerCallback):
    """Before every optimizer step: each router parameter's gradient is a DTensor on every rank, finite
    and nonzero. One all-reduce of the per-parameter magnitudes, NaN where a rank holds no DTensor
    gradient, so the verdict is the same on every rank."""

    def __init__(self, trainer, device: torch.device):
        self.trainer = trainer
        self.device = device
        self.steps: list[bool] = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        grads = [param.grad for param in _router_params(self.trainer.model).values()]
        missing = torch.tensor(float("nan"), device=self.device)
        magnitudes = torch.stack(
            [g.to_local().float().abs().sum() if isinstance(g, DTensor) else missing for g in grads]
        )
        dist.all_reduce(magnitudes)
        self.steps.append(bool(grads) and bool(torch.isfinite(magnitudes).all() and (magnitudes > 0).all()))


class _AtSave(RestorePointSnapshot):
    """The fixed batch's loss at the checkpoint, the resume's ``L_pre``."""

    def __init__(self, trainer, probe: tuple[torch.Tensor, torch.Tensor]):
        super().__init__("save", trainer, capture_optimizer=False)
        self.probe = probe

    def extra(self) -> dict:
        return {"l_pre": fixed_batch_loss(self.trainer.model, *self.probe)}


def _sft_config(output_dir: str, *, save: bool) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_steps=TOTAL_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        max_length=MAX_LENGTH,
        bf16=True,
        # Off for every family: Zaya refuses it, and nothing checked here depends on it.
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="steps" if save else "no",
        save_steps=SAVE_AT_STEP,
        report_to=[],
        logging_nan_inf_filter=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        seed=SEED,
    )


def _checkpoint_checks(ctx, checkpoint: str, router_names: list[str]) -> dict[str, bool]:
    """Rank 0 reads the checkpoint: every file a resume reads, and each router at fp32."""
    checks: dict[str, bool] = {}
    if ctx.rank == 0:
        checks = resume_checkpoint_checks(checkpoint, ctx.world_size)
        saved = load_full_state_dict(checkpoint) or {}
        dtypes = {name: saved[name].dtype if name in saved else None for name in router_names}
        log(f"  routers in the checkpoint: {sorted({str(d) for d in dtypes.values()})}")
        checks["checkpoint_holds_routers_at_fp32"] = bool(dtypes) and all(
            dtype == torch.float32 for dtype in dtypes.values()
        )
    return ctx.broadcast_checks(checks)


@gpu_test_main(exact_world_size=2, prefix="sft_fp32_router_ep1")
def run(ctx):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--lora", action="store_true", help="attention LoRA, router via modules_to_save")
    args = parser.parse_args()
    torch.cuda.set_device(ctx.device)

    base_dir = tiny_family_checkpoint(ctx, args.family)
    train_out = os.path.join(shared_output_dir(ctx), "train_out")
    checkpoint = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")

    def make_trainer(source: str, *, save: bool):
        pc = _parallelism_config()
        model, tokenizer, peft_config = _load(source, args.family, args.lora, pc)
        trainer = DistributedSFTTrainer(
            model=model,
            args=_sft_config(train_out, save=save),
            train_dataset=create_sft_dataset(64, tokenizer, seed=SEED),
            processing_class=tokenizer,
            parallelism_config=pc,
            peft_config=peft_config,
        )
        ctx.on_teardown(trainer.cleanup_ep)
        return trainer, fixed_text_batch(tokenizer, ctx.device, PROBE_TEXT)

    log(f"\n[1/2] {args.family}{' LoRA' if args.lora else ''}: {TOTAL_STEPS} steps, checkpoint at {SAVE_AT_STEP}")
    trainer, probe = make_trainer(base_dir, save=True)
    routers = _router_params(trainer.model)
    log(f"  {len(routers)} router params, e.g. {list(routers)[:2]}")
    checks = {"routers_are_fp32_dtensor_masters": _fp32_dtensor_masters(routers)}
    router_ids = {id(param) for param in routers.values()}
    rest = [p for p in unwrap(trainer.model).parameters() if p.requires_grad and id(p) not in router_ids]
    checks["rest_of_the_trainable_set_is_run_dtype"] = bool(rest) and all(p.dtype == RUN_DTYPE for p in rest)
    before = _router_snapshot(trainer.model)
    grads = _RouterGrads(trainer, ctx.device)
    at_save = _AtSave(trainer, probe)
    trainer.add_callback(grads)
    trainer.add_callback(at_save)
    result = trainer.train()
    uninterrupted = step_losses(trainer)
    checks |= training_run_checks(result, trainer, TOTAL_STEPS, grad_norms=True)
    checks["routers_got_gradient_every_step"] = len(grads.steps) == TOTAL_STEPS and all(grads.steps)
    after = _router_snapshot(trainer.model)
    unmoved = [name for name in before if torch.equal(before[name], after[name])]
    checks["every_router_moved"] = (
        bool(before) and not unmoved and all(torch.isfinite(t).all() for t in after.values())
    )
    log(f"  {len(before) - len(unmoved)}/{len(before)} routers moved")
    finish_phase(trainer)
    if not args.lora:
        checks |= _checkpoint_checks(ctx, checkpoint, list(routers))
    ctx.barrier()

    log(f"\n[2/2] Resume from {checkpoint}")
    source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=base_dir), _parallelism_config()
    )
    trainer, probe = make_trainer(source, save=False)
    capture = ResumeCapture(trainer, *probe, optimizer_state=False)
    trainer.add_callback(capture)
    trainer.train(resume_from_checkpoint=checkpoint)
    resumed = step_losses(trainer)
    checks["resumed_routers_are_fp32_dtensor_masters"] = _fp32_dtensor_masters(_router_params(trainer.model))
    l_pre = (at_save.captured or {}).get("l_pre", float("nan"))
    checks |= resume_continuity_checks(
        capture.capture, l_pre, save_step=SAVE_AT_STEP, loss_tol=TOL.resume_fixed_batch_loss_abs
    )
    deltas = resumed_loss_deltas(uninterrupted, resumed, save_step=SAVE_AT_STEP, total_steps=TOTAL_STEPS)
    checks["resumed_ran_remaining_steps"] = deltas is not None
    metrics = {"final_train_loss": result.training_loss, "router_params": len(routers)}
    if deltas is not None:
        metrics["resumed_loss_max_delta"] = max(deltas)
        checks["resumed_losses_match_uninterrupted"] = max(deltas) < RESUMED_LOSS_TOL[args.lora]
    finish_phase(trainer)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
