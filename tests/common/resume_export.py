"""The save, exact-resume and export phase a trainer's checkpoint owes, under any parallel layout. A resume must
restore per-rank optimizer state bit for bit and replay the unbroken run's later losses and final checkpoint."""

import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from trl import SFTConfig

import halo.checkpoint.parallel.optimizer as optimizer_store
from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.parallel.save import save_tokenizer
from halo.checkpoint.reading import has_whole_model_weight_file, load_full_state_dict
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.checkpoint_io import (
    RestorePointSnapshot,
    loading_problems,
    resume_checkpoint_checks,
    save_context,
    save_schedule,
)
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import pin_deterministic_ep_dispatch, shared_output_dir
from tests.common.memory import cleanup_memory, finish_phase
from tests.common.numerics import max_or_nan, relative_l2
from tests.common.optimizer_state import optimizer_state_matches
from tests.common.rank_log import log, logged_messages
from tests.common.tolerances import TOL
from tests.common.training_checks import resumed_loss_deltas, step_losses, training_run_checks

# Spelled here rather than imported, so a reworded log fails these checks instead of agreeing.
OPTIMIZER_RESTORED = "Restored per-rank optimizer state from {checkpoint}"
WARM_RESTART = "Warm restart"
EXPORT_CHECKS = (
    "export_loads_every_weight",
    "export_is_the_trained_class",
    "export_head_trained",
    "export_backbone_trained",
)
# A real release's EP SFT row: a checkpoint halfway through a run short enough to replay at its depth.
RELEASE_DATA_ROWS = 32
RELEASE_DATA_SEED = 0
RELEASE_MAX_LENGTH = 512
RELEASE_LR = 1e-4
RELEASE_SAVE_STEP = 3
RELEASE_TOTAL_STEPS = 2 * RELEASE_SAVE_STEP


@dataclass(frozen=True)
class ResumeExportRoute:
    """How one trainer trains, resumes and exports; ``resume_checkpoint`` is None for the unbroken run. ``sidecars``
    are the files beside weights and trainer state every checkpoint must carry; ``run_state(trainer)`` reads what they
    restore."""

    build_trainer: Callable
    export_class: type
    dtype: torch.dtype
    attn_implementation: str
    sidecars: tuple[str, ...] = ()
    run_state: Callable | None = None


def save_base_checkpoint(ctx, model, tokenizer, path: str) -> str:
    """COLLECTIVE. ``model`` and ``tokenizer`` saved at the rank-shared ``path`` by rank 0; returns ``path``."""
    if ctx.rank == 0:
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
    ctx.barrier()
    return path


def load_route_model(route: ResumeExportRoute, source: str, parallelism_config, *, resume: bool):
    """COLLECTIVE. ``source`` loaded as the training scripts do; a resume at full checkpoint precision."""
    model, _ = load_distributed_model(
        source,
        parallelism_config,
        dtype=route.dtype,
        trust_remote_code=True,
        attn_implementation=route.attn_implementation,
        model_class=route.export_class,
        preserve_checkpoint_precision=resume,
    )
    return model


def checkpoint_file_checks(checkpoint: str, world_size: int, sidecars: tuple[str, ...]) -> dict[str, bool]:
    """Named checks that ``checkpoint`` holds every resume file and each of ``sidecars``. Disk only: rank 0 calls it
    and shares the verdict."""
    checks = resume_checkpoint_checks(checkpoint, world_size)
    checks |= {name: os.path.isfile(os.path.join(checkpoint, name)) for name in sidecars}
    return checks


def export_checks(export_class: type, checkpoint: str, base: str, dtype: torch.dtype) -> dict[str, bool]:
    """Whether ``checkpoint`` loads through ``export_class`` with no bad weight, and its untied head and some backbone
    tensor moved off ``base``. Rank 0."""
    if not (has_whole_model_weight_file(checkpoint) and has_whole_model_weight_file(base)):
        log(f"  export: no weights at {checkpoint} or {base}")
        return dict.fromkeys(EXPORT_CHECKS, False)
    model, info = export_class.from_pretrained(checkpoint, output_loading_info=True, dtype=dtype)
    problems = loading_problems(info)
    written, initial = load_full_state_dict(checkpoint), load_full_state_dict(base)
    backbone = (f"{model.base_model_prefix}.", "backbone.")
    moved = {
        name
        for name, tensor in written.items()
        if name in initial
        and initial[name].shape == tensor.shape
        and not torch.equal(tensor, initial[name].to(tensor.dtype))
    }
    head = [name for name in written if not name.startswith(backbone)]
    architectures = AutoConfig.from_pretrained(checkpoint).architectures
    log(
        f"  export {checkpoint}: problems={problems or 'none'} architectures={architectures} "
        f"loaded={type(model).__name__} head={head} moved {len(moved)}/{len(written)}"
    )
    return {
        "export_loads_every_weight": not problems,
        "export_is_the_trained_class": architectures
        == AutoConfig.from_pretrained(base).architectures
        == [type(model).__name__],
        "export_head_trained": bool(head) and all(name in moved for name in head),
        "export_backbone_trained": any(name.startswith(backbone) for name in moved),
    }


def restore_log_checks(messages: list[str], checkpoint: str) -> dict[str, bool]:
    """Whether the optimizer restore logged its success and no warm restart. Rank 0's log."""
    restored = OPTIMIZER_RESTORED.format(checkpoint=checkpoint)
    return {
        "optimizer_restore_logged": any(restored in message for message in messages),
        "no_warm_restart_logged": not any(WARM_RESTART in message for message in messages),
    }


def restore_point_checks(saved: dict | None, restored: dict | None, save_step: int) -> dict[str, bool]:
    """What a resume restored vs the state its checkpoint was written from: optimizer state bit for bit, and step and
    LR at ``save_step``."""
    saved, restored = saved or {}, restored or {}
    exact, why = optimizer_state_matches(
        saved.get("optimizer") or {"state": {}}, restored.get("optimizer") or {"state": {}}
    )
    if not exact:
        log(f"  optimizer restore: {why}")
    return {
        "optimizer_restored_bit_exact": exact,
        "step_and_schedule_restored": saved.get("global_step") == save_step
        and restored.get("global_step") == save_step
        and restored.get("sched_last_epoch") == save_step,
    }


def run_state_checks(saved: dict[str, torch.Tensor], restored: dict[str, torch.Tensor]) -> dict[str, bool]:
    equal = bool(saved) and saved.keys() == restored.keys() and all(torch.equal(saved[k], restored[k]) for k in saved)
    return {"run_state_restored": equal}


def replay_checks(
    unbroken: list[float], resumed: list[float], *, save_step: int, total_steps: int
) -> tuple[dict[str, bool], dict[str, float]]:
    """The resumed run's step losses vs the unbroken run's: history carried over as logged, later steps within
    ``TOL.replayed_resume_loss_abs``."""
    deltas = resumed_loss_deltas(unbroken, resumed, save_step=save_step, total_steps=total_steps)
    max_delta = max_or_nan(deltas) if deltas else math.inf
    checks = {
        "history_restored": len(resumed) >= save_step and resumed[:save_step] == unbroken[:save_step],
        "losses_replay_unbroken": max_delta <= TOL.replayed_resume_loss_abs,
    }
    return checks, {"resume_loss_max_delta": max_delta}


def final_weight_checks(unbroken: str, resumed: str) -> tuple[dict[str, bool], dict[str, float]]:
    """The final checkpoints as one vector each, within ``TOL.replayed_resume_weight_rtol`` relative L2. Rank 0."""
    written = [load_full_state_dict(path) if has_whole_model_weight_file(path) else {} for path in (unbroken, resumed)]
    drift = relative_l2(written[1], written[0])
    log(f"  final checkpoint, resumed vs unbroken: relative L2 {drift:.3e} (tol {TOL.replayed_resume_weight_rtol})")
    checks = {"final_weights_match_unbroken": drift <= TOL.replayed_resume_weight_rtol}
    return checks, {"resume_final_weight_rel_l2": drift}


def run_resume_export(
    ctx, route: ResumeExportRoute, parallelism_config, *, base: str, root: str, save_step: int, total_steps: int
) -> tuple[dict[str, bool], dict[str, float]]:
    """COLLECTIVE. The phase's checks and metrics, the resumed run's prefixed ``resume_``. Resumes only when every
    unbroken-run check passed."""
    unbroken_dir, resumed_dir = os.path.join(root, "unbroken"), os.path.join(root, "resumed")
    midway = os.path.join(unbroken_dir, f"checkpoint-{save_step}")
    final = f"checkpoint-{total_steps}"

    log(f"\n  Unbroken run: {total_steps} steps from {base}, a checkpoint every {save_step}")
    trainer = route.build_trainer(
        load_route_model(route, base, parallelism_config, resume=False),
        unbroken_dir,
        max_steps=total_steps,
        save_every=save_step,
        resume_checkpoint=None,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    saved = RestorePointSnapshot("save", trainer, capture_optimizer=True)
    trainer.add_callback(saved)
    run_state = route.run_state(trainer) if route.run_state else {}
    checks = training_run_checks(trainer.train(), trainer, total_steps)
    unbroken_losses = step_losses(trainer)
    finish_phase(trainer)
    saved.trainer = None
    ctx.barrier()
    files = {}
    if ctx.rank == 0:
        for name, checkpoint in (("midway", midway), ("final", os.path.join(unbroken_dir, final))):
            files |= {
                f"{name}_{check}": ok
                for check, ok in checkpoint_file_checks(checkpoint, ctx.world_size, route.sidecars).items()
            }
        files |= export_checks(route.export_class, os.path.join(unbroken_dir, final), base, route.dtype)
    checks |= ctx.broadcast_checks(files)
    if not all(checks.values()):
        log("  Unbroken run FAILED, skipping the resume")
        return checks, {}

    log(f"\n  Resumed run: from {midway} to step {total_steps}")
    resumed = route.build_trainer(
        load_route_model(route, midway, parallelism_config, resume=True),
        resumed_dir,
        max_steps=total_steps,
        save_every=save_step,
        resume_checkpoint=midway,
    )
    ctx.on_teardown(resumed.cleanup_ep)
    restored = RestorePointSnapshot("train_begin", resumed, capture_optimizer=True)
    resumed.add_callback(restored)
    resume_checks = run_state_checks(run_state, route.run_state(resumed)) if route.run_state else {}
    with logged_messages(optimizer_store.logger) as messages:
        result = resumed.train(resume_from_checkpoint=midway)
    resume_checks |= training_run_checks(result, resumed, total_steps)
    resume_checks |= restore_point_checks(saved.captured, restored.captured, save_step)
    replay, metrics = replay_checks(
        unbroken_losses, step_losses(resumed), save_step=save_step, total_steps=total_steps
    )
    resume_checks |= replay
    finish_phase(resumed)
    restored.trainer = None
    ctx.barrier()
    rank0 = {}
    if ctx.rank == 0:
        rank0 = restore_log_checks(messages, midway)
        weights, weight_metrics = final_weight_checks(
            os.path.join(unbroken_dir, final), os.path.join(resumed_dir, final)
        )
        rank0 |= weights
        metrics |= weight_metrics
    resume_checks |= ctx.broadcast_checks(rank0)
    checks |= {f"resume_{name}": ok for name, ok in resume_checks.items()}
    return checks, metrics


def save_gathered_ep(ctx, model, parallelism_config: ParallelismConfig, tokenizer, directory: str) -> None:
    """COLLECTIVE. The gathered EP save and tokenizer, as a run's save writes them."""
    save = save_context(model, parallelism_config, tokenizer=tokenizer, has_ep_layers=True)
    save_tokenizer(save, directory)
    save_ep_gathered_checkpoint(save, directory)
    ctx.barrier()


def run_release_resume(ctx, load: Callable[[ParallelismConfig], tuple]) -> dict:
    """COLLECTIVE. The phase on an EP SFT run of a real release's text class, on SDPA, from a gathered save of what
    ``load`` builds for the config it is given: ``(model, tokenizer)``."""
    # The resumed run replays the unbroken one exactly, which DeepEP's atomic receive order would not.
    pin_deterministic_ep_dispatch()
    parallelism_config = ParallelismConfig(ep_size=ctx.world_size)
    root = shared_output_dir(ctx)
    base = os.path.join(root, "base")
    model, tokenizer = load(parallelism_config)
    save_gathered_ep(ctx, model, parallelism_config, tokenizer, base)
    del model
    cleanup_memory()
    train_dataset = create_sft_dataset(RELEASE_DATA_ROWS, tokenizer, seed=RELEASE_DATA_SEED)

    def build_trainer(model, output_dir, *, max_steps, save_every, resume_checkpoint):
        return DistributedSFTTrainer(
            model=model,
            args=SFTConfig(
                output_dir=output_dir,
                max_steps=max_steps,
                **save_schedule(save_every),
                per_device_train_batch_size=1,
                learning_rate=RELEASE_LR,
                bf16=True,
                gradient_checkpointing=True,
                logging_steps=1,
                report_to=[],
                logging_nan_inf_filter=False,
                max_length=RELEASE_MAX_LENGTH,
                dataloader_drop_last=True,
                dataloader_num_workers=0,
            ),
            train_dataset=train_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

    route = ResumeExportRoute(build_trainer, AutoModelForCausalLM, dtype=torch.bfloat16, attn_implementation="sdpa")
    checks, metrics = run_resume_export(
        ctx,
        route,
        parallelism_config,
        base=base,
        root=root,
        save_step=RELEASE_SAVE_STEP,
        total_steps=RELEASE_TOTAL_STEPS,
    )
    return {"checks": checks, "metrics": metrics}
