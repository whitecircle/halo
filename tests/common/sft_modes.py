"""The shared body of the SFT mode suites: one model under one parallel shape, trained on a synthetic
dataset and checked on the finished run and on the shape the model took.

A suite states what is its own (checkpoint, attention kernel, training arguments) as an
:class:`SFTSuite` and its parallel shapes as a ``{key: SFTMode}`` table; :func:`run_sft_suite` runs
the ``--mode`` a manifest row names. Each row runs one mode, since a second EP model in one process
has to rebuild DeepEP's buffers; ``--mode all`` still runs every mode in order for a standalone sweep.
"""

import argparse
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig

from src.data.collators.factory import select_data_collator
from src.distributed.expert_parallel.dispatcher import destroy_all_dispatchers
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from src.training.script_runner import apply_distributed_trainer_config
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.utils import cleanup_memory, gpu_mem_gb, log, max_or_nan, step_losses, training_run_checks

SEED = 42
# The eval split's seed offset, so it never repeats the training samples.
EVAL_SEED_OFFSET = 100
# SFTConfig fields every suite shares: one sample per device, bf16 with gradient checkpointing, a loss
# logged every step, nothing saved or reported. Liger stays off here because the loader already
# applied its kernels.
BASE_SFT_ARGS: Mapping[str, Any] = {
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "bf16": True,
    "gradient_checkpointing": True,
    "use_liger_kernel": False,
    "logging_steps": 1,
    "save_strategy": "no",
    "report_to": "none",
    "logging_nan_inf_filter": False,
    "dataloader_num_workers": 0,
    "dataloader_drop_last": True,
}

TrainerChecks = Callable[[DistributedSFTTrainer], dict[str, bool]]


@dataclass(frozen=True)
class SFTMode:
    """One parallel shape of a suite: its ``ParallelismConfig`` kwargs and what else it changes."""

    parallelism: Mapping[str, Any] = field(default_factory=dict)
    # ``None`` keeps the suite's.
    attn_implementation: str | None = None
    # Over the suite's SFT arguments.
    sft_args: Mapping[str, Any] = field(default_factory=dict)
    # ``(train, eval)`` sample counts; ``None`` keeps the suite's.
    num_samples: tuple[int, int] | None = None
    extra_checks: TrainerChecks | None = None


@dataclass(frozen=True)
class SFTSuite:
    """What every mode of one suite shares.

    ``sft_args`` sit over :data:`BASE_SFT_ARGS` and must name ``max_steps``. ``dataset`` is called as
    ``dataset(num_samples, tokenizer, seed=...)``; an eval count of 0 builds no eval split.
    ``evaluate`` runs an evaluation before and after training (plus the trainer's own at the last
    step) and checks the final eval loss; ``loss_decreased`` adds the first-vs-last step-loss check.
    """

    model_name: str
    attn_implementation: str | None
    sft_args: Mapping[str, Any]
    num_samples: tuple[int, int] = (32, 8)
    dataset: Callable[..., Dataset] = create_sft_dataset
    use_liger_kernel: bool = True
    evaluate: bool = True
    loss_decreased: bool = False
    extra_checks: TrainerChecks | None = None


def run_sft_mode(ctx, suite: SFTSuite, key: str, mode: SFTMode, tokenizer) -> dict[str, bool]:
    """Train ``suite``'s model under ``mode``; its checks are keyed ``<key>_<check>``."""
    parallelism_config = ParallelismConfig(**mode.parallelism)
    attn_implementation = mode.attn_implementation or suite.attn_implementation
    log(f"\n{'=' * 70}\n  SFT {suite.model_name} --mode {key}: {parallelism_config.summary()}\n{'=' * 70}")

    num_train, num_eval = mode.num_samples or suite.num_samples
    train_dataset = suite.dataset(num_train, tokenizer, seed=SEED)
    eval_dataset = suite.dataset(num_eval, tokenizer, seed=SEED + EVAL_SEED_OFFSET) if num_eval else None

    model, _ = load_distributed_model(
        model_name_or_path=suite.model_name,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        use_liger_kernel=suite.use_liger_kernel,
    )
    log(f"  Loaded {type(model).__name__} ({attn_implementation}), GPU memory {gpu_mem_gb():.1f}GB")

    sft_args = {**BASE_SFT_ARGS, **suite.sft_args, **mode.sft_args}
    max_steps = sft_args["max_steps"]
    if suite.evaluate:
        sft_args |= {"eval_strategy": "steps", "eval_steps": max_steps}
    config = SFTConfig(output_dir=os.path.join(ctx.output_dir, key), **sft_args)
    apply_distributed_trainer_config(config, parallelism_config)
    # CP splits every batch across ranks, so each has to pad to a multiple of cp_size; the production
    # script builds the same collator through this factory.
    data_collator = (
        select_data_collator(
            tokenizer=tokenizer, pad_to_multiple_of=parallelism_config.cp_size, use_context_parallel=True
        )
        if parallelism_config.is_cp_mode
        else None
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
        data_collator=data_collator,
    )

    checks = parallel_shape_checks(model, parallelism_config)
    if suite.evaluate:
        barrier()
        log(f"  Initial eval loss: {trainer.evaluate().get('eval_loss', math.nan):.4f}")
    barrier()
    train_result = trainer.train()
    checks |= training_run_checks(
        train_result, trainer, max_steps, grad_norms=True, loss_decreased=suite.loss_decreased
    )
    if suite.evaluate:
        barrier()
        final_eval_loss = trainer.evaluate().get("eval_loss", math.nan)
        checks["final_eval_loss_finite"] = math.isfinite(final_eval_loss)
        log(f"  Final eval loss: {final_eval_loss:.4f}")
    for extra_checks in (suite.extra_checks, mode.extra_checks):
        if extra_checks is not None:
            checks |= extra_checks(trainer)

    trainer.cleanup_ep()
    return {f"{key}_{name}": ok for name, ok in checks.items()}


def run_sft_suite(ctx, suite: SFTSuite, modes: Mapping[str, SFTMode], *, default_mode: str) -> dict:
    """Run the ``--mode`` the launch names (``all`` runs every mode in order) and return the harness
    result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[*modes, "all"], default=default_mode)
    mode = parser.parse_args().mode
    selected = list(modes) if mode == "all" else [mode]
    log(f"\nSFT suite: {suite.model_name}, world {ctx.world_size}, {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"Modes: {selected}")

    # Registry-wide rather than per trainer: a finalizer bound to a trainer would keep each mode's model
    # alive into the next mode.
    ctx.on_teardown(destroy_all_dispatchers)
    ensure_model_downloaded(suite.model_name, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(suite.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    checks: dict[str, bool] = {}
    for key in selected:
        checks |= run_sft_mode(ctx, suite, key, modes[key], tokenizer)
        cleanup_memory()
    return {"checks": checks}


def sft_metric_checks(
    trainer,
    *,
    first_loss_band: tuple[float, float],
    max_grad_norm: float,
    token_accuracy_rises: bool = False,
    eval_loss_logged: bool = False,
) -> dict[str, bool]:
    """Checks on the metrics an SFT run logs, past what :func:`training_run_checks` reads.

    ``initial_loss_reasonable`` (the first step loss inside ``first_loss_band``: a pretrained model on
    the math data, not an untrained or collapsed one), ``grad_norms_reasonable`` (every logged norm
    under ``max_grad_norm``, the ceiling a missing TP reduction blows past), ``token_accuracy_valid``
    (every logged ``mean_token_accuracy`` in [0, 1]), with ``token_accuracy_rises``
    ``token_accuracy_increased`` (last above first), and with ``eval_loss_logged`` ``eval_loss_finite``
    (at least one eval loss logged, all finite: a silently skipped eval must not pass). The token
    accuracy checks apply only when the run logs it.
    """
    history = trainer.state.log_history
    losses = step_losses(trainer)
    grad_norms = [entry["grad_norm"] for entry in history if "grad_norm" in entry]
    accuracies = [entry["mean_token_accuracy"] for entry in history if "mean_token_accuracy" in entry]
    low, high = first_loss_band
    checks = {
        "initial_loss_reasonable": bool(losses) and low <= losses[0] <= high,
        "grad_norms_reasonable": bool(grad_norms) and max_or_nan(grad_norms) < max_grad_norm,
    }
    if accuracies:
        checks["token_accuracy_valid"] = all(0.0 <= accuracy <= 1.0 for accuracy in accuracies)
        if token_accuracy_rises:
            checks["token_accuracy_increased"] = len(accuracies) >= 2 and accuracies[-1] > accuracies[0]
    if eval_loss_logged:
        eval_losses = [entry["eval_loss"] for entry in history if "eval_loss" in entry]
        checks["eval_loss_finite"] = bool(eval_losses) and all(math.isfinite(loss) for loss in eval_losses)
    log(f"  Grad norms: {[f'{norm:.2f}' for norm in grad_norms]} (ceiling {max_grad_norm})")
    log(f"  Token accuracy: {[f'{accuracy:.4f}' for accuracy in accuracies]}")
    for name, ok in checks.items():
        log(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return checks
