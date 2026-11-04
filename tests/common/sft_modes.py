"""The shared body of the SFT mode suites: one model under one parallel shape, trained on a synthetic dataset and
checked. Each manifest row runs one mode, since a second EP model in one process must rebuild DeepEP's buffers."""

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.distributed.expert_parallel.deepep.dispatcher import destroy_all_dispatchers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import barrier
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import load_peft_model
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.memory import cleanup_memory, gpu_mem_gb
from tests.common.numerics import max_or_nan
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log
from tests.common.sft_collator import build_sft_collator
from tests.common.training_checks import step_losses, training_run_checks

SEED = 42
# So the eval split never repeats the training samples.
EVAL_SEED_OFFSET = 100
# Liger stays off because the loader already applied its kernels.
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
    parallelism: Mapping[str, Any] = field(default_factory=dict)
    # None keeps the suite's.
    attn_implementation: str | None = None
    sft_args: Mapping[str, Any] = field(default_factory=dict)
    # None keeps the suite's.
    num_samples: tuple[int, int] | None = None
    extra_checks: TrainerChecks | None = None
    # :func:`~tests.common.adapters.load_peft_model` kwargs: the model loads through the production adapter path.
    adapters: Mapping[str, Any] | None = None
    # Runs on the built trainer before training; returns the checks to run after.
    before_train: Callable[[DistributedSFTTrainer], TrainerChecks] | None = None
    # False keeps GptOss's pretrained sinks live, as on-policy RL loads them.
    reset_sinks: bool = True


@dataclass(frozen=True)
class SFTSuite:
    """What every mode of one suite shares; ``sft_args`` must name ``max_steps``, and an eval count of 0 builds no eval
    split."""

    model_name: str
    attn_implementation: str | None
    sft_args: Mapping[str, Any]
    num_samples: tuple[int, int] = (32, 8)
    dataset: Callable[..., Dataset] = create_sft_dataset
    use_liger_kernel: bool = True
    evaluate: bool = True
    loss_decreased: bool = False
    loss_band: tuple[float, float] | None = None
    extra_checks: TrainerChecks | None = None


def run_sft_mode(ctx, suite: SFTSuite, key: str, mode: SFTMode, tokenizer) -> dict[str, bool]:
    """Train ``suite``'s model under ``mode``; its checks are keyed ``<key>_<check>``."""
    parallelism_config = ParallelismConfig(**mode.parallelism)
    attn_implementation = mode.attn_implementation or suite.attn_implementation
    log(f"\n{'=' * 70}\n  SFT {suite.model_name} --mode {key}: {parallelism_config.summary()}\n{'=' * 70}")

    num_train, num_eval = mode.num_samples or suite.num_samples
    train_dataset = suite.dataset(num_train, tokenizer, seed=SEED)
    eval_dataset = suite.dataset(num_eval, tokenizer, seed=SEED + EVAL_SEED_OFFSET) if num_eval else None

    if mode.adapters is None:
        model, _ = load_distributed_model(
            model_name_or_path=suite.model_name,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            use_liger_kernel=suite.use_liger_kernel,
            reset_sinks=mode.reset_sinks,
        )
        peft_config = None
    else:
        model, _, peft_config = load_peft_model(
            parallelism_config=parallelism_config,
            model_name=suite.model_name,
            attn_implementation=attn_implementation,
            use_liger_kernel=suite.use_liger_kernel,
            reset_sinks=mode.reset_sinks,
            **mode.adapters,
        )
    log(f"  Loaded {type(model).__name__} ({attn_implementation}), GPU memory {gpu_mem_gb():.1f}GB")

    sft_args = {**BASE_SFT_ARGS, **suite.sft_args, **mode.sft_args}
    max_steps = sft_args["max_steps"]
    if suite.evaluate:
        sft_args |= {"eval_strategy": "steps", "eval_steps": max_steps}
    config = SFTConfig(output_dir=os.path.join(ctx.output_dir, key), **sft_args)
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
        data_collator=build_sft_collator(tokenizer, config, model.config, parallelism_config),
        peft_config=peft_config,
    )

    checks = parallel_shape_checks(model, parallelism_config)
    after_train = mode.before_train(trainer) if mode.before_train is not None else None
    if suite.evaluate:
        barrier()
        log(f"  Initial eval loss: {trainer.evaluate().get('eval_loss', math.nan):.4f}")
    barrier()
    train_result = trainer.train()
    checks |= training_run_checks(
        train_result,
        trainer,
        max_steps,
        loss_band=suite.loss_band,
        grad_norms=True,
        loss_decreased=suite.loss_decreased,
    )
    if suite.evaluate:
        barrier()
        final_eval_loss = trainer.evaluate().get("eval_loss", math.nan)
        checks["final_eval_loss_finite"] = math.isfinite(final_eval_loss)
        log(f"  Final eval loss: {final_eval_loss:.4f}")
    for extra_checks in (suite.extra_checks, mode.extra_checks, after_train):
        if extra_checks is not None:
            checks |= extra_checks(trainer)

    trainer.cleanup_ep()
    return {f"{key}_{name}": ok for name, ok in checks.items()}


def load_suite_tokenizer(ctx, model_name: str):
    # Registry-wide: a finalizer bound to a trainer would keep each mode's model alive into the next.
    ctx.on_teardown(destroy_all_dispatchers)
    ensure_model_downloaded(model_name, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def run_sft_suite(ctx, suite: SFTSuite, modes: Mapping[str, SFTMode], mode: str) -> dict:
    """Run ``mode`` (``all`` runs every mode in order) and return the harness result."""
    selected = list(modes) if mode == "all" else [mode]
    log(f"\nSFT suite: {suite.model_name}, world {ctx.world_size}, {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"Modes: {selected}")

    tokenizer = load_suite_tokenizer(ctx, suite.model_name)
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
    """Checks on the metrics an SFT run logs, past what :func:`training_run_checks` reads. ``max_grad_norm`` is the
    ceiling a missing TP reduction blows past; a skipped eval must not pass. Token-accuracy checks apply only when
    logged."""
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
