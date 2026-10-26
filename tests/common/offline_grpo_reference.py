"""Mapped-reference fixtures shared by offline GRPO storage and trainer tests, and the run-start reference
lifecycle (:func:`run_reference_lifecycle`) the non-CP full fine-tuning GPU suites run on their layouts."""

import math
import os
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import torch
import torch.distributed as dist
from accelerate import PartialState
from datasets import Dataset

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.optimizers.adamw_bf16 import AdamWBF16
from src.runtime.consensus import DeferredRankFailure
from src.runtime.filesystem import is_fs_save_rank
from src.trainers.grpo.offline.reference import OfflineGRPOReferenceMixin
from src.trainers.grpo.offline.reference_cache import ReferenceScoreCache
from src.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from tests.common.checkpoint_io import RestorePointSnapshot
from tests.common.distributed import shared_output_dir
from tests.common.offline_grpo import (
    build_offline_grpo_trainer,
    doubled_head_kl_verdict,
    full_logits_kl,
    offline_grpo_config,
    offline_grpo_dataset,
    pure_kl_batch,
    reference_oracle_rows,
    resumed_export_verdict,
    save_offline_moe_base,
    swept_reference_error,
)
from tests.common.tolerances import TOL
from tests.common.utils import (
    cleanup_memory,
    finish_phase,
    log,
    optimizer_state_matches,
    resumed_loss_deltas,
    step_losses,
)

SETTINGS = {"max_prompt_length": 8, "max_completion_length": 6, "pad_token_id": 0}
_SEED = 1337
_KL_BETA = 0.2
_STEPS = 2
_SAVE_STEP = 1


def reference_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "prompt_input_ids": [[11, 12], [21, 22, 23], [31]],
            "completion_input_ids": [[13, 14], [24], []],
            "group_id": [0, 1, 2],
        }
    )


def reference_rows() -> list[torch.Tensor]:
    return [torch.tensor([-0.25, -1.5]), torch.tensor([-0.75]), torch.empty(0)]


def mapped_scores(output_dir, dataset, rows):
    cache = ReferenceScoreCache(output_dir, dp_size=1)
    guard = DeferredRankFailure("Preparing fixture reference rows", exc_type=ValueError)
    try:
        if is_fs_save_rank():
            guard.run(lambda: cache.append_rows(0, rows))
        guard.reject()
        return cache.finish(dataset)
    except BaseException:
        cache.discard()
        raise


def attach_reference(trainer, dataset, split, rows, *, settings=SETTINGS):
    scores = mapped_scores(trainer.output_dir, dataset, rows)
    identity = trainer._reference_split_identity(dataset, split, settings)
    return trainer._attach_scored_reference_logps(dataset, split, scores, identity=identity)


def restore_reference(trainer, dataset, split, *, settings=SETTINGS):
    identity = trainer._reference_split_identity(dataset, split, settings)
    return trainer._restore_reference_logps_or_none(dataset, split, identity=identity)


class _CheckpointBase:
    def _persist_trainer_sidecars(self, checkpoint_dir):
        pass

    def _rotate_checkpoints_after_sidecars(self, trial):
        self.rotated = True


class ReferenceStorageTrainer(OfflineGRPOReferenceMixin, _CheckpointBase):
    def __init__(self, output_dir, *, checkpoint=None, step=1):
        PartialState()
        self.output_dir = output_dir
        self.args = SimpleNamespace(output_dir=output_dir, resume_from_checkpoint=None)
        self.ref_model = None
        self.state = SimpleNamespace(global_step=step)
        self.rotated = False
        self._init_reference_logps(resume_checkpoint=checkpoint)

    def save_checkpoint(self):
        self._persist_trainer_sidecars(os.path.join(str(self.output_dir), f"checkpoint-{self.state.global_step}"))
        self._rotate_checkpoints_after_sidecars(None)

    def _get_output_dir(self, trial):
        return str(self.output_dir)


class _RunDirs(NamedTuple):
    """The lifecycle's directories under the suite's shared output directory."""

    base: str
    run: str
    checkpoint: str
    continuous: str
    export: str


def _lifecycle_trainer(ctx, parallelism, source, output, train, evaluation=None, *, checkpoint=None, beta=_KL_BETA):
    """The suite's trainer on ``source``; one resumed from ``checkpoint`` must restore its reference scores,
    so a re-sweep raises."""
    args = offline_grpo_config(
        output,
        steps=_STEPS,
        save_steps=_SAVE_STEP,
        seed=_SEED,
        kl_beta=beta,
        evaluate=evaluation is not None,
        save=evaluation is not None,
    )
    if checkpoint is None:
        return build_offline_grpo_trainer(ctx, source, parallelism(), args, train, evaluation)
    with patch.object(OfflineGRPOTrainer, "_sweep_reference_logps", side_effect=AssertionError("reswept reference")):
        return build_offline_grpo_trainer(ctx, source, parallelism(), args, train, evaluation, checkpoint=checkpoint)


def _uninterrupted_run(ctx, parallelism, label, dirs, train, evaluation, expected, *, snapshot, layout_checks):
    """The uninterrupted run's checks, and what its resume is held to: the run-start reference rows, the
    step losses and the save-point capture."""
    trainer = _lifecycle_trainer(ctx, parallelism, dirs.base, dirs.run, train, evaluation)
    checks = {
        "noncp_full_ft_without_live_reference": not trainer.parallelism_config.is_cp_mode
        and trainer.ref_model is None,
        "distinct_rows_make_order_check_nonvacuous": any(not torch.equal(expected[0], row) for row in expected[2::2]),
        **layout_checks(trainer),
    }
    checks["swept_reference_matches_full_logits_in_dataset_order"] = (
        swept_reference_error(trainer.train_dataset[REF_PER_TOKEN_LOGPS_COLUMN], expected) < TOL.logprob_atol
    )
    references = {split: storage.values.clone() for split, storage in trainer._reference_storage_by_split.items()}
    batch = pure_kl_batch(trainer)
    checks["independent_kl_oracle_is_nonzero"], checks["kl_matches_independent_full_logits"] = doubled_head_kl_verdict(
        trainer, batch, lambda model: full_logits_kl(model, batch, _KL_BETA), f"{label} KL"
    )
    saved = snapshot("save", trainer, capture_optimizer=True)
    trainer.add_callback(saved)
    result = trainer.train()
    losses = step_losses(trainer)
    checks["train_and_evaluate_with_kl"] = (
        result.global_step == _STEPS
        and math.isfinite(result.training_loss)
        and math.isfinite(trainer.evaluate()["eval_loss"])
    )
    checks["production_bf16_optimizer"] = isinstance(
        getattr(trainer.optimizer, "optimizer", trainer.optimizer), AdamWBF16
    )
    trainer.save_model(dirs.continuous)
    checks["reference_sidecar_saved"] = os.path.isfile(os.path.join(dirs.checkpoint, REFERENCE_LOGPS_FILE))
    captured = saved.captured or {}
    finish_phase(trainer)
    saved.trainer = None
    return checks, references, losses, captured


def _resumed_run(ctx, parallelism, label, dirs, train, evaluation, references, losses, captured, *, restore_checks):
    """The run resumed from the save step: references restored without a re-sweep, optimizer, scheduler and
    losses replayed exactly, and its export bit-equal to the uninterrupted run's. Returns checks and metrics."""
    resumed = _lifecycle_trainer(
        ctx, parallelism, dirs.checkpoint, dirs.run, train, evaluation, checkpoint=dirs.checkpoint
    )
    checks = restore_checks(resumed, captured)
    for name, split in (("train", "training"), ("eval", "evaluation")):
        checks[f"{name}_reference_restored_bit_exact"] = torch.equal(
            references[split], resumed._reference_storage_by_split[split].values
        )
    restored = RestorePointSnapshot("train_begin", resumed, capture_optimizer=True)
    resumed.add_callback(restored)
    resumed.train(resume_from_checkpoint=dirs.checkpoint)
    resumed_state = restored.captured or {}
    saved_optimizer = captured.get("optimizer") or {"state": {}}
    optimizer_exact, reason = optimizer_state_matches(saved_optimizer, resumed_state.get("optimizer") or {"state": {}})
    checks["optimizer_restore_has_state_coverage"] = bool(saved_optimizer["state"])
    checks["optimizer_restored_bit_exact"] = optimizer_exact
    if not optimizer_exact:
        log(f"{label} optimizer restore: {reason}")
    checks["scheduler_and_step_restored"] = (
        resumed_state.get("global_step") == _SAVE_STEP and resumed_state.get("sched_last_epoch") == _SAVE_STEP
    )
    deltas = resumed_loss_deltas(losses, step_losses(resumed), save_step=_SAVE_STEP, total_steps=_STEPS)
    checks["resumed_loss_matches_continuous"] = bool(deltas) and max(deltas) < TOL.replayed_resume_loss_abs
    checks["resumed_evaluation_finite"] = math.isfinite(resumed.evaluate()["eval_loss"])
    resumed.save_model(dirs.export)
    dist.barrier()
    if ctx.rank == 0:
        (
            checks["resumed_export_bit_exact"],
            checks["resumed_step_updates_weights"],
            checks["hf_export_loads_and_scores"],
        ) = resumed_export_verdict(dirs.continuous, dirs.export, dirs.checkpoint, ctx.device)
    metrics = ctx.metrics(resumed)
    finish_phase(resumed)
    restored.trainer = None
    return checks, metrics


def run_reference_lifecycle(
    ctx,
    parallelism,
    label,
    *,
    save_base=save_offline_moe_base,
    snapshot=RestorePointSnapshot,
    layout_checks=lambda trainer: {},
    restore_checks=lambda resumed, captured: {},
) -> dict:
    """COLLECTIVE. A non-CP full fine-tune's run-start KL scores through train, resume, evaluation and HF
    export, held to a full-logits oracle: the ``gpu_test_main`` body of the layout suites.

    ``parallelism()`` builds each trainer's ``ParallelismConfig`` and ``save_base(path, seed)`` writes the
    base checkpoint on rank 0. A layout adds its own checks on the uninterrupted trainer
    (``layout_checks(trainer)``) and on the resumed one against the save point ``snapshot`` captured
    (``restore_checks(resumed, captured)``); ``label`` names the layout in the log.
    """
    shared = shared_output_dir(ctx)
    run = os.path.join(shared, "train")
    dirs = _RunDirs(
        base=os.path.join(shared, "base"),
        run=run,
        checkpoint=os.path.join(run, f"checkpoint-{_SAVE_STEP}"),
        continuous=os.path.join(shared, "continuous"),
        export=os.path.join(shared, "export"),
    )
    if ctx.rank == 0:
        save_base(dirs.base, _SEED)
    dist.barrier()
    train, evaluation = offline_grpo_dataset(16), offline_grpo_dataset(8, 3)
    expected = reference_oracle_rows(
        _lifecycle_trainer(ctx, parallelism, dirs.base, os.path.join(shared, "oracle"), train, beta=0.0)
    )
    checks, references, losses, captured = _uninterrupted_run(
        ctx, parallelism, label, dirs, train, evaluation, expected, snapshot=snapshot, layout_checks=layout_checks
    )
    cleanup_memory()
    resumed_checks, metrics = _resumed_run(
        ctx, parallelism, label, dirs, train, evaluation, references, losses, captured, restore_checks=restore_checks
    )
    return {"checks": ctx.broadcast_checks(checks | resumed_checks), "metrics": metrics}
