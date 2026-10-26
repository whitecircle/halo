"""Checkpoint save and resume orchestration: the saver ladder, :func:`load_resume_weights`, the optimizer
shard store and the sidecars, with a hook for a trainer's own and rotation deferred until they are on
disk. Its zero-arg ``super()`` calls must resolve past every sibling mixin to the base Trainer, which
``tests/cpu/trainers/test_mixin_composition.py`` checks.
"""

import os
from functools import partial

import torch.nn as nn
from accelerate.logging import get_logger
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, rotate_checkpoints

from src.checkpoint.config_export import finalize_exported_config
from src.checkpoint.hub_layout import save_pretrained_layout
from src.checkpoint.model_card import with_halo_tags
from src.checkpoint.parallel.adapters import save_resume_adapter
from src.checkpoint.parallel.completeness import (
    commit_trainer_state,
    remove_stale_completion_markers,
    trainer_state_withheld,
)
from src.checkpoint.parallel.context import CheckpointContext, CheckpointLoadContext
from src.checkpoint.parallel.optimizer import OptimizerShardStore
from src.checkpoint.parallel.resume_source import resolve_resume_checkpoint
from src.checkpoint.parallel.resume_weights import load_resume_weights
from src.checkpoint.parallel.save import save_checkpoint
from src.checkpoint.parallel.sidecars import restore_balancing_biases, save_balancing_biases, save_lr_scheduler
from src.checkpoint.weights_source import config_schema_source
from src.distributed.fsdp import reshard_fsdp2_modules
from src.model_loading.tokenizer_setup import pristine_model_max_length
from src.models.attention.gpt_oss_sinks import gpt_oss_sinks_restored
from src.models.config_access import config_export_ready, restore_config_fields
from src.models.unwrapping import base_transformers_model
from src.runtime.collectives import barrier
from src.runtime.consensus import DeferredRankFailure
from src.runtime.filesystem import fs_aware_makedirs

logger = get_logger(__name__, log_level="info")


class CheckpointingMixin:
    """Checkpoint save, resume and sidecars for :class:`DistributedTrainerMixin`."""

    # Set by :meth:`_mark_model_save_collectives_done` at the end of every ``save_model``; read by
    # :meth:`_save_checkpoint` to decide whether a recorded failure may be deferred to the collective
    # rejection or must raise on this rank now.
    _model_save_collectives_done: bool = False
    # True while :meth:`_save_checkpoint`'s base save runs ``save_model``, whose context then marks a
    # training checkpoint (tensors at their live dtype) rather than an export.
    _writing_training_checkpoint: bool = False

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        """COLLECTIVE. ``Trainer.train`` with ``resume_from_checkpoint=True`` resolved to the newest checkpoint
        complete on every rank (:func:`resolve_resume_checkpoint`), where the base would take the
        highest-numbered one whether or not its save completed."""
        if resume_from_checkpoint is True:
            resume_from_checkpoint = resolve_resume_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")
        return super().train(resume_from_checkpoint, *args, **kwargs)

    def _checkpoint_load_context(self) -> CheckpointLoadContext:
        """The current model/optimizer/scheduler for a resume path, over the bare ``_fsdp_wrapped`` so
        accelerate-managed FSDP falls through to the base Trainer."""
        return CheckpointLoadContext(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            parallelism_config=self.parallelism_config,
            has_ep_layers=self._has_ep_layers,
            fsdp_wrapped=self._fsdp_wrapped,
            tp_rank=self._get_tp_rank(),
            super_load_optimizer_and_scheduler=super()._load_optimizer_and_scheduler,
            allow_optimizer_warm_restart=self.allow_optimizer_warm_restart,
            base_owns_whole_weight_load=self._accelerate_manages_fsdp,
        )

    def _optimizer_store(self) -> OptimizerShardStore:
        """The per-rank optimizer shard store over the current load context."""
        return OptimizerShardStore(self._checkpoint_load_context())

    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: nn.Module = None, *, for_best_model: bool = False
    ) -> None:
        """Parallelism-aware weight resume (:func:`load_resume_weights`), the base Trainer's own load
        where that declines, then the state saved beside the weights."""
        ctx = self._checkpoint_load_context()
        if not load_resume_weights(ctx, resume_from_checkpoint, model, for_best_model=for_best_model):
            super()._load_from_checkpoint(resume_from_checkpoint, model)
        self._restore_state_beside_weights(resume_from_checkpoint, for_best_model=for_best_model)

    def _restore_state_beside_weights(self, checkpoint: str, *, for_best_model: bool) -> None:
        """Restore the router-balancing biases and, except for a best-model load, the trainer sidecars."""
        restore_balancing_biases(self._top_level_model(), checkpoint, is_pp_mode=self.parallelism_config.is_pp_mode)
        if not for_best_model:
            self._restore_trainer_sidecars(checkpoint)

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        """Optimizer and LR-scheduler resume through the shard store.

        The flag mutes the step-pre-hooks: the restore's zero-LR init step would otherwise run the
        deferred EP sweep at the restored ``global_step`` and trip the first real step's guard.
        """
        self._restoring_optimizer_state = True
        try:
            return self._optimizer_store().load(checkpoint)
        finally:
            self._restoring_optimizer_state = False

    def _load_best_model(self) -> None:
        """``load_best_model_at_end`` through :meth:`_load_from_checkpoint`: the base's ``strict=False`` load
        would accept every EP-fused, CP-prefixed or stage-local name as unexpected and keep the last weights."""
        checkpoint = self.state.best_model_checkpoint
        if not checkpoint:
            return
        logger.info(f"Loading best model from {checkpoint} (parallelism-aware)")
        self._load_from_checkpoint(checkpoint, for_best_model=True)

    def _save_checkpoint(self, model, trial):
        """COLLECTIVE. Save a checkpoint whose every file is on disk before it can vouch for itself.

        The base save runs with rotation and the Hub push deferred and ``trainer_state.json`` withheld;
        then come the resume adapter, the sidecars and, in sharded modes, the per-rank optimizer shards
        that replace the base's rank-0 ``optimizer.pt``; only then is the trainer state published,
        pushed and rotated (:meth:`_publish_checkpoint`). A preemption at any point leaves the previous
        checkpoint intact and this one passed over on resume. Tensors keep their live dtype
        (``CheckpointContext.training_checkpoint``), so fp32 masters are written unrounded.
        """
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self._get_output_dir(trial=trial), checkpoint_folder)
        # Any run type: a marker left in a directory this save rewrites is stale by definition.
        remove_stale_completion_markers(output_dir)
        # Do not force save_only_model for EP/CP here: it drops scheduler.pt and RNG.
        guard = DeferredRankFailure(f"checkpoint write to step {self.state.global_step}")
        save_total_limit, push_to_hub = self.args.save_total_limit, self.args.push_to_hub
        # Deferred to _publish_checkpoint, not dropped.
        self.args.save_total_limit, self.args.push_to_hub = None, False
        self._model_save_collectives_done = False
        self._writing_training_checkpoint = True
        try:
            with trainer_state_withheld(self.state):
                guard.run(partial(super()._save_checkpoint, model, trial))
        finally:
            self.args.save_total_limit, self.args.push_to_hub = save_total_limit, push_to_hub
            self._writing_training_checkpoint = False
        # The fence covers the base's writer-local tail (rank-0 optimizer.pt, RNG,
        # trainer_state.json), so an ENOSPC there reaches every rank as a diagnostic rather than
        # leaving them blocked in the next collective. It cannot cover the region before that: the
        # base save calls this class's ``save_model``, whose saver gathers are world collectives, so
        # deferring a failure raised there would take this rank to ``reject``'s all_gather while its
        # peers block in a gather it never entered. That one is re-raised rank-locally instead.
        if guard.reason is not None and not self._model_save_collectives_done:
            raise RuntimeError(
                f"Checkpoint save at step {self.state.global_step} failed on this rank before the "
                f"save's collectives completed: {guard.reason}"
            )
        guard.reject()
        self._save_merged_checkpoint_resume_adapter(output_dir)
        # save_only_model drops scheduler.pt on every mode, re-warming the LR from step 0 on resume.
        if self.args.save_only_model:
            save_lr_scheduler(self.lr_scheduler, output_dir)
        self._persist_trainer_sidecars(output_dir)
        if not self.args.save_only_model and (store := self._optimizer_store()).writes_shards:
            # Sync before modifying checkpoint files (base Trainer's post-save I/O is async across ranks).
            barrier()
            store.save(output_dir)
        self._publish_checkpoint(output_dir, trial)

    def _publish_checkpoint(self, checkpoint_dir: str, trial) -> None:
        """COLLECTIVE. Commit the complete checkpoint's trainer state, then run what the base save runs after
        it: the Hub push (which uploads the directory as it stands), then rotation."""
        commit_trainer_state(checkpoint_dir)
        if self.args.push_to_hub:
            self._push_from_checkpoint(checkpoint_dir)
        self._rotate_checkpoints_after_sidecars(trial)

    def _rotate_checkpoints_after_sidecars(self, trial) -> None:
        """The base save's rotation, run once the new checkpoint is complete; never after a failed save,
        whose old checkpoints are the only complete ones."""
        if self.args.should_save:
            rotate_checkpoints(
                output_dir=self._get_output_dir(trial=trial),
                save_total_limit=self.args.save_total_limit,
                best_model_checkpoint=self.state.best_model_checkpoint,
                use_mtime=True,
            )

    def _save_merged_checkpoint_resume_adapter(self, checkpoint_dir: str) -> None:
        """COLLECTIVE. Write a ``merge_expert_lora_on_save`` checkpoint's unmerged adapters, which it resumes from
        (:func:`~src.checkpoint.parallel.adapters.save_resume_adapter`), under ``save_only_model`` too; no-op for
        any other run. A trainer whose payload folds adapters overrides it. The final export carries none."""
        if not self.parallelism_config.merge_expert_lora_on_save:
            return
        save_resume_adapter(self._checkpoint_context(), checkpoint_dir)

    def _persist_trainer_sidecars(self, checkpoint_dir: str) -> None:
        """Write a trainer's own resume state into ``checkpoint_dir``; none by default.

        Called on every rank of every checkpoint save, before rotation. An override fences its write
        (:func:`~src.runtime.collectives.barrier_on_exit`, :class:`DeferredRankFailure`), lists its trainer
        ahead of :class:`DistributedTrainerMixin`, and reads it back in :meth:`_restore_trainer_sidecars`.
        """

    def _restore_trainer_sidecars(self, checkpoint: str) -> None:
        """Read back what :meth:`_persist_trainer_sidecars` wrote, on every rank of a resume (never a best-model
        load), through :func:`~src.checkpoint.parallel.reading.consensus_read`; nothing by default."""

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        """Save the model parallelism-aware through the saver ladder (``save_checkpoint``).

        PEFT models save adapters only; otherwise the per-mode saver gathers and writes the full
        model. A False return (no active parallelism) falls back to the base Trainer.

        Every writer runs under ``pristine_model_max_length``: the run's sequence budget is pinned on
        the tokenizer as a truncation default, and ``save_pretrained`` would otherwise persist it as
        the exported model's served context.
        """
        output_dir = output_dir or self.args.output_dir
        # A mid-training save can land while the FSDP2 modules still hold the transient unsharded
        # params a forward registered — objects the optimizer never stepped. Every writer below then
        # reads the wrong tensors, and the PEFT path additionally mis-routes on them: its DTensor
        # probe sees plain tensors, takes the rank-0 ``save_pretrained`` branch, and PEFT's
        # shared-tensor scan raises on the first param whose storage that branch cannot address.
        reshard_fsdp2_modules(self._top_level_model())
        fs_aware_makedirs(output_dir)

        restore_config_fields(getattr(self, "_pristine_special_token_ids", []))
        save_balancing_biases(self._top_level_model(), output_dir, is_pp_mode=self.parallelism_config.is_pp_mode)

        ctx = self._checkpoint_context()
        with pristine_model_max_length(ctx.tokenizer):
            self._write_model_payload(ctx, output_dir, _internal_call)
        self._mark_model_save_collectives_done()

    def _write_model_payload(self, ctx: CheckpointContext, output_dir: str, _internal_call: bool) -> None:
        """Write the model itself: the saver the ladder picks, or the base Trainer's save where none owns it.

        The one step of :meth:`save_model` a trainer with its own model layout overrides; the steps
        around it (the reshard, the special-token restore, the balancing sidecar, the max-length pin,
        the collectives mark) stay shared. Runs on every rank: the savers gather.
        """
        if not save_checkpoint(ctx, output_dir):
            # The base save has no re-emission seam: restore the sinks FA2 drops to None for the
            # write, serialize the config with its run-scoped router mutations restored, and keep its
            # stock save_pretrained from re-adding a prefix a text-only load stripped (the parallel
            # paths get all three through save_model_config / the gathered state dict).
            with (
                gpt_oss_sinks_restored(ctx.model),
                config_export_ready(getattr(ctx.model, "config", None)),
                save_pretrained_layout(base_transformers_model(ctx.model)),
            ):
                super().save_model(output_dir, _internal_call=_internal_call)
            # The base save's config write lands on the ranks HF Trainer writes from; the
            # parallel paths get the same rewrites through save_model_config, and this path owes
            # the identical set — a family-less vendor config (Bailing/Ling) or a source-schema
            # family (Step-3.7) saved single-GPU/DDP is unreadable to the merge tools and the
            # pinned server otherwise, with nothing wrong at train time to show for it.
            if self.args.should_save:
                finalize_exported_config(ctx.model.config, output_dir, source=config_schema_source(ctx.model))

    def _mark_model_save_collectives_done(self) -> None:
        """Record that this save's collectives are behind it: the fence in :meth:`_save_checkpoint`
        defers a failure only past this mark, and re-raises anything earlier on the spot.

        EVERY ``save_model`` implementation calls it as its last step, an override of the mixin's
        included: its gathers are collectives just the same, and an unmarked save would downgrade a
        writer-local tail failure to a rank-local raise.
        ``tests/cpu/checkpoint/test_save_fence_placement.py`` fails if an override forgets.
        """
        self._model_save_collectives_done = True

    def create_model_card(self, *args, tags: str | list[str] | None = None, **kwargs):
        """The base trainer's card with the Halo Hub tags beside its own.

        The card this adds them to is TRL's ``output_dir/README.md``, written before every checkpoint
        from its ``_tag_names`` alone; ``push_to_hub`` already merges the model's ``model_tags``.
        """
        return super().create_model_card(*args, tags=with_halo_tags(tags), **kwargs)

    def _checkpoint_context(self) -> CheckpointContext:
        """Snapshot the trainer state a checkpoint saver needs (:meth:`CheckpointContext.capture`)."""
        return CheckpointContext.capture(self)
