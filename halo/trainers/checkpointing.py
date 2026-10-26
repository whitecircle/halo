"""Checkpoint save and resume orchestration: the saver ladder, :func:`load_resume_weights`, the optimizer
shard store and the sidecars, with hooks for a trainer's own and rotation deferred until they are on disk.
Its zero-arg ``super()`` calls must resolve past every sibling mixin to the base Trainer.
"""

import logging
import os
from functools import partial

import torch.nn as nn
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, rotate_checkpoints

from halo.checkpoint.atomic import apply_umask_mode
from halo.checkpoint.config_export import finalize_exported_config
from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from halo.checkpoint.hub_layout import save_pretrained_layout
from halo.checkpoint.model_card import with_halo_tags
from halo.checkpoint.parallel.adapters import save_resume_adapter
from halo.checkpoint.parallel.completeness import (
    commit_trainer_state,
    remove_stale_completion_markers,
    trainer_state_withheld,
)
from halo.checkpoint.parallel.context import CheckpointContext, CheckpointLoadContext
from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.checkpoint.parallel.resume_source import resolve_resume_checkpoint
from halo.checkpoint.parallel.resume_weights import load_resume_weights
from halo.checkpoint.parallel.save import save_checkpoint
from halo.checkpoint.parallel.sidecars import restore_balancing_biases, save_balancing_biases, save_lr_scheduler
from halo.checkpoint.weights_source import config_schema_source
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.model_loading.tokenizer_setup import pristine_model_max_length
from halo.models.attention.gpt_oss_sinks import gpt_oss_sinks_restored
from halo.models.config_access import config_export_ready, restore_config_fields
from halo.models.unwrapping import base_transformers_model
from halo.runtime.collectives import barrier
from halo.runtime.consensus import DeferredRankFailure
from halo.runtime.filesystem import fs_aware_makedirs, is_fs_save_rank
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


class CheckpointingMixin:
    """Checkpoint save, resume and sidecars for :class:`DistributedTrainerMixin`."""

    # Whether this save's collectives are behind it, the line before which a failure raises rank-locally.
    _model_save_collectives_done: bool = False
    # True while :meth:`_save_checkpoint` runs ``save_model``: tensors keep their live dtype, unlike an export.
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

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        """Save the model through the saver ladder (``save_checkpoint``), the base Trainer's save where no
        saver owns it, under ``pristine_model_max_length``: the run's sequence budget pinned on the tokenizer
        would otherwise persist as the exported model's served context."""
        output_dir = output_dir or self.args.output_dir
        # A mid-training save can land while FSDP2 modules hold the transient unsharded params a forward
        # registered, which the optimizer never stepped: every writer would read them, and the PEFT path's
        # DTensor probe would mis-route to the rank-0 save_pretrained branch, whose shared-tensor scan raises.
        reshard_fsdp2_modules(self._top_level_model())
        fs_aware_makedirs(output_dir)

        restore_config_fields(getattr(self, "_pristine_special_token_ids", []))
        save_balancing_biases(self._top_level_model(), output_dir, is_pp_mode=self.parallelism_config.is_pp_mode)

        ctx = self._checkpoint_context()
        with pristine_model_max_length(ctx.tokenizer):
            self._write_model_payload(ctx, output_dir, _internal_call)
        self._mark_model_save_collectives_done()

    def create_model_card(self, *args, tags: str | list[str] | None = None, **kwargs):
        """The base trainer's card (TRL's ``output_dir/README.md``, written from its ``_tag_names`` alone) with
        the Halo Hub tags beside its own."""
        return super().create_model_card(*args, tags=with_halo_tags(tags), **kwargs)

    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: nn.Module = None, *, for_best_model: bool = False
    ) -> None:
        """Parallelism-aware weight resume (:func:`load_resume_weights`), the base Trainer's own load
        where that declines, then the state saved beside the weights."""
        ctx = self._weights_load_context()
        if not load_resume_weights(ctx, resume_from_checkpoint, model, for_best_model=for_best_model):
            self._load_weights_through_base(resume_from_checkpoint, model)
        self._restore_state_beside_weights(resume_from_checkpoint, for_best_model=for_best_model)

    def _weights_load_context(self) -> CheckpointLoadContext:
        """The context the weight resume reads; a trainer whose saves write a sub-module's names re-points it."""
        return self._checkpoint_load_context()

    def _load_weights_through_base(self, checkpoint: str, model: nn.Module | None) -> None:
        """The base Trainer's own weight load, where the parallelism-aware one declines."""
        super()._load_from_checkpoint(checkpoint, model)

    def _checkpoint_load_context(self) -> CheckpointLoadContext:
        """The current model/optimizer/scheduler for a resume path, over the bare ``_fsdp_wrapped`` so
        accelerate-managed FSDP falls through to the base Trainer."""
        return CheckpointLoadContext(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            parallelism_config=self.parallelism_config,
            has_ep_layers=bool(self._ep_layers()),
            fsdp_wrapped=self._fsdp_wrapped,
            tp_rank=self._get_tp_rank(),
            super_load_optimizer_and_scheduler=super()._load_optimizer_and_scheduler,
            allow_optimizer_warm_restart=self.allow_optimizer_warm_restart,
            base_owns_whole_weight_load=self._accelerate_manages_fsdp,
        )

    def _restore_state_beside_weights(self, checkpoint: str, *, for_best_model: bool) -> None:
        """Restore the router-balancing biases and, except for a best-model load, the trainer sidecars."""
        restore_balancing_biases(self._top_level_model(), checkpoint, is_pp_mode=self.parallelism_config.is_pp_mode)
        if not for_best_model:
            self._restore_trainer_sidecars(checkpoint)

    def _restore_trainer_sidecars(self, checkpoint: str) -> None:
        """Read back what :meth:`_persist_trainer_sidecars` wrote, on every rank of a resume (never a best-model
        load), through :func:`~halo.checkpoint.parallel.reading.consensus_read`; nothing by default."""

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        """Optimizer and LR-scheduler resume through the shard store, with the step-pre-hooks muted: the
        restore's zero-LR init step would otherwise run the deferred EP sweep and trip the first step's guard."""
        self._restoring_optimizer_state = True
        try:
            return self._optimizer_store().load(checkpoint)
        finally:
            self._restoring_optimizer_state = False

    def _optimizer_store(self) -> OptimizerShardStore:
        """The per-rank optimizer shard store over the current load context."""
        return OptimizerShardStore(self._checkpoint_load_context())

    def _load_best_model(self) -> None:
        """``load_best_model_at_end`` through :meth:`_load_from_checkpoint`: the base's ``strict=False`` load
        would accept every EP-fused, CP-prefixed or stage-local name as unexpected and keep the last weights."""
        checkpoint = self.state.best_model_checkpoint
        if not checkpoint:
            return
        if is_global_main_process():
            logger.info(f"Loading best model from {checkpoint} (parallelism-aware)")
        self._load_from_checkpoint(checkpoint, for_best_model=True)

    def _save_checkpoint(self, model, trial):
        """COLLECTIVE. Save a checkpoint whose every file is on disk before it can vouch for itself.

        The base save runs with rotation and the Hub push deferred and ``trainer_state.json`` withheld; then
        the resume adapter, the sidecars and the per-rank optimizer shards are written, and only then is the
        trainer state published, pushed and rotated, so a preemption leaves this one passed over on resume.
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
        # The fence defers only a failure in the base's writer-local tail (rank-0 optimizer.pt, RNG,
        # trainer_state.json). One raised inside save_model's world-collective gathers re-raises here:
        # deferred, it would take this rank to reject's all_gather while its peers block in that gather.
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

    def _save_merged_checkpoint_resume_adapter(self, checkpoint_dir: str) -> None:
        """COLLECTIVE. Write a ``merge_expert_lora_on_save`` checkpoint's unmerged adapters, which it resumes from
        (:func:`~halo.checkpoint.parallel.adapters.save_resume_adapter`), under ``save_only_model`` too; no-op for
        any other run. A trainer whose payload folds adapters overrides it. The final export carries none."""
        if not self.parallelism_config.merge_expert_lora_on_save:
            return
        save_resume_adapter(self._checkpoint_context(), checkpoint_dir)

    def _persist_trainer_sidecars(self, checkpoint_dir: str) -> None:
        """Write a trainer's own resume state into ``checkpoint_dir`` on every rank, before rotation; none by
        default. An override fences its write and reads it back in :meth:`_restore_trainer_sidecars`."""

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

    def _checkpoint_context(self) -> CheckpointContext:
        """The trainer state a checkpoint saver reads: current object references, the resolved write gate
        and shard size."""
        model = self._top_level_model()
        return CheckpointContext(
            model=model,
            parallelism_config=self.parallelism_config,
            has_ep_layers=bool(self._ep_layers()),
            toolkit_fsdp2=self._fsdp_wrapped and not self._accelerate_manages_fsdp,
            accelerate_manages_fsdp=self._accelerate_manages_fsdp,
            is_save_rank=is_fs_save_rank(),
            max_shard_size=self.save_max_shard_size or DEFAULT_MAX_SHARD_SIZE,
            save_sharded_ep=self.save_sharded_ep,
            has_expert_lora=model_has_expert_lora(model),
            cp_wrapper=self._find_cp_wrapper(),
            tokenizer=getattr(self, "processing_class", None),
            pp_wrapper_state=self._pp_wrapper_state,
            training_checkpoint=self._writing_training_checkpoint,
        )

    def _write_model_payload(self, ctx: CheckpointContext, output_dir: str, _internal_call: bool) -> None:
        """Write the model itself, on every rank: the saver the ladder picks, or the base Trainer's save where
        none owns it. The one step of :meth:`save_model` a trainer with its own model layout overrides."""
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
                apply_umask_mode(output_dir)

    def _mark_model_save_collectives_done(self) -> None:
        """Record that this save's collectives are behind it: the fence in :meth:`_save_checkpoint` defers a
        failure only past this mark. EVERY ``save_model`` implementation, an override included, calls it last:
        an unmarked save would downgrade a writer-local tail failure to a rank-local raise."""
        self._model_save_collectives_done = True
