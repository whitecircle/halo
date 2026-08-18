"""Checkpoint save / resume and the sidecars the base Trainer knows nothing about.

Routes every write and every restore through :mod:`src.distributed.checkpoint` (the saver ladder,
the loader, the per-rank optimizer shard store) and adds what a parallel run needs on top:
the LR-scheduler and router-balancing-bias sidecars, rotation deferred until they are on disk, and a
parallelism-aware best-model load.

Mixed into :class:`~src.trainers.mixins.base.DistributedTrainerMixin` — a mixin class because the context
factories read the trainer's live state. Its zero-arg ``super()`` calls must resolve past every
sibling mixin to the base Trainer; ``tests/cpu/trainers/test_mixin_composition.py`` fails if one
ever interposes.
"""

import inspect
import os
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.logging import get_logger
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # Only for accelerate isinstance checks
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, rotate_checkpoints

from src.checkpoint.config_export import (
    checkpoint_source_ref,
    finalize_exported_config,
)
from src.checkpoint.format import (
    DEFAULT_MAX_SHARD_SIZE,
    OPTIMIZER_STATE_FILES,
    ROUTER_BALANCING_BIASES_FILE,
    SCHEDULER_STATE_FILE,
)
from src.distributed.checkpoint.context import CheckpointContext, CheckpointLoadContext
from src.distributed.checkpoint.coordination import consensus_read
from src.distributed.checkpoint.loader import CheckpointLoader
from src.distributed.checkpoint.optimizer import OptimizerShardStore
from src.distributed.checkpoint.peft import PeftAdapterSaver, find_peft_model
from src.distributed.checkpoint.save import save_checkpoint
from src.distributed.expert_parallel.expert_weights import has_ep_lora
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.runtime import (
    DeferredRankFailure,
    barrier,
    barrier_on_exit,
    broadcast_from_rank0,
    fs_aware_makedirs,
    fs_aware_save_rank,
    get_global_rank,
    get_global_world_size,
    is_global_main_process,
    rank_consensus,
)
from src.models.loading.config_levels import config_export_ready, restore_special_token_ids
from src.models.loading.tokenizer_setup import pristine_model_max_length
from src.models.moe_balancing import BALANCING_BIASES_ATTR
from src.models.patches.gpt_oss_sinks import gpt_oss_sinks_restored

logger = get_logger(__name__, log_level="info")


def _model_arg_tolerant(load_fn):
    """Adapt a base ``_load_from_checkpoint`` to the loader's 2-arg call.

    HF's Trainer takes ``(resume_from_checkpoint, model=None)``; sentence-transformers' base
    (``EmbeddingTrainer``'s MRO) takes the checkpoint alone, so the loader's uniform 2-arg call
    would TypeError every embedding resume. Bind ``model`` only where the target accepts it.
    """
    if "model" in inspect.signature(load_fn).parameters:
        return load_fn
    return lambda resume_from_checkpoint, model=None: load_fn(resume_from_checkpoint)


class CheckpointingMixin:
    """Checkpoint save, resume and sidecars for :class:`DistributedTrainerMixin`."""

    # Set by :meth:`_mark_model_save_collectives_done` at the end of every ``save_model``; read by
    # :meth:`_save_checkpoint` to decide whether a recorded failure may be deferred to the collective
    # rejection or must raise on this rank now.
    _model_save_collectives_done: bool = False

    def _checkpoint_load_context(self) -> CheckpointLoadContext:
        """Capture the current model/optimizer/scheduler for a resume path (rebuilt per call so the
        references are current). Uses the bare ``_fsdp_wrapped`` so accelerate-managed FSDP falls
        through to the base Trainer."""
        config = self.parallelism_config
        return CheckpointLoadContext(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            parallelism_config=config,
            is_pp_mode=config.is_pp_mode,
            is_cp_mode=config.is_cp_mode,
            is_tp_mode=config.is_tp_mode,
            has_ep_layers=self._has_ep_layers,
            fsdp_wrapped=self._fsdp_wrapped,
            tp_rank=self._get_tp_rank(),
            tp_size=config.tp_size,
            super_load_from_checkpoint=_model_arg_tolerant(super()._load_from_checkpoint),
            super_load_optimizer_and_scheduler=super()._load_optimizer_and_scheduler,
        )

    def _checkpoint_loader(self) -> CheckpointLoader:
        """The weight-restore half of a resume."""
        return CheckpointLoader(self._checkpoint_load_context())

    def _optimizer_store(self) -> OptimizerShardStore:
        """The per-rank optimizer-shard + LR-scheduler half, over the same context."""
        return OptimizerShardStore(self._checkpoint_load_context())

    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: nn.Module = None, *, for_best_model: bool = False
    ) -> None:
        """Parallelism-aware weight resume — delegates to the checkpoint loader."""
        self._checkpoint_loader().load_model(resume_from_checkpoint, model, for_best_model=for_best_model)
        # Balancing biases live outside the model checkpoint; without this a resumed run re-imbalances from zero.
        self._restore_router_balancing_biases(resume_from_checkpoint)

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        """FSDP2 optimizer-shard + LR-scheduler resume — delegates to the checkpoint loader.

        The flag mutes the step-pre-hooks for the duration: torch's ``set_optimizer_state_dict``
        materializes empty state via ``_init_optim_state`` — a zero-LR ``optimizer.step()`` — which
        would otherwise run the deferred EP sweep at the restored ``global_step`` and make the first
        real step's sweep trip its own non-idempotency guard.
        """
        self._restoring_optimizer_state = True
        try:
            return self._optimizer_store().load(checkpoint)
        finally:
            self._restoring_optimizer_state = False

    def _load_best_model(self) -> None:
        """``load_best_model_at_end`` — through the checkpoint loader, like every other resume path.

        The base implementation does a plain ``load_state_dict(..., strict=False)``. Under EP/CP/PP
        the live model's names are EP-fused, CP-wrapper-prefixed or stage-local, so every key lands in
        ``unexpected_keys``, ``strict=False`` swallows them, and the run logs "Loading best model"
        while keeping the LAST weights — which ``save_model`` then exports as the best. Silent, at the
        end of a multi-day run. Routing it here reuses the same name/shape handling as resume.
        """
        checkpoint = self.state.best_model_checkpoint
        if not checkpoint:
            return
        logger.info(f"Loading best model from {checkpoint} (parallelism-aware)")
        self._load_from_checkpoint(checkpoint, for_best_model=True)

    def _save_checkpoint(self, model, trial):
        """Save a checkpoint, handling sharded optimizer states correctly.

        Under mixin FSDP2 (including EP/CP, whose non-expert params are FSDP2 DTensors and expert
        params plain per-rank tensors) and pure TP, the base Trainer would save only rank 0's
        optimizer view — replace its optimizer.pt with per-rank shards + the topology-fingerprint
        meta (see :meth:`OptimizerShardStore.save`). Non-sharded modes (single
        process, replicated DDP — incl. ep_size==1 grouped-GEMM MoE without FSDP2) keep the base
        Trainer's optimizer.pt, whose replicated state round-trips as-is.

        Checkpoint rotation (the base save's LAST step — an rmtree of old checkpoints) is
        neutralized for the ``super()`` call and re-run only once every toolkit sidecar is on
        disk: with ``save_total_limit: 1`` a preemption between the base's rotation and the
        optimizer-shard writes would otherwise leave exactly one checkpoint with no optimizer
        state. For the same reason the base's rank-0 optimizer.pt stays in place until its
        replacement shards are written.
        """
        # Do NOT force save_only_model for EP/CP here — it drops scheduler.pt + RNG.
        guard = DeferredRankFailure(f"checkpoint write to step {self.state.global_step}")
        save_total_limit = self.args.save_total_limit
        self.args.save_total_limit = None  # rotation is deferred below, not dropped
        self._model_save_collectives_done = False
        try:
            guard.run(partial(super()._save_checkpoint, model, trial))
        finally:
            self.args.save_total_limit = save_total_limit
        # The fence covers the base's writer-local tail (rank-0 optimizer.pt, RNG,
        # trainer_state.json), so an ENOSPC there reaches every rank as a diagnostic instead of
        # stranding them in the next collective. It cannot cover the region before that: the base
        # save calls this class's ``save_model``, whose saver gathers are world collectives, so
        # deferring a failure raised there would take this rank to ``reject``'s all_gather while its
        # peers block in a gather it never entered. That one is re-raised rank-locally instead.
        if guard.reason is not None and not self._model_save_collectives_done:
            raise RuntimeError(
                f"Checkpoint save at step {self.state.global_step} failed on this rank before the "
                f"save's collectives completed: {guard.reason}"
            )
        guard.reject()
        # save_only_model drops scheduler.pt on every mode, re-warming the LR from step 0 on resume.
        self._persist_lr_scheduler_for_resume(trial)

        # Pure TP skips FSDP2 but keeps per-rank TP optimizer shards; one optimizer.pt clobbers them.
        pure_tp = self.parallelism_config.is_tp_mode and not self._fsdp_wrapped
        if (not self._fsdp_wrapped and not pure_tp) or self.args.save_only_model:
            self._rotate_checkpoints_after_sidecars(trial)
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self._get_output_dir(trial=trial), checkpoint_folder)

        # Sync before modifying checkpoint files (base Trainer's post-save I/O is async across ranks).
        barrier()

        self._optimizer_store().save(output_dir)

        # The base's optimizer.pt/.bin is rank 0's view alone — wrong under sharding, but deleted
        # only now, with its replacement shards + meta on disk (never a window with neither). Both
        # spellings: the resume side checks both. FS-aware save rank, not rank 0; fenced because a
        # raise on one node's writer would strand peers in the next collective.
        with barrier_on_exit():
            if fs_aware_save_rank():
                for name in OPTIMIZER_STATE_FILES:
                    incorrect_optim = os.path.join(output_dir, name)
                    if os.path.exists(incorrect_optim):
                        os.remove(incorrect_optim)

        self._rotate_checkpoints_after_sidecars(trial)

    def _rotate_checkpoints_after_sidecars(self, trial) -> None:
        """The base save's checkpoint rotation, run only after every sidecar of the new checkpoint
        is complete (see :meth:`_save_checkpoint`). Same call the base makes: the save rank(s) by
        ``args.should_save``, mtime order, the best checkpoint protected. Deliberately NOT in a
        ``finally`` — after a failed save the old checkpoints are the only good ones, and rotating
        would delete them in favor of the torn new one.
        """
        if self.args.should_save:
            rotate_checkpoints(
                output_dir=self._get_output_dir(trial=trial),
                save_total_limit=self.args.save_total_limit,
                best_model_checkpoint=self.state.best_model_checkpoint,
                use_mtime=True,
            )

    def _persist_lr_scheduler_for_resume(self, trial) -> None:
        """Write ``scheduler.pt`` even under ``save_only_model`` (which makes the base Trainer drop it).

        The structure-independent LR scheduler must resume so the schedule continues from the resumed
        step. Written on the FS-aware save rank(s); no-op without a scheduler or when save_only_model
        is False (already written).
        """
        if self.lr_scheduler is None or not getattr(self.args, "save_only_model", False):
            return
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self._get_output_dir(trial=trial), checkpoint_folder)
        with barrier_on_exit():
            if fs_aware_save_rank():
                os.makedirs(output_dir, exist_ok=True)
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_STATE_FILE))
                if is_global_main_process():
                    logger.info(f"✓ Persisted LR scheduler to {output_dir} (EP/CP resume under save_only_model)")

    def _persist_router_balancing_biases(self, output_dir: str) -> None:
        """Save DeepSeek-V3 router balancing biases so a resume continues balancing, not resets it.

        The biases are replica-identical (all-reduced every step) plain attributes absent from the
        model checkpoint; without this a resumed run re-inits them to zero and re-imbalances. Written
        on the FS-aware save rank(s); no-op when the model has no balancing router.

        Under PP each stage holds different routers, so local names are mapped through the stage's
        ``global_parameter_name`` and gathered to rank 0, which merges them and broadcasts the ONE
        merged dict under unsplit-model names; stage peers contribute replica-identical duplicates
        that merge idempotently. Gather-then-broadcast, not ``all_gather_object``: only the FS-aware
        save rank(s) consume the merge, and an all-gather makes every rank materialize one unpickled
        bias dict PER RANK (512 of them at world 512, each growing with the expert count).

        Fenced like every other sidecar write: this runs before the saver's own collectives, so
        an ENOSPC on one node's save rank would otherwise raise here while its peers walk on into
        those collectives and wait out the watchdog.
        """
        model = self._top_level_model()
        biases = {
            name: getattr(module, BALANCING_BIASES_ATTR).detach().cpu()
            for name, module in model.named_modules()
            if getattr(module, BALANCING_BIASES_ATTR, None) is not None
        }
        if self.parallelism_config.is_pp_mode:
            biases = {model.global_parameter_name(name): bias for name, bias in biases.items()}
            # Non-None on rank 0 alone — what gather_object requires, and what keeps the per-rank
            # copies on the one rank that merges them.
            gathered: list[dict | None] | None = [None] * get_global_world_size() if get_global_rank() == 0 else None
            dist.gather_object(biases, gathered, dst=0)
            merged: dict = {}
            for rank_biases in gathered or ():
                merged.update(rank_biases or {})
            # FS-aware writer: a rank-0-only write leaves nodes 1..N without the file on a non-shared
            # FS, so the merge has to reach every node's save rank.
            merged = broadcast_from_rank0(merged)
            with barrier_on_exit():
                if merged and fs_aware_save_rank():
                    torch.save(merged, os.path.join(output_dir, ROUTER_BALANCING_BIASES_FILE))
            return
        with barrier_on_exit():
            if biases and fs_aware_save_rank():
                torch.save(biases, os.path.join(output_dir, ROUTER_BALANCING_BIASES_FILE))

    def _restore_router_balancing_biases(self, checkpoint: str) -> None:
        """Restore router balancing biases written by :meth:`_persist_router_balancing_biases`.

        When the file is absent on every rank, warns if the model carries balancing biases (a
        ``bias_update`` MoE resumed from a pre-sidecar checkpoint re-imbalances with no other
        symptom) and returns. Copies each saved bias back in place; a name absent from the current
        model is skipped, and a live router missing from the file warns and keeps its zero-init
        biases. A sidecar that matches NO live router is the loud case in the other direction: the
        trained biases are dropped, which is what a weight-sync RL leg does to a ``bias_update``
        checkpoint. A saved bias whose shape differs from the live one raises (consensus'd), since
        ``copy_`` would broadcast it. The read goes through :func:`consensus_read`, so it is
        all-or-nothing across ranks: a node with a missing or torn file cannot silently keep
        zero-init biases while its peers restore — a persistent cross-node routing-bias divergence.
        """
        saved, path = consensus_read(
            os.path.join(checkpoint, ROUTER_BALANCING_BIASES_FILE),
            partial(torch.load, map_location="cpu", weights_only=True),
            what=ROUTER_BALANCING_BIASES_FILE,
            checkpoint=checkpoint,
        )
        if path is None:
            model = self._top_level_model()
            if any(getattr(module, BALANCING_BIASES_ATTR, None) is not None for module in model.modules()):
                # main_process_only=False: under PP only the stages holding routers can see this.
                logger.warning(
                    f"[rank {get_global_rank()}] No {ROUTER_BALANCING_BIASES_FILE} in {checkpoint} but the "
                    f"model carries router balancing biases — they stay zero-init and balancing restarts "
                    f"from scratch (checkpoint predates the balancing-bias sidecar?).",
                    main_process_only=False,
                )
            return  # no balancing biases anywhere
        model = self._top_level_model()
        restored = 0
        missing: list[str] = []
        mismatched: list[str] = []
        for name, module in model.named_modules():
            bias = getattr(module, BALANCING_BIASES_ATTR, None)
            if bias is None:
                continue
            # Mirrors the save-side remap: without it stage 0's biases match every stage's locals.
            key = model.global_parameter_name(name) if self.parallelism_config.is_pp_mode else name
            if key not in saved:
                missing.append(key)
                continue
            value = saved[key]
            # copy_ BROADCASTS, so a checkpoint from a different expert count would fill the live
            # bias with one saved entry instead of failing — silent, and it routes every token.
            if tuple(value.shape) != tuple(bias.shape):
                mismatched.append(f"{key}: saved {tuple(value.shape)} vs live {tuple(bias.shape)}")
                continue
            bias.copy_(value.to(bias.device, bias.dtype))
            restored += 1
        # Consensus'd, not raised where it was found: under PP the drift is stage-local, and a
        # rank-local raise here would strand the peers in the next collective.
        if not rank_consensus(not mismatched)[0]:
            raise RuntimeError(
                f"{ROUTER_BALANCING_BIASES_FILE} at {checkpoint} holds router balancing biases whose "
                f"shape does not match the live routers — a different expert count or router layout. "
                f"Mismatches on this rank: {mismatched[:5] or ['(reported by another rank)']}"
            )
        # A collective, entered by every rank: under PP a stage legitimately holds no router at all,
        # so only a world-wide zero means the sidecar matched nothing.
        any_restored = rank_consensus(restored > 0)[1]
        if saved and not any_restored and not missing:
            # main_process_only=False: the balancing route is resolved per model, and under PP only
            # the router-holding stages could see it at all.
            logger.warning(
                f"[rank {get_global_rank()}] {path} carries {len(saved)} trained router balancing "
                f"bias(es) but no live router in the world accepts one, so they are DROPPED. This run's balancing "
                f"mode resolved to none — weight-sync RL downgrades both bias modes, since the sync "
                f"ships parameters only. Families whose bias exports into a native checkpoint slot "
                f"keep routing as trained (frozen); side-buffer families re-balance from zero, and "
                f"this run's own saves emit no sidecar to resume from.",
                main_process_only=False,
            )
        if missing:
            # main_process_only=False: under PP the miss can be stage-local, invisible to rank 0.
            logger.warning(
                f"[rank {get_global_rank()}] {len(missing)} router(s) absent from {path} keep zero-init "
                f"balancing biases (checkpoint from a different router layout?): "
                f"{missing[:5]}{' ...' if len(missing) > 5 else ''}",
                main_process_only=False,
            )
        if restored and is_global_main_process():
            logger.info(f"✓ Restored router balancing biases for {restored} routers from {path}")

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        """Save the model parallelism-aware through the saver ladder (``save_checkpoint``).

        PEFT models save adapters only; otherwise the per-mode saver gathers and writes the full
        model. A False return (no active parallelism) falls back to the base Trainer.

        Every writer runs under ``pristine_model_max_length``: the run's sequence budget is pinned on
        the tokenizer as a truncation default, and ``save_pretrained`` would otherwise persist it as
        the exported model's served context.
        """
        output_dir = output_dir or self.args.output_dir
        # A mid-training save can land while the FSDP2 modules still hold the transient UNSHARDED
        # params a forward registered — objects the optimizer never stepped. Every writer below then
        # reads the wrong tensors, and the PEFT path additionally MIS-ROUTES on them: its
        # DTensor probe sees plain tensors, takes the rank-0 ``save_pretrained`` branch, and PEFT's
        # shared-tensor scan raises on the first param whose storage that branch cannot address.
        reshard_fsdp2_modules(self._top_level_model())
        fs_aware_makedirs(output_dir)

        restore_special_token_ids(getattr(self, "_pristine_special_token_ids", []))
        self._persist_router_balancing_biases(output_dir)

        ctx = self._checkpoint_context()
        with pristine_model_max_length(ctx.tokenizer):
            peft_model = find_peft_model(ctx.model)
            # merge_expert_lora_on_save wants a merged base, so it takes the mode's saver even with adapters.
            # Under accelerate FSDP v1 adapters are flat-param shards, so the base Trainer's save owns that layout.
            wants_merged_base = ctx.has_expert_lora and ctx.merge_expert_lora_on_save
            if peft_model is not None and not ctx.accelerate_manages_fsdp and not wants_merged_base:
                PeftAdapterSaver().save(ctx, peft_model, output_dir)
            elif not save_checkpoint(ctx, output_dir):
                # The base save has no re-emission seam: restore the sinks FA2 drops to None for the
                # write, and serialize the config with its run-scoped router mutations restored (the
                # parallel paths get both through save_model_config / the gathered state dict).
                with (
                    gpt_oss_sinks_restored(ctx.model),
                    config_export_ready(getattr(ctx.model, "config", None)),
                ):
                    super().save_model(output_dir, _internal_call=_internal_call)
                # The base save's config write lands on the ranks HF Trainer writes from; the
                # parallel paths get the same rewrites through save_model_config, and this path owes
                # the identical set — a family-less vendor config (Bailing/Ling) or a source-schema
                # family (Step-3.7) saved single-GPU/DDP is unreadable to the merge tools and the
                # pinned server otherwise, with nothing wrong at train time to show for it.
                if self.args.should_save:
                    finalize_exported_config(ctx.model.config, output_dir, source=checkpoint_source_ref(ctx.model))
        self._mark_model_save_collectives_done()

    def _mark_model_save_collectives_done(self) -> None:
        """Record that this save's collectives are behind it — the fence in :meth:`_save_checkpoint`
        defers a failure only past this mark, and re-raises anything earlier on the spot.

        EVERY ``save_model`` implementation calls it as its last step, the trainers that replace
        the mixin's included (embedding): their gathers are collectives just the same, and an
        unmarked save would downgrade a writer-local tail failure to a rank-local raise.
        ``tests/cpu/checkpoint/test_save_fence_placement.py`` fails if an override forgets.
        """
        self._model_save_collectives_done = True

    def _checkpoint_context(self) -> CheckpointContext:
        """Snapshot the trainer state a checkpoint saver needs (the one place ``self`` is read).

        Captures object references plus the resolved write gate and shard size so a saver never
        reaches into the trainer. ``model`` is the unwrapped module; a strategy needing a rank width
        reads it off ``parallelism_config``.
        """
        config = self.parallelism_config
        model = self._top_level_model()
        return CheckpointContext(
            model=model,
            parallelism_config=config,
            is_pp_mode=config.is_pp_mode,
            is_cp_mode=config.is_cp_mode,
            is_tp_mode=config.is_tp_mode,
            is_ep_tp_mode=config.is_ep_tp_mode,
            has_ep_layers=self._has_ep_layers,
            fsdp_wrapped=self._fsdp_wrapped or isinstance(self.model, FSDP),
            accelerate_manages_fsdp=self._accelerate_manages_fsdp,
            is_save_rank=fs_aware_save_rank(),
            max_shard_size=getattr(self.args, "save_max_shard_size", None) or DEFAULT_MAX_SHARD_SIZE,
            save_sharded_ep=self.save_sharded_ep,
            has_expert_lora=has_ep_lora(model),
            merge_expert_lora_on_save=config.merge_expert_lora_on_save,
            cp_wrapper=self._find_cp_wrapper(),
            tokenizer=getattr(self, "processing_class", None),
            pp_wrapper_state=self._pp_wrapper_state,
        )
