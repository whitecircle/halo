"""Checkpoint weight loading on resume, the counterpart to :mod:`.save`.

:class:`CheckpointLoader` implements the weight restore paths; optimizer state and the LR scheduler
are handled by :class:`~src.distributed.checkpoint.optimizer.OptimizerShardStore`, adapters by
:func:`~src.distributed.checkpoint.peft.restore_adapters`. Resume policy only — the file reads it
drives (:class:`StreamingCheckpointReader`, :func:`read_checkpoint_key_set`) live in the format leaf,
shared with the standalone tools.
"""

from __future__ import annotations

import itertools
import logging
import os

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.tensor import DTensor, distribute_tensor

from src.checkpoint.config_export import LOADED_WEIGHTS_FROM_ATTR
from src.checkpoint.format import (
    ADAPTER_WEIGHT_NAMES,
    has_whole_model_weight_file,
    is_sharded_checkpoint,
    load_full_state_dict,
    read_checkpoint_key_set,
    read_specific_keys_from_checkpoint,
)
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.coordination import KEY_PREVIEW_COUNT, all_ranks_ok, joined_streaming_reader
from src.distributed.checkpoint.peft import restore_adapters
from src.distributed.checkpoint.save import reject_unhandled_pp_axes
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.runtime import (
    DeferredRankFailure,
    barrier,
    broadcast_from_rank0,
    is_global_main_process,
    is_multi_rank_run,
    reject_across_ranks,
)
from src.models.structure import (
    persistent_buffers,
    unwrap_framework_wrappers,
    unwrap_model,
)

logger = logging.getLogger(__name__)

# Smallest share of the live model's logical elements a checkpoint must supply to count as a resume.
# A fraction of total numel, not of the key count: MoE experts are few keys but most of the bytes, so a
# foreign expert namespace matches nearly every key while losing most of the model.
_MIN_RESUME_COVERAGE_FRACTION = 0.5


def _weights_read_from(model) -> str | None:
    """Where the live model's weights were read, or None for a from-config random init.

    ``load_distributed_model`` stamps ``_loaded_weights_from`` (None under ``init_from_scratch``,
    whose model carries the checkpoint's ``_name_or_path`` while holding random weights). A model
    constructed elsewhere (tests, user code) falls back to ``config._name_or_path``, which for any
    ``from_pretrained`` construction is also where the weights came from.
    """
    live = unwrap_model(model)
    if hasattr(live, LOADED_WEIGHTS_FROM_ATTR):
        return getattr(live, LOADED_WEIGHTS_FROM_ATTR)
    return str(getattr(getattr(live, "config", None), "_name_or_path", "")) or None


def _built_from_checkpoint(live_source: str | None, checkpoint: str) -> bool:
    """Whether a model whose weights were read from ``live_source`` was built from ``checkpoint``.

    ``realpath`` identity, so a symlinked or relative spelling of the same directory still counts;
    ``None`` (an ``init_from_scratch`` build) never does. Resolution is node-local, so callers join
    the verdict across ranks rather than branching on their own: on a non-shared filesystem a
    per-rank answer splits the world.
    """
    return live_source is not None and os.path.realpath(live_source) == os.path.realpath(checkpoint)


def _construction_whence(live_source: str | None, subject: str = "model") -> str:
    """Where ``subject``'s live weights came from, for the messages that name it."""
    if live_source is None:
        return f"this {subject} read no weights at all (init_from_scratch)"
    return f"this {subject} was loaded from '{live_source}'"


def _report_unmatched_coverage(
    mode: str, checkpoint: str, unmatched: set[str], matched_numel: int, total_numel: int
) -> None:
    """Warn about the live tensors a resume above the coverage floor leaves at their init values.

    Rank 0 only, warn-only: past the floor the remaining gaps are legitimate (a tied lm_head, a fresh
    task head), so they are named rather than raised. Below the floor the caller raises instead.
    """
    if not unmatched:
        return
    logger.warning(
        f"{mode} resume: {len(unmatched)} live tensors "
        f"({total_numel - matched_numel:,} of {total_numel:,} parameters) are absent "
        f"from {checkpoint} and keep their construction values "
        f"(first few: {sorted(unmatched)[:KEY_PREVIEW_COUNT]})."
    )


def resume_numel_coverage(model, checkpoint_keys: set[str]) -> tuple[bool, set[str], int, int]:
    """Whether ``checkpoint_keys`` covers at least half of ``model``'s live parameters on resume.

    Weighted by logical element count rather than key count: MoE experts are two keys per layer but
    most of the bytes, so a checkpoint written under a different expert namespace matches almost
    every key while losing the majority of the model's parameters. ``t.shape.numel()`` is the logical
    size, so a DTensor shard counts at full weight and the ratio is sharding-agnostic.

    Live FQNs come from named_parameters/named_buffers rather than ``model.state_dict()``, which
    under FSDP2 would reshard on the calling rank alone (this runs on rank 0 only). Non-persistent
    buffers (rotary inv_freq and friends) are excluded, since ``state_dict`` never writes them.
    ``torch.compile`` is peeled first: an ``OptimizedModule``'s parameters are named ``_orig_mod.*``,
    so every live FQN would miss every checkpoint key, while ``set_model_state_dict`` strips that
    prefix itself.

    Returns ``(coverage_ok, unmatched_names, matched_numel, total_numel)``.
    """
    model = unwrap_framework_wrappers(model)
    non_persistent = {
        f"{prefix}.{name}" if prefix else name
        for prefix, module in model.named_modules()
        for name in module._non_persistent_buffers_set
    }
    live = {
        name: tensor.shape.numel()
        for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers())
        if name not in non_persistent
    }
    matched_numel = sum(numel for name, numel in live.items() if name in checkpoint_keys)
    total_numel = sum(live.values())
    unmatched = {name for name in live if name not in checkpoint_keys}
    # Below the floor is not a legitimate resume; smaller gaps (tied lm_head, task heads) are.
    covered = matched_numel >= total_numel * _MIN_RESUME_COVERAGE_FRACTION
    return covered, unmatched, matched_numel, total_numel


class CheckpointLoader:
    """Parallelism-aware model-weight resume.

    Load paths mutate ``ctx.model`` in place. EP/CP transform the model in ``__init__``, so their
    weights are reloaded by ``load_distributed_model`` rather than here; their optimizer state
    resumes from the per-rank shards when the topology fingerprint matches
    (:class:`~src.distributed.checkpoint.optimizer.OptimizerShardStore`), and the LR scheduler from
    ``scheduler.pt``.
    """

    def __init__(self, ctx: CheckpointLoadContext):
        self.ctx = ctx

    @staticmethod
    def _reject_sharded_resume(checkpoint: str) -> None:
        """Raise uniformly (rank-0 check, broadcast) when the resume target is a per-rank EP sharded
        save: the files are intact but each holds partial tensors, which would otherwise degrade
        into the torn-checkpoint fallback."""
        if broadcast_from_rank0(is_global_main_process() and is_sharded_checkpoint(checkpoint)):
            raise ValueError(
                f"{checkpoint} is a per-rank EP-sharded checkpoint; merge it first "
                f"(scripts/after_training/merge_ep_shards.py) and resume from the merged directory."
            )

    def _needs_skip_weight_load(self) -> bool:
        """EP/CP transform the model structure, so the HF-format checkpoint weights can't be loaded
        into the transformed model — the model was already loaded correctly by load_distributed_model.
        """
        return self.ctx.has_ep_layers or self.ctx.is_cp_mode

    def load_model(self, resume_from_checkpoint: str, model=None, *, for_best_model: bool = False) -> None:
        """Load model weights, routing by mode.

        - EP/CP/EP+CP/EP+TP: skip (already loaded via load_distributed_model; HF-format keys
          incompatible with EP-fused/CP-wrapped structure).
        - FSDP2 only: set_model_state_dict() distributes full-tensor weights into DTensor params.
        - TP (pure, or TP+DP): each rank distributes the full tensors into its DTensor shards (see
          :meth:`_load_tp`).
        - PP: this rank's global-named tensors from the merged index into the stage (see
          :meth:`_load_pp_stage`).
        - Other: falls through to base Trainer.
        """
        ctx = self.ctx
        if ctx.is_pp_mode:
            # Dispatched first: a stage's re-based local names misresolve under every other path.
            return self._load_pp_stage(resume_from_checkpoint, model)
        if self._is_extras_only_checkpoint(resume_from_checkpoint, model if model is not None else ctx.model):
            # Frozen-base runs save only wrapper params; a full load misreads that as a mismatch.
            restore_adapters(
                resume_from_checkpoint, model if model is not None else ctx.model, is_cp_mode=ctx.is_cp_mode
            )
            self._restore_extra_trained_params(resume_from_checkpoint, model if model is not None else ctx.model)
            return

        if self._needs_skip_weight_load():
            live_source = _weights_read_from(model if model is not None else ctx.model)
            # Base weights only load at construction, so both refusals below turn on whether this
            # checkpoint ships base weights (an adapter-only one has none to reload). Decided on rank 0:
            # ``realpath`` resolves locally, so per-rank branching splits the world on a non-shared FS.
            has_base = False
            built_from_ckpt = False
            read_failed = False
            if is_global_main_process():
                built_from_ckpt = _built_from_checkpoint(live_source, resume_from_checkpoint)
                try:
                    has_base = bool(read_checkpoint_key_set(resume_from_checkpoint)) or is_sharded_checkpoint(
                        resume_from_checkpoint
                    )
                except Exception as e:
                    read_failed = True
                    logger.warning(f"Unreadable checkpoint key set at {resume_from_checkpoint}: {e}")
            built_from_ckpt = broadcast_from_rank0(built_from_ckpt)
            ships_base_weights = broadcast_from_rank0(has_base)
            if broadcast_from_rank0(read_failed):
                # Degrading to "no base weights" would take the adapter-only path and continue on the
                # base weights, which is the failure this guard exists to stop.
                raise RuntimeError(
                    f"Unreadable checkpoint key set at {resume_from_checkpoint} (torn index or "
                    f"safetensors header) — cannot decide whether it ships base weights. Repair or "
                    f"re-save the checkpoint, then resume."
                )

            if ships_base_weights and for_best_model:
                raise ValueError(
                    f"load_best_model_at_end cannot reload {resume_from_checkpoint} under EP/CP "
                    f"full fine-tune — base weights only load at construction, so the export "
                    f"would silently carry the LAST weights. Export the best checkpoint directly "
                    f"instead."
                )
            if ships_base_weights and not built_from_ckpt:
                whence = _construction_whence(live_source)
                raise ValueError(
                    f"EP/CP resume requires the model to be constructed FROM the checkpoint: "
                    f"{whence}, not '{resume_from_checkpoint}' — the run would silently continue on "
                    f"those weights. Launch via the training scripts (they repoint "
                    f"model_name_or_path at the checkpoint), or pass a model loaded from the "
                    f"checkpoint directory."
                )
            # Base weights came from load_distributed_model, but LoRA adapters are fresh zero-init.
            restore_adapters(
                resume_from_checkpoint, model if model is not None else ctx.model, is_cp_mode=ctx.is_cp_mode
            )
            # Wrapper-level trained params are dropped by the base-only reload too.
            self._restore_extra_trained_params(resume_from_checkpoint, model if model is not None else ctx.model)
            if is_global_main_process():
                mode = "+".join(
                    m
                    for m in [
                        ("EP" if ctx.has_ep_layers else ""),
                        ("CP" if ctx.is_cp_mode else ""),
                    ]
                    if m
                )
                logger.info(
                    f"Skipping checkpoint base-weight reload for {mode} mode "
                    f"(model already loaded via load_distributed_model). Adapters (if any) and "
                    f"trainer state are restored from {resume_from_checkpoint}."
                )
            return

        if ctx.fsdp_wrapped and not ctx.is_tp_mode:
            return self._load_fsdp2(resume_from_checkpoint, model, for_best_model=for_best_model)

        if not ctx.is_tp_mode:
            return ctx.super_load_from_checkpoint(resume_from_checkpoint, model)

        return self._load_tp(resume_from_checkpoint, model, for_best_model=for_best_model)

    def _load_tp(self, checkpoint: str, model=None, *, for_best_model: bool = False) -> None:
        """Load a gathered checkpoint into a TP model's DTensor params.

        Both TP mechanisms (HF's ``tp_plan`` styles on a dense model, and the toolkit's
        attention-only ``parallelize_module``) place every sharded param as a DTensor on the TP mesh,
        so each rank reads the checkpoint's full tensor and ``distribute_tensor``s it to the live
        param's own placements before ``copy_``, the inverse of the load's ``shard_param``
        (``tests/cpu/parallelism/test_tp_load_inverse_of_save.py``). Params TP shards by hand (GptOss
        sinks, ``model._tp_sharded_non_dtensor``) are sliced by ``tp_rank``; the rest are replicated.

        TP+DP is excluded: FSDP2's 2-D placement stacks a strided dp shard over a strided tp shard for
        packed projections, which ``distribute_tensor`` does not invert (right shape, wrong rows), so
        that layout only resumes a model constructed from the checkpoint and raises otherwise.

        The read is per-rank and streamed one tensor at a time, and ``distribute_tensor`` is
        collective-free (``src_data_rank=None``). Readability, reader construction, the key set and
        the coverage verdict are each joined across ranks before any tensor is written.
        """
        ctx = self.ctx
        if model is None:
            model = ctx.model

        self._reject_sharded_resume(checkpoint)
        checkpoint_keys: set[str] = set()
        if is_global_main_process():
            try:
                checkpoint_keys = read_checkpoint_key_set(checkpoint)
            except Exception as e:
                logger.warning(f"Torn/unreadable model checkpoint at {checkpoint}: {e}")
        if not broadcast_from_rank0(bool(checkpoint_keys)):
            logger.warning(
                f"No readable model weights (model.safetensors[.index.json] / pytorch_model.bin) "
                f"found at {checkpoint} on global rank 0, falling back to standard checkpoint "
                f"loading (all ranks)"
            )
            return ctx.super_load_from_checkpoint(checkpoint, model)

        # A model constructed from this checkpoint already holds these weights (the training scripts
        # repoint model_name_or_path at the checkpoint on resume), so the re-read is waste. Decided on
        # rank 0, since ``realpath`` resolves locally. Best-model loads must still read: the live
        # weights trained past it.
        live_source = _weights_read_from(model)
        constructed_from_ckpt = broadcast_from_rank0(_built_from_checkpoint(live_source, checkpoint))
        if not for_best_model and constructed_from_ckpt:
            if is_global_main_process():
                logger.info(f"TP resume: model was constructed from {checkpoint}; skipping the weight reload.")
            self._restore_extra_trained_params(checkpoint, model)
            return
        if ctx.fsdp_wrapped:
            if for_best_model:
                remedy = "load_best_model_at_end is unsupported under TP+DP — export the best checkpoint directly"
            else:
                whence = _construction_whence(live_source)
                remedy = (
                    f"Resume with a model constructed FROM the checkpoint ({whence}) — launch via the "
                    f"training scripts, which repoint the load"
                )
            raise RuntimeError(
                f"TP+DP cannot reload {checkpoint} into the live model: FSDP2 over TP stacks a strided "
                f"dp shard on the tp shard, a 2-D placement distribute_tensor does not invert for packed "
                f"projections (right shape, wrong rows), so the reload is refused rather than risked. "
                f"{remedy}."
            )

        unwrapped = unwrap_framework_wrappers(model)
        live: dict[str, torch.Tensor] = dict(unwrapped.named_parameters())
        live.update(persistent_buffers(unwrapped))
        hand_sliced = dict(getattr(unwrap_model(model), "_tp_sharded_non_dtensor", None) or ())

        with joined_streaming_reader(checkpoint, live, what="TP checkpoint") as reader:
            # Coverage gate: the load below writes only matching keys, so a checkpoint written for
            # another wrapper layout would leave most of the model at its base weights. The verdict is
            # joined and the matched key count agreed with rank 0's, so a per-node copy that lost keys
            # cannot pass on one rank alone.
            coverage_ok, unmatched, matched_numel, total_numel = resume_numel_coverage(model, reader.available)
            same_keys = len(reader.available) == broadcast_from_rank0(len(reader.available))
            if not all_ranks_ok(coverage_ok and same_keys):
                raise RuntimeError(
                    f"TP checkpoint resume from {checkpoint}: fewer than half of the live model's "
                    f"PARAMETERS match the checkpoint's keys on at least one rank — resume would "
                    f"silently continue from base weights. The checkpoint was written for a "
                    f"different model or wrapper layout, or a per-node copy is incomplete."
                )
            if is_global_main_process():
                _report_unmatched_coverage("TP", checkpoint, unmatched, matched_numel, total_numel)
                if unexpected := sorted(checkpoint_keys - live.keys()):
                    logger.warning(
                        f"TP resume: {len(unexpected)} checkpoint keys have no live tensor: {unexpected[:KEY_PREVIEW_COUNT]}"
                    )

            with torch.no_grad():
                for name in sorted(reader.available):
                    target = live[name]
                    data = target.data if isinstance(target, torch.nn.Parameter) else target
                    value = reader.get(name).to(data.dtype)
                    if isinstance(data, DTensor):
                        value = distribute_tensor(value, data.device_mesh, data.placements, src_data_rank=None)
                    else:
                        dim = next((d for suffix, d in hand_sliced.items() if name.endswith(suffix)), None)
                        if dim is not None:
                            value = value.chunk(ctx.tp_size, dim=dim)[ctx.tp_rank]
                    data.copy_(value)
                    del value

        if is_global_main_process():
            logger.info(f"✓ TP checkpoint loaded from {checkpoint} ({len(live)} live tensors, tp_size={ctx.tp_size})")
        barrier()

    def _load_pp_stage(self, checkpoint: str, model=None) -> None:
        """Load a PP checkpoint's global-named tensors into this rank's pipeline stage.

        The PP save wrote one complete-tensor shard per stage under the unsplit model's global names
        (merged standard HF index); the stage's ``global_parameter_name`` supplies the inverse map
        back to its local names, and only this rank's keys are read. Tensors are copied into the live
        (possibly FSDP2-sharded) params via ``distribute_tensor`` + ``copy_``, as in the adapter
        restore, so every collective stays inside this stage's own DP mesh; a
        ``broadcast_from_rank0`` ``set_model_state_dict`` would cross stages holding different layers.

        Global keys make the weights topology-independent: the same directory resumes onto a
        different ``pp_size``. The optimizer shards are stage-local and are rejected under a changed
        split by the fingerprint / ``pp_stage_partition`` gates in
        :meth:`~src.distributed.checkpoint.optimizer.OptimizerShardStore.load`.
        """
        ctx = self.ctx
        stage = unwrap_model(model if model is not None else ctx.model)
        reject_unhandled_pp_axes(ctx.parallelism_config, "resume")
        self._reject_sharded_resume(checkpoint)

        # EP experts only load at construction, so a model not built from the checkpoint resumes base
        # experts. The signal is ``_weights_read_from`` (None for a from-config init), not
        # ``config._name_or_path``, which survives a build that read no weights. Both inputs are
        # rank-local (``ep_moe_layers()`` is stage-dependent, ``realpath`` resolves per node), so
        # every rank joins the verdict before acting rather than raising inside the gate.
        live_source = _weights_read_from(stage)
        reason = None
        if stage.ep_moe_layers() and not _built_from_checkpoint(live_source, checkpoint):
            whence = _construction_whence(live_source, "stage")
            reason = (
                f"a stage's EP expert weights only load at construction, so resume requires the model "
                f"to be constructed FROM the checkpoint: {whence}, not '{checkpoint}' — its experts "
                f"would resume from BASE weights. Launch via the training scripts "
                f"(prepare_distributed_resume repoints model_name_or_path at the checkpoint), "
                f"or pass a model loaded from the checkpoint directory."
            )
        reject_across_ranks(reason, "PP+EP resume construction identity", exc_type=ValueError)

        # The stage's own map, identical to the one the save walked, so resume reads exactly what was written.
        local_by_global = stage.checkpoint_name_map()

        # Consensus first: a torn shard must raise rank-uniformly rather than mid-collective.
        # Constructing the reader is what validates the shards, so it happens ahead of both joins and
        # serves tensors afterwards, one at a time rather than the whole stage at once.
        with joined_streaming_reader(checkpoint, local_by_global, what="PP checkpoint") as reader:
            # Coverage gate: a requested key absent from the checkpoint would keep base weights.
            missing = sorted(set(local_by_global) - reader.available)
            if not all_ranks_ok(not missing):
                raise RuntimeError(
                    f"PP resume from {checkpoint}: {len(missing)} of this stage's tensors are absent "
                    f"from the checkpoint on at least one rank — resume would silently continue from "
                    f"base weights for them. First few missing: {missing[:KEY_PREVIEW_COUNT]}. Resume from a complete "
                    f"checkpoint saved under the identical pipeline topology."
                )

            live: dict[str, torch.Tensor] = dict(stage.named_parameters())
            live.update(dict(stage.named_buffers()))
            with torch.no_grad():
                # Sorted: distribute_tensor issues mesh collectives, so key order must match per stage.
                for global_name in sorted(local_by_global):
                    local_name = local_by_global[global_name]
                    target = live.get(local_name)
                    if target is None:
                        raise RuntimeError(
                            f"PP resume: stage state_dict key '{local_name}' has no live "
                            f"parameter/buffer to load into — a state-dict hook the PP loader does "
                            f"not understand."
                        )
                    value = reader.get(global_name).to(target.dtype)
                    data = target.data if isinstance(target, torch.nn.Parameter) else target
                    if isinstance(data, DTensor):
                        # Default ``src_data_rank``: the stage's mesh rank 0 broadcasts its read, so
                        # the stage's DP replicas hold one node's bytes and the collective stays inside
                        # the stage's mesh (``_load_tp`` slices per rank with ``src_data_rank=None``).
                        value = distribute_tensor(value, data.device_mesh, data.placements)
                    data.copy_(value)
                    del value

        if is_global_main_process():
            logger.info(f"✓ PP stage checkpoint loaded from {checkpoint} ({len(local_by_global)} tensors)")
        barrier()

    def _is_extras_only_checkpoint(self, resume_from_checkpoint: str, model) -> bool:
        """Whether the checkpoint contains only the model's declared extra trained params.

        Rank-uniform (rank 0 reads the key set, decision broadcast), since the verdict gates
        collective restore paths. False when the model declares no extras or the checkpoint has base
        weights.
        """
        unwrapped = unwrap_framework_wrappers(model)
        names = set(getattr(unwrapped, "_extra_checkpoint_param_names", ()) or ())
        if not names:
            return False
        verdict = False
        if is_global_main_process():
            try:
                keys = read_checkpoint_key_set(resume_from_checkpoint)
                verdict = bool(keys) and keys <= names
            except Exception as e:
                logger.warning(f"Unreadable checkpoint key set at {resume_from_checkpoint}: {e}")
        return broadcast_from_rank0(verdict)

    def _restore_extra_trained_params(self, resume_from_checkpoint: str, model) -> None:
        """Restore wrapper-level trained params that are not base-model weights (EP/CP skip path).

        A wrapper adding trained params after ``from_pretrained`` would otherwise leave them at init,
        since the base reload never sees them. Params are declared via
        ``_extra_checkpoint_param_names``; only those keys are read and broadcast into the live
        (possibly FSDP2-sharded) params. Collective-safe: rank 0 reads, every rank enters
        ``broadcast_from_rank0`` ``set_model_state_dict``. No-op when none are declared; an
        unreadable checkpoint raises rather than warns.
        """
        unwrapped = unwrap_framework_wrappers(model)
        names = tuple(getattr(unwrapped, "_extra_checkpoint_param_names", ()) or ())
        if not names:
            return

        def _read_declared_extras() -> dict[str, torch.Tensor]:
            try:
                return read_specific_keys_from_checkpoint(resume_from_checkpoint, names)
            except Exception as e:
                raise RuntimeError(
                    f"unreadable checkpoint at {resume_from_checkpoint} ({type(e).__name__}: {e}). "
                    f"{list(names)} are this run's trained state, so continuing would resume them at "
                    f"INITIALIZATION under a resumed step count, LR schedule and dataloader "
                    f"position — a silent restart. Resume from a complete checkpoint."
                ) from e

        found_extras: dict[str, torch.Tensor] = {}
        # Fenced rather than raised directly: the read is rank 0's alone and the broadcast below is
        # collective, so a lone raise would leave the peers waiting there.
        guard = DeferredRankFailure(f"extra trained param restore from {resume_from_checkpoint}")
        if is_global_main_process():
            found_extras = guard.run(_read_declared_extras) or {}
            if guard.reason is None:  # a failed read is reported uniformly by reject() below
                missing = [n for n in names if n not in found_extras]
                if missing:
                    logger.warning(
                        f"Resume: extra trained param(s) {missing} not found in {resume_from_checkpoint}; "
                        f"they stay at initialization. (Restored: {sorted(found_extras)})"
                    )
                else:
                    logger.info(
                        f"Restored extra trained param(s) {sorted(found_extras)} from {resume_from_checkpoint}."
                    )
        guard.reject()

        if is_multi_rank_run():
            # Only rank 0's dict is used under broadcast_from_rank0, but every rank must call in.
            options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False)
            set_model_state_dict(model, found_extras, options=options)
        else:
            # Unwrapped names: a torch.compile'd model's params are ``_orig_mod.*``, so no declared
            # key would match.
            named = dict(unwrapped.named_parameters())
            for key, value in found_extras.items():
                param = named.get(key)
                if param is not None:
                    param.data.copy_(value.to(device=param.device, dtype=param.dtype))
        del found_extras
        barrier()

    def _load_fsdp2(self, resume_from_checkpoint: str, model=None, *, for_best_model: bool = False) -> None:
        """Load weights into an FSDP2-wrapped model. Plain load_state_dict() fails on DTensor params;
        set_model_state_dict() distributes full-tensor weights into them.
        """
        ctx = self.ctx
        if model is None:
            model = ctx.model

        # Best-model loads run after the final eval left the tree unsharded; every branch below
        # (set_model_state_dict, the adapter restorer, the base-Trainer fallback) must write into the
        # sharded DTensors, not the transient unsharded buffers.
        reshard_fsdp2_modules(model)

        # set_model_state_dict(broadcast_from_rank0) is collective: decide presence once on rank 0.
        source_has_file = broadcast_from_rank0(has_whole_model_weight_file(resume_from_checkpoint))

        if not source_has_file:
            # PEFT runs save adapter-only checkpoints; route through the DTensor-aware restorer.
            adapter_present = broadcast_from_rank0(
                any(os.path.isfile(os.path.join(resume_from_checkpoint, name)) for name in ADAPTER_WEIGHT_NAMES)
            )
            if adapter_present:
                restore_adapters(resume_from_checkpoint, model, is_cp_mode=self.ctx.is_cp_mode)
                barrier()
                return
            logger.warning(
                f"No model weights found at {resume_from_checkpoint} on global rank 0, "
                f"falling back to standard checkpoint loading (all ranks)"
            )
            return ctx.super_load_from_checkpoint(resume_from_checkpoint, model)

        # A model already constructed from this checkpoint holds exactly these weights, so re-reading a
        # 100B+ state dict is waste. Keyed on where the weights were read (an ``init_from_scratch``
        # build matches on ``_name_or_path`` yet holds random weights) and decided on rank 0, since
        # ``realpath`` resolves locally. Best-model loads must still read: the live weights trained on.
        weights_source = _weights_read_from(model)
        constructed_from_ckpt = broadcast_from_rank0(_built_from_checkpoint(weights_source, resume_from_checkpoint))
        if not for_best_model and constructed_from_ckpt:
            if is_global_main_process():
                logger.info(f"FSDP2 resume: model was constructed from {resume_from_checkpoint}; skipping re-load.")
            self._restore_extra_trained_params(resume_from_checkpoint, model)
            return

        # Only rank 0's dict is used under broadcast_from_rank0, but every rank must call in, so
        # readability is agreed first: a rank-0 raise would leave peers in set_model_state_dict.
        self._reject_sharded_resume(resume_from_checkpoint)
        read_ok = True
        state_dict: dict = {}
        if is_global_main_process():
            try:
                # None = no weight file the loader recognizes; treated as unreadable, as on the TP path.
                state_dict = load_full_state_dict(resume_from_checkpoint, device="cpu") or {}
                read_ok = bool(state_dict)
                logger.info(f"Loading FSDP2 checkpoint from {resume_from_checkpoint} ({len(state_dict)} keys)")
            except Exception as e:
                logger.warning(f"Torn/unreadable model checkpoint at {resume_from_checkpoint}: {e}")
                read_ok = False
        if not broadcast_from_rank0(read_ok):
            # Uniform fallback: the base loader re-reads and raises on every rank.
            return ctx.super_load_from_checkpoint(resume_from_checkpoint, model)

        # Coverage gate, decided on rank 0 and broadcast: the load below is strict=False, so keys that
        # do not match the live FQNs apply as a no-op and the run continues from the base weights. Key
        # sets rather than _IncompatibleKeys: under broadcast_from_rank0 only rank 0 sees the real one,
        # and it reports a wholesale miss as `unexpected` with `missing_keys` empty. Live FQNs come
        # from named_parameters/named_buffers, not ``state_dict()``, which reshards on this rank alone.
        coverage_ok = True
        if is_global_main_process():
            coverage_ok, unmatched, matched_numel, total_numel = resume_numel_coverage(model, set(state_dict))
            if coverage_ok:
                _report_unmatched_coverage("FSDP2", resume_from_checkpoint, unmatched, matched_numel, total_numel)
        if not broadcast_from_rank0(coverage_ok):
            raise RuntimeError(
                f"FSDP2 checkpoint resume from {resume_from_checkpoint}: fewer than half of the "
                f"live model's PARAMETERS match the checkpoint's keys — resume would silently "
                f"continue from base weights. The checkpoint was written for a different model "
                f"or wrapper layout (a different expert namespace, most likely)."
            )

        options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False)
        set_model_state_dict(model, state_dict, options=options)
        del state_dict

        if is_global_main_process():
            logger.info(f"✓ FSDP2 checkpoint loaded from {resume_from_checkpoint}")

        barrier()
