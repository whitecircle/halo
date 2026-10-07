"""Checkpoint weight loading — the resume-side counterpart to :mod:`.save`.

:class:`CheckpointLoader` implements the weight restore paths; optimizer state and the LR scheduler
are handled by :class:`~src.distributed.checkpoint.optimizer.OptimizerShardStore`, adapters by
:func:`~src.distributed.checkpoint.peft.restore_adapters`. :func:`weights_read_from` and
:func:`built_from_checkpoint` are its construction-identity probe, public for the trainers whose own
restores refuse a model built from the checkpoint. Resume policy only — the file reads it
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
    has_adapter_weight_file,
    has_whole_model_weight_file,
    is_sharded_checkpoint,
    load_full_state_dict,
    missing_resume_adapter_reason,
    read_checkpoint_key_set,
    resolve_checkpoint_weights,
    resume_adapter_dir,
    resume_adapter_on_own_weights_reason,
    unmarked_merged_checkpoint_reason,
)
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.coordination import all_ranks_ok, joined_streaming_reader
from src.distributed.checkpoint.peft import find_peft_model, restore_adapters
from src.distributed.checkpoint.save import reject_unhandled_pp_axes
from src.distributed.expert_parallel.expert_weights import has_ep_lora
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.pipeline_parallel.lazy_loader import PER_NODE_PLACEMENT_REMEDY
from src.distributed.runtime import (
    barrier,
    broadcast_from_rank0,
    copy_full_tensor,
    is_global_main_process,
    reject_across_ranks,
)
from src.log import KEY_PREVIEW_COUNT
from src.models.structure import (
    persistent_buffers,
    unwrap_framework_wrappers,
    unwrap_model,
)

logger = logging.getLogger(__name__)

# Smallest share of the live model's logical elements a checkpoint must supply for a resume to be real.
# A fraction of total numel, not of the key count: MoE experts are few keys but most of the bytes, so a
# foreign expert namespace matches nearly every key while losing most of the model.
_MIN_RESUME_COVERAGE_FRACTION = 0.5


def weights_read_from(model) -> str | None:
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


def built_from_checkpoint(live_source: str | None, checkpoint: str) -> bool:
    """Whether a model whose weights were read from ``live_source`` was built from ``checkpoint``.

    ``realpath`` identity, so a symlinked or relative spelling of the same directory still counts;
    ``None`` (an ``init_from_scratch`` build) never does. Resolution is node-local, so every caller
    joins the verdict across ranks rather than branching on its own — on a non-shared filesystem a
    per-rank answer splits the world.
    """
    return live_source is not None and os.path.realpath(live_source) == os.path.realpath(checkpoint)


def _trains_adapters(model) -> bool:
    """Whether ``model`` carries adapters, attention PEFT or native EP expert LoRA (structural, so
    rank-uniform)."""
    unwrapped = unwrap_framework_wrappers(model)
    return find_peft_model(unwrapped) is not None or has_ep_lora(unwrapped)


def _construction_whence(live_source: str | None, subject: str = "model") -> str:
    """Where ``subject``'s live weights actually came from, for the refusals that name it."""
    if live_source is None:
        return f"this {subject} read no weights at all (init_from_scratch)"
    return f"this {subject} was loaded from '{live_source}'"


def _report_unmatched_coverage(
    mode: str, checkpoint: str, unmatched: set[str], matched_numel: int, total_numel: int
) -> None:
    """Warn about the live tensors a resume above the coverage floor leaves at their init values.

    Rank 0's report, warn-only: past the floor the remaining gaps are legitimate (a tied lm_head, a
    fresh task head), so they are named rather than raised — below it the caller raises instead.
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
    """Whether ``checkpoint_keys`` covers at least half of ``model``'s live PARAMETERS on resume.

    Weighted by logical element count, never by key count: MoE experts are two keys per layer but
    most of the bytes, so a checkpoint written under a different expert namespace matches almost
    every key while losing the majority of the model's parameters — a key-count gate reads that as a
    healthy resume. ``t.shape.numel()`` is the LOGICAL size, so a DTensor shard counts at full weight
    and the ratio is sharding-agnostic.

    Live FQNs from named_parameters/named_buffers, NEVER ``model.state_dict()``: under FSDP2 that
    would reshard on the calling rank alone (this runs on rank 0 only). Non-persistent buffers
    (rotary inv_freq and friends) are excluded — ``state_dict`` never writes them, so counting them
    reports a healthy resume as partially absent. ``torch.compile`` is peeled first: an
    ``OptimizedModule``'s parameters are named ``_orig_mod.*``, so every live FQN would miss every
    checkpoint key and the gate would refuse a resume ``set_model_state_dict`` handles (torch strips
    the compiler prefix itself).

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
    # Under the floor is never a legitimate resume; smaller gaps (tied lm_head, task heads) are real.
    covered = matched_numel >= total_numel * _MIN_RESUME_COVERAGE_FRACTION
    return covered, unmatched, matched_numel, total_numel


def _absent_stage_shards_reason(checkpoint: str, stage_keys, pp_rank: int) -> str | None:
    """Why this filesystem's copy of ``checkpoint`` lacks shard files holding this stage's tensors.

    Read off the merged index alone, the only layout a PP save writes. ``None`` when every such file
    is present or the index itself cannot be read, which the joined reader reports as a torn save.
    """
    try:
        layout = resolve_checkpoint_weights(checkpoint)
    except (OSError, ValueError):
        return None
    if layout.index is None:
        return None
    absent = sorted(
        {
            shard
            for key, shard in layout.weight_map.items()
            if key in stage_keys and not os.path.isfile(layout.path(shard))
        }
    )
    if not absent:
        return None
    return (
        f"PP stage {pp_rank}: {checkpoint} indexes this stage's tensors in shard file(s) absent from this "
        f"node's copy ({absent[:KEY_PREVIEW_COUNT]}). {PER_NODE_PLACEMENT_REMEDY}"
    )


class CheckpointLoader:
    """Parallelism-aware model-weight resume.

    Load paths mutate ``ctx.model`` in place. EP/CP transform the model in ``__init__``, so their
    weights are reloaded by ``load_distributed_model`` (not here); their optimizer state resumes
    from the per-rank shards when the topology fingerprint matches
    (:class:`~src.distributed.checkpoint.optimizer.OptimizerShardStore`), and the LR scheduler from
    ``scheduler.pt``.
    """

    def __init__(self, ctx: CheckpointLoadContext):
        self.ctx = ctx

    @staticmethod
    def _reject_sharded_resume(checkpoint: str) -> None:
        """Raise uniformly (rank-0 check, broadcast) when the resume target is a per-rank EP
        sharded save — intact files, partial tensors — instead of letting it degrade into the
        torn-checkpoint fallback's confusing failure."""
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
          incompatible with EP-fused/CP-wrapped structure), then restore the adapters (see
          :meth:`_restore_adapters_onto_constructed_model`).
        - FSDP2 only: set_model_state_dict() distributes full-tensor weights into DTensor params.
        - TP (pure, or TP+DP): each rank distributes the full tensors into its DTensor shards (see
          :meth:`_load_tp`).
        - PP: skipped for a stage constructed from the checkpoint, else this rank's global-named
          tensors from the merged index into the stage (see :meth:`_load_pp_stage`).
        - Other: falls through to base Trainer.
        """
        ctx = self.ctx
        if ctx.is_pp_mode:
            # Dispatched first: a stage's re-based local names misresolve under every other path.
            return self._load_pp_stage(resume_from_checkpoint, model, for_best_model=for_best_model)
        if self._needs_skip_weight_load():
            live_model = model if model is not None else ctx.model
            live_source = weights_read_from(live_model)
            # Base weights only load at construction, so both refusals below turn on whether this
            # checkpoint ships base weights (an adapter-only one has none to reload). Decided on rank 0:
            # ``realpath`` resolves locally, so per-rank branching splits the world on a non-shared FS.
            has_base = False
            built_from_ckpt = False
            read_failed = False
            merged_resume_adapter = None
            if is_global_main_process():
                built_from_ckpt = built_from_checkpoint(live_source, resume_from_checkpoint)
                merged_resume_adapter = resume_adapter_dir(resume_from_checkpoint)
                try:
                    has_base = bool(read_checkpoint_key_set(resume_from_checkpoint)) or is_sharded_checkpoint(
                        resume_from_checkpoint
                    )
                except Exception as e:
                    read_failed = True
                    logger.warning(f"Unreadable checkpoint key set at {resume_from_checkpoint}: {e}")
            built_from_ckpt = broadcast_from_rank0(built_from_ckpt)
            ships_base_weights = broadcast_from_rank0(has_base)
            merged_resume_adapter = broadcast_from_rank0(merged_resume_adapter)
            if broadcast_from_rank0(read_failed):
                # Degrading to "no base weights" would take the adapter-only path and silently continue
                # on base weights — the failure this guard exists to stop.
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
            # Base weights came from load_distributed_model, but LoRA adapters are fresh zero-init.
            self._restore_adapters_onto_constructed_model(
                resume_from_checkpoint,
                live_model,
                live_source=live_source,
                merged_resume_adapter=merged_resume_adapter,
                ships_base_weights=ships_base_weights,
                built_from_ckpt=built_from_ckpt,
            )
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

    def _restore_adapters_onto_constructed_model(
        self,
        checkpoint: str,
        model,
        *,
        live_source: str | None,
        merged_resume_adapter: str | None,
        ships_base_weights: bool,
        built_from_ckpt: bool,
    ) -> None:
        """The adapter half of an EP/CP resume, whose base weights were read at construction.

        A ``merge_expert_lora_on_save`` checkpoint (``merged_resume_adapter`` set, off its marker)
        resumes its unmerged adapters onto a model built from the BASE: its own weights already hold
        the delta, so a model built from them would take it twice. Any other checkpoint that ships
        base weights must be what the model was built from, and its adapters, if any, sit at its
        root. Every input is rank-0-decided and broadcast, and the adapter read is a consensus, so
        each refusal fires on every rank or none.
        """
        is_cp_mode = self.ctx.is_cp_mode
        if merged_resume_adapter is not None:
            if built_from_ckpt:
                raise ValueError(resume_adapter_on_own_weights_reason(checkpoint))
            if restore_adapters(merged_resume_adapter, model, is_cp_mode=is_cp_mode) is None:
                raise RuntimeError(missing_resume_adapter_reason(checkpoint))
            return
        if ships_base_weights and not built_from_ckpt:
            whence = _construction_whence(live_source)
            raise ValueError(
                f"EP/CP resume requires the model to be constructed FROM the checkpoint: "
                f"{whence}, not '{checkpoint}' — the run would silently continue on "
                f"those weights. Launch via the training scripts (they repoint "
                f"model_name_or_path at the checkpoint), or pass a model loaded from the "
                f"checkpoint directory."
            )
        restored = restore_adapters(checkpoint, model, is_cp_mode=is_cp_mode)
        if ships_base_weights and restored is None and _trains_adapters(model):
            # Only a merged save ships base weights from an adapter run.
            raise ValueError(unmarked_merged_checkpoint_reason(checkpoint))

    def _load_tp(self, checkpoint: str, model=None, *, for_best_model: bool = False) -> None:
        """Load a gathered checkpoint into a TP model's DTensor params.

        Both TP mechanisms — HF's ``tp_plan`` styles on a dense model and the toolkit's
        attention-only ``parallelize_module`` — place every sharded param as a DTensor on the TP
        mesh, so each rank reads the checkpoint's full tensor and ``distribute_tensor``s it to the
        live param's own placements before ``copy_``: per style the exact inverse of the load's
        ``shard_param`` (``tests/cpu/parallelism/test_tp_load_inverse_of_save.py``), and the route
        the PP stage load and the adapter restore take. Params TP shards by hand (GptOss sinks,
        ``model._tp_sharded_non_dtensor``) are sliced by ``tp_rank``; everything else is replicated.

        TP+DP is the exception: FSDP2's 2-D placement stacks a strided dp shard over a strided tp
        shard for packed projections, where ``distribute_tensor`` returns the right shape with the
        wrong rows — so that layout only resumes a model constructed FROM the checkpoint (the
        training scripts repoint the load) and raises otherwise.

        Per-rank read, streamed one tensor at a time: a rank slices its own shard, so it needs only
        its node's copy, and ``distribute_tensor`` is collective-free (``src_data_rank=None``).
        Readability, reader construction, the key set and the coverage verdict are each joined across
        ranks before any tensor is written, so a torn or key-losing per-node copy sends every rank to
        the same verdict instead of resuming base weights on one.
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

        # A model CONSTRUCTED from this checkpoint already holds these weights (the training scripts
        # repoint model_name_or_path at the checkpoint on resume), so the re-read is waste. Decided on
        # rank 0, since ``realpath`` resolves locally. Best-model loads must still read: the live
        # weights trained past it.
        live_source = weights_read_from(model)
        constructed_from_ckpt = broadcast_from_rank0(built_from_checkpoint(live_source, checkpoint))
        if not for_best_model and constructed_from_ckpt:
            if is_global_main_process():
                logger.info(f"TP resume: model was constructed from {checkpoint}; skipping the weight reload.")
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
            # Coverage gate: the load below writes only the keys that match, so a checkpoint written
            # for another wrapper layout would resume BASE weights for most of the model. Verdict
            # joined, and the matched key count agreed with rank 0's, so a per-node copy that lost
            # keys cannot pass on one rank alone.
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

    def _load_pp_stage(self, checkpoint: str, model=None, *, for_best_model: bool = False) -> None:
        """Load a PP checkpoint's global-named tensors into this rank's pipeline stage.

        A stage ``load_distributed_model`` constructed FROM this checkpoint on every rank already holds
        exactly these tensors: the stage-aware loader read them, configured FP32 masters at FP32, and
        the reload would cast the same stored values to the same live dtypes. The re-read is skipped
        then, as on the FSDP2 and TP paths; a best-model load always reads, since the live weights
        trained past it.

        The PP save wrote one complete-tensor shard per stage under the UNSPLIT model's global
        names (merged standard HF index); the stage's ``global_parameter_name`` supplies the
        inverse map back to its local names, and only this rank's keys are read. Tensors are
        copied into the live (possibly FSDP2-sharded) params via ``distribute_tensor`` + ``copy_``
        — the proven adapter-restore pattern — so every collective stays inside this stage's own
        DP mesh; a ``broadcast_from_rank0`` ``set_model_state_dict`` would cross stages holding
        different layers.

        Because the shards are keyed globally, the WEIGHTS are topology-independent: the same
        directory resumes onto a different ``pp_size``. The optimizer shards are not — they are
        stage-local, and the fingerprint / ``pp_stage_partition`` gates in
        :meth:`~src.distributed.checkpoint.optimizer.OptimizerShardStore.load` refuse them under a
        changed split.
        """
        ctx = self.ctx
        stage = unwrap_model(model if model is not None else ctx.model)
        reject_unhandled_pp_axes(ctx.parallelism_config, "resume")
        self._reject_sharded_resume(checkpoint)

        # EP experts only load at construction, so a model not built from the checkpoint resumes base
        # experts. The signal is ``weights_read_from`` (None for a from-config init), not
        # ``config._name_or_path``, which survives a build that read no weights. Both inputs are
        # rank-local (``ep_moe_layers()`` is stage-dependent, ``realpath`` resolves per node), so
        # every rank joins the verdict before acting rather than raising inside the gate.
        live_source = weights_read_from(stage)
        reason = None
        if stage.ep_moe_layers() and not built_from_checkpoint(live_source, checkpoint):
            whence = _construction_whence(live_source, "stage")
            reason = (
                f"a stage's EP expert weights only load at construction, so resume requires the model "
                f"to be constructed FROM the checkpoint: {whence}, not '{checkpoint}' — its experts "
                f"would resume from BASE weights. Launch via the training scripts "
                f"(prepare_distributed_resume repoints model_name_or_path at the checkpoint), "
                f"or pass a model loaded from the checkpoint directory."
            )
        reject_across_ranks(reason, "PP+EP resume construction identity", exc_type=ValueError)

        # Only load_distributed_model's own stamp vouches for an exact read: a stage built some other
        # way from the same directory may hold its FP32 masters rounded. Joined, since ``realpath``
        # resolves per node and a stage that skipped would leave its mesh peers in the reload.
        constructed_everywhere = all_ranks_ok(
            built_from_checkpoint(getattr(stage, LOADED_WEIGHTS_FROM_ATTR, None), checkpoint)
        )
        if constructed_everywhere and not for_best_model:
            if is_global_main_process():
                logger.info(f"PP resume: every stage was constructed from {checkpoint}; skipping the weight reload.")
            return

        # A best-model load follows an evaluation whose forward-only drives can leave the stage's FSDP2
        # modules unsharded; the copies below must land in the sharded DTensors.
        reshard_fsdp2_modules(stage)

        # The stage's own map, identical to the one the save walked, so resume reads exactly what was written.
        local_by_global = stage.checkpoint_name_map()
        reject_across_ranks(
            _absent_stage_shards_reason(checkpoint, local_by_global, ctx.parallelism_config.pp_rank),
            "PP resume shard placement",
        )

        # Consensus first: a torn shard must raise rank-uniformly, not strand peers mid-collective.
        # Opening the shards is what validates them, so the reader is built here, ahead of both
        # joins, and only serves tensors afterwards — one at a time, never the whole stage at once.
        with joined_streaming_reader(checkpoint, local_by_global, what="PP checkpoint") as reader:
            # Coverage gate: a requested key absent from the checkpoint would silently keep BASE weights.
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
                # Sorted: distribute_tensor issues mesh collectives — same key order on every stage rank.
                for global_name in sorted(local_by_global):
                    local_name = local_by_global[global_name]
                    target = live.get(local_name)
                    if target is None:
                        raise RuntimeError(
                            f"PP resume: stage state_dict key '{local_name}' has no live "
                            f"parameter/buffer to load into — a state-dict hook the PP loader does "
                            f"not understand."
                        )
                    # A DTensor target is a collective inside this stage's own mesh.
                    copy_full_tensor(target, reader.get(global_name))

        if is_global_main_process():
            logger.info(f"✓ PP stage checkpoint loaded from {checkpoint} ({len(local_by_global)} tensors)")
        barrier()

    def _load_fsdp2(self, resume_from_checkpoint: str, model=None, *, for_best_model: bool = False) -> None:
        """Load weights into an FSDP2-wrapped model. Plain load_state_dict() fails on DTensor params;
        set_model_state_dict() distributes full-tensor weights into them.
        """
        ctx = self.ctx
        if model is None:
            model = ctx.model

        # Best-model loads run after the final eval left the tree unsharded; every branch below
        # (set_model_state_dict, the adapter restorer, the base-Trainer fallback) must write into
        # the sharded DTensors — the source of truth — not the transient unsharded buffers.
        reshard_fsdp2_modules(model)

        # set_model_state_dict(broadcast_from_rank0) is collective — decide presence once on rank 0.
        source_has_file = broadcast_from_rank0(has_whole_model_weight_file(resume_from_checkpoint))

        if not source_has_file:
            # PEFT runs save adapter-only checkpoints; route through the DTensor-aware restorer.
            adapter_present = broadcast_from_rank0(has_adapter_weight_file(resume_from_checkpoint))
            if adapter_present:
                restore_adapters(resume_from_checkpoint, model, is_cp_mode=self.ctx.is_cp_mode)
                barrier()
                return
            logger.warning(
                f"No model weights found at {resume_from_checkpoint} on global rank 0, "
                f"falling back to standard checkpoint loading (all ranks)"
            )
            return ctx.super_load_from_checkpoint(resume_from_checkpoint, model)

        # A model already CONSTRUCTED from this checkpoint holds exactly these weights, so re-reading a
        # 100B+ state dict is waste. Keyed on where weights were actually READ (an ``init_from_scratch``
        # build matches on ``_name_or_path`` yet holds random weights) and decided on rank 0, since
        # ``realpath`` resolves locally. Best-model loads must still read: the live weights trained on.
        weights_source = weights_read_from(model)
        constructed_from_ckpt = broadcast_from_rank0(built_from_checkpoint(weights_source, resume_from_checkpoint))
        if not for_best_model and constructed_from_ckpt:
            if is_global_main_process():
                logger.info(f"FSDP2 resume: model was constructed from {resume_from_checkpoint}; skipping re-load.")
            return

        # Only rank 0's dict is used under broadcast_from_rank0, but every rank must call in — agree on
        # readability first, else a rank-0 raise strands peers in set_model_state_dict.
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
            # Uniform fallback: the base loader re-reads and raises on every rank, failing loud.
            return ctx.super_load_from_checkpoint(resume_from_checkpoint, model)

        # Coverage gate, decided on rank 0 and broadcast: the load below is strict=False, so keys that do
        # not match the live FQNs apply as a no-op and the run continues from BASE weights. Key sets, not
        # _IncompatibleKeys — under broadcast_from_rank0 only rank 0 sees the real one, and it reports a
        # wholesale miss as `unexpected` with `missing_keys` empty. Live FQNs come from
        # named_parameters/named_buffers, never ``state_dict()``, which reshards on this rank alone.
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
