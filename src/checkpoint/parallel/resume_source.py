"""Which checkpoint a run resumes from (:func:`resolve_resume_checkpoint`), and which weights its model is
built from (:func:`resolve_resume_weights_source`)."""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from functools import partial

from src.checkpoint.adapters import has_adapter_weight_file, missing_resume_adapter_reason, resume_adapter_dir
from src.checkpoint.atomic import fsync_directory
from src.checkpoint.format import CHECKPOINT_STEP_DIR_RE, INCOMPLETE_CHECKPOINTS_DIR_NAME, TRAINER_STATE_FILE
from src.checkpoint.parallel.completeness import complete_on_every_rank
from src.checkpoint.reading import has_whole_model_weight_file, is_ep_sharded_checkpoint
from src.runtime.collectives import broadcast_from_rank0, combine_across_ranks
from src.runtime.consensus import DeferredRankFailure
from src.runtime.filesystem import is_fs_save_rank
from src.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


class ResumeWeightsKind(StrEnum):
    """What a resume checkpoint's weights are: merged weights resumed from their resume adapter (or marked
    so with no adapter file), whole weights, an adapter, or none of these (an unmerged sharded save)."""

    MERGED_ADAPTER = "merged_adapter"
    MERGED_ADAPTER_MISSING = "merged_adapter_missing"
    FULL = "full"
    ADAPTER = "adapter"
    INVALID = "invalid"


def _incomplete_reason(checkpoints: list[str]) -> str:
    return (
        f"Resume checkpoint(s) {checkpoints} are incomplete on at least one node: {TRAINER_STATE_FILE}, "
        f"which a save publishes last, is missing — a save that stopped partway, or a node whose copy is "
        f"incomplete. Resume from an earlier complete checkpoint or remove the incomplete one."
    )


def _world_step_directories(output_dir: str, requested: str | None) -> list[str]:
    """COLLECTIVE. The step directory names any FS-aware save rank lists, newest first, merged on rank 0,
    which also checks ``requested`` exists; a failure raises the same exception on every rank."""
    local: list[str] | Exception = []
    if is_fs_save_rank() and os.path.isdir(output_dir):
        try:
            local = [
                name
                for name in os.listdir(output_dir)
                if CHECKPOINT_STEP_DIR_RE.match(name) and os.path.isdir(os.path.join(output_dir, name))
            ]
        except OSError as e:
            local = e

    # Returned, not raised: combine_across_ranks would re-box it as a RuntimeError, and every rank raises it
    # under its own type below.
    def merge(gathered: list[list[str] | Exception]) -> list[str] | Exception:
        failure = next((item for item in gathered if isinstance(item, Exception)), None)
        if failure is None and requested is not None and not os.path.isdir(requested):
            failure = ValueError(
                f"resume_from_checkpoint path does not exist: '{requested}'. Fix the path, or set "
                f"resume_from_checkpoint: true to auto-detect the last checkpoint in output_dir (or remove "
                f"it to start from scratch)."
            )
        if failure is not None:
            return failure
        names = {name for item in gathered for name in item}
        return sorted(names, key=lambda name: int(CHECKPOINT_STEP_DIR_RE.match(name).group(1)), reverse=True)

    merged = combine_across_ranks(local, merge, what="the checkpoint step directories")
    if isinstance(merged, Exception):
        raise merged
    return merged


def _move_into_holding(output_dir: str, checkpoints: list[str]) -> None:
    held = [path for path in checkpoints if os.path.isdir(path)]
    if not held:
        return
    holding = os.path.join(output_dir, INCOMPLETE_CHECKPOINTS_DIR_NAME)
    os.makedirs(holding, exist_ok=True)
    for path in held:
        name = os.path.basename(path)
        target, copy = os.path.join(holding, name), 1
        while os.path.exists(target):
            target, copy = os.path.join(holding, f"{name}.{copy}"), copy + 1
        os.rename(path, target)
    fsync_directory(output_dir)


def resolve_resume_checkpoint(output_dir: str, requested: str | None = None) -> str | None:
    """COLLECTIVE. The checkpoint a run resumes from, the same on every rank.

    ``requested`` must be complete on every rank; ``None`` picks the newest step directory of
    ``output_dir`` complete on every rank (``trainer_state.json``, published last, present on every
    node), or ``None`` when there is none. Raises, moving nothing, when nothing complete is found.

    Every incomplete step directory is then moved into :data:`INCOMPLETE_CHECKPOINTS_DIR_NAME`: one
    numbered above this run's next saves would otherwise be the newest by step, the order rotation falls
    back to, and rotation would keep it and delete the complete checkpoints.
    """
    names = _world_step_directories(output_dir, requested)
    candidates = [os.path.join(output_dir, name) for name in names]
    complete = [complete_on_every_rank(path) for path in candidates]
    if requested is not None:
        checkpoint = requested if complete_on_every_rank(requested) else None
        if checkpoint is None:
            raise RuntimeError(_incomplete_reason([requested]))
    else:
        checkpoint = next((path for path, ok in zip(candidates, complete, strict=True) if ok), None)
        if checkpoint is None and candidates:
            raise RuntimeError(_incomplete_reason(candidates))
    incomplete = [path for path, ok in zip(candidates, complete, strict=True) if not ok]
    if incomplete:
        # Each save rank moves its own copies out of the step-directory namespace.
        guard = DeferredRankFailure(f"Moving the incomplete checkpoints out of {output_dir}")
        if is_fs_save_rank():
            guard.run(partial(_move_into_holding, output_dir, incomplete))
        guard.reject()
    if is_global_main_process():
        if incomplete:
            logger.warning(
                f"Moved {[os.path.basename(path) for path in incomplete]} into "
                f"{os.path.join(output_dir, INCOMPLETE_CHECKPOINTS_DIR_NAME)}: {TRAINER_STATE_FILE} is missing "
                f"on at least one node, so their saves never completed."
            )
        if checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {checkpoint}")
    return checkpoint


def classify_resume_weights(checkpoint: str) -> ResumeWeightsKind:
    """What ``checkpoint``'s weights are, off its on-disk layout alone.

    The merged class is read off the marker its save writes last, since its weights are as loadable as a
    full fine-tune's. A sharded EP index outranks any whole-model file beside it: a directory carrying
    both is a gathered save torn by a later sharded one.
    """
    merged_adapter_dir = resume_adapter_dir(checkpoint)
    if merged_adapter_dir is not None:
        if has_adapter_weight_file(merged_adapter_dir):
            return ResumeWeightsKind.MERGED_ADAPTER
        return ResumeWeightsKind.MERGED_ADAPTER_MISSING
    if has_whole_model_weight_file(checkpoint) and not is_ep_sharded_checkpoint(checkpoint):
        return ResumeWeightsKind.FULL
    if has_adapter_weight_file(checkpoint):
        return ResumeWeightsKind.ADAPTER
    return ResumeWeightsKind.INVALID


def resolve_resume_weights_source(checkpoint: str | None, model_config, parallelism_config) -> str:
    """COLLECTIVE (rank 0's verdict). The ``model_name_or_path`` the policy loads weights from on resume.

    The checkpoint itself under the EP wrappers, CP and TP, whose resumes read base weights only at
    construction (under the default ``use_grouped_gemm``, every torchrun resume); the base for an adapter
    or merged-with-resume-adapter checkpoint, whose adapter the resume restores onto it (a merged
    checkpoint's own weights already hold the delta). A checkpoint neither loadable nor an adapter raises.
    ``model_config`` is left unchanged, so reference/teacher models still resolve the base.
    """
    base = model_config.model_name_or_path
    if checkpoint is None:
        return base
    # TP is listed on its own rather than riding on ``needs_ep_wrappers`` (true only via
    # ``use_grouped_gemm``), so turning that knob off still leaves TP resume working.
    if not (parallelism_config.needs_ep_wrappers or parallelism_config.is_cp_mode or parallelism_config.is_tp_mode):
        return base

    decision = classify_resume_weights(checkpoint) if is_global_main_process() else None
    decision = broadcast_from_rank0(decision)

    if decision == ResumeWeightsKind.INVALID:
        raise ValueError(
            f"Cannot resume {parallelism_config.mode_string or 'EP/CP'} training from '{checkpoint}': it has no "
            "directly-loadable full weights (model.safetensors[.index.json] / pytorch_model.bin) and no adapter "
            "checkpoint (adapter_model.safetensors/.bin). The EP/CP checkpoint loader skips the weight reload, so "
            "training would silently continue from BASE weights. If this is a sharded EP checkpoint, merge it "
            "first (scripts/after_training/merge_ep_shards.py) and resume from the merged directory."
        )
    if decision == ResumeWeightsKind.MERGED_ADAPTER_MISSING:
        # Before the policy load, which would build the whole base only for the loader to refuse.
        raise ValueError(missing_resume_adapter_reason(checkpoint))
    if decision == ResumeWeightsKind.FULL:
        if is_global_main_process():
            logger.info(
                f"EP/CP resume: loading policy weights from checkpoint '{checkpoint}' "
                f"(the Trainer checkpoint loader skips the EP/CP weight reload)."
            )
        return checkpoint
    if decision == ResumeWeightsKind.MERGED_ADAPTER and is_global_main_process():
        logger.info(
            f"Resume: '{checkpoint}' holds merged weights for serving; the policy loads the base "
            f"'{base}' and the resume restores the checkpoint's resume adapter onto it."
        )
    # ADAPTER / MERGED_ADAPTER: the frozen base stays at model_name_or_path; the resume restores the adapter.
    return base
