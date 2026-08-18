"""Training-environment setup: output-dir validation, HF cache wiring, seed, tracking env vars,
and checkpoint-resume detection.

Filesystem-aware: shared FS writes from global rank 0, per-node local FS from each node's local
rank 0. Read-side work follows ``DIST_INPUT_SHARED_FILESYSTEM``, write-side
``DIST_OUTPUT_SHARED_FILESYSTEM``; both fall back to the ``DIST_SHARED_FILESYSTEM`` umbrella.
"""

from __future__ import annotations

import functools
import hashlib
import os
import time
import traceback
from pathlib import Path

import datasets
import torch.distributed as dist
from accelerate.logging import get_logger
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from src.checkpoint.format import (
    ADAPTER_WEIGHT_NAMES,
    has_whole_model_weight_file,
    is_sharded_checkpoint,
)
from src.data.pipeline.processing import ensure_cache_dir
from src.distributed.expert_parallel.dispatcher import destroy_all_dispatchers
from src.distributed.filesystem import OUTPUT_FS_PROBE_PREFIX, RUN_LOG_DIR_NAME
from src.distributed.runtime import (
    barrier,
    broadcast_from_rank0,
    fs_aware_load_rank,
    fs_aware_save_rank,
    is_global_main_process,
    rank_consensus,
    reject_across_ranks,
)
from src.models.loading.dtype import configure_float32_matmul_precision
from src.models.patches.attention import anchor_jit_cache_dir, ensure_fa4_kernel_cache_env
from src.training.parser import is_true_string
from src.training.run_logging import setup_logging

logger = get_logger(__name__)

# Prefix match: the name carries a per-run suffix (``.halo_fs_probe_<ns>``). A job killed inside the
# probe window leaves the file behind, and the next launch would reject an output_dir that holds only
# toolkit-internal artifacts.
_INTERNAL_PREFIXES = (OUTPUT_FS_PROBE_PREFIX,)


def _is_internal_artifact(name: str) -> bool:
    """Whether an ``output_dir`` entry was created by the toolkit itself, not by a previous run."""
    return name == RUN_LOG_DIR_NAME or name.startswith(_INTERNAL_PREFIXES)


def _validate_output_dir(output_dir: str) -> None:
    """Raise ``ValueError`` if output_dir exists and holds anything beyond internal artifacts.

    An unreadable directory gets the same verdict, so the ``OSError`` is translated to a
    ``ValueError`` here rather than escaping as a second error type through the cross-rank seam.
    """
    if os.path.exists(output_dir):
        try:
            entries = os.listdir(output_dir)
        except OSError as e:
            raise ValueError(
                f"Output directory '{output_dir}' exists but could not be read ({e}). Fix its "
                f"permissions or mount, or use a new output_dir."
            ) from e
        contents = [item for item in entries if not _is_internal_artifact(item)]
        if contents:
            raise ValueError(
                f"Output directory '{output_dir}' already exists and is not empty. "
                f"Found {len(contents)} items: {contents[:5]}{'...' if len(contents) > 5 else ''}. "
                f"Set resume_from_checkpoint to resume training, or use a new output_dir "
                f"(or clear this one) to avoid overwriting previous runs."
            )


def _validate_output_dir_across_ranks(output_dir: str) -> None:
    """Run ``_validate_output_dir`` on the FS-aware setup ranks and raise on every rank alike.

    A rank-local raise would desync the job, so the per-node verdict goes through the cross-rank
    rejection seam, which preserves the ``ValueError`` type callers of this guard match on.
    """
    validation_error: str | None = None
    if fs_aware_save_rank():
        try:
            _validate_output_dir(output_dir)
        except ValueError as e:
            validation_error = str(e)
    reject_across_ranks(validation_error, "output_dir validation", exc_type=ValueError)


def run_training(main_fn):
    """Wrap a training entry point so distributed teardown runs on every exit path.

    Usage: ``run_training(main)()`` under ``if __name__ == "__main__":``. Dispatchers are destroyed
    before the process group: a DeepEP Gin buffer outliving the group communicator faults with a
    sticky ``cudaErrorIllegalAddress`` that hides the original traceback. The traceback is printed
    before that teardown because ``destroy_process_group`` is itself a collective — a failing rank
    blocks there while its peers are still inside it, and an uncaught exception prints only once the
    ``finally`` returns, so the error would surface as a hang.
    """

    @functools.wraps(main_fn)
    def wrapper(*args, **kwargs):
        try:
            return main_fn(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            destroy_all_dispatchers()
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()

    return wrapper


def setup_training_environment(args, training_config, script_name: str = "train") -> None:
    """Setup common training environment: logging, seeds, and environment variables."""
    # Both kernel caches, anchored on HF_HOME, before any model/attention import: they read their env
    # lazily at first compile, and a cold cache re-benchmarks per packed-row width (fla keys on
    # ceil(tokens/1024)) while every peer waits in the next collective.
    ensure_fa4_kernel_cache_env()
    anchor_jit_cache_dir("TRITON_CACHE_DIR", "triton_cache")

    # Process-global so every entry point gets it: the embedding script's SentenceTransformer branch
    # does not load through load_distributed_model, and the image's TF32 default collapses adjacent
    # RoPE positions past 2048 tokens.
    configure_float32_matmul_precision()

    # output_dir may carry per-rank strftime codes; broadcast rank 0's value so all ranks agree.
    training_config.output_dir = broadcast_from_rank0(getattr(training_config, "output_dir", None))

    # Config-derived → rank-uniform; a resume request defers this guard to detect_resume_checkpoint.
    skip_validation = getattr(training_config, "resume_from_checkpoint", None) or getattr(
        training_config, "overwrite_output_dir", False
    )
    if not skip_validation:
        _validate_output_dir_across_ranks(training_config.output_dir)

    # Read-side election, unlike the per-node default the coordinated dataset ops use.
    hf_datasets_cache = ensure_cache_dir(writer_rank=fs_aware_load_rank)
    datasets.config.HF_DATASETS_CACHE = Path(hf_datasets_cache)
    os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache

    setup_logging(logger, training_config)
    _setup_tracking_env_vars(args, training_config, script_name)

    set_seed(training_config.seed)
    barrier()


def _setup_tracking_env_vars(args, training_config, script_name: str) -> None:
    """Set experiment-tracking env vars (wandb, clearml)."""
    os.environ["WANDB_PROJECT"] = args.project_name
    os.environ["CLEARML_PROJECT"] = args.project_name

    run_name = (
        training_config.run_name.split("/")[-1]
        if training_config.run_name
        else f"{script_name}-{os.path.basename(training_config.output_dir)}"
    )
    os.environ["CLEARML_TASK"] = run_name
    # transformers passes args.run_name straight to wandb.init and ignores WANDB_NAME, so the
    # shortened name reaches wandb only through the config.
    training_config.run_name = run_name

    # Broadcast unconditionally: gating on a per-rank env var makes this a rank-divergent collective.
    run_id = (
        os.environ.get("WANDB_RUN_ID")
        or hashlib.md5(f"{training_config.output_dir}:{int(time.time())}".encode()).hexdigest()[:8]
    )
    os.environ["WANDB_RUN_ID"] = broadcast_from_rank0(run_id)


def detect_resume_checkpoint(training_config) -> str | None:
    """Detect the checkpoint to resume from, per ``training_config.resume_from_checkpoint``.

    ``True`` auto-detects the last checkpoint in ``output_dir``; ``str`` uses that path and raises
    when it does not exist (falling back to a from-scratch run would overwrite the output_dir);
    ``None``/``False`` skips resume. Rank 0 decides and broadcasts the detection error too, so every
    rank raises the same exception with its original type. Auto-detecting nothing means the run is
    fresh, so the output_dir guard ``setup_training_environment`` skipped runs here instead.
    """
    resume = getattr(training_config, "resume_from_checkpoint", None)
    if not resume:
        return None

    checkpoint = detection_error = None
    if is_global_main_process():
        try:
            checkpoint = _detect_checkpoint_path(resume, training_config.output_dir)
        except (ValueError, OSError) as e:
            detection_error = e

    checkpoint, detection_error = broadcast_from_rank0((checkpoint, detection_error))
    if detection_error is not None:
        raise detection_error

    if checkpoint is None and not getattr(training_config, "overwrite_output_dir", False):
        _validate_output_dir_across_ranks(training_config.output_dir)

    # On a non-shared FS each node reads its own trainer_state.json, so verify everywhere.
    state_on_every_rank = (
        checkpoint is None or rank_consensus(os.path.isfile(os.path.join(checkpoint, "trainer_state.json")))[0]
    )
    if not state_on_every_rank:
        raise RuntimeError(
            f"Resume checkpoint {checkpoint} is incomplete on at least one node (missing "
            f"trainer_state.json) — torn save on a non-shared filesystem. Resume from an "
            f"earlier complete checkpoint or remove the torn one."
        )

    return checkpoint


def _detect_checkpoint_path(resume, output_dir: str) -> str | None:
    """Detect checkpoint path on the main process. Raises for an explicit path that does not exist."""
    if isinstance(resume, str) and not is_true_string(resume):
        if not os.path.isdir(resume):
            raise ValueError(
                f"resume_from_checkpoint path does not exist: '{resume}'. "
                f"Fix the path, or set resume_from_checkpoint: true to auto-detect the last "
                f"checkpoint in output_dir (or remove it to start from scratch)."
            )
        return resume

    if not os.path.isdir(output_dir):
        logger.warning(
            f"resume_from_checkpoint=True but output_dir '{output_dir}' "
            f"does not exist. Starting training from scratch."
        )
        return None

    last_ckpt = get_last_checkpoint(output_dir)
    if last_ckpt is not None:
        logger.info(f"Auto-detected last checkpoint: {last_ckpt}")
        return last_ckpt

    logger.warning(
        f"resume_from_checkpoint=True but no checkpoint found in '{output_dir}'. Starting training from scratch."
    )
    return None


def _checkpoint_has_full_model_weights(checkpoint: str) -> bool:
    """Whether ``checkpoint`` holds directly-loadable full model weights.

    A sharded EP save reuses the index filename but holds only partial tensors, and that rejection
    takes precedence over any whole-model file beside it: a directory carrying both is a gathered
    save torn by a later sharded one, so its gathered half is stale.
    """
    return has_whole_model_weight_file(checkpoint) and not is_sharded_checkpoint(checkpoint)


def _classify_resume_checkpoint(checkpoint: str) -> str:
    """Classify a resume checkpoint as ``"full"`` (loadable weights), ``"adapter"`` (adapter-only),
    or ``"invalid"`` (neither, e.g. an unmerged sharded save). Pure function of the on-disk layout.

    Both adapter spellings count (``ADAPTER_WEIGHT_NAMES``): ``PeftAdapterSaver`` falls back to
    ``adapter_model.bin``, and the loader restores either."""
    if _checkpoint_has_full_model_weights(checkpoint):
        return "full"
    if any(os.path.isfile(os.path.join(checkpoint, name)) for name in ADAPTER_WEIGHT_NAMES):
        return "adapter"
    return "invalid"


def resolve_resume_weights_source(checkpoint: str | None, model_config, parallelism_config) -> str:
    """The ``model_name_or_path`` the policy model should load weights from on resume.

    The Trainer's loader skips the checkpoint weight reload under the EP wrappers (their fused params
    cannot ingest HF-format keys) and CP, and a TP model constructed from the checkpoint skips the
    re-read entirely, so those resumes return the checkpoint dir. Under the default
    ``use_grouped_gemm`` that covers every torchrun resume, dense included.

    ``model_config`` is left unchanged so reference/teacher models and the dataset-compat check still
    resolve the base path. No-resume and adapter-only cases return the base; a checkpoint that is
    neither loadable nor an adapter raises, since training would otherwise continue from the base
    weights. Decided on rank 0 and broadcast for FS-agnostic consistency.
    """
    base = model_config.model_name_or_path
    if checkpoint is None:
        return base
    # TP is listed on its own rather than riding on ``needs_ep_wrappers`` (true only via
    # ``use_grouped_gemm``), so turning that knob off still leaves TP resume working.
    if not (parallelism_config.needs_ep_wrappers or parallelism_config.is_cp_mode or parallelism_config.is_tp_mode):
        return base

    decision = _classify_resume_checkpoint(checkpoint) if is_global_main_process() else None
    decision = broadcast_from_rank0(decision)

    if decision == "invalid":
        raise ValueError(
            f"Cannot resume {parallelism_config.mode_string or 'EP/CP'} training from '{checkpoint}': "
            f"it has no directly-loadable full weights (model.safetensors[.index.json] / "
            f"pytorch_model.bin) and no adapter checkpoint (adapter_model.safetensors/.bin). "
            f"The EP/CP checkpoint loader skips "
            f"the weight reload, so training would silently continue from BASE weights. If this is a "
            f"sharded EP checkpoint, merge it first (scripts/after_training/merge_ep_shards.py) and "
            f"resume from the merged directory."
        )
    if decision == "full":
        if is_global_main_process():
            logger.info(
                f"EP/CP resume: loading policy weights from checkpoint '{checkpoint}' "
                f"(the Trainer checkpoint loader skips the EP/CP weight reload)."
            )
        return checkpoint
    # "adapter": frozen base stays at model_name_or_path; the loader restores the adapter.
    return base


def prepare_distributed_resume(training_config, model_config, parallelism_config) -> tuple[str | None, str]:
    """Resolve checkpoint resume for a distributed run, before the model is loaded.

    Returns ``(resume_checkpoint, policy_weights_source)``: the checkpoint for trainer state, and the
    ``model_name_or_path`` the policy loads weights from (:func:`resolve_resume_weights_source`).
    ``model_config`` is not mutated."""
    checkpoint = detect_resume_checkpoint(training_config)
    weights_source = resolve_resume_weights_source(checkpoint, model_config, parallelism_config)
    return checkpoint, weights_source
