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
from pathlib import Path

import datasets
import torch.distributed as dist
from accelerate.logging import get_logger
from transformers import set_seed

from src.checkpoint.format import REFERENCE_CACHE_DIR_NAME
from src.checkpoint.parallel.resume_source import resolve_resume_checkpoint, resolve_resume_weights_source
from src.data.pipeline.processing import ensure_cache_dir
from src.distributed.expert_parallel.deepep.dispatcher import destroy_all_dispatchers
from src.kernels.jit_cache import anchor_jit_cache_dir, ensure_fa4_kernel_cache_env
from src.log import KEY_PREVIEW_COUNT, RUN_LOG_DIR_NAME
from src.model_loading.dtype import configure_float32_matmul_precision
from src.runtime.collectives import barrier, broadcast_from_rank0
from src.runtime.consensus import reject_across_ranks
from src.runtime.filesystem import OUTPUT_FS_PROBE_PREFIX, is_fs_load_rank, is_fs_save_rank
from src.runtime.ranks import is_global_main_process
from src.training.parser import is_true_string
from src.training.run_logging import setup_logging

logger = get_logger(__name__)

# Probe suffixes and reference-cache parents can outlive a failed run; NFS may retain live mmap inodes.
_INTERNAL_PREFIXES = (OUTPUT_FS_PROBE_PREFIX, REFERENCE_CACHE_DIR_NAME)


class TrainingStoppedEarly(SystemExit):
    """A run an early stop ended: it exits non-zero, after the teardown :func:`run_training` gives a finished
    run, since the stop's verdict is rank-uniform and every peer leaves training at the same step."""


def _is_internal_artifact(name: str) -> bool:
    """Whether an ``output_dir`` entry is toolkit scratch or logging, not saved training state."""
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
                f"Found {len(contents)} items: {contents[:KEY_PREVIEW_COUNT]}"
                f"{'...' if len(contents) > KEY_PREVIEW_COUNT else ''}. "
                f"Set resume_from_checkpoint to resume training, or use a new output_dir "
                f"(or clear this one) to avoid overwriting previous runs."
            )


def _validate_output_dir_across_ranks(output_dir: str) -> None:
    """Run ``_validate_output_dir`` on the FS-aware setup ranks and raise on every rank alike.

    A rank-local raise would desync the job, so the per-node verdict goes through the cross-rank
    rejection seam, which preserves the ``ValueError`` type callers of this guard match on.
    """
    validation_error: str | None = None
    if is_fs_save_rank():
        try:
            _validate_output_dir(output_dir)
        except ValueError as e:
            validation_error = str(e)
    reject_across_ranks(validation_error, "output_dir validation", exc_type=ValueError)


def run_training(main_fn):
    """Wrap a training entry point so distributed teardown runs on the SUCCESS path and on a
    :class:`TrainingStoppedEarly` exit only.

    Usage: ``run_training(main)()`` under ``if __name__ == "__main__":``. Dispatchers are destroyed
    before the process group — a DeepEP Gin buffer outliving the group communicator faults with a
    sticky ``cudaErrorIllegalAddress`` that hides the original traceback.

    Any other exception propagates with no teardown at all. Both halves are collectives
    (``destroy_all_dispatchers`` opens with a barrier, ``destroy_process_group`` is one), and a rank
    failing mid-step — an OOM in backward, say — has peers still inside the step's own collective,
    so entering another parks the failed rank in the NCCL watchdog for the full timeout while
    torchrun sees no exit to act on. Exiting non-zero instead lets the launcher reap the peers;
    the process exit frees the communicators.
    """

    @functools.wraps(main_fn)
    def wrapper(*args, **kwargs):
        try:
            result = main_fn(*args, **kwargs)
        except TrainingStoppedEarly:
            _tear_down_distributed()
            raise
        _tear_down_distributed()
        return result

    return wrapper


def _tear_down_distributed() -> None:
    """Destroy the EP dispatchers, then the process group. Collective."""
    destroy_all_dispatchers()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def setup_training_environment(args, training_config, script_name: str) -> None:
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

    # Config-derived → rank-uniform; a resume request defers this guard to detect_resume_checkpoint.
    skip_validation = getattr(training_config, "resume_from_checkpoint", None) or getattr(
        training_config, "overwrite_output_dir", False
    )
    if not skip_validation:
        _validate_output_dir_across_ranks(training_config.output_dir)

    # Read-side election, unlike the per-node default the coordinated dataset ops use.
    hf_datasets_cache = ensure_cache_dir(writer_rank=is_fs_load_rank)
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

    ``True`` (or its YAML string forms) auto-detects the newest checkpoint in ``output_dir`` complete on
    every rank, and a path must be complete on every rank (degrading to a from-scratch run would
    overwrite the output_dir); ``None``/``False`` skips resume. Either way the incomplete step
    directories are moved out of rotation's sight (:func:`resolve_resume_checkpoint`). Auto-detecting
    nothing means the run is fresh, so the output_dir guard ``setup_training_environment`` skipped
    re-runs here.
    """
    resume = getattr(training_config, "resume_from_checkpoint", None)
    if not resume:
        return None
    auto = not isinstance(resume, str) or is_true_string(resume)
    checkpoint = resolve_resume_checkpoint(training_config.output_dir, None if auto else resume)
    if checkpoint is None:
        if is_global_main_process():
            logger.warning(
                f"resume_from_checkpoint=True but no checkpoint found in '{training_config.output_dir}'. "
                f"Starting training from scratch."
            )
        if not getattr(training_config, "overwrite_output_dir", False):
            _validate_output_dir_across_ranks(training_config.output_dir)
    return checkpoint


def prepare_distributed_resume(training_config, model_config, parallelism_config) -> tuple[str | None, str]:
    """Resolve checkpoint resume for a distributed run, before the model is loaded.

    Returns ``(resume_checkpoint, policy_weights_source)``: the checkpoint for trainer state, and the
    ``model_name_or_path`` the policy loads weights from (:func:`resolve_resume_weights_source`).
    ``model_config`` is not mutated."""
    checkpoint = detect_resume_checkpoint(training_config)
    weights_source = resolve_resume_weights_source(checkpoint, model_config, parallelism_config)
    return checkpoint, weights_source
