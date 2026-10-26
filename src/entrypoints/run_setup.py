"""Run setup before any model loads: the ``output_dir`` guard, caches, log verbosity, tracking env,
seed, resume detection and the ``run.log`` console tee. Read-side work follows
``DIST_INPUT_SHARED_FILESYSTEM``, write-side ``DIST_OUTPUT_SHARED_FILESYSTEM``.
"""

import atexit
import contextlib
import hashlib
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import datasets
import transformers
from transformers import set_seed

from src.checkpoint.format import REFERENCE_CACHE_DIR_NAME
from src.checkpoint.parallel.resume_source import resolve_resume_checkpoint, resolve_resume_weights_source
from src.configs.parser import is_true_string
from src.data.pipeline.coordinated_map import ensure_cache_dir
from src.kernels.jit_cache import anchor_jit_cache_dir, ensure_fa4_kernel_cache_env
from src.log import KEY_PREVIEW_COUNT, RUN_LOG_DIR_NAME, RUN_LOG_FILE
from src.model_loading.dtype import configure_float32_matmul_precision
from src.runtime.collectives import barrier, broadcast_from_rank0
from src.runtime.consensus import reject_across_ranks
from src.runtime.filesystem import OUTPUT_FS_PROBE_PREFIX, is_fs_load_rank, is_fs_save_rank
from src.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# Seconds allowed for tee to flush its block-buffered tail once the fds are restored at exit. It is
# draining an already-closed pipe, so a longer wait would only mask a stuck process.
_TEE_DRAIN_TIMEOUT_S = 5

# The run.log this process's fds are redirected into: a path, not a bool, since the redirect is
# one-shot and a later call naming another output_dir must be reported.
_tee_log_path: str | None = None


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
        # Toolkit scratch and logging, not training state: probe suffixes and reference-cache parents can
        # outlive a failed run, and NFS may retain live mmap inodes.
        contents = [
            item
            for item in entries
            if item != RUN_LOG_DIR_NAME and not item.startswith((OUTPUT_FS_PROBE_PREFIX, REFERENCE_CACHE_DIR_NAME))
        ]
        if contents:
            raise ValueError(
                f"Output directory '{output_dir}' already exists and is not empty. "
                f"Found {len(contents)} items: {contents[:KEY_PREVIEW_COUNT]}"
                f"{'...' if len(contents) > KEY_PREVIEW_COUNT else ''}. "
                f"Set resume_from_checkpoint to resume training, or use a new output_dir "
                f"(or clear this one) to avoid overwriting previous runs."
            )


def _validate_output_dir_across_ranks(output_dir: str) -> None:
    """COLLECTIVE: run ``_validate_output_dir`` on the FS-aware setup ranks and raise on every rank
    alike, as the ``ValueError`` callers of this guard match on."""
    validation_error: str | None = None
    if is_fs_save_rank():
        try:
            _validate_output_dir(output_dir)
        except ValueError as e:
            validation_error = str(e)
    reject_across_ranks(validation_error, "output_dir validation", exc_type=ValueError)


def _setup_tracking_env_vars(args, training_config, script_name: str) -> None:
    """COLLECTIVE: set the wandb and clearml env vars, with one run id agreed across ranks."""
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


def configure_training_run(args, training_config, script_name: str, *, overwrite_output_dir: bool) -> None:
    """COLLECTIVE: set the run up before any model loads: kernel caches, fp32 matmul precision, the
    ``output_dir`` guard, the datasets cache, log verbosity (the root level is set earlier, by
    ``parse_script_args``), tracking env vars and the seed."""
    # Both kernel caches, anchored on HF_HOME, before any model/attention import: they read their env
    # lazily at first compile, and a cold cache re-benchmarks per packed-row width (fla keys on
    # ceil(tokens/1024)) while every peer waits in the next collective.
    ensure_fa4_kernel_cache_env()
    anchor_jit_cache_dir("TRITON_CACHE_DIR", "triton_cache")

    # Process-global so every entry point gets it: the embedding script's SentenceTransformer branch
    # does not load through load_distributed_model, and the image's TF32 default collapses adjacent
    # RoPE positions past 2048 tokens.
    configure_float32_matmul_precision()

    # Config-derived → rank-uniform; a resume request defers this guard to resume_checkpoint_from_config.
    if not (getattr(training_config, "resume_from_checkpoint", None) or overwrite_output_dir):
        _validate_output_dir_across_ranks(training_config.output_dir)

    # Read-side election, unlike the per-node default the coordinated dataset ops use.
    hf_datasets_cache = ensure_cache_dir(writer_rank=is_fs_load_rank)
    datasets.config.HF_DATASETS_CACHE = Path(hf_datasets_cache)
    os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache

    # log_level is passive by default, so the main process logs at info.
    if training_config.should_log:
        transformers.logging.set_verbosity_info()
    else:
        transformers.logging.set_verbosity_error()
    logger.setLevel(training_config.get_process_log_level())
    _setup_tracking_env_vars(args, training_config, script_name)

    set_seed(training_config.seed)
    barrier()


def resume_checkpoint_from_config(training_config, *, overwrite_output_dir: bool) -> str | None:
    """COLLECTIVE: the checkpoint ``training_config.resume_from_checkpoint`` names, or ``None``.

    ``True`` (or its YAML string forms) picks the newest checkpoint in ``output_dir`` complete on every
    rank, and a path must be complete on every rank; ``None``/``False`` skips resume. Incomplete step
    directories are moved out of rotation's sight (:func:`resolve_resume_checkpoint`). Picking nothing
    means a fresh run, so the ``output_dir`` guard :func:`configure_training_run` skipped runs here.
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
        if not overwrite_output_dir:
            _validate_output_dir_across_ranks(training_config.output_dir)
    return checkpoint


def prepare_distributed_resume(
    training_config, model_config, parallelism_config, *, overwrite_output_dir: bool
) -> tuple[str | None, str]:
    """COLLECTIVE: ``(resume_checkpoint, policy_weights_source)`` before the model loads: the checkpoint
    for trainer state, and where the policy reads its weights (:func:`resolve_resume_weights_source`)."""
    checkpoint = resume_checkpoint_from_config(training_config, overwrite_output_dir=overwrite_output_dir)
    weights_source = resolve_resume_weights_source(checkpoint, model_config, parallelism_config)
    return checkpoint, weights_source


def install_log_tee(output_dir: str | None) -> None:
    """Tee the run's console output into ``<output_dir>/log/run.log``.

    Redirects fds 1 and 2 through a ``tee`` subprocess so logging, ``print``, progress bars and native
    torch/NCCL writes all land in the file while still showing live; a log directory or ``tee`` that
    cannot be set up is warned. A no-op without ``output_dir`` or off ``is_fs_save_rank``: run.log is
    an output artifact, so it follows the checkpoint writers' rank-agreed filesystem flags, and a
    per-node env override cannot leave two nodes writing into one file.
    """
    global _tee_log_path
    if not output_dir or not is_fs_save_rank():
        return

    log_dir = os.path.join(output_dir, RUN_LOG_DIR_NAME)
    log_path = os.path.abspath(os.path.join(log_dir, RUN_LOG_FILE))
    if _tee_log_path is not None:
        if log_path != _tee_log_path:
            logger.warning(
                "Console output is already tee'd to %s; this process's fds cannot be redirected a "
                "second time, so %s will NOT receive it. Run one output_dir per process.",
                _tee_log_path,
                log_path,
            )
        return

    try:
        os.makedirs(log_dir, exist_ok=True)
        tee = subprocess.Popen(["tee", "-a", log_path], stdin=subprocess.PIPE, close_fds=False)
    except OSError as exc:
        logger.warning("Console output is not persisted to %s (%s); the run continues without it.", log_path, exc)
        return
    if tee.stdin is None:
        tee.kill()
        return

    # Restored at exit so tee sees EOF and flushes its block-buffered tail (a kill would drop it).
    saved_out, saved_err = os.dup(1), os.dup(2)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(tee.stdin.fileno(), 1)
    os.dup2(tee.stdin.fileno(), 2)
    tee.stdin.close()

    def _drain_tee() -> None:
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        try:
            tee.wait(timeout=_TEE_DRAIN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            tee.kill()

    atexit.register(_drain_tee)
    _tee_log_path = log_path
