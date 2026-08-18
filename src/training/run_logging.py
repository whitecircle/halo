"""Per-rank transformers verbosity and the ``run.log`` console tee."""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import subprocess
import sys
from typing import TYPE_CHECKING

import transformers

from src.distributed.filesystem import RUN_LOG_DIR_NAME
from src.distributed.runtime import fs_aware_save_rank

if TYPE_CHECKING:
    # Importing TrainingArguments eagerly pulls the whole TRL/peft/pandas stack; the YAML parser
    # imports this module for install_log_tee alone.
    from transformers import TrainingArguments

logger = logging.getLogger(__name__)

# The run.log this process's fds are already redirected into. Stored as a path rather than a bool
# because the redirect is process-global and one-shot: a later call naming a different output_dir
# must be reported rather than ignored.
_TEE_LOG_PATH: str | None = None

# Seconds allowed for tee to flush its block-buffered tail once the fds are restored at exit. It is
# draining an already-closed pipe, so a longer wait would only mask a stuck process.
_TEE_DRAIN_TIMEOUT_S = 5


def setup_logging(logger, training_args: TrainingArguments):
    """Set logging verbosity from training arguments (log_level is passive by default → info)."""
    if training_args.should_log:
        transformers.logging.set_verbosity_info()
    else:
        transformers.logging.set_verbosity_error()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)


def install_log_tee(output_dir: str | None) -> None:
    """Persist the run's full console output to ``<output_dir>/log/run.log``.

    Redirects fds 1 and 2 through a ``tee`` subprocess so logging, ``print``, progress bars, and
    native torch/NCCL writes all land in the file while still showing live. No-op without
    ``output_dir``, off the writer rank, or with no ``tee``.

    The writer rank is ``fs_aware_save_rank``: run.log is an output artifact, so it follows the
    checkpoint writers' predicate and their rank-agreed shared-filesystem flags
    (``init_distributed`` has already run, in ``H4ArgumentParser.parse``), so a per-node env override
    cannot leave two nodes both writing into one ``run.log``.
    """
    global _TEE_LOG_PATH
    if not output_dir or not fs_aware_save_rank():
        return

    log_dir = os.path.join(output_dir, RUN_LOG_DIR_NAME)
    log_path = os.path.abspath(os.path.join(log_dir, "run.log"))
    if _TEE_LOG_PATH is not None:
        if log_path != _TEE_LOG_PATH:
            logger.warning(
                "Console output is already tee'd to %s; this process's fds cannot be redirected a "
                "second time, so %s will NOT receive it. Run one output_dir per process.",
                _TEE_LOG_PATH,
                log_path,
            )
        return

    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        return

    try:
        tee = subprocess.Popen(["tee", "-a", log_path], stdin=subprocess.PIPE, close_fds=False)
    except (FileNotFoundError, OSError):
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
    _TEE_LOG_PATH = log_path
