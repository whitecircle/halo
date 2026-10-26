"""Process logging: the root handler and third-party suppression installed at package import, the training
run's root level, CLI verbosity for standalone tools, the warn-once helper, the key-preview count and the
run-log names under ``output_dir``."""

import logging
import sys
from collections.abc import Hashable
from typing import Any

# How many offending keys a message lists before truncating.
KEY_PREVIEW_COUNT = 5

# The run's log directory under ``output_dir``, and the console tee's file in it.
RUN_LOG_DIR_NAME = "log"
RUN_LOG_FILE = "run.log"

_SILENCED_LOGGERS = (
    "faiss",
    "faiss.loader",
    "datasets",
    "datasets.arrow_dataset",
    "datasets.fingerprint",
    "fsspec",
    "fsspec.local",
    "urllib3",
    "filelock",
    "httpcore",
    "httpx",
    "botocore",
    "boto3",
    "s3transfer",
    "aiobotocore",
    "s3fs",
)


def configure_root_logging() -> None:
    """Install the stdout root handler and pin the chatty third-party loggers to WARNING."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger().setLevel(logging.WARNING)
    for name in _SILENCED_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_training_logging() -> None:
    """Raise the root logger to INFO for a training run. Every rank logs at the root level; a line for global
    rank 0 only is guarded at its call with ``runtime.ranks.is_global_main_process()``."""
    logging.getLogger().setLevel(logging.INFO)


def info_logger(name: str) -> logging.Logger:
    """The logger ``name`` set to INFO, for lines a tool must print under the WARNING root it leaves
    unconfigured."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def configure_cli_logging(verbose: bool = False) -> None:
    """Set the root logger level for a standalone ``scripts/`` entry point (``force``: the package
    import already configured the root)."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        force=True,
    )


def warn_once(
    logger: logging.Logger,
    seen: set[Hashable],
    key: Hashable,
    message: str,
    *args: Any,
) -> bool:
    """Log ``message`` at WARNING the first time ``key`` enters the caller's ``seen``; return whether
    it did."""
    if key in seen:
        return False
    seen.add(key)
    logger.warning(message, *args)
    return True
