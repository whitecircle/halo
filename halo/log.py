"""Process logging: root handler, third-party suppression, CLI verbosity, warn-once and the run-log names."""

import logging
import sys
from collections.abc import Hashable
from typing import Any

# How many offending keys a message lists before truncating.
KEY_PREVIEW_COUNT = 5

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
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger().setLevel(logging.WARNING)
    for name in _SILENCED_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_training_logging() -> None:
    """Every rank logs at INFO; guard a rank-0-only line at its call with
    ``runtime.ranks.is_global_main_process()``."""
    logging.getLogger().setLevel(logging.INFO)


def info_logger(name: str) -> logging.Logger:
    """``name`` at INFO, for lines a tool must print under the WARNING root it leaves unconfigured."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def configure_cli_logging(verbose: bool = False) -> None:
    """Root level for a standalone ``scripts/`` entry point; ``force``, as the package import already configured it."""
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
    """Log at WARNING the first time ``key`` enters ``seen``; return whether it did."""
    if key in seen:
        return False
    seen.add(key)
    logger.warning(message, *args)
    return True
