"""Process logging: the root handler and third-party suppression installed at package import, CLI
verbosity for standalone tools, and the warn-once helper.

The body is stdlib-only, but importing this module executes ``src/__init__.py`` (torch,
transformers) like any other module in the package.
"""

import logging
import sys
from collections.abc import Hashable
from typing import Any

# How many offending keys a message lists before truncating.
KEY_PREVIEW_COUNT = 5

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


def info_logger(name: str) -> logging.Logger:
    """The module logger ``name`` with its own level at INFO, for a module whose INFO records matter.

    :func:`configure_root_logging` pins the root to WARNING, so a plain child logger drops every INFO
    record. Plain logging rather than accelerate's adapter: entry points that initialize no
    accelerate state (the S3 CLI, ``scripts/inference/*``) reach these modules too.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def configure_cli_logging(verbose: bool = False) -> None:
    """Set the root logger level for a standalone ``scripts/`` entry point.

    ``force=True`` is required: the package import already pinned the root logger to WARNING, and a
    plain ``basicConfig`` would be a no-op against an existing configuration. Library code does not
    call this.
    """
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
    """Log ``message`` at WARNING the first time ``key`` is seen, and return whether it did.

    ``seen`` is supplied by the caller, so the scope of "once" (per backend, per method, per shape)
    sits next to the state it guards and a test can re-arm a single entry.
    """
    if key in seen:
        return False
    seen.add(key)
    logger.warning(message, *args)
    return True
