"""Process logging: the root handler and third-party suppression installed at package import, CLI
verbosity for standalone tools, and the warn-once idiom.

Stdlib-only in its body, so every layer can reach it, but importing it executes ``src/__init__.py``
(torch, transformers) like any other module in the package.
"""

import logging
import sys
from collections.abc import Hashable
from typing import Any

# How many offending keys a message lists before truncating — one cap for every layer.
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


def configure_cli_logging(verbose: bool = False) -> None:
    """Hand the root logger to a standalone ``scripts/`` entry point at its own level.

    ``force`` is what makes it take effect: the package import already pinned the root to WARNING,
    against which a plain ``basicConfig`` is a silent no-op. Library code never calls this.
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
    exc_info: bool = False,
) -> bool:
    """Log ``message`` at WARNING the first time ``key`` is seen, and return whether it did.

    The caller owns ``seen``, so the scope of "once" (per backend, per method, per shape) lives
    next to the state it guards and a test can re-arm exactly the entry it exercises.
    """
    if key in seen:
        return False
    seen.add(key)
    logger.warning(message, *args, exc_info=exc_info)
    return True
