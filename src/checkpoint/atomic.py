"""Staged checkpoint-file writes, with ordinary umask permissions and durable publication."""

import errno
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

import torch

FILE_STAGING_SUFFIX = ".tmp"
_LINK_COPY_ERRNOS = {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOSYS, errno.EOPNOTSUPP}


def create_staged_file(directory: str | Path, filename: str) -> Path:
    """Create an exclusive sibling under the same ``0o666``-and-umask mode as a normal file."""
    staged = Path(directory) / f".{filename}.{uuid.uuid4().hex}{FILE_STAGING_SUFFIX}"
    os.close(os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
    return staged


def is_staged_file(name: str, filename: str) -> bool:
    """Recognize interrupted staging writes, including names without the ``.tmp`` suffix."""
    prefix = f".{filename}."
    return name.startswith(prefix) and len(name) > len(prefix)


def atomic_torch_save(path: str, payload: Callable[[], object], previous: str | None = None) -> None:
    """Publish a complete payload, reusing an unchanged immutable file where available.

    The payload stays lazy so an unchanged reference table is not materialized again. Sync the
    file before its rename and the directory afterward, before a caller rotates older checkpoints.
    """
    destination = Path(path)
    directory = destination.parent
    directory.mkdir(parents=True, exist_ok=True)
    staged = create_staged_file(directory, destination.name)
    try:
        if previous is not None and os.path.isfile(previous):
            staged.unlink()
            try:
                os.link(previous, staged)
            except OSError as exc:
                if exc.errno not in _LINK_COPY_ERRNOS:
                    raise
                shutil.copyfile(previous, staged)
        else:
            torch.save(payload(), staged)
        with staged.open("rb") as completed:
            os.fsync(completed.fileno())
        os.replace(staged, destination)
        fsync_directory(directory)
    finally:
        staged.unlink(missing_ok=True)


def fsync_directory(directory: str | Path) -> None:
    """Persist a directory's renamed or linked entries before their predecessors can rotate."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
