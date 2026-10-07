"""Staged checkpoint-file writes, with ordinary umask permissions and durable publication."""

import errno
import os
import pickle
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

import torch

FILE_STAGING_SUFFIX = ".tmp"
# A complete file held off its published name until the rest of its checkpoint is on disk.
WITHHELD_FILE_SUFFIX = ".uncommitted"
# ``<name>.staged``: the async-GRPO ``prefetch_pending`` sidecars a pre-1.1 main checkpoint can carry
# from an interrupted write. An export must not ship it.
_LEGACY_STAGED_SUFFIX = ".staged"
_LINK_COPY_ERRNOS = {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOSYS, errno.EOPNOTSUPP}


def create_staged_file(directory: str | Path, filename: str) -> Path:
    """Create an exclusive sibling under the same ``0o666``-and-umask mode as a normal file."""
    staged = Path(directory) / f".{filename}.{uuid.uuid4().hex}{FILE_STAGING_SUFFIX}"
    os.close(os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
    return staged


def withheld_file_name(filename: str) -> str:
    """The name ``filename`` waits under until its checkpoint is complete."""
    return f".{filename}{WITHHELD_FILE_SUFFIX}"


def is_atomic_staging_file(name: str) -> bool:
    """Whether ``name`` is an unpublished stage a crash left: one :func:`create_staged_file` made, a
    :func:`withheld_file_name`, or an older checkpoint's staging spelling."""
    if name.endswith(_LEGACY_STAGED_SUFFIX):
        return True
    return name.startswith(".") and name.endswith((FILE_STAGING_SUFFIX, WITHHELD_FILE_SUFFIX))


def link_or_copy_file(source: str | Path, destination: str | Path) -> None:
    """Hard-link ``source`` to ``destination``, copying it where the filesystem refuses links."""
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno not in _LINK_COPY_ERRNOS:
            raise
        shutil.copyfile(source, destination)


def atomic_torch_save(
    path: str, payload: Callable[[], object], previous: str | None = None, *, pickle_module=pickle
) -> None:
    """Publish a complete payload, reusing an unchanged immutable file where available.

    The payload stays lazy so an unchanged reference table is not materialized again. Sync the
    file before its rename and the directory afterward, before a caller rotates older checkpoints.
    ``pickle_module`` is ``torch.save``'s, for a payload the stdlib pickler refuses.
    """
    destination = Path(path)
    directory = destination.parent
    directory.mkdir(parents=True, exist_ok=True)
    staged = create_staged_file(directory, destination.name)
    try:
        if previous is not None and os.path.isfile(previous):
            staged.unlink()
            link_or_copy_file(previous, staged)
        else:
            torch.save(payload(), staged, pickle_module=pickle_module)
        publish_staged_file(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def publish_staged_file(staged: str | Path, destination: str | Path) -> None:
    """Sync ``staged``, rename it over ``destination`` and sync their directory, in that order."""
    with open(staged, "rb") as completed:
        os.fsync(completed.fileno())
    os.replace(staged, destination)
    fsync_directory(Path(destination).parent)


def fsync_directory(directory: str | Path) -> None:
    """Persist a directory's renamed or linked entries before their predecessors can rotate."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
