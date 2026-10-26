"""Staged checkpoint-file writes, with ordinary umask permissions and durable publication, and the staged
directory publish of a tool that may be replacing the only copy of its input."""

import errno
import os
import pickle
import shutil
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

FILE_STAGING_SUFFIX = ".tmp"
# A complete file held off its published name until the rest of its checkpoint is on disk.
WITHHELD_FILE_SUFFIX = ".uncommitted"
# A staged directory publish builds beside the target and swaps in; a failed swap prints these paths.
PUBLISH_STAGING_SUFFIX = ".halo-staging"
DISPLACED_SUFFIX = ".halo-displaced"
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
    """Whether ``name`` is an unpublished stage a crash left: one :func:`create_staged_file` made, or a
    :func:`withheld_file_name`."""
    return name.startswith(".") and name.endswith((FILE_STAGING_SUFFIX, WITHHELD_FILE_SUFFIX))


def link_or_copy_file(source: str | Path, destination: str | Path) -> None:
    """Hard-link ``source`` to ``destination``, copying it where the filesystem refuses links."""
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno not in _LINK_COPY_ERRNOS:
            raise
        shutil.copyfile(source, destination)


def fsync_directory(directory: str | Path) -> None:
    """Persist a directory's renamed or linked entries before their predecessors can rotate."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_staged_file(staged: str | Path, destination: str | Path) -> None:
    """Sync ``staged``, rename it over ``destination`` and sync their directory, in that order."""
    with open(staged, "rb") as completed:
        os.fsync(completed.fileno())
    os.replace(staged, destination)
    fsync_directory(Path(destination).parent)


@contextmanager
def staged_file(directory: str | Path, filename: str, *, durable: bool = True) -> Iterator[Path]:
    """Yield an exclusive staged sibling of ``directory/filename``, published there on a clean exit.

    ``durable`` publishes through :func:`publish_staged_file`; without it the rename alone. The stage
    never outlives the block.
    """
    staged = create_staged_file(directory, filename)
    try:
        yield staged
        destination = Path(directory) / filename
        if durable:
            publish_staged_file(staged, destination)
        else:
            os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def atomic_torch_save(
    path: str, payload: Callable[[], object], previous: str | None = None, *, pickle_module=pickle
) -> None:
    """Durably publish ``payload()`` at ``path``, hard-linking ``previous`` instead where it exists.

    The payload stays lazy so an unchanged file is not materialized again; ``pickle_module`` is
    ``torch.save``'s.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with staged_file(destination.parent, destination.name) as staged:
        if previous is not None and os.path.isfile(previous):
            staged.unlink()
            link_or_copy_file(previous, staged)
        else:
            torch.save(payload(), staged, pickle_module=pickle_module)


def clear_staging_path(target: str | os.PathLike, suffix: str = PUBLISH_STAGING_SUFFIX) -> str:
    """``target`` + ``suffix`` (:data:`PUBLISH_STAGING_SUFFIX` or :data:`DISPLACED_SUFFIX`), cleared of what
    an earlier run left there.

    Found at the start of a run it belongs to a run that failed before its swap, so it is never the only
    copy; left after a failed swap it is, and the caller must name it in its error and keep it.
    """
    path = f"{os.fspath(target)}{suffix}"
    shutil.rmtree(path, ignore_errors=True)
    return path
