"""Staged, durably published checkpoint-file writes under umask permissions, and staged directory publishes."""

import errno
import os
import pickle
import shutil
import stat
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import save_file

_FILE_STAGING_SUFFIX = ".tmp"
# A complete file waits as ``.<name>.uncommitted`` until the rest of its checkpoint is on disk.
WITHHELD_FILE_SUFFIX = ".uncommitted"
# A staged directory publish builds beside the target and swaps in; a failed swap prints these paths.
PUBLISH_STAGING_SUFFIX = ".halo-staging"
DISPLACED_SUFFIX = ".halo-displaced"
_LINK_COPY_ERRNOS = {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOSYS, errno.EOPNOTSUPP}


def create_staged_file(directory: str | Path, filename: str) -> Path:
    """Create an exclusive sibling under the same ``0o666``-and-umask mode as a normal file."""
    staged = Path(directory) / f".{filename}.{uuid.uuid4().hex}{_FILE_STAGING_SUFFIX}"
    os.close(os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
    return staged


def is_atomic_staging_file(name: str) -> bool:
    """Whether ``name`` is an unpublished stage a crash left, from :func:`create_staged_file` or a withheld file."""
    return name.startswith(".") and name.endswith((_FILE_STAGING_SUFFIX, WITHHELD_FILE_SUFFIX))


def _is_same_entry(path: str | Path, other: str | Path) -> bool:
    """Whether two paths name one directory entry, however their directories are reached, not two links to a file."""
    path, other = Path(path), Path(other)
    return path.name == other.name and os.path.samefile(path.parent, other.parent)


def link_or_copy_file(source: str | Path, destination: str | Path) -> None:
    """Hard-link, or copy with mode and times where links are refused; ``destination`` is replaced, never written
    through, and refused when it is the entry holding ``source``'s data."""
    if os.path.lexists(destination) and _is_same_entry(os.path.realpath(source), destination):
        raise ValueError(
            f"{destination} names {source} itself: replacing it would delete the only copy. Write to a "
            f"directory other than the source's."
        )
    Path(destination).unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno not in _LINK_COPY_ERRNOS:
            raise
        shutil.copy2(source, destination)


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
    """Yield an exclusive staged sibling, published on a clean exit (``durable`` adds fsyncs); dies with the block."""
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
    """Durably publish ``payload()`` at ``path``, hard-linking ``previous`` where it exists, hence the lazy payload."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with staged_file(destination.parent, destination.name) as staged:
        if previous is not None and os.path.isfile(previous):
            link_or_copy_file(previous, staged)
        else:
            torch.save(payload(), staged, pickle_module=pickle_module)


def apply_umask_mode(path: str | Path) -> None:
    """Give ``path``, or every ``*.safetensors`` directly under it, the mode a new file there gets (safetensors creates
    ``0600``). Read off a probe file, so umask and default ACLs apply without the thread-racy ``os.umask`` swap."""
    path = Path(path)
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    if not files:
        return
    probe = create_staged_file(files[0].parent, "mode-probe")
    try:
        mode = stat.S_IMODE(probe.stat().st_mode)
    finally:
        probe.unlink()
    for file in files:
        os.chmod(file, mode)


def save_safetensors_file(tensors: dict[str, torch.Tensor], path: str | Path, *, metadata=None) -> None:
    """``safetensors.torch.save_file`` under the mode every other checkpoint file gets."""
    save_file(tensors, path, metadata=metadata)
    apply_umask_mode(path)


def clear_staging_path(target: str | os.PathLike, suffix: str = PUBLISH_STAGING_SUFFIX) -> str:
    """``target`` + ``suffix``, cleared of what an earlier run left. Found at run start that is never the only copy;
    left by a failed swap it is, and the caller must name it in its error and keep it."""
    path = f"{os.fspath(target)}{suffix}"
    shutil.rmtree(path, ignore_errors=True)
    return path
