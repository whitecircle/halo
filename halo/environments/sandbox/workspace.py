"""A program's working directory, as seen by the side that runs the program. The program owns the directory and is
not trusted, so nothing here follows a link, blocks on a FIFO, recurses, or raises on what the program left behind.
Standard library only."""

import contextlib
import errno
import os
import stat
from dataclasses import dataclass
from typing import NamedTuple

_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_NOT_A_REGULAR_FILE = (errno.ELOOP, errno.EISDIR, errno.ENXIO)


class SessionPathError(ValueError):
    """A session path that is a link, is not a regular file, or leaves the working directory."""


class Owner(NamedTuple):
    uid: int
    gid: int


@dataclass(frozen=True, order=True)
class Entry:
    name: str
    kind: int


class _Identity(NamedTuple):
    device: int
    inode: int


@dataclass
class _Frame:
    identity: _Identity
    name: str
    subdirectories: list[str]


def grant_owner(name: str, bits: int, dir_fd: int | None = None) -> None:
    """Add ``bits`` to the owner permissions of a directory or regular file. An entry that refuses stays as it is."""
    # CPython reports a link that fchmodat refuses as NotImplementedError, or as ValueError under a dir_fd.
    with contextlib.suppress(OSError, NotImplementedError, ValueError):
        mode = os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        if (stat.S_ISDIR(mode) or stat.S_ISREG(mode)) and mode & bits != bits:
            os.chmod(name, stat.S_IMODE(mode) | bits, dir_fd=dir_fd, follow_symlinks=False)


def _identity(fd: int) -> _Identity:
    info = os.fstat(fd)
    return _Identity(info.st_dev, info.st_ino)


def _sweep_entries(dir_fd: int, *, remove: bool) -> list[str]:
    """Restore owner access to each entry of the directory, or delete each non-directory when ``remove``.
    Returns the names of its subdirectories."""
    try:
        with os.scandir(dir_fd) as scan:
            entries = list(scan)
    except OSError:
        return []
    subdirectories = []
    for entry in entries:
        try:
            is_directory = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if is_directory:
            grant_owner(entry.name, stat.S_IRWXU, dir_fd)
            subdirectories.append(entry.name)
        elif remove:
            with contextlib.suppress(OSError):
                os.unlink(entry.name, dir_fd=dir_fd)
        else:
            grant_owner(entry.name, stat.S_IRUSR | stat.S_IWUSR, dir_fd)
    return subdirectories


def walk_tree(top: str, *, remove: bool) -> None:
    """Restore owner access to ``top`` and everything under it, or delete it all when ``remove``. Never raises.

    The walk keeps one descriptor open, goes down by name without following a link, and comes back up through
    ``..``, so a tree of any depth costs no recursion and no descriptor budget. It stops where ``..`` is not the
    directory it came from, which means a process moved the tree during the walk."""
    grant_owner(top, stat.S_IRWXU)
    try:
        fd = os.open(top, _DIR_OPEN_FLAGS)
    except OSError:
        return
    try:
        path = [_Frame(_identity(fd), top, _sweep_entries(fd, remove=remove))]
        while True:
            if path[-1].subdirectories:
                name = path[-1].subdirectories.pop()
                try:
                    child = os.open(name, _DIR_OPEN_FLAGS, dir_fd=fd)
                except OSError:
                    continue
                os.close(fd)
                fd = child
                path.append(_Frame(_identity(fd), name, _sweep_entries(fd, remove=remove)))
                continue
            finished = path.pop()
            if not path:
                break
            child, fd = fd, os.open("..", _DIR_OPEN_FLAGS, dir_fd=fd)
            os.close(child)
            if _identity(fd) != path[-1].identity:
                return
            if remove:
                with contextlib.suppress(OSError):
                    os.rmdir(finished.name, dir_fd=fd)
    except OSError:
        return
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
    if remove:
        with contextlib.suppress(OSError):
            os.rmdir(top)


class Workspace:
    """The working directory at ``path``. Everything created in it is given to ``owner``, the user the program runs
    as; ``None`` when the program runs as this process's own user."""

    def __init__(self, path: str, owner: Owner | None = None):
        self.path = path
        self.owner = owner

    def hand_over(self) -> None:
        if self.owner is not None:
            os.chown(self.path, self.owner.uid, self.owner.gid, follow_symlinks=False)

    def open(self, name: str, flags: int) -> int:
        """A descriptor on the regular file ``name``. Raises :class:`SessionPathError` for anything else."""
        target = os.path.normpath(os.path.join(os.path.realpath(self.path), name))
        if os.path.realpath(target) != target:
            raise SessionPathError(f"session path {name!r} is a link or resolves outside the working directory")
        if flags & os.O_CREAT:
            self._create_directories(os.path.dirname(target))
        try:
            # O_NONBLOCK: opening a FIFO must not wait for its other end.
            fd = os.open(target, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644)
        except OSError as error:
            if error.errno in _NOT_A_REGULAR_FILE:
                raise SessionPathError(f"session path {name!r} is not a regular file") from error
            raise
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise SessionPathError(f"session path {name!r} is not a regular file")
        if self.owner is not None:
            os.fchown(fd, self.owner.uid, self.owner.gid)
        return fd

    def _create_directories(self, path: str) -> None:
        if os.path.isdir(path):
            return
        self._create_directories(os.path.dirname(path))
        with contextlib.suppress(FileExistsError):
            os.mkdir(path)
            if self.owner is not None:
                os.chown(path, self.owner.uid, self.owner.gid, follow_symlinks=False)

    def write(self, name: str, content: str) -> None:
        with os.fdopen(self.open(name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC), "w", encoding="utf-8") as file:
            file.write(content)

    def read(self, name: str) -> str | None:
        """The file's text, or ``None`` when ``name`` is missing or is not a regular file."""
        try:
            fd = self.open(name, os.O_RDONLY)
        except (SessionPathError, FileNotFoundError, NotADirectoryError):
            return None
        with os.fdopen(fd, encoding="utf-8", errors="replace") as file:
            return file.read()

    def files(self) -> list[str]:
        found: list[str] = []
        for directory, _subdirectories, names in os.walk(self.path):
            found.extend(os.path.relpath(os.path.join(directory, name), self.path) for name in names)
        return sorted(found)

    def entries(self) -> set[Entry]:
        """The top-level entries with their file types. A link counts as a link."""
        found: set[Entry] = set()
        for name in os.listdir(self.path):
            with contextlib.suppress(FileNotFoundError):
                found.add(Entry(name, stat.S_IFMT(os.lstat(os.path.join(self.path, name)).st_mode)))
        return found

    def keep_only(self, entries: set[Entry]) -> None:
        """Remove every top-level entry that is not in ``entries``."""
        self.restore_owner_access()
        for entry in self.entries() - entries:
            path = os.path.join(self.path, entry.name)
            if entry.kind == stat.S_IFDIR:
                walk_tree(path, remove=True)
            else:
                with contextlib.suppress(FileNotFoundError, PermissionError):
                    os.remove(path)

    def restore_owner_access(self) -> None:
        walk_tree(self.path, remove=False)

    def remove(self) -> None:
        walk_tree(self.path, remove=True)
