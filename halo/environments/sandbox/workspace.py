"""A program's working directory, as the side that runs the program sees it: the file, listing and removal primitives
every backend shares. Standard library only, so the hosted backends can ship this file into their sandboxes.

The program owns the directory and is not trusted. Between runs it can replace any entry with a link to a file outside,
a FIFO or a device, take its own permissions away, and nest directories past any recursion budget. Nothing here
follows a link, blocks on a FIFO, recurses, or raises on what the program left."""

import contextlib
import errno
import os
import stat

_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

# (name, file type) per directory entry; lstat, so a link is a link.
EntryKinds = set[tuple[str, int]]


class SessionPathError(ValueError):
    """A session path the host must not touch: a link, a non-regular file, or one resolving outside the workdir."""


def grant_owner(name: str, bits: int, dir_fd: int | None = None) -> None:
    """Add ``bits`` to the owner permissions of the directory or regular file ``name``; an entry that refuses stays.
    ``fchmodat`` refuses a link raced in after the type check; CPython raises that as ``NotImplementedError``, or as
    ``ValueError`` relative to a directory descriptor."""
    with contextlib.suppress(OSError, NotImplementedError, ValueError):
        mode = os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        if (stat.S_ISDIR(mode) or stat.S_ISREG(mode)) and mode & bits != bits:
            os.chmod(name, stat.S_IMODE(mode) | bits, dir_fd=dir_fd, follow_symlinks=False)


def _identity(fd: int) -> tuple[int, int]:
    """``(st_dev, st_ino)``, which ``..`` must match on the way back up."""
    info = os.fstat(fd)
    return info.st_dev, info.st_ino


def _sweep_entries(dir_fd: int, *, remove: bool) -> list[str]:
    """Give the owner back access to each entry of ``dir_fd``, or with ``remove`` delete each non-directory.
    Returns the subdirectories' names."""
    try:
        with os.scandir(dir_fd) as scan:
            entries = list(scan)
    except OSError:
        return []
    subdirs = []
    for entry in entries:
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if is_dir:
            grant_owner(entry.name, stat.S_IRWXU, dir_fd)
            subdirs.append(entry.name)
        elif remove:
            with contextlib.suppress(OSError):
                os.unlink(entry.name, dir_fd=dir_fd)
        else:
            grant_owner(entry.name, stat.S_IRUSR | stat.S_IWUSR, dir_fd)
    return subdirs


def walk_tree(top: str, *, remove: bool) -> None:
    """Give the owner back access to ``top`` and everything under it, or with ``remove`` delete it all. Never raises.
    A program can take those bits away and nest directories past any recursion or descriptor budget, so the walk
    holds one descriptor at a time, descends by name without following a link and climbs through ``..``, stopping
    where ``..`` is not the directory it came from (an escaped child moved the tree)."""
    grant_owner(top, stat.S_IRWXU)
    try:
        fd = os.open(top, _DIR_OPEN_FLAGS)
    except OSError:
        return
    try:
        # Per directory on the path down: identity, name, subdirectories left.
        frames = [(_identity(fd), top, _sweep_entries(fd, remove=remove))]
        while True:
            pending = frames[-1][2]
            if pending:
                name = pending.pop()
                try:
                    child = os.open(name, _DIR_OPEN_FLAGS, dir_fd=fd)
                except OSError:
                    continue
                parent, fd = fd, child
                os.close(parent)
                frames.append((_identity(fd), name, _sweep_entries(fd, remove=remove)))
                continue
            _, name, _ = frames.pop()
            if not frames:
                break
            child, fd = fd, os.open("..", _DIR_OPEN_FLAGS, dir_fd=fd)
            os.close(child)
            if _identity(fd) != frames[-1][0]:
                return
            if remove:
                with contextlib.suppress(OSError):
                    os.rmdir(name, dir_fd=fd)
    except OSError:
        return
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
    if remove:
        with contextlib.suppress(OSError):
            os.rmdir(top)


class Workspace:
    """The directory at ``path``. ``owner`` is the ``(uid, gid)`` the program runs as, to whom everything created
    here is handed; ``None`` where the program runs as this process's own user."""

    def __init__(self, path: str, owner: tuple[int, int] | None = None):
        self.path = path
        self.owner = owner

    def hand_over(self) -> None:
        """Give the directory itself to the program's user."""
        if self.owner is not None:
            os.chown(self.path, *self.owner, follow_symlinks=False)

    def open(self, name: str, flags: int) -> int:
        """Open the regular file ``name`` following no link at any component; what it creates goes to the owner.
        Anything else raises :class:`SessionPathError`, and ``O_NONBLOCK`` keeps a FIFO from blocking the caller."""
        dest = os.path.normpath(os.path.join(os.path.realpath(self.path), name))
        if os.path.realpath(dest) != dest:
            raise SessionPathError(f"session path {name!r} is a link or resolves outside the working directory")
        if flags & os.O_CREAT:
            self._create_dirs(os.path.dirname(dest))
        try:
            fd = os.open(dest, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644)
        except OSError as exc:
            # ELOOP: a link raced in at the final component; EISDIR: a write met a directory; ENXIO: a write met a FIFO.
            if exc.errno in (errno.ELOOP, errno.EISDIR, errno.ENXIO):
                raise SessionPathError(f"session path {name!r} is not a regular file") from exc
            raise
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise SessionPathError(f"session path {name!r} is not a regular file")
        if self.owner is not None:
            os.fchown(fd, *self.owner)
        return fd

    def _create_dirs(self, path: str) -> None:
        """``os.makedirs(path, exist_ok=True)``, handing each directory it creates to the owner."""
        if os.path.isdir(path):
            return
        self._create_dirs(os.path.dirname(path))
        with contextlib.suppress(FileExistsError):
            os.mkdir(path)
            if self.owner is not None:
                os.chown(path, *self.owner, follow_symlinks=False)

    def write(self, name: str, content: str) -> None:
        """Create or overwrite ``name`` with ``content``, which UTF-8 must be able to encode."""
        with os.fdopen(self.open(name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC), "w", encoding="utf-8") as fh:
            fh.write(content)

    def read(self, name: str) -> str | None:
        """The file's text, or ``None`` when ``name`` is not a regular file, so a planted link never carries a file
        from outside into the trajectory. Decoded as UTF-8 with replacement."""
        try:
            fd = self.open(name, os.O_RDONLY)
        except (SessionPathError, FileNotFoundError, NotADirectoryError):
            return None
        with os.fdopen(fd, encoding="utf-8", errors="replace") as fh:
            return fh.read()

    def files(self) -> list[str]:
        """Relative paths of everything under the directory that is not itself a directory."""
        found: list[str] = []
        for root, _dirs, names in os.walk(self.path):
            found.extend(os.path.relpath(os.path.join(root, name), self.path) for name in names)
        return sorted(found)

    def entries(self) -> EntryKinds:
        """The top-level entries with their file types."""
        kinds: EntryKinds = set()
        for name in os.listdir(self.path):
            with contextlib.suppress(FileNotFoundError):
                kinds.add((name, stat.S_IFMT(os.lstat(os.path.join(self.path, name)).st_mode)))
        return kinds

    def keep_only(self, entries: EntryKinds) -> None:
        """Remove every top-level entry not in ``entries``, an entry the program swapped for another type included."""
        self.restore_owner_access()
        for name, kind in self.entries() - entries:
            path = os.path.join(self.path, name)
            if kind == stat.S_IFDIR:
                walk_tree(path, remove=True)
            else:
                # Refused right after the restore means an escaped child locked it; it stays.
                with contextlib.suppress(FileNotFoundError, PermissionError):
                    os.remove(path)

    def restore_owner_access(self) -> None:
        """Undo permissions the program took away from its own tree."""
        walk_tree(self.path, remove=False)

    def remove(self) -> None:
        """Delete the directory and everything under it. Never raises; what the walk cannot remove stays."""
        walk_tree(self.path, remove=True)
