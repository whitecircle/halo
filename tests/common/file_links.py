"""The filesystem conditions ``halo.checkpoint.atomic.link_or_copy_file`` and the tools over it branch on, forced:
a refused hard link, a directory reached by a second path ``realpath`` does not resolve, and a known umask under
which a file that took the umask instead of its source's mode shows."""

import errno
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

COMMON_UMASK = 0o022
# A mode no ``0o666``-and-umask creation under COMMON_UMASK yields: a carried file that has it kept its source's.
OWNER_ONLY_MODE = 0o600


def refuse_hard_links(monkeypatch) -> None:
    """Fail every ``os.link`` the way a link between two filesystems fails."""

    def across_filesystems(*args, **kwargs):
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV))

    monkeypatch.setattr(os, "link", across_filesystems)


def mount_again(monkeypatch, directory: Path, alias: Path) -> None:
    """Make ``alias`` reach ``directory`` as a second bind mount of it does, by a path ``os.path.realpath`` keeps
    distinct: a directory symlink, with ``realpath`` reduced to ``abspath`` so that it is not resolved."""
    alias.symlink_to(directory, target_is_directory=True)
    monkeypatch.setattr(os.path, "realpath", os.path.abspath)


@contextmanager
def process_umask(mask: int = COMMON_UMASK) -> Iterator[None]:
    """Run the block under ``mask``, restoring the previous umask after it."""
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)
