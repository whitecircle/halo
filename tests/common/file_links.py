"""The filesystem conditions ``link_or_copy_file`` and the tools over it branch on, forced."""

import errno
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

COMMON_UMASK = 0o022
# A mode no ``0o666``-and-umask creation yields: a carried file that has it kept its source's.
OWNER_ONLY_MODE = 0o600


def refuse_hard_links(monkeypatch) -> None:
    """Fail every ``os.link`` the way a link between two filesystems fails."""

    def across_filesystems(*args, **kwargs):
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV))

    monkeypatch.setattr(os, "link", across_filesystems)


def mount_again(monkeypatch, directory: Path, alias: Path) -> None:
    """Make ``alias`` reach ``directory`` as a second bind mount does: a symlink, with ``realpath`` reduced to
    ``abspath``."""
    alias.symlink_to(directory, target_is_directory=True)
    monkeypatch.setattr(os.path, "realpath", os.path.abspath)


@contextmanager
def process_umask(mask: int = COMMON_UMASK) -> Iterator[None]:
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)
