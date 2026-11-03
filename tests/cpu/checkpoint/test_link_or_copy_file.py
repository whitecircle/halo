"""``link_or_copy_file``: a hard link where possible, else a copy that reads like the link.
An existing destination is replaced, never written through, since it may link to another checkpoint's file."""

import os
import stat

import pytest

from halo.checkpoint.atomic import link_or_copy_file
from tests.common.file_links import OWNER_ONLY_MODE, mount_again, process_umask, refuse_hard_links

# An mtime no file this test writes gets on its own.
SOURCE_MTIME_NS = 1_000_000_000_000_000_000
OTHER_CHECKPOINT_BYTES = b"another checkpoint's optimizer state"
SOURCE_BYTES = b"optimizer state"


def _source(tmp_path):
    source = tmp_path / "optimizer_shard_00000.pt"
    source.write_bytes(SOURCE_BYTES)
    source.chmod(OWNER_ONLY_MODE)
    os.utime(source, ns=(SOURCE_MTIME_NS, SOURCE_MTIME_NS))
    return source


def test_on_one_filesystem_the_destination_is_the_source_file(tmp_path):
    source = _source(tmp_path)
    destination = tmp_path / "merged" / source.name
    destination.parent.mkdir()

    link_or_copy_file(source, destination)

    assert not destination.is_symlink()
    assert os.path.samefile(source, destination), "the file was copied, not hard-linked"


def test_a_refused_link_copies_with_the_source_mode_and_mtime(tmp_path, monkeypatch):
    source = _source(tmp_path)
    destination = tmp_path / "merged.pt"
    refuse_hard_links(monkeypatch)

    with process_umask():
        link_or_copy_file(source, destination)

    copied = destination.stat()
    assert copied.st_ino != source.stat().st_ino
    assert destination.read_bytes() == source.read_bytes()
    assert stat.S_IMODE(copied.st_mode) == OWNER_ONLY_MODE, "the copy took the umask's mode, not the source's"
    assert copied.st_mtime_ns == SOURCE_MTIME_NS


@pytest.mark.parametrize("links", [True, False], ids=["link", "copy"])
def test_an_existing_destination_is_replaced_not_written_through(tmp_path, monkeypatch, links):
    """A merge re-run finds the earlier file there, possibly a link to another checkpoint's."""
    other = tmp_path / "other_checkpoint.pt"
    other.write_bytes(OTHER_CHECKPOINT_BYTES)
    destination = tmp_path / "merged.pt"
    os.link(other, destination)
    source = _source(tmp_path)
    if not links:
        refuse_hard_links(monkeypatch)

    link_or_copy_file(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert other.read_bytes() == OTHER_CHECKPOINT_BYTES, "the write went through the old link"


@pytest.mark.parametrize("reached_by", ["same_path", "second_mount"])
def test_the_sources_own_entry_is_refused_and_survives(tmp_path, monkeypatch, reached_by):
    """It would delete the only copy; a second mount is the path ``realpath`` does not see through."""
    save = tmp_path / "save"
    save.mkdir()
    source = _source(save)
    if reached_by == "same_path":
        destination = source
    else:
        mount_again(monkeypatch, save, tmp_path / "mounted")
        destination = tmp_path / "mounted" / source.name

    with pytest.raises(ValueError, match="itself"):
        link_or_copy_file(source, destination)

    assert source.read_bytes() == SOURCE_BYTES, "the source was removed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
