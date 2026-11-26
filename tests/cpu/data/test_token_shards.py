#!/usr/bin/env python
"""datatrove token folders read as packed rows: the windows and document lengths of folders datatrove 0.10.1 wrote
(``tests/common/datatrove_shards.json``, 2- and 4-byte ids), held to each document tokenized apart from datatrove; every
document separator trained where the tokenizer pads with it; the empty files a task with no input leaves; the open
files held at their cap; a file cut short under the run, told apart from a read cut short; an S3 folder fetched once and
a changed prefix refused, not replaced; and every refusal of a folder that cannot be read as written."""

import os
import pickle
import re
from pathlib import Path

import numpy as np
import pytest

from halo.data.collators.factory import select_data_collator
from halo.data.sources import token_shards
from halo.data.sources.token_shards import TokenShards
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from tests.common.datatrove_shards import (
    END_OF_TEXT,
    load_record,
    recorded_documents,
    write_recorded_folder,
    write_token_folder,
)
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_config, load_cached_tokenizer

# 7, 16 and 40 each give a window holding several documents and one cut mid-document, and a file tail shorter than a
# window except 7 on code's 105 tokens; 1 makes every token its own window.
WINDOW_LENGTHS = (1, 7, 16, 40)
RECORDED_SOURCES = tuple(load_record()["sources"])


def _expected_rows(files: dict[str, list[list[int]]], max_length: int) -> list[dict[str, list[int]]]:
    """The rows of ``files`` (each its documents' ids) by the reader's contract: per file in name order, whole windows
    of ``max_length``, cut at every document end strictly inside a window."""
    rows = []
    for name in sorted(files):
        tokens = [token for document in files[name] for token in document]
        ends = np.cumsum([len(document) for document in files[name]]).tolist()
        for begin in range(0, len(tokens) - max_length + 1, max_length):
            end = begin + max_length
            cuts = [begin, *(e for e in ends if begin < e < end), end]
            window = tokens[begin:end]
            rows.append({"input_ids": window, "labels": window, "seq_lengths": np.diff(cuts).tolist()})
    return rows


def test_the_recorded_folders_read_as_their_documents(tmp_path, subtests):
    assert RECORDED_SOURCES == ("web", "code")
    for source in RECORDED_SOURCES:
        folder = write_recorded_folder(source, tmp_path / source)
        for max_length in WINDOW_LENGTHS:
            with subtests.test(f"{source}-{max_length}"):
                expected = _expected_rows(recorded_documents(source), max_length)
                shards = TokenShards(folder, max_length)
                assert len(shards) == len(expected) > 0
                assert [shards[index] for index in range(len(shards))] == expected


def test_a_window_never_crosses_a_file_and_each_tail_is_dropped(tmp_path):
    """web's files hold 111 and 47 tokens: two windows of 40 from the first (31 dropped), one from the second (7)."""
    shards = TokenShards(write_recorded_folder("web", tmp_path), 40)
    assert [file.num_tokens for file in shards.files] == [111, 47]
    second_file = [token for document in recorded_documents("web")["00001_unshuffled.ds"] for token in document]
    assert len(shards) == 3
    assert shards[2]["input_ids"] == second_file[:40]


def test_the_token_width_and_tokenizer_come_from_the_metadata(tmp_path):
    """Qwen3's vocabulary of 151,669 ids takes 4 bytes a token, GPT-2's 50,257 takes 2."""
    record = load_record()["sources"]
    for source, token_bytes in (("web", 4), ("code", 2)):
        shards = TokenShards(write_recorded_folder(source, tmp_path / source), 8)
        assert record[source]["token_bytes"] == token_bytes
        assert shards.token_bytes == token_bytes
        assert {file.tokenizer for file in shards.files} == {record[source]["tokenizer"]}


def test_empty_files_and_an_orphan_metadata_file_are_skipped(tmp_path):
    """web's third task got no document and left empty files; the merger left code.ds.metadata with no .ds."""
    web = write_recorded_folder("web", tmp_path / "web")
    code = write_recorded_folder("code", tmp_path / "code")
    assert (web / "00002_unshuffled.ds").stat().st_size == 0
    assert (code / "code.ds.metadata").is_file()
    assert [file.name for file in TokenShards(web, 8).files] == ["00000_unshuffled.ds", "00001_unshuffled.ds"]
    assert [file.name for file in TokenShards(code, 8).files] == ["000_code.ds"]


def test_the_layout_writer_reproduces_datatrove_bytes(tmp_path, subtests):
    """The synthetic sources of the mix tests are written by this writer, so it must write what datatrove writes."""
    for source in RECORDED_SOURCES:
        with subtests.test(source):
            recorded = write_recorded_folder(source, tmp_path / "recorded" / source)
            spec = load_record()["sources"][source]
            written = write_token_folder(
                tmp_path / "written" / source,
                recorded_documents(source),
                token_bytes=spec["token_bytes"],
                tokenizer=spec["tokenizer"],
            )
            for name in recorded_documents(source):
                for suffix in ("", ".index"):
                    assert (written / f"{name}{suffix}").read_bytes() == (recorded / f"{name}{suffix}").read_bytes()
                metadata = (recorded / f"{name}.metadata").read_text().splitlines()[:2]
                assert (written / f"{name}.metadata").read_text().splitlines()[:2] == metadata


def test_rows_are_ints_and_lengths_sum_to_the_window(tmp_path):
    """The packing collator reads ``seq_lengths`` only as a list, ids from 4-byte files must stay exact, and the
    declared columns are what every row holds."""
    shards = TokenShards(write_recorded_folder("web", tmp_path), 16)
    for index in range(len(shards)):
        row = shards[index]
        assert tuple(row) == shards.column_names == ("input_ids", "labels", "seq_lengths")
        assert type(row["seq_lengths"]) is list and sum(row["seq_lengths"]) == 16
        assert all(type(token) is int for token in row["input_ids"])
    assert max(token for index in range(len(shards)) for token in shards[index]["input_ids"]) == 151643


def test_every_separator_trains_where_the_tokenizer_pads_with_it(tmp_path):
    """Qwen3 pads with the separator the recorded folder ends each document with: the script's packing collator masks
    exactly each document's first position, never a separator for being the pad id."""
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    separator = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    assert tokenizer.pad_token_id == separator != tokenizer.eos_token_id
    collator = select_data_collator(
        tokenizer, train_on_completions_only=False, packing=True, model_config=load_cached_config(QWEN3_0_6B)
    )
    shards = TokenShards(write_recorded_folder("web", tmp_path), 16)
    batch = collator([shards[index] for index in range(len(shards))])
    ids, labels, positions = (batch[key][0].tolist() for key in ("input_ids", "labels", "position_ids"))
    separators = [index for index, token in enumerate(ids) if token == separator]
    assert len(separators) > 1 and all(positions[index] > 0 for index in separators)
    assert labels == [
        LABEL_IGNORE_INDEX if position == 0 else token for token, position in zip(ids, positions, strict=True)
    ]


def _break(folder, how: str) -> None:
    """Damage the recorded web folder as ``how`` names."""
    first = folder / "00000_unshuffled.ds"
    if how == "missing index":
        (folder / "00000_unshuffled.ds.index").unlink()
    elif how == "missing metadata":
        (folder / "00000_unshuffled.ds.metadata").unlink()
    elif how == "no width":
        (folder / "00000_unshuffled.ds.metadata").write_text("Qwen/Qwen3-0.6B\n111\n111 T")
    elif how == "width 3":
        (folder / "00000_unshuffled.ds.metadata").write_text("Qwen/Qwen3-0.6B|3\n111\n111 T")
    elif how == "partial token":
        first.write_bytes(first.read_bytes()[:-1])
    elif how == "index past the tokens":
        first.write_bytes(first.read_bytes()[:-4])
    elif how == "empty index":
        (folder / "00000_unshuffled.ds.index").write_bytes(b"")
    elif how == "mixed widths":
        write_token_folder(folder, {"00003_extra.ds": [[1, 2, 3]]}, token_bytes=2)
    elif how == "unknown tokenizer":
        (folder / "00000_unshuffled.ds.metadata").write_text("Unknown Tokenizer|2|2\n222\n222 T")
    elif how == "only empty files":
        for name in ("00000_unshuffled.ds", "00001_unshuffled.ds"):
            (folder / name).write_bytes(b"")


_REFUSALS = {
    "missing index": "has no 00000_unshuffled.ds.index beside it",
    "missing metadata": "has no 00000_unshuffled.ds.metadata beside it",
    "no width": r"not '<tokenizer>\|<bytes per token>'",
    "width 3": r"not '<tokenizer>\|<bytes per token>'",
    "partial token": "does not end at the 110 tokens",
    "index past the tokens": "does not end at the 110 tokens",
    "empty index": "does not end at the 111 tokens",
    "mixed widths": r"mixes token widths \{2: '00003_extra.ds', 4: '00000_unshuffled.ds'\}",
    "unknown tokenizer": r"starts with 'Unknown Tokenizer\|2\|2': DocumentTokenizerMerger .* 2-byte ids",
    "only empty files": "holds no non-empty .ds file",
}


def test_a_folder_that_cannot_be_read_as_written_is_refused(tmp_path, subtests):
    for how, message in _REFUSALS.items():
        with subtests.test(how):
            folder = write_recorded_folder("web", tmp_path / how.replace(" ", "_"))
            TokenShards(folder, 8)
            _break(folder, how)
            with pytest.raises(ValueError, match=message):
                TokenShards(folder, 8)


def test_a_missing_folder_and_a_window_longer_than_every_file_are_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        TokenShards(tmp_path / "absent", 8)
    folder = write_recorded_folder("web", tmp_path / "web")
    assert len(TokenShards(folder, 111)) == 1
    with pytest.raises(ValueError, match="no window of max_length=112 tokens: its largest file holds 111"):
        TokenShards(folder, 112)


def test_the_content_digest_follows_the_tokens_and_boundaries_not_the_sizes(tmp_path):
    documents = {"a.ds": [[5, 6, 7, 8], [9, 10]], "b.ds": [[11] * 9]}
    variants = {
        "reordered": {"a.ds": [[9, 10], [5, 6, 7, 8]], "b.ds": [[11] * 9]},
        "other boundaries": {"a.ds": [[5, 6], [7, 8, 9, 10]], "b.ds": [[11] * 9]},
    }
    first = TokenShards(write_token_folder(tmp_path / "first", documents, token_bytes=2), 3)
    copy = TokenShards(write_token_folder(tmp_path / "copy", documents, token_bytes=2), 3)
    assert first.content_digest() == copy.content_digest()
    for label, files in variants.items():
        other = TokenShards(write_token_folder(tmp_path / label.replace(" ", "_"), files, token_bytes=2), 3)
        assert [file.num_tokens for file in other.files] == [file.num_tokens for file in first.files]
        assert other.content_digest() != first.content_digest(), label
    boundaries = TokenShards(tmp_path / "other_boundaries", 3)
    assert [boundaries[index]["input_ids"] for index in range(len(boundaries))] == [
        first[index]["input_ids"] for index in range(len(first))
    ]


def _large_folder(tmp_path, files: int = 3, tokens: int = 1 << 18):
    rng = np.random.default_rng(0)
    documents = {
        f"{index:05d}.ds": [rng.integers(0, 1 << 20, size=tokens // 4).tolist() for _ in range(4)]
        for index in range(files)
    }
    return write_token_folder(tmp_path, documents, token_bytes=4), documents


def test_a_pickled_folder_carries_no_mapped_bytes_and_reads_the_same(tmp_path):
    """A spawned loader worker gets the dataset by pickle; mapped arrays would pickle every byte they map."""
    folder, _ = _large_folder(tmp_path)
    shards = TokenShards(folder, 4096)
    rows = [shards[index] for index in range(0, len(shards), 7)]
    payload = pickle.dumps(shards)
    assert len(payload) < 64 * 1024 < sum(file.num_tokens for file in shards.files)
    restored = pickle.loads(payload)
    assert [restored[index] for index in range(0, len(restored), 7)] == rows


class _FakeS3Client:
    """S3 at its boundary: a prefix holding the recorded web folder, with a content fingerprint the test moves."""

    fingerprint = "etag-1"
    downloads: list[str] = []

    def __init__(self, bucket: str):
        self.bucket = bucket

    def download_folder(self, key: str, local_path: str) -> str:
        type(self).downloads.append(f"{self.bucket}/{key}")
        write_recorded_folder("web", Path(local_path))
        return local_path

    def content_fingerprint(self, key: str) -> str:
        return type(self).fingerprint


def test_an_s3_folder_is_fetched_once_and_a_changed_prefix_is_refused_not_replaced(tmp_path, monkeypatch):
    """A run reading the cached copy, and a checkpoint trained on it, need its files as they are."""
    monkeypatch.setattr(token_shards, "S3Client", _FakeS3Client)
    monkeypatch.setattr(token_shards, "HALO_S3_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(_FakeS3Client, "downloads", [])
    expected = _expected_rows(recorded_documents("web"), 16)
    folder = token_shards.fetch_s3_token_folder("s3://bucket/tok/web/")
    assert folder.is_relative_to(tmp_path / "cache")
    assert [TokenShards(folder, 16)[index] for index in range(len(expected))] == expected
    assert token_shards.fetch_s3_token_folder("s3://bucket/tok/web") == folder
    assert _FakeS3Client.downloads == ["bucket/tok/web"]
    monkeypatch.setattr(_FakeS3Client, "fingerprint", "etag-2")
    cached = re.escape(str(folder))
    with pytest.raises(ValueError, match=f"changed since it was cached in {cached}.*Remove {cached} to fetch"):
        token_shards.fetch_s3_token_folder("s3://bucket/tok/web")
    assert _FakeS3Client.downloads == ["bucket/tok/web"]
    assert [TokenShards(folder, 16)[index] for index in range(len(expected))] == expected


def test_rows_stay_exact_when_files_are_closed_and_reopened(tmp_path, monkeypatch):
    folder, documents = _large_folder(tmp_path, tokens=1 << 12)
    expected = _expected_rows(documents, 512)
    monkeypatch.setattr(token_shards, "_MAX_OPEN_FILES", 1)
    shards = TokenShards(folder, 512)
    assert shards.max_open_descriptors == 2 < 2 * len(shards.files)
    order = [index for round_ in range(3) for index in range(round_, len(shards), 5)]
    rows, held = [], []
    for index in order:
        rows.append(shards[index])
        held.append(len(shards._open_files))
    assert rows == [expected[index] for index in order]
    assert held == [1] * len(order), "a read leaves more files open than the cap"


def test_a_file_cut_short_under_the_run_is_refused_at_its_next_row(tmp_path):
    """The scan found every window whole; a read past the new end would hand the collator a short row."""
    folder, _ = _large_folder(tmp_path, tokens=1 << 12)
    shards = TokenShards(folder, 512)
    last = len(shards) - 1
    assert shards[last]["input_ids"], "premise: the file is open before it is cut"
    tail = folder / shards.files[-1].name
    tail.write_bytes(tail.read_bytes()[:-4])
    with pytest.raises(
        OSError, match=rf"Token file {re.escape(str(tail))} ends before window {last}\b.*changed while"
    ):
        shards[last]


def test_a_read_cut_short_is_read_on_to_the_window_end(tmp_path, monkeypatch):
    """A read an I/O error or a FUSE mount cuts short returns the bytes it read, which is no change to the folder."""
    folder, documents = _large_folder(tmp_path, tokens=1 << 12)
    expected = _expected_rows(documents, 512)
    real_pread = os.pread
    monkeypatch.setattr(token_shards.os, "pread", lambda fd, count, offset: real_pread(fd, min(count, 100), offset))
    shards = TokenShards(folder, 512)
    assert [shards[index] for index in range(len(shards))] == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
