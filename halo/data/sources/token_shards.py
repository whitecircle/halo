"""datatrove ``DocumentTokenizer`` output folders, local or fetched from S3 into the download cache, read as packed rows
of ``max_length`` tokens with the lengths of the document pieces each holds, and the open-file limit their open files
need. Halo reads the layout and never imports datatrove."""

import hashlib
import io
import logging
import os
import resource
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from halo.data.sources.dataset_cache import HALO_S3_DATASET_CACHE_DIR, s3_cache_key, serve_cached_tree
from halo.data.sources.paths import parse_s3_uri
from halo.data.sources.s3_client import S3Client
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# The token widths datatrove writes, keyed by the byte count after the last '|' on a file's first metadata line.
TOKEN_DTYPES = {2: np.dtype("<u2"), 4: np.dtype("<u4")}
# Evenly spaced windows whose tokens the content digest and the vocabulary check read.
SAMPLED_WINDOWS = 16
# Each document's end in tokens, cumulative within its file.
_INDEX_DTYPE = np.dtype("<u8")
# The tokenizer field datatrove writes when it had no tokenizer name, as DocumentTokenizerMerger does over inputs
# without metadata, guessing the token width.
_UNKNOWN_TOKENIZER = "Unknown Tokenizer"
# .ds files one folder keeps open, each holding two descriptors (its tokens' handle and its index's map); the least
# recently read close.
_MAX_OPEN_FILES = 512
# File descriptors left beside the folders' for the rest of a training process: CUDA, NCCL, loader pipes, checkpoints.
_RESERVED_OPEN_FILES = 512


@dataclass(frozen=True)
class TokenFile:
    """A non-empty ``.ds`` file: its token count, its token width in bytes and the tokenizer its metadata names."""

    name: str
    num_tokens: int
    token_bytes: int
    tokenizer: str


def fetch_s3_token_folder(uri: str) -> Path:
    """The local copy of the token folder at ``uri`` in the S3 download cache, fetched whole when absent. A copy whose
    prefix changed since is refused, never replaced: a run reads its files as it goes, and a checkpoint resumes only
    onto the files it trained on. Not collective: concurrent callers serialize on the cache directory's lock."""
    bucket, key = parse_s3_uri(uri)
    client = S3Client(bucket=bucket)
    cache_path = os.path.join(HALO_S3_DATASET_CACHE_DIR, s3_cache_key(bucket, key), "tokens")
    serve_cached_tree(
        cache_path,
        uri,
        lambda staging: client.download_folder(key, staging),
        lambda: client.content_fingerprint(key),
        stale_message=(
            f"S3 token folder {uri} changed since it was cached in {cache_path}. The copy is not replaced, since a run "
            f"reading it and the checkpoints trained on it need its files as they are. Remove {cache_path} to fetch "
            f"the new state, upload the new state under a new prefix, or set the source's path to {cache_path} to "
            f"keep training on the cached copy."
        ),
        replace_stale=False,
    )
    return Path(cache_path)


def _read_token_file(path: Path, size: int) -> TokenFile:
    index, metadata = path.with_name(f"{path.name}.index"), path.with_name(f"{path.name}.metadata")
    if not index.is_file():
        raise ValueError(
            f"Token file {path} has no {index.name} beside it. Document boundaries are read from that index, never "
            f"from a separator token. Copy the index with the tokens, or re-run DocumentTokenizer with save_index on."
        )
    if not metadata.is_file():
        raise ValueError(
            f"Token file {path} has no {metadata.name} beside it, whose first line gives the token width. Copy it "
            f"with the tokens, or re-run DocumentTokenizer with save_final_metadata on (its default)."
        )
    lines = metadata.read_text(encoding="utf-8").splitlines()
    tokenizer, _, width = (lines[0] if lines else "").rpartition("|")
    token_bytes = int(width) if width.isdigit() else None
    if token_bytes not in TOKEN_DTYPES:
        raise ValueError(
            f"{metadata} starts with {lines[:1]}, not '<tokenizer>|<bytes per token>' with 2 or 4 bytes as "
            f"DocumentTokenizer writes it. Re-run DocumentTokenizer on this source."
        )
    if tokenizer.startswith(_UNKNOWN_TOKENIZER):
        raise ValueError(
            f"{metadata} starts with {lines[0]!r}: DocumentTokenizerMerger writes {_UNKNOWN_TOKENIZER!r} when its "
            f"inputs carry no metadata, and then reads them as {token_bytes}-byte ids whatever width they were "
            f"written at, which garbles a source of 4-byte ids. Re-run DocumentTokenizer with save_final_metadata on "
            f"(its default), then the merger."
        )
    num_tokens = size // token_bytes
    index_size = index.stat().st_size
    last_end = (
        int(np.fromfile(index, dtype=_INDEX_DTYPE, count=1, offset=index_size - _INDEX_DTYPE.itemsize)[0])
        if index_size and index_size % _INDEX_DTYPE.itemsize == 0
        else None
    )
    if last_end != num_tokens:
        raise ValueError(
            f"{index} does not end at the {num_tokens} tokens of {path.name} (its last document ends at "
            f"{last_end}, in a file of {index_size} bytes): the two come from different writes, such as a folder "
            f"still being written or copied, or files of two runs mixed. Copy the folder again once its writer has "
            f"finished."
        )
    return TokenFile(path.name, num_tokens, token_bytes, tokenizer)


def _scan_folder(folder: Path) -> tuple[TokenFile, ...]:
    """The non-empty ``.ds`` files in name order, all of one token width; an empty one is what a DocumentTokenizer task
    that received no document leaves."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Token source folder {folder} does not exist or is not a directory.")
    files = []
    for name in sorted(path.name for path in folder.glob("*.ds") if path.is_file()):
        size = (folder / name).stat().st_size
        if size:
            files.append(_read_token_file(folder / name, size))
    if not files:
        raise ValueError(
            f"Token source folder {folder} holds no non-empty .ds file. Point it at a DocumentTokenizer (or "
            f"DocumentTokenizerMerger) output folder."
        )
    widths = sorted({file.token_bytes for file in files})
    if len(widths) > 1:
        by_width = {width: next(file.name for file in files if file.token_bytes == width) for width in widths}
        raise ValueError(
            f"Token source folder {folder} mixes token widths {by_width}: one source is one tokenization. Split the "
            f"folder into one source per tokenizer, or re-tokenize it with one tokenizer."
        )
    return tuple(files)


class TokenShards(Dataset):
    """One DocumentTokenizer output folder cut into windows of ``max_length`` tokens, file by file in name order: a
    window never crosses a file, and each file's tail shorter than a window is dropped. A window is a packed row as the
    ``wrapped`` packing writes one, ``input_ids`` with ``seq_lengths`` cut at the document ends of the file's index,
    and ``labels`` equal to the ids: the packing collator then masks only each document's first position, where
    without them it would also mask every id equal to the pad id, the document separator of tokenizers that pad with
    it (Qwen3's and gpt-oss's ``<|endoftext|>``)."""

    # Every row's keys, for the trainer and data checks that ask a dataset for its columns.
    column_names = ("input_ids", "labels", "seq_lengths")

    def __init__(self, folder: str | Path, max_length: int):
        if max_length < 1:
            raise ValueError(f"max_length must be at least 1 token, got {max_length}.")
        self.folder = Path(folder)
        self.max_length = max_length
        self.files = _scan_folder(self.folder)
        self._window_starts = np.cumsum([0, *(file.num_tokens // max_length for file in self.files)])
        self.num_windows = int(self._window_starts[-1])
        if self.num_windows == 0:
            raise ValueError(
                f"Token source folder {self.folder} holds no window of max_length={max_length} tokens: its largest "
                f"file holds {max(file.num_tokens for file in self.files)}. Lower max_length, or merge the files "
                f"with DocumentTokenizerMerger."
            )
        self._open_files: OrderedDict[str, tuple[io.FileIO, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        if not 0 <= index < self.num_windows:
            raise IndexError(f"window {index} is outside the {self.num_windows} windows of {self.folder}")
        file_index = int(np.searchsorted(self._window_starts, index, side="right")) - 1
        file = self.files[file_index]
        begin = (index - int(self._window_starts[file_index])) * self.max_length
        end = begin + self.max_length
        handle, ends = self._open(file.name)
        dtype = TOKEN_DTYPES[file.token_bytes]
        # Positional reads of the window: a map would fault in the device's whole read-ahead around each cold row.
        # A read cut short by an I/O error returns what it read, so only a read of nothing is the end of the file.
        size, offset, data = self.max_length * dtype.itemsize, begin * dtype.itemsize, b""
        while len(data) < size:
            chunk = os.pread(handle.fileno(), size - len(data), offset + len(data))
            if not chunk:
                raise OSError(
                    f"Token file {self.folder / file.name} ends before window {index}, which the scan found whole: "
                    f"the folder changed while the run read it. Restore the folder as it was at startup, or start a "
                    f"new run."
                )
            data += chunk
        tokens = np.frombuffer(data, dtype=dtype)
        inside = ends[np.searchsorted(ends, np.uint64(begin), "right") : np.searchsorted(ends, np.uint64(end), "left")]
        cuts = np.concatenate(([begin], inside.astype(np.int64), [end]))
        ids = tokens.astype(np.int64).tolist()
        return {"input_ids": ids, "labels": list(ids), "seq_lengths": np.diff(cuts).tolist()}

    def __getstate__(self) -> dict:
        """Pickled for a spawned loader worker without its open files, which the worker opens itself."""
        return {**self.__dict__, "_open_files": OrderedDict()}

    @property
    def token_bytes(self) -> int:
        return self.files[0].token_bytes

    @property
    def max_open_descriptors(self) -> int:
        """The most file descriptors this folder keeps open: two per open ``.ds`` file, up to the cap."""
        return 2 * min(len(self.files), _MAX_OPEN_FILES)

    def content_digest(self) -> str:
        """A digest of ``SAMPLED_WINDOWS`` evenly spaced windows, not of every token: folders whose files match in
        name and size differ in it only where a sampled window differs, as a whole folder re-tokenized or reshuffled
        does, while one file rewritten alone can pass."""
        hasher = hashlib.blake2b(digest_size=8)
        for index in self._sampled_windows():
            row = self[index]
            hasher.update(np.asarray(row["input_ids"], dtype="<u8").tobytes())
            hasher.update(np.asarray(row["seq_lengths"], dtype="<u8").tobytes())
        return hasher.hexdigest()

    def verify_token_ids_below(self, vocab_size: int) -> None:
        """Raise on a sampled window holding an id the model's embedding has no row for, which the first step that
        draws it would index past."""
        for index in self._sampled_windows():
            highest = max(self[index]["input_ids"])
            if highest >= vocab_size:
                raise ValueError(
                    f"Token source folder {self.folder}: window {index} holds token id {highest}, past the model's "
                    f"vocabulary of {vocab_size}, so it was tokenized for another model (its metadata names "
                    f"{self.files[0].tokenizer!r}). Re-tokenize it with the model's tokenizer."
                )

    def _sampled_windows(self) -> list[int]:
        last = self.num_windows - 1
        return sorted({step * last // (SAMPLED_WINDOWS - 1) for step in range(SAMPLED_WINDOWS)})

    def _open(self, name: str) -> tuple[io.FileIO, np.ndarray]:
        """``(tokens handle, index map)`` of file ``name``, kept for its next row: an open per row would be a metadata
        request on a network filesystem."""
        opened = self._open_files.get(name)
        if opened is None:
            opened = self._open_files[name] = (
                open(self.folder / name, "rb", buffering=0),  # noqa: SIM115 - closed on eviction
                np.memmap(self.folder / f"{name}.index", dtype=_INDEX_DTYPE, mode="r"),
            )
            if len(self._open_files) > _MAX_OPEN_FILES:
                self._open_files.popitem(last=False)[1][0].close()
        else:
            self._open_files.move_to_end(name)
        return opened


def ensure_open_file_limit(folders: Sequence[TokenShards]) -> None:
    """Raise this process's soft open-file limit, toward its hard one, to hold every descriptor ``folders`` can keep
    open plus a reserve for the rest of the run; raise ``ValueError`` when the hard limit is below that, since a read
    past it fails mid-run on a bare ``Too many open files``. Loader workers inherit the raised limit."""
    descriptors = sum(folder.max_open_descriptors for folder in folders)
    needed = descriptors + _RESERVED_OPEN_FILES
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY or soft >= needed:
        return
    if hard != resource.RLIM_INFINITY and hard < needed:
        raise ValueError(
            f"The token folders can hold {descriptors:,} file descriptors open (two per .ds file, its tokens and its "
            f"index, for at most {_MAX_OPEN_FILES:,} files per folder), which with {_RESERVED_OPEN_FILES} descriptors "
            f"for the rest of the run needs an open-file limit of {needed:,}, past this process's hard limit of "
            f"{hard:,}. Raise the hard limit "
            f"the launcher grants (docker run --ulimit nofile={needed}:{needed}, or nofile in limits.conf), or merge "
            f"each folder's files into fewer with DocumentTokenizerMerger."
        )
    resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
    if is_global_main_process():
        logger.info(f"Raised the open-file soft limit from {soft:,} to {needed:,} for the token folders' files.")
