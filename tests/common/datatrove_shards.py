"""datatrove 0.10.1's ``DocumentTokenizer`` output for two sources (2- and 4-byte token ids), recorded with each document
tokenized apart from datatrove, and a writer of the same file layout for synthetic sources. Re-record with datatrove
importable: ``python -m tests.common.datatrove_shards --write``."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from halo.data.sources.token_shards import TOKEN_DTYPES

RECORD_PATH = Path(__file__).with_suffix(".json")
DATATROVE_VERSION = "0.10.1"
END_OF_TEXT = "<|endoftext|>"


@dataclass(frozen=True)
class _RecordedSource:
    tokenizer: str
    # Each input file's documents; JsonlReader hands file k to task k when there are at least as many tasks.
    input_files: tuple[tuple[str, ...], ...]
    tasks: int
    # The merger's save_filename, or None to record the tokenizer's own output folder.
    merged_as: str | None


_RECORDED_SOURCES = {
    # Three tasks over two input files: the third task gets no document and writes empty files.
    "web": _RecordedSource(
        tokenizer="Qwen/Qwen3-0.6B",
        input_files=(
            (
                "The quick brown fox jumps over the lazy dog.",
                "Short.",
                "Pretraining mixes weigh several sources of text by the tokens each one holds, and a weight above a "
                "source's size repeats it.",
                "Yes.",
                "A longer document " + " ".join(f"word{i}" for i in range(24)) + ".",
            ),
            (
                "Data folders written by one tokenizer task hold their documents back to back.",
                "Ok.",
                "Document boundaries come from the index file, never from a separator token, since the separator "
                "differs by recipe.",
                "End of the second file.",
            ),
        ),
        tasks=3,
        merged_as=None,
    ),
    # One task, then the merger, which names its file NNN_<save_filename>.ds and leaves a total metadata file with no
    # .ds beside it.
    "code": _RecordedSource(
        tokenizer="openai-community/gpt2",
        input_files=(
            (
                "def add(a, b):\n    return a + b\n",
                "x = 1",
                "for i in range(10):\n    print(i * i)\n",
                "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n",
                "pass",
                "import numpy as np\nprint(np.arange(4))\n",
            ),
        ),
        tasks=1,
        merged_as="code",
    ),
}


def load_record() -> dict:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def write_recorded_folder(source: str, directory: Path, *, data_file: str | None = None) -> Path:
    """The files datatrove wrote for ``source``, byte for byte, in ``directory``; with ``data_file``, only that
    ``.ds`` and its index and metadata, as one task wrote them."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in load_record()["sources"][source]["files"].items():
        if data_file is None or name == data_file or name.startswith(f"{data_file}."):
            (directory / name).write_bytes(bytes.fromhex(payload))
    return directory


def recorded_documents(source: str) -> dict[str, list[list[int]]]:
    """Per non-empty ``.ds`` file, its documents' token ids as the tokenizer produced them outside datatrove."""
    files = load_record()["sources"][source]["documents"]
    return {name: [document["ids"] for document in documents] for name, documents in files.items()}


def write_token_folder(
    directory: Path,
    files: Mapping[str, Sequence[Sequence[int]]],
    *,
    token_bytes: int,
    tokenizer: str = "synthetic",
) -> Path:
    """Write each ``files`` entry (its documents' token ids) in DocumentTokenizer's layout: ``<name>`` holds the ids
    little-endian, ``<name>.index`` every document's uint64 end and ``<name>.metadata`` the tokenizer and width line,
    then the token count. An entry with no document writes the empty files a task with no input leaves."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, documents in files.items():
        ids = [token for document in documents for token in document]
        ends = np.cumsum([len(document) for document in documents], dtype=np.uint64)
        (directory / name).write_bytes(np.asarray(ids, dtype=TOKEN_DTYPES[token_bytes]).tobytes())
        (directory / f"{name}.index").write_bytes(ends.astype("<u8").tobytes())
        (directory / f"{name}.metadata").write_text(f"{tokenizer}|{token_bytes}\n{len(ids)}\n{len(ids)} T")
    return directory


def _tokenized_apart(tokenizer_name: str, text: str) -> list[int]:
    """What DocumentTokenizer(eos_token=END_OF_TEXT) writes for ``text``, computed without it: the tokenizer's ids with
    no template, then the end-of-text id."""
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    return tokenizer.encode(text, add_special_tokens=False).ids + [tokenizer.token_to_id(END_OF_TEXT)]


def _expected_files(name: str, spec: _RecordedSource) -> dict[str, list[dict]]:
    documents = [
        [{"text": text, "ids": _tokenized_apart(spec.tokenizer, text)} for text in texts] for texts in spec.input_files
    ]
    if spec.merged_as is not None:
        return {f"000_{spec.merged_as}.ds": [document for file in documents for document in file]}
    return {f"{task:05d}_unshuffled.ds": file for task, file in enumerate(documents)}


def _run_datatrove(spec: _RecordedSource, work: Path) -> Path:
    """The folder datatrove writes for ``spec``."""
    # Absent from the image by design: Halo reads the layout only, so re-recording puts datatrove on PYTHONPATH.
    from datatrove.executor import LocalPipelineExecutor  # noqa: PLC0415
    from datatrove.pipeline.readers import JsonlReader  # noqa: PLC0415
    from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger  # noqa: PLC0415
    from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer  # noqa: PLC0415

    inputs = work / "input"
    inputs.mkdir(parents=True)
    for index, texts in enumerate(spec.input_files):
        lines = [json.dumps({"text": text, "id": f"{index}-{row}"}) for row, text in enumerate(texts)]
        (inputs / f"{index:02d}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tokenized = work / "tokenized"
    LocalPipelineExecutor(
        pipeline=[
            JsonlReader(str(inputs)),
            DocumentTokenizer(
                output_folder=str(tokenized),
                tokenizer_name_or_path=spec.tokenizer,
                eos_token=END_OF_TEXT,
                shuffle_documents=False,
            ),
        ],
        tasks=spec.tasks,
        workers=1,
        logging_dir=str(work / "logs-tokenize"),
    ).run()
    if spec.merged_as is None:
        return tokenized
    merged = work / "merged"
    LocalPipelineExecutor(
        pipeline=[
            DocumentTokenizerMerger(
                input_folder=str(tokenized), output_folder=str(merged), save_filename=spec.merged_as, shuffle=False
            )
        ],
        tasks=1,
        workers=1,
        logging_dir=str(work / "logs-merge"),
    ).run()
    return merged


def _record() -> dict:
    sources = {}
    with tempfile.TemporaryDirectory() as scratch:
        for name, spec in _RECORDED_SOURCES.items():
            folder = _run_datatrove(spec, Path(scratch) / name)
            files = {path.name: path.read_bytes().hex() for path in sorted(folder.iterdir()) if path.is_file()}
            documents = _expected_files(name, spec)
            written = {file for file, payload in files.items() if file.endswith(".ds") and payload}
            if written != set(documents):
                raise SystemExit(f"{name}: datatrove wrote non-empty {sorted(written)}, expected {sorted(documents)}")
            vocabulary = Tokenizer.from_pretrained(spec.tokenizer).get_vocab_size()
            sources[name] = {
                "tokenizer": spec.tokenizer,
                "token_bytes": 2 if vocabulary <= 1 << 16 else 4,
                "documents": documents,
                "files": files,
            }
    return {
        "_about": (
            f"Written by datatrove {DATATROVE_VERSION}: JsonlReader, then DocumentTokenizer(eos_token={END_OF_TEXT!r}, "
            "shuffle_documents=False), and for 'code' DocumentTokenizerMerger(shuffle=False). 'files' holds every "
            "file of the output folder as hex; 'documents' holds each non-empty .ds file's documents, tokenized by "
            "tokenizers' Tokenizer.encode(text, add_special_tokens=False) plus the end-of-text id, apart from "
            "datatrove's writer."
        ),
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Re-record {RECORD_PATH.name} with datatrove {DATATROVE_VERSION}.")
    parser.add_argument("--write", action="store_true", help="rewrite the record (needs datatrove importable)")
    if not parser.parse_args().write:
        parser.error("pass --write to re-record")
    RECORD_PATH.write_text(json.dumps(_record(), indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
