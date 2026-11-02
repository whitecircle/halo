#!/usr/bin/env python
"""The streaming shard writer's background part writes change when bytes reach disk, never which bytes.

Entered as a context manager, ``StreamingShardWriter`` hands each finished part to
``HALO_CHECKPOINT_WRITE_THREADS`` background threads and fills the next part meanwhile. Pinned here:
every file a gathered save writes (dense and every EP-gathered MoE family) is byte-identical to the
in-line writer's; the parts in flight stay within the thread count, which is what bounds the writer's
host RAM; ``close()`` returns only once every part it names is on disk, and names the files already
there (an external table's) beside them; a model's external table is written in line while its parts
are still in flight, and the finalize waits those out before naming both in the index; a failed
background write raises its own exception only once no other write is running, and on every rank of a
gathered save, through the stream's guard or the finalize's; the block's exit waits out every write
whatever the caller raised, logging a failed one the caller never collected; outside a block (the
pipeline-stage writer) parts are written in line; a negative thread count raises when any writer is built, a
pipeline stage's never-entered one included.

    python tests/cpu/checkpoint/test_background_shard_writer.py
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import weakref
from functools import partial
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

import halo.checkpoint.parallel.writing as gathered_writing
import halo.checkpoint.writing as writing
from halo.checkpoint.format import PP_WRAPPER_PART_PREFIX, SAFETENSORS_INDEX_FILE
from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.parallel.save import save_fsdp2_checkpoint
from halo.checkpoint.parallel.writing import chunked_saveable_tensors, stream_gathered_checkpoint
from halo.checkpoint.writing import (
    StreamingShardWriter,
    normalize_gathered_state_dict,
    resolve_checkpoint_write_threads,
)
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from tests.common.checkpoint_io import file_digests, loading_problems, save_context
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN4_EXP_CONFIG
from tests.common.parallelism import single_process_ep_config
from tests.common.qwen4_exp import (
    NGRAM_TABLE_PATH,
    build_tiny_qwen4_exp_checkpoint,
    load_host_table_model,
    transformers_forwards,
)
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

THREADS_ENV = "HALO_CHECKPOINT_WRITE_THREADS"
WRITER_THREAD_PREFIX = "halo-shard-writer"
# Small enough that every tiny model below splits into several parts.
SHARD_SIZE = "16KB"
# The tiny Flash-Next model's streamed weights in a handful of parts, so slowing each write keeps the test short.
TABLE_MODEL_SHARD_SIZE = "256KB"
WORLD_SIZE = 2
_REAL_SAVE = writing.save_safetensors_file


def _record_write_threads(monkeypatch) -> list[str]:
    """Route every part write through a recorder of the thread it ran on."""
    real_save = writing.save_safetensors_file
    threads: list[str] = []

    def recording_save(tensors, path, **kwargs):
        threads.append(threading.current_thread().name)
        return real_save(tensors, path, **kwargs)

    monkeypatch.setattr(writing, "save_safetensors_file", recording_save)
    return threads


def _save_with_threads(monkeypatch, tmp_path, threads: int, save) -> tuple[dict[str, str], list[str]]:
    """``save(output_dir)`` under ``threads`` background writers: the files' digests and the threads that wrote."""
    output = tmp_path / f"threads{threads}"
    output.mkdir()
    with monkeypatch.context() as patch:
        patch.setenv(THREADS_ENV, str(threads))
        write_threads = _record_write_threads(patch)
        save(str(output))
    return file_digests(output), write_threads


def _assert_identical_across_thread_counts(monkeypatch, tmp_path, save) -> None:
    inline, inline_threads = _save_with_threads(monkeypatch, tmp_path, 0, save)
    assert len(inline_threads) > 2, f"premise: the save splits into several parts, wrote {len(inline_threads)}"
    assert all(not name.startswith(WRITER_THREAD_PREFIX) for name in inline_threads)
    for threads in (1, 2):
        background, background_threads = _save_with_threads(monkeypatch, tmp_path, threads, save)
        assert len(background_threads) == len(inline_threads)
        assert all(name.startswith(WRITER_THREAD_PREFIX) for name in background_threads), (
            f"{threads} write threads: a part was written in line, so the comparison proves nothing"
        )
        assert background == inline, f"{threads} write threads changed the bytes on disk"


def _tensors(count: int):
    """``count`` distinct seeded 1 MB float32 tensors, made one at a time so the caller alone holds each."""
    generator = torch.Generator().manual_seed(count)
    for i in range(count):
        yield f"w{i}", torch.randn(250_000, generator=generator)


def test_the_stream_writes_the_same_bytes_with_any_thread_count(tmp_path, monkeypatch):
    """Parts of uneven size, an oversized tensor flushed alone, mixed dtypes: same files, same index."""
    staged = {
        **{key: tensor.to(torch.bfloat16) if i % 3 else tensor for i, (key, tensor) in enumerate(_tensors(9))},
        "huge": torch.randn(2_000_000),
        **{f"t{i}": torch.randn(1000) for i in range(5)},
    }

    def save(output_dir):
        with StreamingShardWriter(output_dir, "2MB", enabled=True) as writer:
            for key, tensor in staged.items():
                writer.add(key, tensor)
            writer.close_as_hf_checkpoint()

    _assert_identical_across_thread_counts(monkeypatch, tmp_path, save)


def test_a_dense_gathered_save_writes_the_same_bytes_with_any_thread_count(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(**{**TINY_QWEN3_CONFIG, "num_hidden_layers": 3, "vocab_size": 256}))

    def save(output_dir):
        stream_gathered_checkpoint(
            model,
            chunked_saveable_tensors(model, retain=True),
            output_dir,
            is_save_rank=True,
            max_shard_size=SHARD_SIZE,
            normalize=partial(normalize_gathered_state_dict, model),
        )

    _assert_identical_across_thread_counts(monkeypatch, tmp_path, save)


@pytest.mark.parametrize("family", sorted(TINY_MOE_FAMILIES))
def test_an_ep_gathered_save_writes_the_same_bytes_with_any_thread_count(family, tmp_path, monkeypatch):
    """Every MoE family's gathered EP save: its own expert layout, hub respell and save-dtype cast per layer."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[family])
    apply_ep_layers(model, single_process_ep_config())

    def save(output_dir):
        save_ep_gathered_checkpoint(save_context(model, has_ep_layers=True, max_shard_size=SHARD_SIZE), output_dir)

    _assert_identical_across_thread_counts(monkeypatch, tmp_path, save)


class _SlowWrites:
    """A part write that takes ``seconds``, counting the writes running at once and failing the parts named.

    Before that, a write is held until ``together`` writes are running and until ``release`` is set, each
    for at most ``hold_seconds``: a held write is still running however slowly the caller proceeds. With
    ``held_parts``, only those parts wait on ``release``.
    """

    def __init__(
        self,
        monkeypatch,
        seconds: float,
        fail_parts: tuple[int, ...] = (),
        *,
        together: int = 1,
        release: threading.Event | None = None,
        hold_seconds: float = 1.0,
        held_parts: tuple[int, ...] | None = None,
    ):
        self.seconds = seconds
        self.fail_parts = fail_parts
        self.together = together
        self.release = release
        self.hold_seconds = hold_seconds
        self.held_parts = held_parts
        self.lock = threading.Condition()
        self.running = 0
        self.peak_running = 0
        self.finished: list[int] = []
        self.parts: list[dict] = []
        self._real = writing.save_safetensors_file
        monkeypatch.setattr(writing, "save_safetensors_file", self)

    def __call__(self, tensors, path, **kwargs):
        part = int(os.path.basename(path).rsplit("-", 1)[1].split(".")[0])
        with self.lock:
            self.parts.append(tensors)
            self.running += 1
            self.peak_running = max(self.peak_running, self.running)
            self.lock.notify_all()
            self.lock.wait_for(lambda: self.running >= self.together, timeout=self.hold_seconds)
        try:
            if self.release is not None and (self.held_parts is None or part in self.held_parts):
                self.release.wait(timeout=self.hold_seconds)
            time.sleep(self.seconds)
            if part in self.fail_parts:
                raise OSError(28, f"No space left on device writing part {part}")
            self._real(tensors, path, **kwargs)
        finally:
            with self.lock:
                self.running -= 1
                self.finished.append(part)


@pytest.mark.parametrize("threads", [1, 2])
def test_parts_in_flight_never_exceed_the_thread_count(tmp_path, monkeypatch, threads):
    """The bound on host RAM: the part filling plus one per thread, each released once written."""
    monkeypatch.setenv(THREADS_ENV, str(threads))
    writes = _SlowWrites(monkeypatch, seconds=0.05, together=threads)
    refs, peak_alive = [], 0
    with StreamingShardWriter(str(tmp_path), "2MB", enabled=True) as writer:
        assert writer.max_resident_bytes == (1 + threads) * writer.max_bytes
        for key, tensor in _tensors(24):
            refs.append(weakref.ref(tensor))
            writer.add(key, tensor)
            del tensor
            peak_alive = max(peak_alive, sum(ref() is not None for ref in refs))
        writer.close()
    assert writes.peak_running == threads, f"{writes.peak_running} parts were written at once, not {threads}"
    # Two 1 MB tensors per 2 MB part: the filling part plus one part per thread.
    assert peak_alive <= 2 * (1 + threads), f"{peak_alive} tensors alive at once"
    assert all(ref() is None for ref in refs), "the writer kept a written part alive past close"
    # Emptied before each write reported done, so no flush waiting on one held it beside the next part.
    assert writes.parts and not any(writes.parts), "a written part kept its tensors after its write finished"


def _writer_errors(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == writing.logger.name and r.levelno >= logging.ERROR]


@pytest.mark.parametrize("threads", [1, 2])
def test_a_failed_write_raises_its_own_error_once_no_write_is_running(tmp_path, monkeypatch, threads):
    """The error surfaces from the next flush or the close, as its own type, with every other write done,
    so a caller's cleanup never races a thread still writing; every later call and the block's exit
    raise it again."""
    monkeypatch.setenv(THREADS_ENV, str(threads))
    writes = _SlowWrites(monkeypatch, seconds=0.05, fail_parts=(2,))
    writer = StreamingShardWriter(str(tmp_path), "2MB", enabled=True)
    with pytest.raises(OSError, match="writing part 2"), writer:
        with pytest.raises(OSError, match="writing part 2"):
            for key, tensor in _tensors(24):
                writer.add(key, tensor)
            writer.close_as_hf_checkpoint()
        assert writes.running == 0, "a write was still running when the failure reached the caller"
        assert 2 in writes.finished
        with pytest.raises(OSError, match="writing part 2"):
            writer.close()
        # Swallowed by the caller, it raises once more at the block's exit.
    assert not (tmp_path / "model.safetensors.index.json").exists()


def test_a_write_error_raised_through_the_block_is_not_logged_again(tmp_path, monkeypatch, caplog):
    """The caller already propagates the write error, so the exit has no uncollected failure to log."""
    monkeypatch.setenv(THREADS_ENV, "1")
    _SlowWrites(monkeypatch, seconds=0.0, fail_parts=(1,))
    with caplog.at_level(logging.ERROR, logger=writing.logger.name), pytest.raises(OSError, match="writing part 1"):
        with StreamingShardWriter(str(tmp_path), "2MB", enabled=True) as writer:
            for key, tensor in _tensors(6):
                writer.add(key, tensor)
            writer.close()
    assert not _writer_errors(caplog), "the exit logged a write error the caller was already raising"


@pytest.mark.parametrize("threads", [1, 2])
def test_close_returns_only_once_every_part_it_names_is_on_disk(tmp_path, monkeypatch, threads):
    """``close()`` hands off the last part itself, so its own join is all that keeps the fragment it
    returns (published by the index exchange) from naming a part still being written."""
    monkeypatch.setenv(THREADS_ENV, str(threads))
    release = threading.Event()
    _SlowWrites(monkeypatch, seconds=0.0, release=release, hold_seconds=0.3)
    with StreamingShardWriter(str(tmp_path), "2MB", enabled=True) as writer:
        for key, tensor in _tensors(5):
            writer.add(key, tensor)
        weight_map, _ = writer.close()
        missing = sorted(name for name in set(weight_map.values()) if not (tmp_path / name).is_file())
        release.set()
    assert len(set(weight_map.values())) == 3, "premise: three parts, the last flushed by close()"
    assert not missing, f"close() returned {missing} before they were written"


@pytest.mark.parametrize("threads", [1, 2])
def test_close_names_written_files_beside_its_parts(tmp_path, monkeypatch, threads):
    """A multi-writer save's fragment holds the files already on disk (an external table's) beside every part,
    the last one joined first, and counts both in its bytes."""
    monkeypatch.setenv(THREADS_ENV, str(threads))
    release = threading.Event()
    _SlowWrites(monkeypatch, seconds=0.0, release=release, hold_seconds=0.3)
    table_file, table_bytes = "model-table00-00001-of-00001.safetensors", 4096
    (tmp_path / table_file).touch()
    with StreamingShardWriter(str(tmp_path), "2MB", enabled=True) as writer:
        for key, tensor in _tensors(5):
            writer.add(key, tensor)
        writer.add_written_files({"table.0": table_file}, table_bytes)
        weight_map, total_bytes = writer.close()
        missing = sorted(name for name in set(weight_map.values()) if not (tmp_path / name).is_file())
        release.set()
    assert weight_map["table.0"] == table_file
    assert len(set(weight_map.values()) - {table_file}) == 3, "premise: three parts, the last flushed by close()"
    assert not missing, f"close() returned {missing} before they were written"
    assert total_bytes == 5 * 1_000_000 + table_bytes


@pytest.mark.parametrize("threads", [1, 2])
def test_a_failed_last_part_raises_from_close_itself(tmp_path, monkeypatch, threads):
    """The part ``close()`` flushes fails with no later ``add`` to report it: ``close()`` raises it."""
    monkeypatch.setenv(THREADS_ENV, str(threads))
    writes = _SlowWrites(monkeypatch, seconds=0.0, fail_parts=(3,), release=threading.Event(), hold_seconds=0.3)
    writer = StreamingShardWriter(str(tmp_path), "2MB", enabled=True)
    with pytest.raises(OSError, match="writing part 3"), writer:
        for key, tensor in _tensors(5):
            writer.add(key, tensor)
        with pytest.raises(OSError, match="writing part 3"):
            writer.close()
        assert sorted(writes.finished) == [1, 2, 3]


@pytest.mark.parametrize("threads", [1, 2])
def test_an_external_table_is_indexed_beside_parts_still_in_flight(tmp_path, monkeypatch, threads):
    """A gathered save of a model holding an external table (Qwen3.8-Flash-Next's host n-gram table) writes the
    table's files in line while streamed parts are still being written; the finalize waits those out before it
    renames them, and its index names both. The files equal the in-line writer's, and the save reloads whole."""
    source, background, inline = (str(tmp_path / name) for name in ("source", "background", "inline"))
    build_tiny_qwen4_exp_checkpoint(source)
    os.makedirs(background)
    os.makedirs(inline)
    model = load_host_table_model(source, dtype=torch.bfloat16)
    ids = torch.randint(0, TINY_QWEN4_EXP_CONFIG["vocab_size"], (2, 29), generator=torch.Generator().manual_seed(0))
    with torch.no_grad(), transformers_forwards():
        expected = model.eval()(input_ids=ids, use_cache=False).logits
    table_keys = {
        f"{NGRAM_TABLE_PATH}.shard_{index}.weight" for index in range(TINY_QWEN4_EXP_CONFIG["split_ngram_parts"])
    }
    monkeypatch.setenv(THREADS_ENV, "0")
    save_fsdp2_checkpoint(save_context(model, max_shard_size=TABLE_MODEL_SHARD_SIZE), inline)
    inline_map = json.loads(Path(inline, SAFETENSORS_INDEX_FILE).read_text())["weight_map"]
    streamed_parts = len(set(inline_map.values()) - {inline_map[key] for key in table_keys})
    assert streamed_parts > 2, "premise: several parts"

    write_tables = gathered_writing.write_external_tables
    in_flight_at_tables: list[int] = []
    release = threading.Event()
    with monkeypatch.context() as patch:
        patch.setenv(THREADS_ENV, str(threads))
        # The finalize flushes the last part, so the one before it is the last in flight beside the tables: held
        # until they are written, however long the save takes to reach them.
        writes = _SlowWrites(patch, seconds=0.0, release=release, hold_seconds=60.0, held_parts=(streamed_parts - 1,))

        def write_tables_mid_stream(table_model, writer):
            with writes.lock:
                # A submitted part counts once its thread runs it; the hold keeps it running from then on.
                writes.lock.wait_for(lambda: writes.running >= 1, timeout=30)
                in_flight_at_tables.append(writes.running)
            try:
                write_tables(table_model, writer)
            finally:
                release.set()

        patch.setattr(gathered_writing, "write_external_tables", write_tables_mid_stream)
        save_fsdp2_checkpoint(save_context(model, max_shard_size=TABLE_MODEL_SHARD_SIZE), background)

    weight_map = json.loads(Path(background, SAFETENSORS_INDEX_FILE).read_text())["weight_map"]
    assert table_keys <= set(weight_map), "the index lost the table's files"
    assert in_flight_at_tables and in_flight_at_tables[0] >= 1, "premise: no part was in flight beside the table"
    assert file_digests(background) == file_digests(inline), "background writes changed the bytes on disk"
    reloaded, info = AutoModelForCausalLM.from_pretrained(background, dtype=torch.bfloat16, output_loading_info=True)
    assert loading_problems(info) == {}
    with torch.no_grad(), transformers_forwards():
        assert torch.equal(reloaded.eval()(input_ids=ids, use_cache=False).logits, expected)


def test_the_block_exit_waits_out_every_write_and_logs_a_failed_one(tmp_path, monkeypatch, caplog):
    """A caller failing mid-stream leaves writes in flight: the exit waits them out, lets the caller's
    own exception through and logs the failed write it never collected, which may be the cause."""
    monkeypatch.setenv(THREADS_ENV, "2")
    release = threading.Event()
    writes = _SlowWrites(monkeypatch, seconds=0.2, fail_parts=(1,), release=release, hold_seconds=10.0)
    with caplog.at_level(logging.ERROR, logger=writing.logger.name):
        with pytest.raises(ValueError, match="the caller's own failure"):
            with StreamingShardWriter(str(tmp_path), "2MB", enabled=True) as writer:
                for key, tensor in _tensors(5):
                    writer.add(key, tensor)
                with writes.lock:
                    assert writes.lock.wait_for(lambda: writes.running == 2, timeout=5), "premise: two writes held"
                assert writes.finished == [], "premise: no write finished before the caller fails"
                release.set()
                raise ValueError("the caller's own failure")
    assert writes.running == 0
    assert sorted(writes.finished) == [1, 2]
    [record] = _writer_errors(caplog)
    assert "ValueError" in record.getMessage()
    assert isinstance(record.exc_info[1], OSError) and "writing part 1" in str(record.exc_info[1])


def test_a_write_failure_no_close_reported_raises_at_the_block_exit(tmp_path, monkeypatch):
    monkeypatch.setenv(THREADS_ENV, "1")
    _SlowWrites(monkeypatch, seconds=0.0, fail_parts=(1,))
    with pytest.raises(OSError, match="writing part 1"):
        with StreamingShardWriter(str(tmp_path), "2MB", enabled=True) as writer:
            for key, tensor in _tensors(3):
                writer.add(key, tensor)


def test_outside_a_block_and_at_zero_threads_every_part_is_written_in_line(tmp_path, monkeypatch):
    """A writer never entered (the pipeline-stage saver) runs no thread to outlive its caller."""
    monkeypatch.setenv(THREADS_ENV, "2")
    write_threads = _record_write_threads(monkeypatch)
    writer = StreamingShardWriter(str(tmp_path / "plain"), "2MB", enabled=True)
    os.makedirs(writer.output_dir)
    for key, tensor in _tensors(6):
        writer.add(key, tensor)
    writer.close()
    assert writer.max_resident_bytes == writer.max_bytes
    monkeypatch.setenv(THREADS_ENV, "0")
    os.makedirs(tmp_path / "zero")
    with StreamingShardWriter(str(tmp_path / "zero"), "2MB", enabled=True) as writer:
        for key, tensor in _tensors(6):
            writer.add(key, tensor)
        writer.close()
    assert len(write_threads) > 2
    assert write_threads == [threading.main_thread().name] * len(write_threads)


@pytest.mark.parametrize(
    ("raw", "threads", "warns"), [(None, 0, False), ("2", 2, False), ("0", 0, False), ("x", 0, True)]
)
def test_the_thread_count_knob(monkeypatch, caplog, raw, threads, warns):
    if raw is None:
        monkeypatch.delenv(THREADS_ENV, raising=False)
    else:
        monkeypatch.setenv(THREADS_ENV, raw)
    with caplog.at_level(logging.WARNING):
        assert resolve_checkpoint_write_threads() == threads
    assert any(THREADS_ENV in r.getMessage() for r in caplog.records) == warns


@pytest.mark.parametrize("enabled", [True, False], ids=["save_rank", "peer_rank"])
def test_a_negative_thread_count_raises_on_every_rank_before_the_block(tmp_path, monkeypatch, enabled):
    """The save rank and every peer refuse it building the writer, before a gathered save's first collective."""
    monkeypatch.setenv(THREADS_ENV, "-3")
    entered = False
    with pytest.raises(ValueError, match="HALO_CHECKPOINT_WRITE_THREADS=-3"):
        with StreamingShardWriter(str(tmp_path), SHARD_SIZE, enabled=enabled):
            entered = True
    assert not entered
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("enabled", [True, False], ids=["stage_writer_rank", "stage_peer_rank"])
def test_a_negative_thread_count_refuses_the_pipeline_stage_writer(tmp_path, monkeypatch, enabled):
    """A pipeline stage builds its writer and writes in line, never entering a block; the count is refused there
    too, as a stage's writer is built on every stage rank ahead of the stage's gathers."""
    monkeypatch.setenv(THREADS_ENV, "-1")
    with pytest.raises(ValueError, match="HALO_CHECKPOINT_WRITE_THREADS=-1"):
        StreamingShardWriter(str(tmp_path), SHARD_SIZE, enabled=enabled, filename_prefix=PP_WRAPPER_PART_PREFIX)


def _raise_on_writer_thread(fail_part: int, tensors, path, **kwargs):
    """Rank 0's part write: part ``fail_part`` fails, and only on a background thread."""
    part = int(os.path.basename(path).rsplit("-", 1)[1].split(".")[0])
    if part == fail_part and threading.current_thread().name.startswith(WRITER_THREAD_PREFIX):
        raise OSError(28, f"No space left on device writing part {part}")
    return _REAL_SAVE(tensors, path, **kwargs)


def _failing_stream_worker(rank: int, root: str, fail_part: int) -> None:
    """Rank 0 writes four one-tensor parts of a gathered save; part ``fail_part`` fails in the background."""
    if rank == 0:
        writing.save_safetensors_file = partial(_raise_on_writer_thread, fail_part)
    chunks = [{f"layer{i}.weight": torch.full((64_000,), float(i))} for i in range(4)]
    try:
        stream_gathered_checkpoint(
            torch.nn.Linear(2, 2),
            chunks,
            os.path.join(root, "checkpoint"),
            is_save_rank=rank == 0,
            max_shard_size="300KB",
            normalize=dict,
        )
        outcome = "saved"
    except Exception as exc:  # the verdict file must be written whatever the rank raised
        outcome = f"{type(exc).__name__}: {exc}"
    with open(os.path.join(root, f"outcome_{rank}.json"), "w") as fh:
        json.dump(outcome, fh)


@pytest.mark.parametrize(
    ("fail_part", "seam"),
    [(1, "gathered checkpoint write"), (4, "gathered checkpoint finalize")],
    ids=["mid-stream", "last-part"],
)
def test_a_background_write_failure_raises_on_every_rank(tmp_path, fail_part, seam):
    """A mid-stream part fails the stream's guard (surfaced at a later flush), the last part the
    finalize's: either way both ranks raise naming the write, rather than the peer finalizing an index
    or waiting in the next gather."""
    os.makedirs(tmp_path / "checkpoint")
    run_gloo_ranks(_failing_stream_worker, WORLD_SIZE, str(tmp_path), fail_part, env={THREADS_ENV: "1"})
    outcomes = [json.loads((tmp_path / f"outcome_{rank}.json").read_text()) for rank in range(WORLD_SIZE)]
    for rank, outcome in enumerate(outcomes):
        assert outcome != "saved", f"rank {rank} finished a save whose part {fail_part} was never written"
        assert seam in outcome and f"writing part {fail_part}" in outcome, f"rank {rank}: {outcome}"
    assert not (tmp_path / "checkpoint" / "model.safetensors.index.json").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
