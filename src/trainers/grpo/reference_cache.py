"""Bounded, filesystem-aware storage of offline GRPO's ragged frozen-reference scores."""

from __future__ import annotations

import errno
import os
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

import pyarrow as pa
import torch
import torch.distributed as dist
from datasets import Dataset

from src.distributed.filesystem import store_reject_across_ranks
from src.distributed.runtime import (
    DeferredRankFailure,
    broadcast_from_rank0,
    collective_device,
    fs_aware_save_rank,
    get_global_rank,
    get_global_world_size,
)

REFERENCE_BUFFER_VALUES = 1 << 18
REFERENCE_BATCH_ROWS = 256


@dataclass
class MappedReferenceScores:
    """Keep the mapped tensor owners alive for Arrow and checkpoint serialization."""

    lengths: torch.Tensor
    values: torch.Tensor
    offsets: torch.Tensor

    def column(self) -> pa.LargeListArray:
        return pa.LargeListArray.from_arrays(
            pa.array(self.offsets.numpy()), pa.array(self.values.numpy(), type=pa.float32())
        )


def reference_cache_writers() -> tuple[int, ...]:
    """Gather actual filesystem owners once, sharing the checkpoint writer predicate."""
    owner = torch.tensor([int(fs_aware_save_rank())], dtype=torch.int64, device=collective_device())
    if get_global_world_size() == 1:
        return (0,) if bool(owner.item()) else ()
    gathered = [torch.empty_like(owner) for _ in range(get_global_world_size())]
    dist.all_gather(gathered, owner)
    return tuple(rank for rank, flag in enumerate(gathered) if bool(flag.item()))


def reference_payload_mismatch(lengths, values, dataset: Dataset) -> str | None:
    """Validate ragged score storage without allocating a full-token validation plane."""
    if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.int64 or lengths.shape != (len(dataset),):
        return "its row lengths are malformed"
    if not isinstance(values, torch.Tensor) or values.dtype != torch.float32 or values.ndim != 1:
        return "its flat float32 reference values are malformed"
    total = 0
    for start, batch in zip(
        range(0, len(dataset), REFERENCE_BATCH_ROWS),
        dataset.select_columns(["completion_input_ids"]).with_format("arrow").iter(batch_size=REFERENCE_BATCH_ROWS),
        strict=True,
    ):
        expected = torch.tensor(
            batch.column("completion_input_ids").combine_chunks().value_lengths().to_numpy(), dtype=torch.int64
        )
        actual = lengths[start : start + expected.numel()]
        if not torch.equal(actual, expected):
            return "its reference lengths do not match the completions"
        total += int(actual.sum())
    if total != values.numel():
        return "its row lengths do not cover its reference values"
    for start in range(0, values.numel(), REFERENCE_BUFFER_VALUES):
        if not bool(torch.isfinite(values[start : start + REFERENCE_BUFFER_VALUES]).all()):
            return "its reference values contain non-finite log-probabilities"
    return None


def mapped_reference_scores(lengths: torch.Tensor, values: torch.Tensor) -> MappedReferenceScores:
    """Checkpoint tensors stay mmap-backed; only row-sized offsets need materializing."""
    offsets = torch.empty(lengths.numel() + 1, dtype=torch.int64)
    offsets[0] = 0
    torch.cumsum(lengths, dim=0, out=offsets[1:])
    return MappedReferenceScores(lengths, values, offsets)


class ReferenceScoreCache:
    """Stream current batches to output-FS writers, then map the ordered token table."""

    def __init__(self, output_dir: str, *, dp_size: int):
        identifier = broadcast_from_rank0(uuid.uuid4().hex if get_global_rank() == 0 else None)
        # Trainer.push_to_hub excludes underscore-prefixed scratch, including NFS's live-map remnants.
        self.directory = os.path.join(os.fspath(output_dir), "_reference_cache", identifier)
        self.dp_size = dp_size
        self.writers = reference_cache_writers()
        self._transfer_buffers = {
            dtype: torch.empty(REFERENCE_BUFFER_VALUES, dtype=dtype, device=collective_device())
            for dtype in (torch.int64, torch.float32)
        }
        guard = DeferredRankFailure("Creating the offline GRPO reference cache")
        if fs_aware_save_rank():
            guard.run(lambda: os.makedirs(self.directory))
        guard.reject()

    def _path(self, shard: int | str, kind: str) -> str:
        return os.path.join(self.directory, f"{shard}.{kind}")

    def discard(self) -> None:
        """Remove only this run's UUID cache, without requiring a healthy process group."""
        if fs_aware_save_rank() and os.path.isdir(self.directory):
            for name in os.listdir(self.directory):
                if not name.startswith(".nfs"):
                    os.unlink(os.path.join(self.directory, name))
            try:
                os.rmdir(self.directory)
            except OSError as exc:
                # NFS silly-renames an unlinked open mmap until its final reader closes it.
                if exc.errno != errno.ENOTEMPTY or any(
                    not name.startswith(".nfs") for name in os.listdir(self.directory)
                ):
                    raise

    def _append(self, shard: int, kind: str, values: torch.Tensor) -> None:
        with open(self._path(shard, kind), "ab") as destination:
            destination.write(memoryview(values.detach().cpu().contiguous().numpy()).cast("B"))

    def append_rows(self, shard: int, rows: Iterable[torch.Tensor]) -> None:
        """Local writer path, also used for exact-token evaluation reuse."""
        for row in rows:
            values = torch.as_tensor(row, dtype=torch.float32).detach().cpu()
            if values.ndim != 1 or not bool(torch.isfinite(values).all()):
                raise ValueError("Reference rows must contain finite, one-dimensional float32 scores")
            self._append(shard, "lengths", torch.tensor([values.numel()], dtype=torch.int64))
            for start in range(0, values.numel(), REFERENCE_BUFFER_VALUES):
                self._append(shard, "values", values[start : start + REFERENCE_BUFFER_VALUES])

    def collect_batch(self, rows: list[torch.Tensor] | None, representatives: dict[int, int]) -> None:
        """Only a DP representative transmits, and only filesystem writers receive score values."""
        rank, world = get_global_rank(), get_global_world_size()
        device = collective_device()
        guard = DeferredRankFailure("Collecting an offline GRPO reference batch")

        def pack():
            lengths = torch.tensor([value.numel() for value in rows], dtype=torch.int64)
            values = torch.cat(rows).float().cpu() if rows else torch.empty(0, dtype=torch.float32)
            if any(value.ndim != 1 for value in rows) or not bool(torch.isfinite(values).all()):
                raise ValueError("Reference batch contains malformed or non-finite scores")
            return lengths, values

        packed = guard.run(pack) if rows is not None else None
        header = torch.tensor(
            [packed[0].numel(), packed[1].numel()] if packed is not None else [0, 0],
            dtype=torch.int64,
            device=device,
        )
        if world > 1:
            headers = torch.empty(world * 2, dtype=torch.int64, device=device)
            dist.all_gather_into_tensor(headers, header)
        else:
            headers = header
        batch_sizes = headers.reshape(world, 2).cpu().tolist()
        for shard, source in representatives.items():
            row_count, value_count = batch_sizes[source]
            if not row_count:
                continue
            for writer in self.writers:
                for kind, count, dtype in (
                    ("lengths", row_count, torch.int64),
                    ("values", value_count, torch.float32),
                ):
                    for start in range(0, count, REFERENCE_BUFFER_VALUES):
                        size = min(REFERENCE_BUFFER_VALUES, count - start)
                        buffer = self._transfer_buffers[dtype][:size]
                        if rank == source:
                            tensor = packed[0] if kind == "lengths" else packed[1]
                            buffer.copy_(tensor[start : start + size])
                            if writer != source:
                                dist.send(buffer, dst=writer)
                        elif rank == writer:
                            dist.recv(buffer, src=source)
                        if rank == writer:
                            guard.run(lambda buffer=buffer, kind=kind, shard=shard: self._append(shard, kind, buffer))
        guard.reject()

    def _map(self, shard: int | str) -> MappedReferenceScores:
        tensors = []
        for kind, dtype in (("lengths", torch.int64), ("values", torch.float32)):
            path = self._path(shard, kind)
            itemsize = torch.tensor([], dtype=dtype).element_size()
            if os.path.getsize(path) % itemsize:
                raise ValueError(f"Malformed reference cache '{path}': truncated {kind}")
            tensors.append(torch.from_file(path, shared=False, size=os.path.getsize(path) // itemsize, dtype=dtype))
        return mapped_reference_scores(*tensors)

    def finish(self, dataset: Dataset) -> MappedReferenceScores:
        try:
            return self._finish(dataset)
        except BaseException as failure:
            try:
                self.discard()
            except Exception as cleanup_failure:
                failure.add_note(f"Reference scratch cleanup also failed: {cleanup_failure}")
            raise

    def _finish(self, dataset: Dataset) -> MappedReferenceScores:
        guard = DeferredRankFailure("Completing the offline GRPO reference cache", exc_type=ValueError)

        def merge():
            for kind in ("lengths", "values"):
                with open(self._path("merged", kind), "wb") as destination:
                    for shard in range(self.dp_size):
                        path = self._path(shard, kind)
                        if os.path.exists(path):
                            with open(path, "rb") as source:
                                shutil.copyfileobj(source, destination, length=REFERENCE_BUFFER_VALUES * 4)
            mapped = self._map("merged")
            mismatch = reference_payload_mismatch(mapped.lengths, mapped.values, dataset)
            if mismatch:
                raise ValueError(f"Incomplete offline GRPO reference cache: {mismatch}")

        if fs_aware_save_rank():
            guard.run(merge)
        store_reject_across_ranks(
            f"reference-cache/{os.path.basename(self.directory)}/merge", guard.reason, guard.what, exc_type=ValueError
        )
        guard = DeferredRankFailure("Mapping the offline GRPO reference cache", exc_type=ValueError)
        mapped = guard.run(lambda: self._map("merged"))
        guard.reject()
        # Every reader must map first; on Linux the tensors retain the unlinked backing storage.
        guard = DeferredRankFailure("Removing the offline GRPO reference scratch files")
        guard.run(self.discard)
        guard.reject()
        return mapped
