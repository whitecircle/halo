"""Bounded, filesystem-aware storage of offline GRPO's ragged frozen-reference scores.

Every file of a launch lives under ``output_dir/_reference_cache/<launch>/``, mapped by every rank, and
nothing is unlinked while mapped (on NFS a removed file stays readable only on the unlinking host). A launch
holds an exclusive lock on ``<launch>.lock`` until it exits; a new cache removes launches whose lock it takes.
"""

from __future__ import annotations

import errno
import fcntl
import functools
import logging
import os
import shutil
import uuid
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass

import pyarrow as pa
import torch
import torch.distributed as dist
from datasets import Dataset

from halo.checkpoint.atomic import link_or_copy_file
from halo.checkpoint.format import REFERENCE_CACHE_DIR_NAME
from halo.env import nccl_timeout
from halo.runtime.collectives import all_gather_cat, broadcast_from_rank0, collective_device
from halo.runtime.consensus import DeferredRankFailure
from halo.runtime.filesystem import is_fs_save_rank
from halo.runtime.ranks import get_global_rank, get_global_world_size
from halo.trainers.reference_logps import REFERENCE_SCAN_ROWS

logger = logging.getLogger(__name__)

# Values per transfer buffer, merge copy, validation slice and digest update: bounds the device
# buffers and host copies whatever the table's size (1 MiB of float32 scores, 2 MiB of int64 lengths).
REFERENCE_BUFFER_VALUES = 1 << 18
# The two flat files of a cache, in the order every writer appends and every reader maps them.
_PAYLOAD_KINDS = (("lengths", torch.int64), ("values", torch.float32))
# NFS's stand-in for a removed file some process still has open; removing it breaks that process.
_NFS_REMNANT_PREFIX = ".nfs"
# The descriptors holding this process's launch locks, by scratch root; closed only by process exit.
_held_launch_locks: dict[str, int | None] = {}


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


@functools.cache
def _launch_id() -> str:
    """This launch's scratch directory name. COLLECTIVE on the first call, which every rank makes from
    its first reference cache, in lockstep."""
    return broadcast_from_rank0(uuid.uuid4().hex if get_global_rank() == 0 else None)


def _remove_scratch(path: str) -> None:
    """Remove a scratch tree, keeping the ``.nfs*`` remnants of files a live process still maps and the
    directories that hold them."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            _remove_scratch(entry.path)
        elif not entry.name.startswith(_NFS_REMNANT_PREFIX):
            os.unlink(entry.path)
    try:
        os.rmdir(path)
    except OSError as exc:
        # Unlinking a file this host still maps leaves its .nfs* stand-in, here or in a kept subdirectory.
        leftovers = [name for name in os.listdir(path) if not name.startswith(_NFS_REMNANT_PREFIX)]
        if exc.errno != errno.ENOTEMPTY or not all(os.path.isdir(os.path.join(path, name)) for name in leftovers):
            raise


def _lock_path(root: str, launch: str) -> str:
    """The lock file beside a launch's directory, held exclusively by that launch's writer while it runs."""
    return os.path.join(root, f"{launch}.lock")


def _try_lock(path: str) -> int | None:
    """An exclusive, non-blocking lock on ``path`` (created if absent), or None where it is held or the
    mount takes no locks."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(descriptor)
        return None
    return descriptor


def _hold_launch_lock(root: str, launch: str) -> None:
    """Lock this launch's scratch for the life of the process, before its directory exists, so a
    directory another launch finds is always one whose lock tells whether its writer still runs."""
    if root in _held_launch_locks:
        return
    _held_launch_locks[root] = _try_lock(_lock_path(root, launch))
    if _held_launch_locks[root] is None:
        logger.warning(
            "Could not lock %s: the mount takes no locks, so other launches in this output_dir keep this "
            "launch's reference scratch, and this launch keeps theirs.",
            _lock_path(root, launch),
        )


def _remove_finished_launches(root: str, launch: str) -> None:
    """Remove the other launches' scratch whose lock nothing holds. A held lock, or one the mount cannot
    take, keeps it: a live launch's mapped files stay, and an unprovable verdict never deletes."""
    for entry in os.scandir(root):
        if entry.name == launch or not entry.is_dir(follow_symlinks=False):
            continue
        lock = _lock_path(root, entry.name)
        try:
            descriptor = _try_lock(lock)
        except OSError:  # a lock file it cannot even open proves nothing either
            descriptor = None
        if descriptor is None:
            continue
        try:
            with suppress(FileNotFoundError):
                _remove_scratch(entry.path)
            with suppress(FileNotFoundError):
                os.unlink(lock)
        finally:
            os.close(descriptor)


def _create_launch_cache_directory(output_dir: str | os.PathLike) -> str:
    """COLLECTIVE. A fresh directory in this launch's scratch; its writers first remove every other
    launch's whose writer no longer runs (:func:`_remove_finished_launches`)."""
    root = os.path.join(os.fspath(output_dir), REFERENCE_CACHE_DIR_NAME)
    launch = _launch_id()
    directory = os.path.join(root, launch, broadcast_from_rank0(uuid.uuid4().hex if get_global_rank() == 0 else None))

    def create():
        os.makedirs(root, exist_ok=True)
        _hold_launch_lock(root, launch)
        _remove_finished_launches(root, launch)
        os.makedirs(directory)

    guard = DeferredRankFailure("Creating the offline GRPO reference cache")
    if is_fs_save_rank():
        guard.run(create)
    guard.reject()
    return directory


def link_checkpoint_into_scratch(output_dir: str | os.PathLike, source: str) -> str:
    """COLLECTIVE. Link (or copy) a checkpoint file into this launch's scratch and return the linked path.

    A resume maps the scratch name, so the mapping never holds the checkpoint's own entry, which
    rotation removes: on NFS a mapped entry's removal leaves an ``.nfs*`` file that keeps the
    checkpoint directory alive and freshly modified, and the newest-mtime rotation then keeps it over a
    complete checkpoint. A source absent from a writer's filesystem stays absent from its scratch, for
    the caller's presence verdict.
    """
    linked = os.path.join(_create_launch_cache_directory(output_dir), os.path.basename(source))
    guard = DeferredRankFailure(f"Staging {source} into the reference scratch")
    if is_fs_save_rank() and os.path.isfile(source):
        guard.run(lambda: link_or_copy_file(source, linked))
    guard.reject()
    return linked


def reference_cache_writers() -> tuple[int, ...]:
    """COLLECTIVE. The global ranks that write the output filesystem, by the checkpoint writer predicate."""
    owner = torch.tensor([int(is_fs_save_rank())], dtype=torch.int64, device=collective_device())
    gathered = all_gather_cat(owner, 0, None, get_global_world_size())
    return tuple(rank for rank, flag in enumerate(gathered.tolist()) if flag)


def reference_payload_mismatch(lengths, values, dataset: Dataset) -> str | None:
    """Validate ragged score storage without allocating a full-token validation plane."""
    if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.int64 or lengths.shape != (len(dataset),):
        return "its row lengths are malformed"
    if not isinstance(values, torch.Tensor) or values.dtype != torch.float32 or values.ndim != 1:
        return "its flat float32 reference values are malformed"
    total = 0
    for start, batch in zip(
        range(0, len(dataset), REFERENCE_SCAN_ROWS),
        dataset.select_columns(["completion_input_ids"]).with_format("arrow").iter(batch_size=REFERENCE_SCAN_ROWS),
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
    """Stream current batches to output-FS writers, then map the ordered token table for the launch."""

    def __init__(self, output_dir: str | os.PathLike, *, dp_size: int):
        self.directory = _create_launch_cache_directory(output_dir)
        self.dp_size = dp_size
        self.writers = reference_cache_writers()
        self._transfer_buffers = {
            dtype: torch.empty(REFERENCE_BUFFER_VALUES, dtype=dtype, device=collective_device())
            for _, dtype in _PAYLOAD_KINDS
        }
        self._transfer_group: dist.ProcessGroup | None = None

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
        """COLLECTIVE. Only a DP representative transmits, and only filesystem writers receive score values."""
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
            [packed[0].numel(), packed[1].numel()] if packed is not None else [0, 0], dtype=torch.int64, device=device
        )
        if world > 1:
            headers = torch.empty(world * 2, dtype=torch.int64, device=device)
            dist.all_gather_into_tensor(headers, header)
            group = self._transfer_group_for(representatives)
        else:
            headers = header
            group = None
        batch_sizes = headers.reshape(world, 2).cpu().tolist()
        for shard, source in representatives.items():
            if not batch_sizes[source][0]:
                continue
            for writer in self.writers:
                for (kind, dtype), count, tensor in zip(
                    _PAYLOAD_KINDS, batch_sizes[source], packed or (None, None), strict=True
                ):
                    for start in range(0, count, REFERENCE_BUFFER_VALUES):
                        size = min(REFERENCE_BUFFER_VALUES, count - start)
                        buffer = self._transfer_buffers[dtype][:size]
                        if rank == source:
                            buffer.copy_(tensor[start : start + size])
                            if writer != source:
                                dist.send(buffer, dst=writer, group=group)
                        elif rank == writer:
                            dist.recv(buffer, src=source, group=group)
                        if rank == writer:
                            guard.run(lambda buffer=buffer, kind=kind, shard=shard: self._append(shard, kind, buffer))
        guard.reject()

    def finish(self, dataset: Dataset) -> MappedReferenceScores:
        """COLLECTIVE. Merge, validate and map the scores, which stay on disk for the launch."""
        try:
            return self._finish(dataset)
        except BaseException as failure:
            try:
                self.discard()
            except Exception as cleanup_failure:
                failure.add_note(f"Reference scratch cleanup also failed: {cleanup_failure}")
            raise
        finally:
            # Every member gets here together: _finish joins its failures across ranks.
            if self._transfer_group is not None:
                dist.destroy_process_group(self._transfer_group)
                self._transfer_group = None

    def discard(self) -> None:
        """Remove this cache after a failure, without requiring a healthy process group. A transfer group
        is left to process teardown: the failure may have stranded a transfer on it."""
        if is_fs_save_rank() and os.path.isdir(self.directory):
            _remove_scratch(self.directory)

    def _append(self, shard: int, kind: str, values: torch.Tensor) -> None:
        with open(self._path(shard, kind), "ab") as destination:
            destination.write(memoryview(values.detach().cpu().contiguous().numpy()).cast("B"))

    def _path(self, shard: int | str, kind: str) -> str:
        return os.path.join(self.directory, f"{shard}.{kind}")

    def _transfer_group_for(self, representatives: dict[int, int]) -> dist.ProcessGroup | None:
        """COLLECTIVE once. The sweep's own point-to-point group over its sources and writers, or None
        when every source writes its own scores.

        NCCL keeps a buffer per connected peer for its communicator's lifetime, so sends over the world
        group would leave each writer O(DP) of them through training; :meth:`finish` destroys this one.
        """
        if self._transfer_group is None and any(
            writer != source for writer in self.writers for source in representatives.values()
        ):
            members = sorted({*self.writers, *representatives.values()})
            self._transfer_group = dist.new_group(members, timeout=nccl_timeout())
        return self._transfer_group

    def _finish(self, dataset: Dataset) -> MappedReferenceScores:
        guard = DeferredRankFailure("Completing the offline GRPO reference cache", exc_type=ValueError)

        def merge():
            for kind, dtype in _PAYLOAD_KINDS:
                with open(self._path("merged", kind), "wb") as destination:
                    for shard in range(self.dp_size):
                        path = self._path(shard, kind)
                        if os.path.exists(path):
                            with open(path, "rb") as source:
                                shutil.copyfileobj(
                                    source, destination, length=REFERENCE_BUFFER_VALUES * dtype.itemsize
                                )
                            os.unlink(path)
            mapped = self._map("merged")
            mismatch = reference_payload_mismatch(mapped.lengths, mapped.values, dataset)
            if mismatch:
                raise ValueError(f"Incomplete offline GRPO reference cache: {mismatch}")

        if is_fs_save_rank():
            guard.run(merge)
        guard.store_reject(f"reference-cache/{os.path.basename(self.directory)}/merge")
        guard = DeferredRankFailure("Mapping the offline GRPO reference cache", exc_type=ValueError)
        mapped = guard.run(lambda: self._map("merged"))
        guard.reject()
        return mapped

    def _map(self, shard: int | str) -> MappedReferenceScores:
        tensors = []
        for kind, dtype in _PAYLOAD_KINDS:
            path = self._path(shard, kind)
            file_size = os.path.getsize(path)
            if file_size % dtype.itemsize:
                raise ValueError(f"Malformed reference cache '{path}': truncated {kind}")
            tensors.append(
                torch.from_file(path, shared=False, size=file_size // dtype.itemsize, dtype=dtype)
                if file_size
                else torch.empty(0, dtype=dtype)
            )
        return mapped_reference_scores(*tensors)
