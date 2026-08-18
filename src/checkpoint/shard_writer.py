"""Incremental safetensors writer for a checkpoint shard.

safetensors has no append handle (``save_file`` takes a whole dict), so bounding the writer's host
memory and honouring ``save_max_shard_size`` are the same mechanism: flush whenever the pending batch
would overflow. A multi-writer caller (the PP save, one rank per stage) hands its weight-map fragment
back through :meth:`StageShardWriter.close` to be merged, because no writer knows the global part
count; a single writer knows it at the last flush, so
:meth:`StageShardWriter.close_as_hf_checkpoint` renames its parts to HF's own names and writes the
index itself.
"""

from __future__ import annotations

import os

import torch
from huggingface_hub.serialization._base import parse_size_to_int
from safetensors.torch import save_file

from src.checkpoint.format import (
    SAFETENSORS_METADATA,
    remove_stale_checkpoint_files,
    shard_file_name,
    write_merged_index,
)


class StageShardWriter:
    """Accumulate tensors and flush them to numbered safetensors parts under one prefix.

    ``enabled=False`` makes every method a no-op, so non-writer ranks, which still enter each
    collective gather, share the call sites without a branch at each one.
    """

    def __init__(self, output_dir: str, filename_prefix: str, max_shard_size: str, *, enabled: bool):
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.max_bytes = parse_size_to_int(max_shard_size)
        self.enabled = enabled
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._part = 0
        self._weight_map: dict[str, str] = {}
        self._part_names: list[str] = []
        self._total_bytes = 0

    def add(self, key: str, tensor: torch.Tensor) -> None:
        """Stage one tensor, flushing the pending batch first if this one would overflow it.

        Flushing before appending is HF's own rule: a part then exceeds the limit only when a single
        tensor does, instead of every part overshooting by its last tensor.

        A key may be staged once. Unlike a state-dict build, a streaming writer cannot resolve a
        second claim: the first tensor is already in a flushed part, so the checkpoint would carry
        the key twice at two shapes with ``total_size`` counting both. A re-added key means the
        caller's chunk sequence claims one tensor from two sources, so it is refused rather than
        overwritten.
        """
        if not self.enabled:
            return
        if key in self._weight_map or key in self._pending:
            raise RuntimeError(
                f"{key!r} was staged twice for the checkpoint in {self.output_dir}: a streamed part "
                f"already claims it, so the second tensor cannot replace the first. Two chunks of "
                f"this save name the same tensor (a hand-sliced param emitted by the parameter walk "
                f"AND by its own gather is the known shape) — the writer refuses rather than write "
                f"the key twice at two shapes."
            )
        nbytes = tensor.numel() * tensor.element_size()
        if self._pending and self._pending_bytes + nbytes > self.max_bytes:
            self._flush()
        self._pending[key] = tensor.contiguous()
        self._pending_bytes += nbytes

    def close(self) -> tuple[dict[str, str], int]:
        """Flush the remainder and return ``(weight_map fragment, bytes written)``.

        Must be called before the index exchange: the index must never name a file that is not on
        disk yet.
        """
        if not self.enabled:
            return {}, 0
        self._flush()
        return self._weight_map, self._total_bytes

    def close_as_hf_checkpoint(self) -> int:
        """Flush, rename the parts to HF-standard names, write the index; return bytes written.

        The single-writer finalizer: what ``from_pretrained`` needs, without ever holding the whole
        state dict. Its part boundaries differ from
        :func:`~src.checkpoint.format.save_sharded_state_dict` (this flushes the open block before an
        oversized tensor rather than carrying it alongside), so both layouts load identically but
        neither the file count nor the key-to-file map is comparable.

        Only valid when this writer owns every ``model*.safetensors`` in the directory: it deletes
        the ones it did not write, because ``from_pretrained`` prefers a leftover single
        ``model.safetensors`` over the index. The sweep runs after the parts are on disk, so a
        re-save transiently holds both checkpoints (2x peak disk).

        A multi-writer caller must use :meth:`close`: renaming needs the global part count.
        """
        if not self.enabled:
            return 0
        self._flush()
        if not self._part:
            # An enabled writer with no parts means the caller gathered nothing, and the sweep below
            # would delete the previous checkpoint and put nothing in its place.
            raise RuntimeError(
                f"Refusing to finalize an empty checkpoint in {self.output_dir}: this writer is the "
                f"designated save rank but no tensors were staged, so there is nothing to name and "
                f"the existing files there would be deleted for nothing."
            )
        renamed = {part_name: shard_file_name(i, self._part) for i, part_name in enumerate(self._part_names, start=1)}
        for part_name, final_name in renamed.items():
            # Overwrites a same-named leftover, which is intended; the stale sweep below then only
            # has to consider names this save did not claim.
            os.replace(os.path.join(self.output_dir, part_name), os.path.join(self.output_dir, final_name))
        weight_map = {key: renamed[part_name] for key, part_name in self._weight_map.items()}
        self._weight_map = weight_map
        self._part_names = list(renamed.values())

        if self._part > 1:
            write_merged_index(self.output_dir, weight_map, {"total_size": self._total_bytes})
        else:
            # A single part is the unsharded ``model.safetensors``, which carries no index of its own.
            remove_stale_checkpoint_files(self.output_dir, set(renamed.values()))
        return self._total_bytes

    def _flush(self) -> None:
        if not self._pending:
            return
        self._part += 1
        # Numbered even when the stage produces a single part, so it cannot collide with a stray
        # `model.safetensors` from another writer.
        name = f"{self.filename_prefix}-{self._part:05d}.safetensors"
        # Without the framework marker safetensors readers do not resolve the file as torch tensors.
        save_file(self._pending, os.path.join(self.output_dir, name), metadata=SAFETENSORS_METADATA)
        self._weight_map.update(dict.fromkeys(self._pending, name))
        self._part_names.append(name)
        self._total_bytes += self._pending_bytes
        self._pending = {}
        self._pending_bytes = 0
