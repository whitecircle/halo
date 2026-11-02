"""Qwen3.8-Flash-Next's hashed n-gram table in host memory: rows gathered on the CPU from page-cache views of its
checkpoint shards and copied to the device per lookup."""

from __future__ import annotations

import torch

from halo.models.external_tables import ExternalTable
from halo.models.gradient_checkpointing import active_recompute_scope


class HostNGramTable(ExternalTable):
    """The frozen table ``Qwen4ExpTextNGramEmbedding`` holds in place of its ``nn.Embedding``.

    A lookup returns the rows the stock embedding would, on the ids' device at ``compute_dtype``, the dtype the
    stock weight would hold (it follows module-wide casts). A checkpointed layer's recompute reads its first
    pass's rows back rather than gathering them again.
    """

    part_key_format = "shard_{index}.weight"

    def __init__(self, rows: int, dim: int, part_count: int, compute_dtype: torch.dtype):
        super().__init__(rows, dim, part_count)
        self.compute_dtype = compute_dtype

    def _apply(self, fn, recurse=True):
        self.compute_dtype = fn(torch.empty(0, dtype=self.compute_dtype)).dtype
        return super()._apply(fn, recurse)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        scope = active_recompute_scope()
        cache = scope.slot(self, "rows") if scope is not None else {}
        if "rows" not in cache:
            cache["rows"] = self._gather(ids)
        return cache["rows"]

    def _gather(self, ids: torch.Tensor) -> torch.Tensor:
        """Rows for ``ids``, gathered part by part into a pinned buffer when they go to an accelerator."""
        if not self.is_attached:
            raise RuntimeError(
                f"{type(self).__name__} has no table attached: the model was built for a host n-gram table but "
                f"never passed through finalize_loaded_model, which attaches it from the checkpoint."
            )
        flat = ids.reshape(-1).to("cpu", torch.int64)
        if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= self.rows):
            raise IndexError(
                f"n-gram ids span [{int(flat.min())}, {int(flat.max())}] but the table holds {self.rows} rows: the "
                f"config's n-gram vocabulary does not describe the attached table."
            )
        pinned = ids.device.type == "cuda"
        rows = torch.empty((flat.numel(), self.dim), dtype=self.parts[0].dtype, pin_memory=pinned)
        owners = torch.bucketize(flat, self.part_ends, right=True)
        order = owners.argsort(stable=True)
        hit, counts = owners[order].unique_consecutive(return_counts=True)
        position = 0
        for part, count in zip(hit.tolist(), counts.tolist(), strict=True):
            picked = order[position : position + count]
            start = int(self.part_ends[part]) - self.parts[part].shape[0]
            rows[picked] = self.parts[part][flat[picked] - start]
            position += count
        rows = rows.view(*ids.shape, self.dim).to(ids.device, non_blocking=pinned)
        return rows.to(self.compute_dtype)
