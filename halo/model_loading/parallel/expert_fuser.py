"""Fusing a per-expert checkpoint's local experts into the model's 3D tensors: ``experts.N.gate_proj.weight [M, H]``
into ``experts.gate_up_proj [E, 2M, H]``, reading only this rank's experts."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from safetensors import safe_open

from halo.checkpoint.reading import StreamingCheckpointReader
from halo.log import KEY_PREVIEW_COUNT
from halo.model_loading.dtype import reject_fp8_tensor
from halo.model_loading.lazy_safetensors.materialization import assign_tensor_to_model, verify_loaded_shape
from halo.model_loading.parallel.lazy_plan import PER_EXPERT_KEY_PATTERN
from halo.models.moe.family_specs import per_expert_fusion_map

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FusionTask:
    """One fused model tensor and the per-expert checkpoint keys it stacks."""

    model_key: str
    kind: str  # "gate_up" or "down"
    # {expert index: {projection suffix: (disk key, shard file)}}, this rank's experts only.
    experts: dict[int, dict[str, tuple[str, str]]]

    @property
    def disk_keys(self) -> list[str]:
        return [disk_key for suffixes in self.experts.values() for disk_key, _ in suffixes.values()]

    @property
    def shard_files(self) -> set[str]:
        return {shard_file for suffixes in self.experts.values() for _, shard_file in suffixes.values()}


def _enclosing_layer_prefix(container_prefix: str) -> str | None:
    """``model.layers.47.mlp.experts.`` → ``model.layers.47.``, cut at the last index; ``None`` without one."""
    parts = [p for p in container_prefix.split(".") if p]
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].isdigit():
            return ".".join(parts[: i + 1]) + "."
    return None


class ExpertFuser:
    """Fuses this rank's ``[ep_start, ep_end)`` experts of a per-expert checkpoint."""

    def __init__(self, ep_start: int, ep_end: int):
        self.ep_start = ep_start
        self.ep_end = ep_end

    def detect_tasks(
        self, weight_map: dict[str, str], disk_to_model: dict[str, str], model_keys: set[str]
    ) -> list[FusionTask]:
        """This rank's fusion tasks, the target resolved through ``disk_to_model``, since re-deriving it from the disk
        key would lose a family needing ``base_model_prefix`` or a hub rename to a silent skip."""
        groups: dict[str, dict[str, dict[int, dict[str, tuple[str, str]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

        for disk_key, shard_file in weight_map.items():
            m = PER_EXPERT_KEY_PATTERN.search(disk_key)
            if not m:
                continue

            expert_idx = int(m.group(2))
            if expert_idx < self.ep_start or expert_idx >= self.ep_end:
                continue

            suffix = disk_key[m.end() :]
            if suffix not in per_expert_fusion_map():
                continue

            kind, _ = per_expert_fusion_map()[suffix]
            mapped = disk_to_model.get(disk_key, disk_key)
            mapped_m = PER_EXPERT_KEY_PATTERN.search(mapped)
            if mapped_m is None:
                raise RuntimeError(
                    f"EP lazy load: per-expert checkpoint key '{disk_key}' maps to model key "
                    f"'{mapped}', which carries no expert index — the disk→model mapping renamed the "
                    f"expert container. Its experts would stay on meta and fail later as 'Cannot copy "
                    f"out of meta tensor'. Load with ep_lazy_loading=False."
                )
            prefix = f"{mapped[: mapped_m.start()]}.{mapped_m.group(1)}."
            groups[prefix][kind][expert_idx][suffix] = (disk_key, shard_file)

        tasks = []
        for layer_prefix, fusion_groups in groups.items():
            for kind, experts in fusion_groups.items():
                model_key = layer_prefix + ("gate_up_proj" if kind == "gate_up" else "down_proj")
                if model_key in model_keys:
                    tasks.append(FusionTask(model_key, kind, experts))
                    continue

                # No fused target: legitimate only when the model stores one module per expert, which the
                # planner already covers; anything else would leave the experts on meta.
                unresolved = sorted(
                    disk_key
                    for suffixes in experts.values()
                    for disk_key, _ in suffixes.values()
                    if disk_to_model.get(disk_key, disk_key) not in model_keys
                )
                if not unresolved:
                    continue

                # A layer the model instantiates nothing of is a declared-but-absent MTP tail (GLM-4
                # num_nextn_predict_layers), whose non-expert keys the planner drops the same way.
                enclosing_layer = _enclosing_layer_prefix(layer_prefix)
                if enclosing_layer is not None and not any(key.startswith(enclosing_layer) for key in model_keys):
                    logger.info(
                        "EP lazy load: skipping %d expert tensors under '%s' — the model instantiates "
                        "no parameter of layer '%s' (multi-token-prediction tail).",
                        len(unresolved),
                        layer_prefix,
                        enclosing_layer,
                    )
                    continue

                raise RuntimeError(
                    f"EP lazy load: per-expert checkpoint keys under '{layer_prefix}' match neither "
                    f"the fused model parameter '{model_key}' nor any per-expert model parameter, so "
                    f"{len(unresolved)} expert tensors would never load — and the model DOES "
                    f"instantiate the rest of that layer, so this is a layout mismatch, not an "
                    f"absent MTP tail. First few: {unresolved[:KEY_PREVIEW_COUNT]}. Load with ep_lazy_loading=False."
                )

        return tasks

    def validate_task(self, task: FusionTask) -> None:
        """Require this rank's full contiguous expert range and both GLU halves before any read: the fuse
        stacks ``sorted(experts)`` into local slots, so a gap would shift every later expert."""
        missing = sorted(set(range(self.ep_start, self.ep_end)) - set(task.experts))
        if missing:
            raise RuntimeError(
                f"EP lazy load: the checkpoint carries {len(task.experts)} of this rank's "
                f"{self.ep_end - self.ep_start} experts for {task.model_key!r} — global expert index(es) "
                f"{missing[:KEY_PREVIEW_COUNT]} are absent from it. Fusing the rest would shift every later expert "
                f"into the wrong local slot."
            )
        if task.kind == "gate_up":
            for expert_idx, suffixes in task.experts.items():
                if {per_expert_fusion_map()[suffix][1] for suffix in suffixes} != {0, 1}:
                    raise RuntimeError(
                        f"EP lazy load: expert {expert_idx} carries only {sorted(suffixes)} "
                        "of its two GLU halves in the checkpoint — fusing a half-present expert would "
                        "install garbage for the missing half."
                    )

    def execute(
        self,
        tasks: list[FusionTask],
        model: nn.Module,
        model_path: str,
        dtype: torch.dtype | None,
        device: str,
        *,
        keep_fp32: frozenset[str] = frozenset(),
        reader: StreamingCheckpointReader | None = None,
        preserve_parameters: bool = False,
    ) -> set[str]:
        """Fuse and assign every task; returns the model keys assigned, for the caller to unplan."""
        if not tasks:
            return set()

        handles = (
            reader
            if reader is not None
            else {
                shard_file: safe_open(os.path.join(model_path, shard_file), framework="pt", device=device)
                for shard_file in set().union(*(task.shard_files for task in tasks))
            }
        )

        fused_keys: set[str] = set()
        try:
            for task in tasks:
                self.validate_task(task)
                tensor = (
                    self._fuse_gate_up(task, handles) if task.kind == "gate_up" else self._fuse_down(task, handles)
                )
                verify_loaded_shape(
                    model,
                    task.model_key,
                    f"{len(task.experts)} fused per-expert checkpoint keys",
                    tensor,
                    shard_dim=0,
                    shard_len=self.ep_end - self.ep_start,
                )
                reject_fp8_tensor(task.model_key, tensor, dtype)
                if dtype is not None and tensor.is_floating_point():
                    target_device = model.get_parameter(task.model_key).device if preserve_parameters else device
                    tensor = tensor.to(
                        device=target_device, dtype=torch.float32 if task.model_key in keep_fp32 else dtype
                    )

                assign_tensor_to_model(model, task.model_key, tensor, preserve_parameter=preserve_parameters)
                fused_keys.add(task.model_key)
        finally:
            if reader is None:
                handles.clear()

        return fused_keys

    def _fuse_gate_up(self, task: FusionTask, handles: dict | StreamingCheckpointReader) -> torch.Tensor:
        """The gate and up halves (resolved by position) → ``gate_up_proj [E_local, 2M, H]``."""
        slices = []
        for expert_idx in sorted(task.experts):
            by_pos = {per_expert_fusion_map()[s][1]: entry for s, entry in task.experts[expert_idx].items()}
            gate = self._read_tensor(handles, *by_pos[0])
            up = self._read_tensor(handles, *by_pos[1])
            slices.append(torch.cat([gate, up], dim=0))
        return torch.stack(slices, dim=0)

    def _fuse_down(self, task: FusionTask, handles: dict | StreamingCheckpointReader) -> torch.Tensor:
        """The single down projection (``down_proj`` or ``w2``) → ``[E_local, H, M]``."""
        return torch.stack(
            [self._read_tensor(handles, *next(iter(task.experts[idx].values()))) for idx in sorted(task.experts)],
            dim=0,
        )

    @staticmethod
    def _read_tensor(handles: dict | StreamingCheckpointReader, key: str, shard: str) -> torch.Tensor:
        if isinstance(handles, StreamingCheckpointReader):
            return handles.get(key)
        return handles[shard].get_slice(key)[...]
