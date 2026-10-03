"""Stream configured FP32 masters back from the checkpoint before parallel wrapping.

HF eager construction may already have rounded those values to the run dtype. Replaying its
conversion/key plan restores only the parameter master set, not a whole FP32 model or its buffers.
The EP planner/fuser owns expert layouts; selection and restore are shared by dense, CP and EP
construction. Native dense TP is the exception to plain pre-wrap storage: its existing 1-D TP
DTensors are rebuilt with their placements and all tied aliases before the trainer adds DP/FSDP2.
"""

from __future__ import annotations

import os
from collections import defaultdict

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from src.checkpoint.format import StreamingCheckpointReader, read_checkpoint_key_set, resolve_checkpoint_weights
from src.distributed.expert_parallel.config import EPConfig, get_num_experts
from src.distributed.expert_parallel.lazy_loader import (
    CheckpointFormat,
    EPWeightPlanner,
    ExpertFuser,
    build_family_key_mapping,
)
from src.distributed.loading.precision import fp32_master_param_keys, verify_fp32_master_coverage
from src.models.loading.dtype import reject_fp8_tensor
from src.models.loading.lazy_safetensors.weights import (
    assign_tensor_to_model,
    materialize_weight_plan,
    verify_loaded_shape,
)


def _install_tp_master(model: nn.Module, aliases: list[str], target: DTensor, tensor: torch.Tensor) -> None:
    """Replace TP storage and every tied owner, without rebinding a DTensor's stale local storage."""
    mesh = target.device_mesh
    if mesh.ndim != 1 or mesh.mesh_dim_names != ("tp",):
        raise TypeError(
            f"FP32-master load of {aliases[0]!r} requires the named 1-D TP mesh before DP/FSDP2 "
            f"wrapping, got {mesh.mesh_dim_names!r}. Restoring an already DP-sharded master would "
            "not invert FSDP2's packed shard layout."
        )
    # Every rank reads the same full tensor. No src-rank broadcast or full_tensor() collective:
    # this replay is rank-local so a read failure can join before later wrapping collectives.
    placement = target.placements[0]
    if isinstance(placement, (Shard, _StridedShard)):
        # PyTorch's placement implements its own split, including packed _StridedShard ordering.
        # _StridedShard deliberately does not satisfy Placement.is_shard() on the pinned torch.
        # Split on CPU first: uploading the full FP32 vocab/expert tensor would defeat TP's peak.
        local = placement._shard_tensor(tensor, mesh, 0, src_data_rank=None)
    elif placement.is_replicate():
        local = tensor
    else:
        raise TypeError(f"FP32-master load of {aliases[0]!r} cannot restore TP placement {placement!r}.")
    master = nn.Parameter(
        DTensor.from_local(
            local.to(device=target.device, dtype=torch.float32),
            mesh,
            target.placements,
            run_check=False,
            shape=target.shape,
            stride=target.stride(),
        ),
        requires_grad=target.requires_grad,
    )
    for name in aliases:
        parent_path, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent_path) if parent_path else model, attr, master)


def restore_fp32_master_parameters(
    model: nn.Module,
    model_path: str,
    ep_config: EPConfig | None = None,
    *,
    keep_non_ep: bool,
    strict: bool = False,
    revision: str | None = None,
) -> None:
    """Reread configured masters for any checkpoint start; resume provenance makes coverage strict.

    Reads remain rank-local. Callers defer failures until after their node-serialized loading region
    and before EP/CP/TP/FSDP2 wrapping. Local paths and already-downloaded Hub revisions share the
    same streaming reader. A fresh task head may be initialized under the normal load gate; a full
    finetune resume cannot use a missing-key exemption for a configured master.
    """
    masters = fp32_master_param_keys(model, ep_config, keep_non_ep=keep_non_ep)
    if not masters:
        return
    source = model_path
    if not os.path.isdir(model_path):
        model_path = snapshot_download(model_path, revision=revision, local_files_only=True)
    state = model.state_dict(keep_vars=True)
    aliases = defaultdict(list)
    for key, param in state.items():
        if isinstance(param, nn.Parameter):
            aliases[id(param)].append(key)
    layout = resolve_checkpoint_weights(model_path)
    disk_keys = read_checkpoint_key_set(model_path)
    weight_map = {key: layout.legacy_bin for key in disk_keys} if layout.legacy_bin is not None else layout.weight_map
    disk_to_model, fanout = build_family_key_mapping(model, list(weight_map), loaded_conversions=True)
    plans = EPWeightPlanner(None).build(weight_map, disk_to_model, set(state), fanout=fanout)
    plans = [plan for plan in plans if plan.model_key in masters]

    tasks = []
    fuser = None
    if CheckpointFormat.detect(weight_map) == CheckpointFormat.INDIVIDUAL:
        fuser = ExpertFuser(0, get_num_experts(model.config))
        tasks = [task for task in fuser.detect_tasks(weight_map, disk_to_model, set(state)) if task[0] in masters]
    fusion_keys = {task[0] for task in tasks}
    plans = [plan for plan in plans if plan.model_key not in fusion_keys]
    requested = {key for plan in plans for key in plan.disk_keys}
    requested.update(
        key for _target, _kind, experts in tasks for slots in experts.values() for key, _sf in slots.values()
    )
    restored = set()
    restored_ids = set()
    with StreamingCheckpointReader(model_path, requested) as reader:
        if tasks:
            restored.update(
                fuser.execute(tasks, model, model_path, torch.float32, "cpu", reader=reader, preserve_parameters=True)
            )
            restored_ids.update(id(state[key]) for key in restored)
        for plan in plans:
            target = state[plan.model_key]
            if id(target) not in restored_ids:
                tensor = materialize_weight_plan(plan, lambda key, _plan: reader.get(key))
                verify_loaded_shape(model, plan.model_key, f"checkpoint key(s) {plan.disk_keys}", tensor)
                reject_fp8_tensor(plan.model_key, tensor, torch.float32)
                if isinstance(target, DTensor):
                    _install_tp_master(model, aliases[id(target)], target, tensor)
                else:
                    assign_tensor_to_model(
                        model,
                        plan.model_key,
                        tensor.to(device=target.device, dtype=torch.float32),
                        preserve_parameter=True,
                    )
                restored_ids.add(id(target))
            restored.add(plan.model_key)
    if strict:
        verify_fp32_master_coverage(state, masters, restored, source=source)
