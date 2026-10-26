"""Configured FP32 parameter masters: which HF parameters train at fp32, and their checkpoint values kept
or restored at fp32 before any parallel wrap.

Selected off the EP layer registry, the accessors the wrappers adopt through, so every loader (eager, lazy
EP, PP stage) keeps one checkpoint-backed set. A native dense TP master is rebuilt as its 1-D TP DTensor.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from src.checkpoint.reading import StreamingCheckpointReader, resolve_checkpoint_weights
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.expert_fuser import ExpertFuser
from src.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from src.distributed.expert_parallel.lazy_plan import ExpertKeyLayout, WeightPlanner
from src.distributed.expert_parallel.parallelize import ep_claimed_blocks
from src.distributed.mesh import MeshDim
from src.log import KEY_PREVIEW_COUNT
from src.model_loading.dtype import is_packed_4bit_parameter, reject_fp8_tensor
from src.model_loading.lazy_safetensors.conversion_selection import family_key_mapping
from src.model_loading.lazy_safetensors.materialization import (
    assign_tensor_to_model,
    materialize_weight_plan,
    verify_loaded_shape,
)
from src.models.moe.config_fields import require_expert_count


def fp32_master_param_keys(
    model: nn.Module, ep_config: EPConfig | None = None, *, keep_non_ep: bool
) -> frozenset[str]:
    """HF parameter keys whose configured training storage is fp32: every non-EP parameter under
    ``keep_non_ep``, and ``ep_config``'s routers and experts through the family's own accessors.

    Shared experts stay at the run dtype, and FSDP-managed ep1 experts ignore ``fp32_experts`` as at wrapper
    init. Selected by object identity, so tied aliases count and a persistent float buffer does not.
    """
    blocks = ep_claimed_blocks(model) if ep_config is not None else []
    ep_params = {id(param) for _path, block in blocks for param in block.parameters()}
    masters = {id(param) for param in model.parameters() if keep_non_ep and id(param) not in ep_params}
    for _path, block in blocks:
        layer_cls = MOE_LAYER_MAP[type(block).__name__]
        if ep_config.fp32_router:
            router = layer_cls._find_gate_or_router(block)
            if isinstance(router, nn.Parameter):
                masters.add(id(router))
            elif router is not None:
                masters.update(id(param) for param in router.parameters())
        if ep_config.keeps_fp32_experts:
            masters.update(id(param) for param in layer_cls._find_experts_container(block).parameters())
    return frozenset(
        name
        for name, param in model.state_dict(keep_vars=True).items()
        if isinstance(param, nn.Parameter)
        and param.is_floating_point()
        and not is_packed_4bit_parameter(param)
        and id(param) in masters
    )


def verify_fp32_master_coverage(
    state: dict[str, torch.Tensor], masters: frozenset[str], loaded: Iterable[str], *, source: str
) -> None:
    """Require every selected master's identity, aliases included, among ``loaded``: a resume cannot keep a
    rounded or random master through a missing-key exemption."""
    loaded_ids = {id(state[key]) for key in loaded}
    missing = sorted(key for key in masters if id(state[key]) not in loaded_ids)
    if missing:
        raise RuntimeError(
            f"FP32-master checkpoint load from {source}: {len(missing)} configured master "
            f"tensor(s) were not restored: {missing[:KEY_PREVIEW_COUNT]}. A full-finetune resume "
            "must restore every master; missing-key exemptions cannot retain rounded or random weights."
        )


def _replace_tp_master(model: nn.Module, aliases: list[str], target: DTensor, local: torch.Tensor) -> None:
    """Replace TP storage and every tied owner, without rebinding a DTensor's stale local storage."""
    mesh = target.device_mesh
    if mesh.ndim != 1 or mesh.mesh_dim_names != (MeshDim.TP,):
        raise TypeError(
            f"FP32-master load of {aliases[0]!r} requires the named 1-D TP mesh before DP/FSDP2 "
            f"wrapping, got {mesh.mesh_dim_names!r}. Restoring an already DP-sharded master would "
            "not invert FSDP2's packed shard layout."
        )
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


def _install_tp_master(model: nn.Module, aliases: list[str], target: DTensor, tensor: torch.Tensor) -> None:
    # Every rank reads the same full tensor: split on CPU, then upload only the local shard.
    placement = target.placements[0]
    if isinstance(placement, (Shard, _StridedShard)):
        local = placement._shard_tensor(tensor, target.device_mesh, 0, src_data_rank=None)
    elif placement.is_replicate():
        local = tensor
    else:
        raise TypeError(f"FP32-master load of {aliases[0]!r} cannot restore TP placement {placement!r}.")
    _replace_tp_master(model, aliases, target, local)


def _promote_exact_bf16_master(
    model: nn.Module,
    aliases: list[str],
    target: nn.Parameter,
    disk_keys: list[str] | tuple[str, ...],
    reader: StreamingCheckpointReader,
) -> bool:
    """BF16 disk values survive BF16/FP32 construction exactly, so promote existing storage only."""
    if (
        target.is_meta
        or target.dtype not in (torch.bfloat16, torch.float32, torch.float64)
        or not all(reader.safetensors_dtype(key) == "BF16" for key in disk_keys)
    ):
        return False
    if isinstance(target, DTensor):
        _replace_tp_master(model, aliases, target, target.to_local())
    else:
        target.data = target.data.float()
    return True


def restore_fp32_master_parameters(
    model: nn.Module,
    local_dir: str,
    ep_config: EPConfig | None = None,
    *,
    keep_non_ep: bool,
    strict: bool = False,
) -> None:
    """Reread the configured masters at fp32 out of the checkpoint in ``local_dir`` (``ModelSource.local_dir``),
    after an eager load and before any EP/CP/TP/FSDP2 wrap; ``strict`` (a resume) requires every one restored.

    Rank-local: callers defer a failure until after their node-serialized loading region.
    """
    masters = fp32_master_param_keys(model, ep_config, keep_non_ep=keep_non_ep)
    if not masters:
        return
    state = model.state_dict(keep_vars=True)
    aliases = defaultdict(list)
    for key, param in state.items():
        if isinstance(param, nn.Parameter):
            aliases[id(param)].append(key)
    layout = resolve_checkpoint_weights(local_dir)
    disk_keys = layout.key_set()
    weight_map = {key: layout.legacy_bin for key in disk_keys} if layout.legacy_bin is not None else layout.weight_map
    disk_to_model, fanout = family_key_mapping(model, list(weight_map), replay_loaded=True)
    plans = WeightPlanner(None).build(weight_map, disk_to_model, set(state), fanout=fanout)
    plans = [plan for plan in plans if plan.model_key in masters]

    tasks = []
    fuser = None
    if ExpertKeyLayout.detect(weight_map) == ExpertKeyLayout.PER_EXPERT:
        fuser = ExpertFuser(0, require_expert_count(model.config))
        tasks = [
            task for task in fuser.detect_tasks(weight_map, disk_to_model, set(state)) if task.model_key in masters
        ]
    fusion_keys = {task.model_key for task in tasks}
    plans = [plan for plan in plans if plan.model_key not in fusion_keys]
    requested = {key for plan in plans for key in plan.disk_keys}
    requested.update(key for task in tasks for key in task.disk_keys)
    restored = set()
    restored_ids = set()
    with StreamingCheckpointReader(local_dir, requested) as reader:
        if tasks:
            replay_tasks = []
            for task in tasks:
                fuser.validate_task(task)
                key = task.model_key
                if _promote_exact_bf16_master(model, aliases[id(state[key])], state[key], task.disk_keys, reader):
                    restored.add(key)
                else:
                    replay_tasks.append(task)
            restored.update(
                fuser.execute(
                    replay_tasks,
                    model,
                    local_dir,
                    torch.float32,
                    state[tasks[0].model_key].device,
                    reader=reader,
                    preserve_parameters=True,
                )
            )
            restored_ids.update(id(state[key]) for key in restored)
        for plan in plans:
            target = state[plan.model_key]
            if id(target) not in restored_ids:
                if not _promote_exact_bf16_master(model, aliases[id(target)], target, plan.disk_keys, reader):
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
        verify_fp32_master_coverage(state, masters, restored, source=local_dir)
