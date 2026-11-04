"""Attach a family's released MTP head after the per-mode load: its tail streams through the lazy planner on a head-only
shell, so the trunk's load and its ignore patterns stay untouched and one path serves every mode."""

from __future__ import annotations

import logging
from collections import Counter

import torch
import torch.nn as nn

from halo.checkpoint.reading import read_checkpoint_config, resolve_checkpoint_weights
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.log import KEY_PREVIEW_COUNT
from halo.model_loading.buffer_repair import apply_buffer_repairs
from halo.model_loading.dtype import resolve_load_dtype
from halo.model_loading.lazy_safetensors.materialization import WeightAction, WeightPlan
from halo.model_loading.liger.cross_entropy import head_token_ce_sum
from halo.model_loading.liger.dispatch import LIGER_APPLIED_CONFIG_ATTR
from halo.model_loading.parallel.expert_fuser import ExpertFuser, FusionTask
from halo.model_loading.parallel.lazy_load import load_planned_weights
from halo.model_loading.parallel.lazy_plan import ExpertKeyLayout, LazyLoadPlan, WeightPlanner
from halo.model_loading.parallel.model_source import ModelSource, resolve_model_source
from halo.model_loading.parallel.whole_model import cast_loaded_parameters
from halo.models.module_tree import backbone_with_layers
from halo.models.moe.config_fields import require_expert_count
from halo.models.mtp.head import (
    MtpForwardTerm,
    attach_mtp_head_module,
    build_mtp_head,
    freeze_trunk_for_head,
    mtp_head_path,
)
from halo.models.mtp.spec import MtpHeadRequest, MtpHeadSpec, require_mtp_head_spec
from halo.runtime.consensus import DeferredRankFailure
from halo.runtime.ranks import get_global_rank, get_local_rank, is_global_main_process

logger = logging.getLogger(__name__)


class _HeadShell(nn.Module):
    """Holds only the head at its release path, so a state dict is the head's keys and the absent-module sweep has
    nothing else to draw. The EP patch it takes registers its forward hook here, on a module nothing calls, so the
    capacity generation still advances once per model forward."""

    def __init__(self, config, head: nn.Module, path: str):
        super().__init__()
        self.config = config
        *parents, name = path.split(".")
        node: nn.Module = self
        for parent in parents:
            child = nn.ModuleDict()
            node.add_module(parent, child)
            node = child
        node.add_module(name, head)


def _reject_stored_quantization(source: ModelSource) -> None:
    config = read_checkpoint_config(source.local_dir, missing_ok=True) or {}
    if config.get("quantization_config"):
        raise ValueError(
            f"{source.name} stores quantized weights, so its MTP tail has no float tensors to train from. Set "
            "mtp_head_source to a checkpoint storing the head in bf16 or fp32."
        )


def _tail_map(source: ModelSource, spec: MtpHeadSpec, path: str, *, resuming: bool) -> dict[str, str]:
    """The tail keys ``source`` stores under the head's path, without the trunk's copies the head shares."""
    weight_map = resolve_checkpoint_weights(source.local_dir).require_safetensors("checkpoint").weight_map
    copies = tuple(f"{path}.{key}" for key, _embedding in spec.trunk_copies)
    tail = {key: shard for key, shard in weight_map.items() if key.startswith(f"{path}.") and key not in copies}
    if not tail and resuming:
        raise ValueError(
            f"{source.name} stores no MTP head under {path}.*, and a resume continues the head its checkpoint holds, so "
            "a head cannot join a run midway. Start a new run with model_name_or_path set to this checkpoint and "
            "mtp_head_source set to one that stores a head, or resume without mtp_loss_weight."
        )
    if not tail:
        raise ValueError(
            f"{source.name} stores no MTP head under {path}.*, so this run has no head to train. Set mtp_head_source "
            "to a checkpoint that stores one, for example the release this model was fine-tuned from."
        )
    return tail


def _full_bank_fusion_tasks(tail: dict[str, str], shell_keys: set[str], config) -> list[FusionTask]:
    """The fusion of every expert a per-expert tail stores, whichever rank will read it, each task checked whole."""
    if ExpertKeyLayout.detect(tail) != ExpertKeyLayout.PER_EXPERT:
        return []
    fuser = ExpertFuser(0, require_expert_count(config))
    tasks = fuser.detect_tasks(tail, {key: key for key in tail}, shell_keys)
    for task in tasks:
        fuser.validate_task(task)
    return tasks


def _reject_tail_mismatch(tail: dict[str, str], shell_keys: set[str], config, source: ModelSource) -> None:
    """Refuse a tail that is not the head this config builds, in either direction: a key left out stays random against
    the rest, and a key beyond them is a head this run would not carry on to its export."""
    tasks = _full_bank_fusion_tasks(tail, shell_keys, config)
    fused = {
        disk_key for task in tasks for suffixes in task.experts.values() for disk_key, _shard in suffixes.values()
    }
    extra = sorted(key for key in tail if key not in shell_keys and key not in fused)
    missing = sorted(shell_keys - set(tail) - {task.model_key for task in tasks})
    if missing or extra:
        raise ValueError(
            f"The MTP tail in {source.name} is not the head this config builds: it lacks {len(missing)} tensor(s) "
            f"{missing[:KEY_PREVIEW_COUNT]} and adds {len(extra)} {extra[:KEY_PREVIEW_COUNT]}. Load it with the config "
            "of the checkpoint it came from, or set mtp_head_source to a matching one."
        )


def _report_tail_plan(plans: list[WeightPlan], fused_keys: set[str]) -> None:
    counts = Counter(plan.action for plan in plans)
    logger.info(
        f"[Rank {get_global_rank()}] MTP head: {counts[WeightAction.EXPERT_SHARD]} expert-sharded, "
        f"{counts[WeightAction.REPLICATE]} replicated, {counts[WeightAction.IGNORE]} ignored, {len(fused_keys)} fused"
    )


def _stream_tail(
    shell: _HeadShell,
    spec: MtpHeadSpec,
    path: str,
    source: ModelSource,
    *,
    ep_config,
    dtype: torch.dtype,
    keep_fp32_params: bool,
    preserve_checkpoint_precision: bool,
) -> bool:
    """COLLECTIVE: load the tail into ``shell``; returns whether any expert tensor arrived as this rank's slice. A
    checkpoint read at its stored precision is a resume's."""
    what = f"MTP head load from {source.name}"
    guard = DeferredRankFailure(what)

    def read_tail() -> tuple[dict[str, str], list[WeightPlan]]:
        _reject_stored_quantization(source)
        tail = _tail_map(source, spec, path, resuming=preserve_checkpoint_precision)
        shell_keys = set(shell.state_dict())
        _reject_tail_mismatch(tail, shell_keys, shell.config, source)
        return tail, WeightPlanner(ep_config).build(tail, {key: key for key in tail}, shell_keys)

    tail, plans = guard.run(read_tail) or ({}, [])
    guard.reject()

    fuser = None
    layout = ExpertKeyLayout.detect(tail)
    if layout == ExpertKeyLayout.PER_EXPERT:
        expert_range = (
            (ep_config.expert_start_idx, ep_config.expert_end_idx)
            if ep_config is not None
            else (0, require_expert_count(shell.config))
        )
        fuser = ExpertFuser(*expert_range)
    base = LazyLoadPlan(
        model=shell,
        plans=plans,
        weight_map=tail,
        shard_files=sorted(set(tail.values())),
        missing_shard_files=[],
        expert_layout=layout,
        disk_to_model={key: key for key in tail},
        model_keys=set(shell.state_dict()),
        dtype=dtype,
    )
    return load_planned_weights(
        base,
        plans,
        model_path=source.local_dir,
        ep_config=ep_config,
        keep_fp32_params=keep_fp32_params,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        what=what,
        label=f"MTP head load (rank {get_global_rank()})",
        report=_report_tail_plan,
        fuser=fuser,
    )


def _draw_random_head(model: nn.Module, spec: MtpHeadSpec, *, dtype: torch.dtype, pc: ParallelismConfig, ep_config):
    """A from-scratch run's head: the full expert bank drawn on CPU from the trunk's stream, which every rank shares, so
    the EP patch slices each rank's range out of one bank. The family's own init, as a directly built head leaves the
    gate and both expert tensors at a zero gradient."""
    head = build_mtp_head(model, spec, device="cpu")
    head.apply(backbone_with_layers(model)._init_weights)
    cast_loaded_parameters(head, dtype, keep_fp32=pc.fp32_non_ep_params, ep_wrapped=ep_config is not None)
    return head


def _tail_source(request: MtpHeadRequest, trunk_source: ModelSource, *, from_checkpoint: bool) -> ModelSource:
    """COLLECTIVE-EQUIVALENT. The resume checkpoint's tail, whose head is the trained one, else ``request.source``
    agreed across ranks, else the model's own."""
    if from_checkpoint:
        if request.source is not None and is_global_main_process():
            logger.warning(
                f"mtp_head_source {request.source!r} is ignored on this resume: the head continues from "
                f"{trunk_source.name}."
            )
        return trunk_source
    return trunk_source if request.source is None else resolve_model_source(request.source, None, tag="mtp_head")


def attach_mtp_head(
    model: nn.Module,
    request: MtpHeadRequest,
    *,
    pc: ParallelismConfig,
    source: ModelSource,
    dtype: torch.dtype,
    wraps_experts: bool,
    init_from_scratch: bool,
    preserve_checkpoint_precision: bool,
) -> nn.Module:
    """COLLECTIVE: the head loaded (or drawn, for a from-scratch run), EP-patched, attached at the release's key path with
    the run's objective hooked into ``model``'s forward. A resume that restores FSDP2 state over a base load reads the
    usual source here, and the restore overwrites it. Every rank enters the same collectives in the same order."""
    spec = require_mtp_head_spec(model.config)
    ep_config = pc.ep_config() if wraps_experts else None
    path = mtp_head_path(model, spec)
    if init_from_scratch:
        head = _draw_random_head(model, spec, dtype=dtype, pc=pc, ep_config=ep_config)
        shell = _HeadShell(model.config, head, path)
        weights_already_sharded = False
    else:
        tail_source = _tail_source(request, source, from_checkpoint=preserve_checkpoint_precision)
        head = build_mtp_head(model, spec)
        shell = _HeadShell(model.config, head, path)
        weights_already_sharded = _stream_tail(
            shell,
            spec,
            path,
            tail_source,
            ep_config=ep_config,
            dtype=resolve_load_dtype(dtype, model.config),
            keep_fp32_params=pc.fp32_non_ep_params,
            preserve_checkpoint_precision=preserve_checkpoint_precision,
        )
    if ep_config is not None:
        apply_ep_layers(shell, ep_config, weights_already_sharded=weights_already_sharded)
    apply_buffer_repairs(shell)
    device = f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu"
    head = shell.get_submodule(path).to(device)
    attach_mtp_head_module(model, head, spec)
    if request.objective.scope == "head_only":
        freeze_trunk_for_head(model, head)
    applied = getattr(model.config, LIGER_APPLIED_CONFIG_ATTR, None) or {}
    MtpForwardTerm(
        model,
        head,
        spec,
        request.objective,
        head_token_ce_sum(model, fused=bool(applied.get("fused_linear_cross_entropy"))),
    )
    if is_global_main_process():
        logger.info(
            f"MTP head attached at {path}: {sum(param.numel() for param in head.parameters())} parameters on this "
            f"rank, scope={request.objective.scope}, loss={request.objective.loss_type}."
        )
    return head
