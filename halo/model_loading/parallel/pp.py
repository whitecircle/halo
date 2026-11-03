"""Stage-aware loads for pipeline parallelism: every rank plans the whole checkpoint against a full meta shell, keeps
only its stage's layers re-based to 0, and streams those tensors, so the load ceiling falls with ``pp_size``."""

from __future__ import annotations

import logging
from dataclasses import replace

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from halo.checkpoint.reading import PER_NODE_PLACEMENT_REMEDY
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.pipeline_parallel.model_rules import reject_unsupported_pp_structure
from halo.distributed.pipeline_parallel.partition import resolve_layer_partition, validate_partition
from halo.distributed.pipeline_parallel.stage import (
    PP_STAGE_PARTITION_ATTR,
    StageLayout,
    rebase_layer_index,
    slice_backbone_to_stage,
)
from halo.log import KEY_PREVIEW_COUNT
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.lazy_safetensors.materialization import WeightAction, WeightPlan
from halo.model_loading.lazy_safetensors.meta_shell import instantiate_on_meta
from halo.model_loading.parallel.expert_fuser import ExpertFuser, FusionTask
from halo.model_loading.parallel.lazy_load import load_planned_weights
from halo.model_loading.parallel.lazy_plan import (
    ExpertKeyLayout,
    LazyLoadPlan,
    is_lazy_loadable,
    lazy_load_prologue,
)
from halo.model_loading.parallel.model_source import ModelSource, agree_lazy_load
from halo.models.config_access import TIE_WORD_EMBEDDINGS_FLAG, tie_flag_configs
from halo.models.module_tree import split_layer_path
from halo.runtime.consensus import reject_across_ranks
from halo.runtime.load_coordination import timed_global_load
from halo.runtime.ranks import get_global_rank

logger = logging.getLogger(__name__)

# Names this loader in every uniform cross-rank rejection it raises.
_PP_LOAD_LABEL = "Stage-aware pipeline load"


class StageKeyFilter:
    """Narrows a whole-model weight plan to one stage's layers ``[lo, hi)`` under ``layer_root`` (the unsplit
    naming the checkpoint shares), re-based to 0; keys outside the layers belong to every stage."""

    def __init__(self, lo: int, hi: int, layer_root: str):
        self.lo = lo
        self.hi = hi
        self.layer_root = layer_root

    def owns(self, model_key: str) -> bool:
        parts = split_layer_path(model_key, self.layer_root)
        return parts is None or self.lo <= parts[0] < self.hi

    def stage_key(self, model_key: str) -> str:
        """``model_key`` in this stage's layer numbering, the inverse of ``global_parameter_name``'s shift."""
        return rebase_layer_index(model_key, self.layer_root, self.layer_root, -self.lo) or model_key

    def model_key(self, stage_key: str) -> str:
        """``stage_key`` back in the unsplit model's numbering, the inverse of :meth:`stage_key`."""
        return rebase_layer_index(stage_key, self.layer_root, self.layer_root, self.lo) or stage_key

    def filter(self, plans: list[WeightPlan]) -> list[WeightPlan]:
        return [replace(plan, model_key=self.stage_key(plan.model_key)) for plan in plans if self.owns(plan.model_key)]


def _instantiate_untied_on_meta(model_path: str, model_class, config, **model_kwargs) -> nn.Module:
    """The meta shell built with weight tying suppressed (``from_pretrained``'s tied-value ``torch.equal`` has
    no meta kernel), the declared flag written back for the pipeline tie gate and the saved stage config."""
    holders = tie_flag_configs(config)
    declared = {path: bool(getattr(holder, TIE_WORD_EMBEDDINGS_FLAG)) for path, holder in holders.items()}
    try:
        for holder in holders.values():
            setattr(holder, TIE_WORD_EMBEDDINGS_FLAG, False)
        model = instantiate_on_meta(model_path, model_class, config, **model_kwargs)
    finally:
        for path, flag in declared.items():
            setattr(holders[path], TIE_WORD_EMBEDDINGS_FLAG, flag)

    # ``from_pretrained`` deep-copies the config, so the shell carries its own suppressed tree.
    built = tie_flag_configs(model.config)
    unrestorable = sorted(set(declared) - set(built))
    if unrestorable:
        raise RuntimeError(
            f"Meta shell for {type(model).__name__} lost the config level(s) {unrestorable} that "
            f"declare {TIE_WORD_EMBEDDINGS_FLAG}, so the declaration suppressed to build it cannot be restored. "
            f"Leaving it suppressed would let a tied checkpoint past the pipeline tie gate and save "
            f"a stage config that silently untied the head."
        )
    for path, flag in declared.items():
        setattr(built[path], TIE_WORD_EMBEDDINGS_FLAG, flag)
    return model


def _split_missing_shard_plans(
    plans: list[WeightPlan],
    missing_files: list[str],
    droppable: tuple[str, ...],
    model_path: str,
    pp_rank: int,
) -> tuple[list[WeightPlan], list[str], str | None]:
    """``(loadable_plans, dropped_keys, reason)``: plans on absent shards drop when all sit under the ``droppable``
    paths the stage build drops (a per-node pipeline save), else the stage-local ``reason`` the caller joins."""
    if not missing_files:
        return plans, [], None
    missing = set(missing_files)
    from_missing = [plan for plan in plans if plan.shard_file in missing]
    if not from_missing:
        return plans, [], None
    undroppable = sorted(
        plan.model_key
        for plan in from_missing
        if not any(plan.model_key == prefix or plan.model_key.startswith(prefix + ".") for prefix in droppable)
    )
    if undroppable:
        return (
            plans,
            [],
            (
                f"PP stage {pp_rank}: {len(undroppable)} planned tensor(s) live in shard file(s) absent from "
                f"{model_path} and are NOT the cross-stage modules this stage drops — the checkpoint on this "
                f"node's filesystem does not carry state the stage trains. {PER_NODE_PLACEMENT_REMEDY} "
                f"First few: {undroppable[:KEY_PREVIEW_COUNT]} (missing: {missing_files[:3]})."
            ),
        )
    dropped_keys = sorted(plan.model_key for plan in from_missing)
    logger.info(
        f"PP stage {pp_rank}: skipping {len(dropped_keys)} cross-stage tensor(s) whose shard files are "
        f"absent locally (dropped at the stage build): {dropped_keys[:KEY_PREVIEW_COUNT]}"
    )
    return [plan for plan in plans if plan.shard_file not in missing], dropped_keys, None


def _missing_fusion_shards_reason(
    fusion_tasks: list[FusionTask], missing_files: list[str], model_path: str, pp_rank: int
) -> str | None:
    """Why this stage's fusion tasks need shard files this filesystem lacks: experts are never droppable
    cross-stage state, so any hit is the wrong-topology refusal."""
    if not fusion_tasks or not missing_files:
        return None
    hit = sorted(set(missing_files).intersection(set().union(*(task.shard_files for task in fusion_tasks))))
    if not hit:
        return None
    return (
        f"PP stage {pp_rank}: the per-expert fusion needs shard file(s) absent from {model_path} "
        f"({hit[:3]}) — the checkpoint on this node's filesystem does not carry experts the stage "
        f"trains. {PER_NODE_PLACEMENT_REMEDY}"
    )


def _unresolved_plans_reason(model: nn.Module, plans: list[WeightPlan], pp_rank: int, lo: int, hi: int) -> str | None:
    """Why a re-based plan names a tensor the sliced stage does not have, or ``None``; stage 0 (an identity
    re-base) never fails it, so the verdict is joined across ranks rather than raised."""
    stage_keys = set(model.state_dict().keys())
    unresolved = sorted({plan.model_key for plan in plans if plan.action is not WeightAction.IGNORE} - stage_keys)
    if not unresolved:
        return None
    return (
        f"PP stage {pp_rank} (layers [{lo}, {hi})): {len(unresolved)} planned checkpoint key(s) "
        f"name no parameter on the sliced stage — the layer re-base does not match the stage's "
        f"own naming. First few: {unresolved[:KEY_PREVIEW_COUNT]}."
    )


def _unmaterialized_reason(model: nn.Module, model_path: str, pp_rank: int, lo: int, hi: int) -> str | None:
    """Why a stage ``state_dict`` tensor is still on meta, or ``None``; equal layer shapes hide it until forward."""
    stranded = sorted(name for name, tensor in model.state_dict().items() if tensor is not None and tensor.is_meta)
    if not stranded:
        return None
    return (
        f"PP stage {pp_rank} (layers [{lo}, {hi})): {len(stranded)} parameter(s)/buffer(s) were "
        f"never materialized from {model_path} and remain on the meta device — they would train "
        f"as garbage or fail at the first forward. First few: {stranded[:KEY_PREVIEW_COUNT]}."
    )


def _plan_stage(
    base: LazyLoadPlan,
    model_path: str,
    pp_rank: int,
    pp_size: int,
    pp_split: list[int] | None,
    fuser: ExpertFuser | None,
) -> tuple[list[WeightPlan], list[str], list[FusionTask], list[tuple[int, int]], StageKeyFilter]:
    """COLLECTIVE: this stage's plans, dropped keys, fusion tasks, partition and key filter; slices the shell.
    Stage-local verdicts are joined on the meta shell, before any rank pays for a weight."""
    model, missing_files = base.model, base.missing_shard_files
    if missing_files:
        logger.info(
            f"[Rank {get_global_rank()}] {len(missing_files)}/{len(base.shard_files)} indexed shard file(s) "
            f"absent at {model_path} (per-node pipeline checkpoint); loaded only locally-present shards."
        )
    reject_unsupported_pp_structure(model)

    partition = resolve_layer_partition(model, pp_size, pp_split)
    lo, hi = partition[pp_rank]
    validate_partition(model, partition)

    layout = StageLayout.of(model)
    stage_filter = StageKeyFilter(lo, hi, layout.layer_root)
    plans, dropped_keys, missing_reason = _split_missing_shard_plans(
        stage_filter.filter(base.plans),
        missing_files,
        layout.dropped_prefixes(pp_rank == 0, pp_rank == pp_size - 1),
        model_path,
        pp_rank,
    )
    reject_across_ranks(missing_reason, _PP_LOAD_LABEL)

    fusion_tasks = []
    if fuser is not None:
        fusion_tasks = [
            replace(task, model_key=stage_filter.stage_key(task.model_key))
            for task in fuser.detect_tasks(base.weight_map, base.disk_to_model, base.model_keys)
            if stage_filter.owns(task.model_key)
        ]
    # Per-expert keys never ride through the plans, so the gate above cannot see their shard files.
    reject_across_ranks(
        _missing_fusion_shards_reason(fusion_tasks, missing_files, model_path, pp_rank), _PP_LOAD_LABEL
    )

    # Sliced only now: the plans above live in the unsplit model's key space, the checkpoint index's.
    slice_backbone_to_stage(model, lo, hi)
    setattr(model, PP_STAGE_PARTITION_ATTR, partition)
    reject_across_ranks(_unresolved_plans_reason(model, plans, pp_rank, lo, hi), _PP_LOAD_LABEL)
    return plans, dropped_keys, fusion_tasks, partition, stage_filter


def _load_stage_lazy(
    model_path: str,
    pp_rank: int,
    pp_size: int,
    *,
    config,
    pp_split: list[int] | None = None,
    ep_config: EPConfig | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = True,
    model_class=None,
    keep_fp32_params: bool = False,
    preserve_checkpoint_precision: bool = False,
    **model_kwargs,
) -> nn.Module:
    """COLLECTIVE: a ``*ForCausalLM`` holding only this stage's layers re-based to 0, its partition on
    ``PP_STAGE_PARTITION_ATTR``; ``preserve_checkpoint_precision`` requires every fp32 master restored."""
    rank = get_global_rank()
    base = lazy_load_prologue(
        model_path,
        config,
        ep_config,
        dtype=dtype,
        meta_init_phase="pp_stage_meta_init",
        # A per-node PP save leaves each node only its own stage's shard files under the full index, which
        # from_pretrained would open; the plan gate decides whether the absent tensors are droppable.
        build_meta_shell=lambda incomplete: _instantiate_untied_on_meta(
            model_path,
            model_class or AutoModelForCausalLM,
            config,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            config_only=incomplete,
            **model_kwargs,
        ),
    )
    model = base.model
    fuser = (
        ExpertFuser(ep_config.expert_start_idx, ep_config.expert_end_idx)
        if ep_config is not None and base.expert_layout == ExpertKeyLayout.PER_EXPERT
        else None
    )
    plans, dropped_keys, fusion_tasks, partition, stage_filter = _plan_stage(
        base, model_path, pp_rank, pp_size, pp_split, fuser
    )
    lo, hi = partition[pp_rank]

    def report(plans: list[WeightPlan], fused_keys: set[str]) -> None:
        live = [plan for plan in plans if plan.action is not WeightAction.IGNORE]
        n_shard = sum(1 for plan in live if plan.action is WeightAction.EXPERT_SHARD)
        logger.info(
            f"[Rank {rank}] PP stage {pp_rank}/{pp_size} layers [{lo}, {hi}) of {partition[-1][1]}: "
            f"{len(live)} tensors ({n_shard} expert-sharded, {len(fused_keys)} fused) from "
            f"{len({plan.shard_file for plan in live})}/{len(base.shard_files)} shards"
        )

    # Stage-dependent raises defer to the fence instead of stranding the other stages. Masters are captured after the
    # slice, in the stage's own numbering, as the plans now are.
    weights_already_sharded = load_planned_weights(
        base,
        plans,
        model_path=model_path,
        ep_config=ep_config,
        keep_fp32_params=keep_fp32_params,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        what=f"PP stage weight load from {model_path}",
        label=f"PP stage {pp_rank}",
        report=report,
        fuser=fuser,
        fusion_tasks=fusion_tasks,
        dropped_keys=dropped_keys,
    )

    if ep_config is not None:
        # Any failure is stage-local and reaches the peers through the join.
        reason = None
        try:
            apply_ep_layers(model, ep_config, weights_already_sharded=weights_already_sharded)
        except Exception as e:  # a stage-local verdict, joined below
            reason = f"EP patching failed on pipeline stage {pp_rank}: {type(e).__name__}: {e}"
        reject_across_ranks(reason, _PP_LOAD_LABEL)
    # External tables are planned in the checkpoint's key space, so a table on a later stage reads its own layer's.
    finalize_loaded_model(model, checkpoint_name=stage_filter.model_key)
    reject_across_ranks(_unmaterialized_reason(model, model_path, pp_rank, lo, hi), _PP_LOAD_LABEL)
    return model


def load_pp_stage_model(
    source: ModelSource,
    pc: ParallelismConfig,
    model_class,
    from_pretrained_kwargs: dict,
    *,
    preserve_checkpoint_precision: bool,
    wraps_experts: bool,
) -> nn.Module:
    """COLLECTIVE: this pipeline stage's decoder layers only, streamed per rank from safetensors, MoE stages
    already EP- or grouped-GEMM-wrapped (``wraps_experts``); ``build_pipeline_stage`` finishes the split."""
    # Before the lazy gate: ep_config() issues new_group on every rank, which no per-rank file probe may precede.
    ep_config = pc.ep_config() if wraps_experts else None
    if not agree_lazy_load(source, lazy_requested=True, layout_supported=is_lazy_loadable):
        raise ValueError(
            f"Pipeline parallelism needs a safetensors checkpoint the stage-aware loader can read, "
            f"but {source.name!r} has none. Falling back to a whole-model load would make "
            f"every rank hold all pipeline_parallel_size={pc.pp_size} stages — the ceiling PP "
            f"exists to remove. Either the checkpoint is not safetensors (convert it), or this "
            f"family's spec declares supports_lazy_loading=False / nests its expert index "
            f"under an extra module (pre-5.14 Zaya checkpoints), in which case the model cannot "
            f"use PP at all (see agent-docs/parallelism/pipeline-parallelism.md)."
        )
    # No reason left to carry (resolve_model_source verified every rank's directory); the join stays in the
    # load's collective sequence.
    reject_across_ranks(None, "Resolving the pipeline stage checkpoint")
    return timed_global_load(
        "PP",
        "pp_stage_lazy",
        lambda: _load_stage_lazy(
            source.local_dir,
            pc.pp_rank,
            pc.pp_size,
            pp_split=pc.pp_split,
            ep_config=ep_config,
            model_class=model_class,
            keep_fp32_params=pc.fp32_non_ep_params,
            preserve_checkpoint_precision=preserve_checkpoint_precision,
            **from_pretrained_kwargs,
        ),
    )
