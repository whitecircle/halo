"""Stage-aware model loading for pipeline parallelism.

Every rank instantiates the full model on the meta device (zero bytes), plans the checkpoint against
that whole-model key space, then keeps only the plans for the decoder layers its stage holds —
re-based onto the stage's own layer numbering — and streams exactly those tensors from safetensors,
so the load-time ceiling falls with ``pp_size`` instead of pinning the whole model to one GPU.

Composition with EP is a filter rather than a second loader: :class:`PPWeightPlanner` narrows
whatever :class:`~src.distributed.expert_parallel.lazy_loader.EPWeightPlanner` produced (an
``EPConfig`` of ``None`` means no expert sharding), so dense and EP stages take one code path.

Embeddings, the final norm and the task head are materialized on every stage and dropped afterwards
by :func:`~src.distributed.pipeline_parallel.stage.build_pipeline_stage`: they are O(vocab x hidden)
once, while the decoder layers scale with depth.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import replace
from functools import partial

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.lazy_loader import (
    CheckpointFormat,
    ExpertFuser,
    lazy_load_prologue,
)
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.pipeline_parallel.split import (
    TIE_WORD_EMBEDDINGS_FLAG,
    resolve_layer_partition,
    resolve_module_attr,
    resolve_pp_spec,
    tie_flag_configs,
    validate_model_structure_supports_pp,
)
from src.distributed.pipeline_parallel.stage import (
    PP_STAGE_PARTITION_ATTR,
    input_embedding_module,
    module_path,
    rebase_layer_index,
    reject_layer_type_rebase,
    resolve_layer_root,
    slice_backbone_to_stage,
    split_layer_path,
)
from src.distributed.runtime import DeferredRankFailure, get_global_rank, get_local_rank, reject_across_ranks
from src.log import KEY_PREVIEW_COUNT
from src.models.loading.lazy_safetensors.meta_shell import instantiate_on_meta
from src.models.loading.lazy_safetensors.weights import (
    SafetensorsWeightLoader,
    WeightAction,
    WeightPlan,
    init_checkpoint_absent_modules,
    materialize_empty_like,
)
from src.models.patches.buffer_fixes import finalize_loaded_model
from src.models.structure import backbone_with_layers

logger = logging.getLogger(__name__)

# Label attached to this loader's cross-rank rejections.
_PP_LOAD_LABEL = "Stage-aware pipeline load"


class PPWeightPlanner:
    """Narrow a whole-model weight plan to one pipeline stage, re-basing its layer indices.

    ``layer_root`` is the dotted prefix of the decoder-layer paths in the unsplit model's naming
    (``model.layers.``, ``model.language_model.layers.``, …) — the space the checkpoint and the meta
    shell share. Keys outside it (embeddings, final norm, task head) are claimed by every stage.
    """

    def __init__(self, lo: int, hi: int, layer_root: str):
        self.lo = lo
        self.hi = hi
        self.layer_root = layer_root

    def owns(self, model_key: str) -> bool:
        """Whether this stage loads ``model_key``."""
        parts = split_layer_path(model_key, self.layer_root)
        return parts is None or self.lo <= parts[0] < self.hi

    def stage_key(self, model_key: str) -> str:
        """``model_key`` re-based onto this stage's layer numbering (identity outside the layers).

        Inverse of :meth:`~src.distributed.pipeline_parallel.stage.PipelineStageModule.global_parameter_name`'s
        index shift; both go through :func:`~src.distributed.pipeline_parallel.stage.rebase_layer_index`.
        """
        return rebase_layer_index(model_key, self.layer_root, self.layer_root, -self.lo) or model_key

    def filter(self, plans: list[WeightPlan]) -> list[WeightPlan]:
        """This stage's plans, with their model keys re-based."""
        return [replace(plan, model_key=self.stage_key(plan.model_key)) for plan in plans if self.owns(plan.model_key)]


def load_pp_stage_model(
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
    **model_kwargs,
) -> nn.Module:
    """Load only this pipeline stage's decoder layers, streamed from safetensors.

    Returns a normal ``*ForCausalLM`` (never a ``PipelineStageModule``) whose decoder-layer list
    already holds just this stage's layers, re-based to 0, so ``.config``, ``get_input_embeddings()``
    and ``resize_token_embeddings()`` keep working for every PP-enabled script.
    ``build_pipeline_stage`` finishes the split — it reads the partition back off
    :data:`~src.distributed.pipeline_parallel.stage.PP_STAGE_PARTITION_ATTR`.

    Raises:
        ValueError: the model's structure cannot be split (see
            :func:`~src.distributed.pipeline_parallel.split.validate_model_structure_supports_pp`).
        RuntimeError: a planned key does not exist on the sliced stage, a live tensor was never
            assigned, or this stage carries no EP MoE layer. Each condition is stage-local, so it is
            routed through :func:`~src.distributed.runtime.reject_across_ranks` and every rank raises.
    """
    rank = get_global_rank()
    device = f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu"

    base = lazy_load_prologue(
        model_path,
        config,
        ep_config,
        rank=rank,
        dtype=dtype,
        meta_init_phase="pp_stage_meta_init",
        # A per-node PP save leaves each node only its own stage's shard files plus the full index
        # (agent-docs/parallelism/pipeline-parallelism.md, Checkpoints). from_pretrained opens every
        # indexed file, so the meta shell is then built config-only and the plan gate below decides
        # whether the absent tensors are the droppable cross-stage ones.
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
    model, dtype = base.model, base.dtype
    missing_files = base.missing_shard_files
    if missing_files:
        logger.info(
            f"[Rank {rank}] {len(missing_files)}/{len(base.shard_files)} indexed shard file(s) absent at "
            f"{model_path} (per-node pipeline checkpoint); loaded only locally-present shards."
        )

    # Runs on the meta shell, before any weight is read.
    validate_model_structure_supports_pp(model)

    partition = resolve_layer_partition(model, pp_size, pp_split)
    lo, hi = partition[pp_rank]
    # Every rank validates every stage's offset and the family's boundary constraints, on the meta
    # shell: a rank-local raise would leave its peers waiting at the first collective below.
    for stage_rank, (stage_lo, _) in enumerate(partition):
        reject_layer_type_rebase(model, stage_lo, stage_rank)
    resolve_pp_spec(backbone_with_layers(model)).validate_partition(model, partition)

    planner = PPWeightPlanner(lo, hi, resolve_layer_root(model))
    plans = planner.filter(base.plans)

    plans, dropped_keys, missing_reason = _split_missing_shard_plans(
        model, plans, missing_files, model_path, pp_rank, pp_size
    )
    reject_across_ranks(missing_reason, _PP_LOAD_LABEL)

    fuser = fusion_tasks = None
    if ep_config is not None and base.ckpt_format == CheckpointFormat.INDIVIDUAL:
        fuser = ExpertFuser(ep_config.expert_start_idx, ep_config.expert_end_idx)
        fusion_tasks = [
            (planner.stage_key(model_key), fusion_type, experts)
            for model_key, fusion_type, experts in fuser.detect_tasks(
                base.weight_map, base.disk_to_model, base.model_keys
            )
            if planner.owns(model_key)
        ]
    # Per-expert keys map to no fused model key, so they never appear in ``plans`` and the
    # missing-shard gate above cannot see them; without this gate the fuser's ``safe_open`` raises
    # rank-local on an absent file, between two collectives.
    reject_across_ranks(
        _missing_fusion_shards_reason(fusion_tasks, missing_files, model_path, pp_rank), _PP_LOAD_LABEL
    )

    # Sliced only now: the plan above is built in the unsplit model's key space, the one the
    # checkpoint index and disk_to_model use.
    slice_backbone_to_stage(model, lo, hi)
    setattr(model, PP_STAGE_PARTITION_ATTR, partition)
    reject_across_ranks(_unresolved_plans_reason(model, plans, pp_rank, lo, hi), _PP_LOAD_LABEL)

    # The rank-local materialization steps below raise stage-dependent errors (a shape mismatch or
    # missing expert confined to one stage's layers); the guard turns those into the collective
    # rejection, so the other stages do not wait at the barrier.
    guard = DeferredRankFailure(f"PP stage weight load from {model_path}")
    fused_keys: set[str] = set()
    if fusion_tasks:
        fused_keys = (
            guard.run(partial(fuser.execute, fusion_tasks, model, model_path, dtype=dtype, device=device)) or set()
        )
        plans = [plan for plan in plans if plan.model_key not in fused_keys]

    if dropped_keys:
        guard.run(partial(_materialize_droppable_keys, model, dropped_keys, device, dtype))
    # dropped_keys count as planned: they are materialized (empty) above, not checkpoint-absent, and
    # the absent-module pass would otherwise re-init their modules and fail its coverage gate.
    planned = {plan.model_key for plan in plans} | fused_keys | set(dropped_keys)
    guard.run(partial(init_checkpoint_absent_modules, model, planned, device, f"PP stage {pp_rank}", dtype=dtype))

    live = [plan for plan in plans if plan.action is not WeightAction.IGNORE]
    n_shard = sum(1 for plan in live if plan.action is WeightAction.EXPERT_SHARD)
    logger.info(
        f"[Rank {rank}] PP stage {pp_rank}/{pp_size} layers [{lo}, {hi}) of {partition[-1][1]}: "
        f"{len(live)} tensors ({n_shard} expert-sharded, {len(fused_keys)} fused) from "
        f"{len({plan.shard_file for plan in live})}/{len(base.shard_files)} shards"
    )

    loader = SafetensorsWeightLoader(model_path, sorted({plan.shard_file for plan in live}), device=device)
    guard.run(partial(loader.load_into_model, model, plans, dtype=dtype))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Collective (all_gather_object): acts as the barrier and raises on any rank's deferred failure.
    guard.reject()
    if ep_config is not None:
        reason = None
        try:
            # Derived, not asserted (see load_ep_model_lazy): claiming "already sharded" for a
            # checkpoint whose expert keys matched no pattern would give every rank experts
            # 0..E_local-1.
            model = patch_moe_model_for_ep(model, ep_config, weights_already_sharded=bool(fused_keys) or n_shard > 0)
        except Exception as e:  # stage-local failure, reported collectively below
            # The common case is a hybrid MoE with a dense prefix (first_k_dense_replace,
            # mlp_only_layers) leaving a leading stage with no MoE block. Every exception is caught,
            # not just that ValueError: patching runs each family's wrapper constructor and
            # detect_num_experts over this stage's layers, so a shape or key error is stage-local
            # too, and one escaping here would leave the peers waiting in the gather below.
            reason = f"EP patching failed on pipeline stage {pp_rank}: {type(e).__name__}: {e}"
        reject_across_ranks(reason, _PP_LOAD_LABEL)
        # Only this stage's layers exist, so the buffers are stage-scoped by construction; EP groups
        # never straddle a stage, so every rank of a stage creates the same set.
        create_ep_buffers(model)

    # Non-persistent buffers stayed meta on the shell; the tie leg is a no-op (PP rejects tied
    # embeddings).
    finalize_loaded_model(model)

    reject_across_ranks(_unmaterialized_reason(model, model_path, pp_rank, lo, hi), _PP_LOAD_LABEL)
    return model


def _instantiate_untied_on_meta(model_path: str, model_class, config, **model_kwargs) -> nn.Module:
    """Build the meta shell with weight tying suppressed, then restore the declared flag on it.

    When the checkpoint carries both tied keys, ``from_pretrained`` decides whether to honour the tie
    by comparing their values with ``torch.equal``, which has no meta kernel: a tied checkpoint would
    fail with ``NotImplementedError: aten::equal`` before
    :func:`~src.distributed.pipeline_parallel.split.validate_model_structure_supports_pp` runs.
    Building untied does not change that gate's verdict — it reads the declared flag and the presence
    of an output embedding, which is what permits a tie-declaring reward/classification head — so the
    flag is restored for the gate and for the stage config it is saved into.
    """
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


def _cross_stage_droppable_prefixes(model: nn.Module, pp_rank: int, pp_size: int) -> tuple[str, ...]:
    """Module paths this stage materializes only to drop at the stage build.

    Mirrors :func:`~src.distributed.pipeline_parallel.stage.build_pipeline_stage`: a non-first stage
    drops the input embedding, a non-last stage replaces the final norm with ``Identity`` and sheds
    the task head. Their checkpoint values are never read on this stage, so a per-node pipeline
    checkpoint that lacks them locally is still complete here. Resolved by the same identity/spec
    probes the stage build uses, not by hardcoded attribute names.
    """
    prefixes: list[str] = []
    backbone = backbone_with_layers(model)
    spec = resolve_pp_spec(backbone)
    backbone_prefix = module_path(model, backbone)
    if pp_rank > 0:
        prefixes.append(module_path(model, input_embedding_module(model)))
    if pp_rank < pp_size - 1:
        norm_attr = resolve_module_attr(backbone, spec.FINAL_NORM_ATTRS, "final norm")
        prefixes.append(f"{backbone_prefix}.{norm_attr}" if backbone_prefix else norm_attr)
        for tail_attr in spec.TAIL_MODULE_ATTRS:
            resolved = resolve_module_attr(backbone, (tail_attr,), "tail module")
            prefixes.append(f"{backbone_prefix}.{resolved}" if backbone_prefix else resolved)
        prefixes.append(resolve_module_attr(model, spec.HEAD_ATTRS, "task head"))
    return tuple(prefixes)


def _split_missing_shard_plans(
    model: nn.Module,
    plans: list[WeightPlan],
    missing_files: list[str],
    model_path: str,
    pp_rank: int,
    pp_size: int,
) -> tuple[list[WeightPlan], list[str], str | None]:
    """Drop the plans whose shard files are absent on this filesystem, if that is legitimate.

    Returns ``(loadable_plans, dropped_keys, reason)``. A per-node pipeline save leaves only this
    node's stage shard, so the only keys a stage may plan out of an absent file are the cross-stage
    modules it drops at the stage build (:func:`_cross_stage_droppable_prefixes`). Anything else
    means the checkpoint lacks state this stage trains; ``reason`` carries the stage-local rejection
    for ``reject_across_ranks``, returned rather than raised so the peers are not left in that gather.
    """
    if not missing_files:
        return plans, [], None
    missing = set(missing_files)
    from_missing = [plan for plan in plans if plan.shard_file in missing]
    if not from_missing:
        return plans, [], None
    droppable = _cross_stage_droppable_prefixes(model, pp_rank, pp_size)
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
                f"node's filesystem does not carry state the stage trains. A per-node pipeline checkpoint "
                f"resumes only onto the same topology and rank placement; for anything else gather every "
                f"node's directory onto one filesystem first. First few: {undroppable[:KEY_PREVIEW_COUNT]} "
                f"(missing: {missing_files[:3]})."
            ),
        )
    dropped_keys = sorted(plan.model_key for plan in from_missing)
    logger.info(
        f"PP stage {pp_rank}: skipping {len(dropped_keys)} cross-stage tensor(s) whose shard files are "
        f"absent locally (dropped at the stage build): {dropped_keys[:KEY_PREVIEW_COUNT]}"
    )
    return [plan for plan in plans if plan.shard_file not in missing], dropped_keys, None


def _missing_fusion_shards_reason(fusion_tasks, missing_files: list[str], model_path: str, pp_rank: int) -> str | None:
    """Why this stage's per-expert fusion tasks reference shard files absent on this filesystem.

    Expert weights are never droppable cross-stage state, so any hit is the wrong-topology rejection
    :func:`_split_missing_shard_plans` issues for planned keys.
    """
    if not fusion_tasks or not missing_files:
        return None
    missing = set(missing_files)
    hit = sorted(
        {
            shard_file
            for _, _, expert_dict in fusion_tasks
            for suffixes in expert_dict.values()
            for _, shard_file in suffixes.values()
            if shard_file in missing
        }
    )
    if not hit:
        return None
    return (
        f"PP stage {pp_rank}: the per-expert fusion needs shard file(s) absent from {model_path} "
        f"({hit[:3]}) — the checkpoint on this node's filesystem does not carry experts the stage "
        f"trains. A per-node pipeline checkpoint resumes only onto the same topology and rank "
        f"placement; for anything else gather every node's directory onto one filesystem first."
    )


def _materialize_droppable_keys(model: nn.Module, keys: list[str], device: str, dtype: torch.dtype | None) -> None:
    """Give the to-be-dropped cross-stage tensors real (uninitialized) storage off the meta device.

    Their values are never read — ``build_pipeline_stage`` drops or replaces these modules — but the
    coverage gate refuses any tensor left on meta, and FSDP cannot wrap one. ``torch.empty`` rather
    than an init pass: an RNG draw here would desynchronize the seed stream against ranks whose
    checkpoint carries the tensors.
    """
    for key in keys:
        materialize_empty_like(model, key, device, dtype)


def _unresolved_plans_reason(model: nn.Module, plans: list[WeightPlan], pp_rank: int, lo: int, hi: int) -> str | None:
    """Why a re-based plan names a tensor the sliced stage does not have, or ``None``.

    Bypassing ``from_pretrained`` means nothing else reports a bad key: the assignment would raise a
    bare ``AttributeError`` deep in an attribute walk, or for a stale name never run at all. Stage 0
    cannot fail this gate (its re-base is the identity), so the verdict is returned for
    :func:`~src.distributed.runtime.reject_across_ranks` rather than raised here.
    """
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
    """Coverage gate: every tensor of the stage's persistent state must hold real data.

    The stage-aware load bypasses ``from_pretrained``, so nothing else reports a missing checkpoint
    key. A tensor left on meta means either the checkpoint does not carry it or the re-base dropped
    its plan, and since every decoder layer shares its shapes an omission is otherwise invisible
    until the first forward. Scope is ``state_dict`` (parameters + persistent buffers), the state a
    checkpoint carries; non-persistent buffers are recomputed by the ``buffer_fixes`` pass.
    """
    stranded = sorted(name for name, tensor in model.state_dict().items() if tensor is not None and tensor.is_meta)
    if not stranded:
        return None
    return (
        f"PP stage {pp_rank} (layers [{lo}, {hi})): {len(stranded)} parameter(s)/buffer(s) were "
        f"never materialized from {model_path} and remain on the meta device — they would train "
        f"as garbage or fail at the first forward. First few: {stranded[:KEY_PREVIEW_COUNT]}."
    )
