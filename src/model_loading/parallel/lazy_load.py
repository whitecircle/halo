"""The weight load both lazy loaders (EP, PP stage) run once their plans are final: fp32-master capture, expert
fusion, the stream and its fence."""

from collections.abc import Callable, Sequence
from functools import partial

import torch

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.expert_fuser import ExpertFuser, FusionTask
from src.distributed.expert_parallel.lazy_plan import LazyLoadPlan
from src.model_loading.lazy_safetensors.materialization import (
    WeightAction,
    WeightPlan,
    materialize_empty_like,
    stream_planned_weights,
)
from src.model_loading.parallel.fp32_masters import fp32_master_param_keys, verify_fp32_master_coverage
from src.runtime.consensus import DeferredRankFailure
from src.runtime.ranks import get_local_rank


def load_planned_weights(
    base: LazyLoadPlan,
    plans: list[WeightPlan],
    *,
    model_path: str,
    ep_config: EPConfig | None,
    keep_fp32_params: bool,
    preserve_checkpoint_precision: bool,
    what: str,
    label: str,
    report: Callable[[list[WeightPlan], set[str]], None],
    fuser: ExpertFuser | None = None,
    fusion_tasks: list[FusionTask] | None = None,
    dropped_keys: Sequence[str] = (),
) -> bool:
    """COLLECTIVE (the closing fence): stream ``plans`` into ``base.model``; returns whether any expert tensor
    arrived as this rank's slice.

    Every step defers a rank-local raise to the fence: fp32-master capture, the per-expert fusion (``fusion_tasks``,
    else detected off ``base``), empty storage for ``dropped_keys``, a resume's master coverage, then ``report``.
    """
    model, dtype = base.model, base.dtype
    device = f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu"
    guard = DeferredRankFailure(what)
    # Read on the shell before a fusion replaces parameters: a resume's coverage check groups tied aliases by
    # this identity-keyed state.
    masters, master_state = guard.run(
        lambda: (
            fp32_master_param_keys(model, ep_config, keep_non_ep=keep_fp32_params),
            model.state_dict(keep_vars=True) if preserve_checkpoint_precision else None,
        )
    ) or (frozenset(), None)
    if fuser is not None and fusion_tasks is None:
        fusion_tasks = guard.run(partial(fuser.detect_tasks, base.weight_map, base.disk_to_model, base.model_keys))
    fused_keys: set[str] = set()
    if fusion_tasks:
        fused_keys = (
            guard.run(
                partial(fuser.execute, fusion_tasks, model, model_path, dtype=dtype, device=device, keep_fp32=masters)
            )
            or set()
        )
        plans = [plan for plan in plans if plan.model_key not in fused_keys]
    # Empty storage, deliberately not an init pass: the stage build drops these modules, and an RNG draw
    # would desynchronize the seed stream against ranks whose checkpoint carries them.
    for key in dropped_keys:
        guard.run(partial(materialize_empty_like, model, key, device, dtype))
    # Dropped keys count as planned (materialized, not absent) and need no master.
    planned = {plan.model_key for plan in plans} | fused_keys | set(dropped_keys)
    if preserve_checkpoint_precision:
        guard.run(partial(verify_fp32_master_coverage, master_state, masters, planned, source=model_path))
    report(plans, fused_keys)
    stream_planned_weights(
        model,
        plans,
        planned,
        model_path=model_path,
        weight_map=base.weight_map,
        device=device,
        dtype=dtype,
        keep_fp32=masters,
        run=guard.run,
        label=label,
    )
    guard.reject()
    return bool(fused_keys) or any(plan.action is WeightAction.EXPERT_SHARD for plan in plans)
