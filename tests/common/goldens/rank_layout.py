"""Rank-layout golden: the process groups, rank coordinates and expert grad-sync regime every grid
shape builds today from each rank's ``ParallelismConfig`` and the ``EPConfig``/``CPConfig`` it hands
out, stored as a digest per shape plus the full layout of each shape in :data:`LITERAL_SHAPES`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from tests.common.goldens import regen_main
from tests.common.goldens.shape_grid import GridGolden, LayoutShape
from tests.common.parallelism import stubbed_process_groups

# The EPConfig attributes that decide where expert gradients are synced; each is job-wide.
EP_GRAD_SYNC_FLAGS = ("num_ep_groups", "needs_expert_grad_sync", "is_deferred_dp", "defer_grad_sync")

# Every axis combination, both EP scopes, the column-block degenerates (one member per domain, one
# global group), EP+ETP on and across domains, pipelines, a 1-GPU domain and odd widths.
LITERAL_SHAPES = (
    LayoutShape(1, 1, 1),
    LayoutShape(2, 2, 2, ep_size=2),
    LayoutShape(4, 4, 4, ep_size=4),
    LayoutShape(8, 8, 8),
    LayoutShape(8, 8, 8, ep_size=2),
    LayoutShape(8, 8, 8, ep_size=8),
    LayoutShape(8, 8, 8, ep_size=8, ep_scope="global"),
    LayoutShape(8, 8, 8, ep_size=2, ep_scope="global"),
    LayoutShape(8, 8, 8, expert_tp_size=2),
    LayoutShape(8, 8, 8, expert_tp_size=8, ep_scope="global"),
    LayoutShape(8, 8, 8, ep_size=4, expert_tp_size=2),
    LayoutShape(8, 8, 8, ep_size=2, expert_tp_size=2),
    LayoutShape(8, 8, 8, ep_size=2, expert_tp_size=4),
    LayoutShape(8, 8, 8, tp_size=2),
    LayoutShape(8, 8, 8, tp_size=8),
    LayoutShape(8, 8, 8, cp_size=2),
    LayoutShape(8, 8, 8, cp_size=8),
    LayoutShape(8, 8, 8, ep_size=2, tp_size=2),
    LayoutShape(8, 8, 8, ep_size=8, tp_size=4),
    LayoutShape(8, 8, 8, ep_size=8, cp_size=8),
    LayoutShape(16, 8, 8),
    LayoutShape(16, 8, 8, ep_size=2),
    LayoutShape(16, 8, 8, ep_size=8),
    LayoutShape(16, 8, 8, ep_size=16, ep_scope="global"),
    LayoutShape(16, 8, 8, ep_size=8, ep_scope="global"),
    LayoutShape(16, 8, 8, ep_size=2, ep_scope="global"),
    LayoutShape(16, 8, 8, ep_size=2, expert_tp_size=8, ep_scope="global"),
    LayoutShape(16, 8, 8, expert_tp_size=2),
    LayoutShape(16, 8, 8, ep_size=16, tp_size=4, ep_scope="global"),
    LayoutShape(16, 8, 8, ep_size=8, cp_size=8),
    LayoutShape(16, 8, 8, tp_size=4),
    LayoutShape(16, 8, 8, cp_size=4),
    LayoutShape(16, 8, 8, pp_size=2),
    LayoutShape(16, 8, 8, pp_size=2, ep_size=8),
    LayoutShape(16, 8, 8, pp_size=2, ep_size=2),
    LayoutShape(16, 8, 8, pp_size=2, expert_tp_size=2),
    LayoutShape(32, 8, 8, pp_size=2, ep_size=16, ep_scope="global"),
    LayoutShape(4, 1, 1, ep_size=4, ep_scope="global"),
    LayoutShape(24, 8, 8, ep_size=6, ep_scope="global"),
    LayoutShape(12, 6, 6, ep_size=6, cp_size=3),
)


def _partition(handles: Iterable[tuple[int, ...] | None]) -> dict | None:
    """The distinct groups the ranks hold and, per rank, the index of the one it holds (``None`` for a
    rank holding none); ``None`` when no rank holds one."""
    held = list(handles)
    groups = sorted({group for group in held if group is not None})
    if not groups:
        return None
    index = {group: position for position, group in enumerate(groups)}
    return {
        "groups": [list(group) for group in groups],
        "of_rank": [None if group is None else index[group] for group in held],
    }


def _job_wide(holders: Sequence, attribute: str):
    """``attribute`` as every holder reports it; raises when the ranks disagree."""
    values = {getattr(holder, attribute) for holder in holders}
    if len(values) != 1:
        raise AssertionError(f"the ranks disagree on {attribute}: {sorted(values)}")
    return values.pop()


def layout_record(shape: LayoutShape) -> dict:
    """What every rank of ``shape`` builds, as JSON-native values."""
    configs = [shape.config_as_rank(rank) for rank in range(shape.world_size)]
    ep_configs, cp_configs = [], []
    for rank, config in enumerate(configs):
        with stubbed_process_groups(shape.world_size, rank):
            ep_configs.append(config.ep_config())
            if config.is_cp_mode:
                cp_configs.append(config.cp_config())
    pipelined = shape.pp_size > 1
    return {
        "data_parallel_size": _job_wide(configs, "data_parallel_size"),
        "num_ep_groups": _job_wide(configs, "num_ep_groups"),
        "dp_rank": [config.get_data_parallel_rank() for config in configs],
        "pp_chain": _partition(tuple(config.pp_chain_ranks()) for config in configs) if pipelined else None,
        "ep_group": _partition(ep.process_group for ep in ep_configs),
        "ep_rank": [ep.ep_rank for ep in ep_configs],
        "dispatch_group": _partition(ep.dispatch_ep_group for ep in ep_configs),
        "dispatch_ep_rank": [ep.dispatch_ep_rank for ep in ep_configs],
        "expert_tp_group": _partition(ep.expert_tp_group for ep in ep_configs),
        "expert_tp_rank": [ep.expert_tp_rank for ep in ep_configs],
        "expert_replica_group": _partition(ep.expert_replica_group for ep in ep_configs),
        "dp_scope_group": _partition(ep.dp_scope_group for ep in ep_configs),
        "cp_group": _partition(cp.process_group for cp in cp_configs) if cp_configs else None,
        "cp_rank": [cp.cp_rank for cp in cp_configs] if cp_configs else None,
        "ep_grad_sync": {attribute: _job_wide(ep_configs, attribute) for attribute in EP_GRAD_SYNC_FLAGS},
    }


GOLDEN = GridGolden(Path(__file__).with_suffix(".json"), layout_record, LITERAL_SHAPES)


if __name__ == "__main__":
    regen_main(GOLDEN.path, GOLDEN.render)
