"""EP group-order golden: the ``new_group`` calls each shape's ``EPConfig`` makes, in order, and which call built each
group a rank keeps. ``new_group`` is collective over the default group, so the record raises where ranks disagree, as a
real job hangs there. A group is identified by its call: same members, two communicators."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from halo.env import nccl_timeout
from tests.common.goldens import regen_main
from tests.common.goldens.shape_grid import GridGolden, LayoutShape
from tests.common.parallelism import StubGroup, stubbed_process_groups

GROUP_FIELDS = ("process_group", "dispatch_ep_group", "expert_tp_group", "expert_replica_group", "dp_scope_group")
# Off its default, so a timeout fixed in code or read at import is written as its value; only a build-time read is
# written as the call.
RECORD_ENV = {"DIST_NCCL_TIMEOUT_MINUTES": "7"}

# Node-local and cross-node EP at expert TP 1, 2 and 4, EP or expert TP alone on two stages, and one-rank EP groups,
# whose creation is skipped.
LITERAL_SHAPES = (
    LayoutShape(8, 8, 8),
    LayoutShape(16, 8, 8, ep_size=8),
    LayoutShape(16, 8, 8, pp_size=2),
    LayoutShape(16, 8, 8, pp_size=2, ep_size=2),
    LayoutShape(8, 8, 8, expert_tp_size=2),
    LayoutShape(8, 8, 8, ep_size=4, expert_tp_size=2),
    LayoutShape(8, 8, 8, ep_size=2, expert_tp_size=4),
    LayoutShape(16, 8, 8, pp_size=2, expert_tp_size=2),
    LayoutShape(16, 8, 8, pp_size=2, expert_tp_size=4),
    LayoutShape(16, 8, 8, ep_size=16, ep_scope="global"),
    LayoutShape(16, 8, 8, ep_size=8, ep_scope="global"),
    LayoutShape(32, 8, 8, pp_size=2, ep_size=16, ep_scope="global"),
    LayoutShape(8, 2, 2, ep_size=4, expert_tp_size=2, ep_scope="global"),
    LayoutShape(8, 4, 4, ep_size=2, expert_tp_size=4, ep_scope="global"),
)


def _argument(name: str, value) -> str:
    """A ``new_group`` keyword as the golden writes it: the watchdog timeout as ``nccl_timeout()`` when that call
    returns it."""
    if name == "timeout" and value == nccl_timeout():
        return "nccl_timeout()"
    return repr(value)


def _call(group: StubGroup) -> list:
    return [list(group), {name: _argument(name, value) for name, value in group.kwargs.items()}]


def _call_index(calls: list[StubGroup], group: StubGroup | None) -> int | None:
    if group is None:
        return None
    for index, built in enumerate(calls):
        if built is group:
            return index
    raise AssertionError(f"group {list(group)} was not built by any of this rank's new_group calls")


def group_order_record(shape: LayoutShape) -> dict:
    """The ``new_group`` calls every rank makes building its ``EPConfig``, the call each group comes from, and
    ``ep_group_idx`` and ``expert_replica_ranks``."""
    orders = []
    built_by = {field: [] for field in GROUP_FIELDS}
    ep_group_idx, expert_replica_ranks = [], []
    with patch.dict(os.environ, RECORD_ENV):
        for rank in range(shape.world_size):
            config = shape.config_as_rank(rank)
            calls: list[StubGroup] = []
            with stubbed_process_groups(shape.world_size, rank, calls):
                ep_config = config.ep_config()
            orders.append([_call(group) for group in calls])
            for field in GROUP_FIELDS:
                built_by[field].append(_call_index(calls, getattr(ep_config, field)))
            ep_group_idx.append(ep_config.ep_group_idx)
            expert_replica_ranks.append(ep_config.expert_replica_ranks)
    diverging = [rank for rank, order in enumerate(orders) if order != orders[0]]
    if diverging:
        raise AssertionError(
            f"{shape.key}: ranks {diverging} make other new_group calls than rank 0, where a real job hangs"
        )
    return {
        "new_group": orders[0],
        "built_by": built_by,
        "ep_group_idx": ep_group_idx,
        "expert_replica_ranks": expert_replica_ranks,
    }


GOLDEN = GridGolden(Path(__file__).with_suffix(".json"), group_order_record, LITERAL_SHAPES)


if __name__ == "__main__":
    regen_main(GOLDEN.path, GOLDEN.render)
