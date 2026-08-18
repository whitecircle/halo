#!/usr/bin/env python
"""``EPConfig`` and ``ParallelismConfig`` must accept and refuse the SAME EP group shapes.

Both classes gate the EP dispatch group: ``ParallelismConfig`` at config time for every training
run, ``EPConfig`` for the 30 files (tests, ``tests/common/pp_equivalence``) that build it directly.
Spelling the fit/divide predicates and their messages separately on each side lets them drift, so a
shape one accepts the other rejects — either a run refused for the wrong reason or, worse, a
directly built EPConfig admitting a topology the run-level gate exists to stop. The predicates live
once in ``group_layout``; this pins the agreement so an edit to one side cannot silently move only
that side.

The message SPELLINGS deliberately differ in their remedy (the two APIs offer different escape
hatches) and in the scope noun under PP, so the assertions pin the shared invariant clause only.

Run: python tests/cpu/parallelism/test_ep_guard_agreement.py
"""

from __future__ import annotations

import pytest

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.group_layout import reject_cross_node_etp_shape
from tests.common.parallelism import make_parallelism_config


def _parallelism_config(*, world_size, domain_size, rank=0, **kwargs):
    return make_parallelism_config(world_size=world_size, gpus_per_node=domain_size, rank=rank, **kwargs)


def _ep_config(*, world_size, domain_size, ep_size, expert_tp_size, scope):
    return EPConfig(
        ep_size=ep_size,
        expert_tp_size=expert_tp_size,
        world_size=world_size,
        gpus_per_node=domain_size,
        node_local=scope == "node",
    )


# (id, world, domain, ep_size, scope, pp_size, shared invariant fragment)
_REFUSED = [
    ("node-group-exceeds-domain", 16, 8, 16, "node", 1, "cannot exceed the NVLink domain"),
    ("node-group-indivisible", 8, 8, 3, "node", 1, "must divide the NVLink domain"),
    ("global-group-exceeds-scope", 16, 8, 32, "global", 1, "EP group size (32) cannot exceed"),
    ("global-group-indivisible", 8, 8, 3, "global", 1, "EP group size (3) must divide"),
    # PP: ParallelismConfig bounds the group by the STAGE world, EPConfig is constructed per stage
    # with that same width as its world — the scope_desc branch the other rows never reach.
    ("global-group-indivisible-pp2", 16, 8, 3, "global", 2, "EP group size (3) must divide"),
]


@pytest.mark.parametrize("case_id,world,domain,ep_size,scope,pp_size,fragment", _REFUSED, ids=[c[0] for c in _REFUSED])
def test_both_configs_refuse_the_same_shape(case_id, world, domain, ep_size, scope, pp_size, fragment):
    """A shape either gate refuses must be refused by both, naming the same invariant."""
    stage_world = world // pp_size

    with pytest.raises(ValueError) as pc_error:
        _parallelism_config(world_size=world, domain_size=domain, ep_size=ep_size, ep_scope=scope, pp_size=pp_size)
    with pytest.raises(ValueError) as ep_error:
        _ep_config(world_size=stage_world, domain_size=domain, ep_size=ep_size, expert_tp_size=1, scope=scope)

    assert fragment in str(pc_error.value), pc_error.value
    assert fragment in str(ep_error.value), ep_error.value


# (id, world, domain, ep_size, expert_tp_size, scope)
_ACCEPTED = [
    ("no-ep", 8, 8, 1, 1, "node"),
    ("node-ep2", 8, 8, 2, 1, "node"),
    ("node-ep-fills-domain", 8, 8, 8, 1, "node"),
    ("node-ep2-etp2", 8, 8, 2, 2, "node"),
    ("global-single-group", 16, 8, 16, 1, "global"),
]


@pytest.mark.parametrize("case_id,world,domain,ep_size,expert_tp_size,scope", _ACCEPTED, ids=[c[0] for c in _ACCEPTED])
def test_both_configs_accept_the_same_shape(case_id, world, domain, ep_size, expert_tp_size, scope):
    """The other half: a shape the run-level gate admits must not be refused by EPConfig."""
    _parallelism_config(
        world_size=world,
        domain_size=domain,
        ep_size=ep_size,
        expert_tp_size=expert_tp_size,
        ep_scope=scope,
    )
    _ep_config(world_size=world, domain_size=domain, ep_size=ep_size, expert_tp_size=expert_tp_size, scope=scope)


def test_cross_node_etp_shape_refusal_names_the_invariant():
    """The third shared predicate. EPConfig enforces it at group construction (which needs a live
    process group, so it is unreachable single-process); ParallelismConfig enforces it at config
    time. Both route through this one predicate, asserted here on its own."""
    with pytest.raises(ValueError, match="one ETP group per NVLink domain"):
        reject_cross_node_etp_shape(4, 8, 2, 2, remedy="Set expert_tp_size=8.")

    with pytest.raises(ValueError) as error:
        _parallelism_config(world_size=16, domain_size=8, ep_size=4, expert_tp_size=4, ep_scope="global")
    assert "one ETP group per NVLink domain" in str(error.value), error.value

    # Anti-vacuity: the shape that DOES form one ETP group per domain passes the same predicate.
    reject_cross_node_etp_shape(8, 8, 2, 2, remedy="unreachable")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
