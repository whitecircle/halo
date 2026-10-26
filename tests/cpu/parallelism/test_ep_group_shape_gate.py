#!/usr/bin/env python
"""``ParallelismConfig`` refuses every EP group shape the EP process groups cannot tile.

It is the one gate: ``EPConfig`` is built from a validated config (``ParallelismConfig.ep_config``) and
does not re-check the shape, so a refusal missing here would only surface at group construction, after
the model loads. The shapes it admits build their groups.

Run: python tests/cpu/parallelism/test_ep_group_shape_gate.py
"""

from __future__ import annotations

import re

import pytest

from tests.common.parallelism import built_ep_config, make_parallelism_config

# (id, world, ep_size, expert_tp_size, ep_scope, pp_size, message), on 8-GPU NVLink domains.
_REFUSED = [
    ("node-group-exceeds-domain", 16, 16, 1, "node", 1, "EP group size (16) cannot exceed the NVLink domain (8)"),
    ("node-group-indivisible", 8, 3, 1, "node", 1, "EP group size (3) must divide the NVLink domain (8)"),
    ("global-group-exceeds-scope", 16, 32, 1, "global", 1, "EP group size (32) cannot exceed world size (16)"),
    ("global-group-indivisible", 8, 3, 1, "global", 1, "EP group size (3) must divide world size (8)"),
    # Under PP a group is bounded by one stage's world, and the message names that unit.
    ("global-group-indivisible-pp2", 16, 3, 1, "global", 2, "EP group size (3) must divide stage world size"),
    ("cross-node-etp-shape", 16, 4, 4, "global", 1, "Cross-node EP+ETP supports one ETP group per NVLink domain"),
]


@pytest.mark.parametrize(
    "case_id,world,ep_size,expert_tp_size,ep_scope,pp_size,message", _REFUSED, ids=[c[0] for c in _REFUSED]
)
def test_a_shape_the_groups_cannot_tile_is_refused_at_config_time(
    case_id, world, ep_size, expert_tp_size, ep_scope, pp_size, message
):
    with pytest.raises(ValueError, match=re.escape(message)):
        make_parallelism_config(
            world_size=world, ep_size=ep_size, expert_tp_size=expert_tp_size, ep_scope=ep_scope, pp_size=pp_size
        )


# (id, world, ep_size, expert_tp_size, ep_scope) of a shape the config admits, on 8-GPU NVLink domains.
_ACCEPTED = [
    ("no-ep", 8, 1, 1, "node"),
    ("node-ep2", 8, 2, 1, "node"),
    ("node-ep-fills-domain", 8, 8, 1, "node"),
    ("node-ep2-etp2", 8, 2, 2, "node"),
    ("global-single-group", 16, 16, 1, "global"),
    ("cross-node-one-etp-group-per-domain", 16, 2, 8, "global"),
]


@pytest.mark.parametrize("case_id,world,ep_size,expert_tp_size,ep_scope", _ACCEPTED, ids=[c[0] for c in _ACCEPTED])
def test_a_shape_the_config_admits_builds_its_groups(case_id, world, ep_size, expert_tp_size, ep_scope):
    config = make_parallelism_config(
        world_size=world, ep_size=ep_size, expert_tp_size=expert_tp_size, ep_scope=ep_scope
    )

    ep_config = built_ep_config(config)

    assert ep_config.num_ep_groups == config.num_ep_groups


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
