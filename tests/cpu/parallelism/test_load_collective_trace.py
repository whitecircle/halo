#!/usr/bin/env python
"""What every branch of the loader ladder issues, from a local directory and from the Hub, pinned against
the loader collective-trace golden: process groups in build order with their members, collectives,
device meshes, c10d-store operations, node-load phases and the model builds between them.

Run: python tests/cpu/parallelism/test_load_collective_trace.py
"""

import json
from unittest.mock import patch

import pytest
from transformers import AutoConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.models.moe_balancing import config_has_experts, has_ep_wrapper_class
from tests.common.goldens.load_collective_trace import (
    GOLDEN_PATH,
    LOAD_CASES,
    build_checkpoints,
    golden_key,
    trace_case,
    trace_rank,
)

GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
CASES = {case.key: case for case in LOAD_CASES}


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    return build_checkpoints(tmp_path_factory.mktemp("load_collective_trace"))


def test_the_cases_are_the_golden_cases():
    assert list(CASES) == list(GOLDEN)
    assert all(list(GOLDEN[case.key]) == [golden_key(rank) for rank in case.ranks] for case in LOAD_CASES)


@pytest.mark.parametrize("case", LOAD_CASES, ids=lambda case: case.key)
def test_the_load_issues_the_golden_trace(case, checkpoints):
    assert trace_case(case, checkpoints) == GOLDEN[case.key]


def test_a_case_loads_a_moe_family_no_ep_layer_class_wraps(checkpoints):
    configs = [
        AutoConfig.from_pretrained(checkpoints.root / name) for name in {case.checkpoint for case in LOAD_CASES}
    ]
    assert any(config_has_experts(config) and not has_ep_wrapper_class(config) for config in configs)


@pytest.mark.parametrize(
    ("key", "host"),
    [
        pytest.param("ep lazy", {"DIST_INPUT_SHARED_FILESYSTEM": "0"}, id="per-node input under a shared case"),
        pytest.param("ep lazy", {"ACCELERATE_USE_FSDP": "true"}, id="an accelerate launch marker"),
        pytest.param(
            "ep lazy per-node filesystem",
            {"DIST_INPUT_SHARED_FILESYSTEM": "1"},
            id="shared input under a per-node case",
        ),
    ],
)
def test_the_hosts_launch_environment_does_not_reach_the_trace(key, host, checkpoints, monkeypatch):
    """Each host setting, read by the recording, would scope the load's phases differently or refuse it."""
    for name, value in host.items():
        monkeypatch.setenv(name, value)
    assert trace_case(CASES[key], checkpoints) == GOLDEN[key]


def test_building_the_cp_groups_before_the_ep_groups_breaks_the_trace(checkpoints):
    """Anti-vacuity: both configs cache their groups, so building the CP config first moves its
    ``new_group`` calls ahead of the EP ones, which on a real job pairs one rank's EP group build
    with a peer's CP group build."""
    ep_config = ParallelismConfig.ep_config

    def cp_first(config):
        config.cp_config()
        return ep_config(config)

    with patch.object(ParallelismConfig, "ep_config", cp_first):
        reordered = checkpoints.portable(trace_rank(CASES["ep+cp lazy"], 0, checkpoints).events)
    golden = GOLDEN["ep+cp lazy"][golden_key(0)]
    assert sorted(reordered) == sorted(golden), "premise: the reorder issues the same events"
    assert reordered != golden


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
