#!/usr/bin/env python
"""What every branch of the loader ladder issues, pinned against the loader collective-trace golden."""

import json
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from halo.distributed import rank_uniform_env
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.moe.config_fields import has_routed_experts
from halo.models.moe.family_specs import find_moe_family_spec_for_config
from tests.common.collective_recorder import recording_rank
from tests.common.goldens.load_collective_trace import (
    GOLDEN_PATH,
    GPUS_PER_NODE,
    LOAD_CASES,
    WORLD_SIZE,
    build_checkpoints,
    cpu_model_builds,
    golden_key,
    trace_case,
    trace_rank,
)
from tests.common.parallelism import make_parallelism_config

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
    assert any(has_routed_experts(config) and find_moe_family_spec_for_config(config) is None for config in configs)


def test_a_moe_no_ep_layer_class_wraps_loads_stock_experts_with_no_ep_groups(checkpoints):
    config = make_parallelism_config(world_size=WORLD_SIZE, gpus_per_node=GPUS_PER_NODE, max_concurrent_loading=1)
    source = checkpoints.source(CASES["undistributed moe without an EP class"])
    with (
        recording_rank(0, WORLD_SIZE, GPUS_PER_NODE) as trace,
        cpu_model_builds(trace),
        patch.object(rank_uniform_env, "_env_uniformity_verified", True),
    ):
        model, _ = load_distributed_model(source, config, attn_implementation="eager", trust_remote_code=True)
    assert not [event for event in trace.events if event.startswith("new_group")]
    assert not named_ep_layers(model)
    stock = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16).state_dict()
    state = model.state_dict()
    assert state.keys() == stock.keys()
    for key, tensor in stock.items():
        assert torch.equal(state[key], tensor), key
    assert model(input_ids=torch.tensor([[1, 2, 3]])).logits.isfinite().all()


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
    """Each host setting would scope the load's phases differently or refuse it."""
    for name, value in host.items():
        monkeypatch.setenv(name, value)
    assert trace_case(CASES[key], checkpoints) == GOLDEN[key]


def test_building_the_cp_groups_before_the_ep_groups_breaks_the_trace(checkpoints):
    """Anti-vacuity: on a real job this pairs one rank's EP group build with a peer's CP group build."""
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
