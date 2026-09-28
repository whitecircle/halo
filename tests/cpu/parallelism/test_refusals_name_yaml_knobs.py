#!/usr/bin/env python
"""The parallelism refusals below name the knobs a user sets and print launch lines that run as
written; the ep1 expert-sharding refusal links the docs for its trade-off.

A run is configured through ``expert_parallel_size`` / ``tensor_parallel_size`` /
``expert_tensor_parallel_size`` / ``context_parallel_size``; ``ep_size`` / ``tp_size`` /
``expert_tp_size`` / ``cp_size`` are the dataclass fields behind them, and a refusal that prints
those names points at a key no YAML accepts. ``ep_group_size`` is derived, not set, so the messages
may keep it. Each case below trips its real gate.

Run: python tests/cpu/parallelism/test_refusals_name_yaml_knobs.py
"""

import re
from types import SimpleNamespace

import pytest

from src.distributed.parallelism_config import accelerate_launch_rejection
from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.parallelism import make_parallelism_config
from tests.common.utils import REPO_ROOT, load_script_module

INTERNAL_FIELD = re.compile(r"\b(ep_size|tp_size|expert_tp_size|cp_size)\b")
EP1_SHARDING_DOC = "agent-docs/parallelism/data-parallelism.md#ep1-expert-sharding"


@pytest.mark.parametrize(
    ("shape", "knobs"),
    [
        pytest.param(
            {"ep_size": 2, "tp_size": 4, "world_size": 8, "gpus_per_node": 8, "ep_scope": "node"},
            ("expert_parallel_size (2)", "tensor_parallel_size (4)"),
            id="ep_not_a_multiple_of_tp",
        ),
        pytest.param(
            {"tp_size": 16, "world_size": 16, "gpus_per_node": 8},
            ("tensor_parallel_size (16) must divide the NVLink domain",),
            id="tp_straddles_domains",
        ),
        pytest.param(
            {"ep_size": 8, "tp_size": 4, "world_size": 16, "gpus_per_node": 8, "ep_scope": "node"},
            ("expert_parallel_size=8", "tensor_parallel_size=4"),
            id="multi_domain_multi_group_ep_tp",
        ),
        pytest.param(
            {"ep_size": 2, "expert_tp_size": 3, "world_size": 48, "gpus_per_node": 8, "ep_scope": "global"},
            ("expert_tensor_parallel_size (3) must divide the NVLink domain",),
            id="etp_does_not_divide_domain",
        ),
        pytest.param(
            {"ep_size": 4, "world_size": 8, "gpus_per_node": 8, "ep_scope": "node"},
            ("expert_parallel_size=4 on a single 8-GPU NVLink domain",),
            id="racy_single_domain_multi_group_ep",
        ),
        pytest.param(
            {"ep_size": 2, "world_size": 8, "gpus_per_node": 8, "fsdp_reshard_after_forward": True},
            ("expert_parallel_size=2", "expert_tensor_parallel_size=1"),
            id="zero3_with_expert_distribution",
        ),
        pytest.param(
            {"tp_size": 2, "world_size": 8, "gpus_per_node": 8, "fsdp_reshard_after_forward": True},
            ("tensor_parallel_size=2",),
            id="zero3_with_tp_and_dp",
        ),
    ],
)
def test_config_refusal_names_the_yaml_knobs(shape, knobs):
    with pytest.raises(ValueError) as err:
        make_parallelism_config(**shape)
    message = str(err.value)
    for knob in knobs:
        assert knob in message, message
    assert not INTERNAL_FIELD.search(message), message


def test_ep1_sharding_refusal_points_at_the_docs_instead_of_a_benchmark():
    """The trade-off figure lives in the owning doc; the refusal links it rather than restating a
    number that drifts from the table."""
    with pytest.raises(ValueError) as err:
        make_parallelism_config(fsdp_shard_ep1_experts=False, tp_size=2, world_size=8, gpus_per_node=8)
    message = str(err.value)
    assert EP1_SHARDING_DOC in message, message
    assert "%" not in message, message
    page, anchor = EP1_SHARDING_DOC.split("#")
    check_links = load_script_module("scripts/docs/check_links.py")
    assert anchor in check_links.parse_markdown((REPO_ROOT / page).read_text()).anchors


def test_stock_adamw_refusal_names_the_yaml_knobs():
    stub = SimpleNamespace(parallelism_config=SimpleNamespace(ep_group_size=8))
    with pytest.raises(ValueError, match="mixed torch.Tensor and DTensor") as err:
        DistributedTrainerMixin._refuse_stock_optimizer_on_mixed_params(stub, "adamw_torch_fused")
    message = str(err.value)
    assert "expert_parallel_size=1" in message, message
    assert not INTERNAL_FIELD.search(message), message


def test_accelerate_refusal_launch_lines_pass_the_config(monkeypatch):
    """The entry scripts take the YAML config as their positional, so a printed torchrun line
    without it fails before the run starts."""
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    message = accelerate_launch_rejection(make_parallelism_config(ep_size=2, world_size=2, gpus_per_node=2))
    script_lines = [line for line in message.splitlines() if "scripts/training/" in line]
    assert len(script_lines) == 3, message
    assert all("<config>" in line for line in script_lines), message


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
