#!/usr/bin/env python
"""Parallelism refusals name the knobs a user sets and print launch lines that run as written.
``ep_size`` etc. are dataclass fields no YAML accepts; ``ep_group_size`` is derived, so the messages may keep it."""

import re
from types import SimpleNamespace

import pytest

from halo.distributed.context_parallel.sequence_chunks import cp_chunk_bounds
from halo.distributed.parallelism_rules import reject_custom_parallelism_under_accelerate
from halo.distributed.tensor_parallel.parallelize_attention import validate_tp_head_divisibility
from halo.model_loading.parallel.load_rules import reject_unsupported_model
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.moe.expert_lora_spec import reject_expert_lora_with_expert_tp
from halo.trainers.trainer_rules import reject_stock_adamw_on_mixed_params
from tests.common.parallelism import make_parallelism_config

INTERNAL_FIELD = re.compile(
    r"\b(ep_size|tp_size|expert_tp_size|cp_size|pp_size|pp_split|pp_schedule|pp_microbatches)\b|\b(CP|TP) size\b"
)
EP1_SHARDING_DOC = "agent-docs/parallelism/data-parallelism.md#ep1-expert-sharding"


class _ModelConfig:
    """Only the fields a gate reads."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _assert_names_knobs(message: str, knobs) -> None:
    for knob in knobs:
        assert knob in message, message
    assert not INTERNAL_FIELD.search(message), message


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
        pytest.param({"ep_size": 0}, ("expert_parallel_size must be >= 1",), id="size_below_one"),
        pytest.param(
            {"cp_size": 16, "world_size": 8, "gpus_per_node": 8},
            ("context_parallel_size (16) cannot exceed the NVLink domain",),
            id="cp_wider_than_domain",
        ),
        pytest.param(
            {"tp_size": 16, "world_size": 8, "gpus_per_node": 8},
            ("tensor_parallel_size (16) cannot exceed world size",),
            id="tp_wider_than_world",
        ),
        pytest.param(
            {"tp_size": 2, "cp_size": 2, "world_size": 8, "gpus_per_node": 8},
            ("tensor_parallel_size == context_parallel_size",),
            id="unsupported_axis_set",
        ),
        pytest.param(
            {"ep_size": 2, "cp_size": 2, "world_size": 8, "gpus_per_node": 8, "ep_scope": "node"},
            ("expert_parallel_size=2 * expert_tensor_parallel_size=1",),
            id="node_local_ep_cp",
        ),
        pytest.param(
            {"ep_size": 3, "world_size": 8, "gpus_per_node": 8, "ep_scope": "global"},
            ("expert_parallel_size(3) * expert_tensor_parallel_size(1)",),
            id="ep_group_does_not_tile",
        ),
        pytest.param(
            {"tp_size": 2, "use_hsdp": True, "world_size": 8, "gpus_per_node": 8},
            ("tensor_parallel_size=2", "expert_tensor_parallel_size=1"),
            id="hsdp_with_tp",
        ),
        pytest.param(
            {"ep_size": 2, "use_hsdp": True, "world_size": 8, "gpus_per_node": 8},
            ("expert_parallel_size=2",),
            id="hsdp_with_ep",
        ),
        pytest.param({"pp_schedule": "nonsense"}, ("pipeline_schedule must be one of",), id="pp_schedule"),
        pytest.param(
            {"pp_split": [1, 1], "world_size": 8, "gpus_per_node": 8},
            ("pipeline_split=[1, 1] is only meaningful with pipeline_parallel_size > 1",),
            id="pp_knob_without_pp",
        ),
        pytest.param(
            {"pp_size": 3, "world_size": 8, "gpus_per_node": 8},
            ("divisible by pipeline_parallel_size (3)",),
            id="pp_does_not_divide_world",
        ),
        pytest.param(
            {"pp_size": 2, "pp_split": [1], "world_size": 16, "gpus_per_node": 8},
            ("pipeline_split has 1 entries but pipeline_parallel_size=2",),
            id="pp_split_length",
        ),
    ],
)
def test_config_refusal_names_the_yaml_knobs(shape, knobs):
    with pytest.raises(ValueError) as err:
        make_parallelism_config(**shape)
    _assert_names_knobs(str(err.value), knobs)


@pytest.mark.parametrize(
    ("shape", "model_config", "knob"),
    [
        pytest.param(
            {"ep_size": 2},
            _ModelConfig(num_experts=3),
            "expert_parallel_size values dividing 3",
            id="experts_not_divisible",
        ),
        pytest.param(
            {"expert_tp_size": 2},
            _ModelConfig(num_experts=8, moe_intermediate_size=7),
            "Use an expert_tensor_parallel_size dividing 7",
            id="expert_ffn_not_divisible",
        ),
    ],
)
def test_model_shape_refusal_names_the_yaml_knobs(shape, model_config, knob):
    config = make_parallelism_config(world_size=2, gpus_per_node=2, **shape)
    with pytest.raises(ValueError) as err:
        reject_unsupported_model(config, model_config, sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=False)
    _assert_names_knobs(str(err.value), (knob,))


@pytest.mark.parametrize(
    ("refuse", "knob"),
    [
        pytest.param(
            lambda: make_parallelism_config(world_size=16, gpus_per_node=8, cp_size=16),
            "context_parallel_size (16) cannot exceed the NVLink domain (8)",
            id="cp_locality",
        ),
        pytest.param(
            lambda: cp_chunk_bounds(7, 0, 2), "divisible by context_parallel_size 2", id="cp_sequence_length"
        ),
        pytest.param(
            lambda: validate_tp_head_divisibility(_ModelConfig(num_attention_heads=6, num_key_value_heads=6), 4),
            "divisible by tensor_parallel_size (4)",
            id="tp_heads",
        ),
        pytest.param(
            reject_expert_lora_with_expert_tp,
            "Expert LoRA is not supported with expert_tensor_parallel_size > 1",
            id="expert_lora_under_etp",
        ),
    ],
)
def test_runtime_refusal_names_the_yaml_knobs(refuse, knob):
    with pytest.raises(ValueError) as err:
        refuse()
    _assert_names_knobs(str(err.value), (knob,))


def test_ep1_sharding_refusal_points_at_the_docs_instead_of_a_benchmark():
    """The figure lives in the owning doc rather than a number that drifts."""
    with pytest.raises(ValueError) as err:
        make_parallelism_config(fsdp_shard_ep1_experts=False, tp_size=2, world_size=8, gpus_per_node=8)
    message = str(err.value)
    assert EP1_SHARDING_DOC in message, message
    assert "%" not in message, message


def test_stock_adamw_refusal_names_the_yaml_knobs():
    with pytest.raises(ValueError, match="mixed torch.Tensor and DTensor") as err:
        reject_stock_adamw_on_mixed_params("adamw_torch_fused", SimpleNamespace(ep_group_size=8), None)
    _assert_names_knobs(str(err.value), ("expert_parallel_size=1",))


def test_accelerate_refusal_launch_lines_pass_the_config(monkeypatch):
    """The entry scripts take the YAML config as their positional."""
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    with pytest.raises(ValueError) as err:
        reject_custom_parallelism_under_accelerate(make_parallelism_config(ep_size=2, world_size=2, gpus_per_node=2))
    message = str(err.value)
    script_lines = [line for line in message.splitlines() if "scripts/training/" in line]
    assert len(script_lines) == 3, message
    assert all("<config>" in line for line in script_lines), message


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
