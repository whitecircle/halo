#!/usr/bin/env python
"""The embedding trainer refuses in-place-injected LoRA under expert parallelism and on the EP layers.
The save folds adapters over rank-local expert shards or the layers' own layout, so the checkpoint loads nowhere."""

import ast
import inspect
import textwrap

import pytest

from halo.trainers.embedding.embedding_rules import reject_injected_lora_parallelism
from halo.trainers.embedding.trainer import EmbeddingTrainer
from tests.common.parallelism import make_parallelism_config

REFUSAL = "LoRA for embedding training is not supported with Expert Parallelism"


@pytest.mark.parametrize(
    "shape", [{"ep_size": 2}, {"ep_size": 1, "expert_tp_size": 2}, {"ep_size": 2, "expert_tp_size": 2}]
)
def test_injected_lora_is_refused_wherever_an_ep_group_holds_expert_shards(shape):
    with pytest.raises(ValueError, match=REFUSAL):
        reject_injected_lora_parallelism(make_parallelism_config(**shape), has_injected_lora=True, has_ep_layers=True)


def test_injected_lora_is_refused_on_the_grouped_gemm_ep_layers_without_an_ep_group():
    config = make_parallelism_config(use_grouped_gemm=True)
    assert not config.has_ep_group, "premise: expert_parallel_size 1 forms no EP group"

    with pytest.raises(ValueError, match="use_grouped_gemm: false"):
        reject_injected_lora_parallelism(config, has_injected_lora=True, has_ep_layers=True)


def test_injected_lora_passes_on_stock_experts_and_ep_passes_without_it():
    reject_injected_lora_parallelism(make_parallelism_config(), has_injected_lora=True, has_ep_layers=False)
    reject_injected_lora_parallelism(
        make_parallelism_config(use_grouped_gemm=False), has_injected_lora=True, has_ep_layers=False
    )
    reject_injected_lora_parallelism(make_parallelism_config(ep_size=2), has_injected_lora=False, has_ep_layers=True)


def test_the_gate_runs_in_the_trainers_init_on_the_models_own_ep_layers():
    """``use_grouped_gemm`` alone leaves dense or stock-expert MoEs without EP layers, and the fold uses hub names."""
    source = textwrap.dedent(inspect.getsource(EmbeddingTrainer.__init__))
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reject_injected_lora_parallelism"
    ]
    assert len(calls) == 1
    (probe,) = [ast.unparse(keyword.value) for keyword in calls[0].keywords if keyword.arg == "has_ep_layers"]
    assert "self._ep_layers()" in probe, probe


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
