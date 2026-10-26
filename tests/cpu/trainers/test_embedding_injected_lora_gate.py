#!/usr/bin/env python
"""The embedding trainer refuses in-place-injected LoRA under expert parallelism.

The save folds the adapters over the backbone's own parameters, which under EP or expert TP hold this rank's
expert shards under their local names, so the folded checkpoint would load nowhere.

Run: pytest tests/cpu/trainers/test_embedding_injected_lora_gate.py
"""

import ast
import inspect
import textwrap

import pytest

from halo.trainers.embedding.embedding_rules import reject_injected_lora_parallelism
from halo.trainers.embedding.trainer import EmbeddingTrainer
from tests.common.parallelism import make_parallelism_config


@pytest.mark.parametrize(
    "shape", [{"ep_size": 2}, {"ep_size": 1, "expert_tp_size": 2}, {"ep_size": 2, "expert_tp_size": 2}]
)
def test_injected_lora_is_refused_wherever_an_ep_group_holds_expert_shards(shape):
    with pytest.raises(ValueError, match="LoRA for embedding training is not supported with Expert Parallelism"):
        reject_injected_lora_parallelism(make_parallelism_config(**shape), has_injected_lora=True)


def test_injected_lora_passes_without_an_ep_group_and_ep_passes_without_it():
    reject_injected_lora_parallelism(make_parallelism_config(), has_injected_lora=True)
    reject_injected_lora_parallelism(make_parallelism_config(ep_size=2), has_injected_lora=False)


def test_the_gate_runs_in_the_trainers_init():
    source = textwrap.dedent(inspect.getsource(EmbeddingTrainer.__init__))
    called = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "reject_injected_lora_parallelism" in called


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
