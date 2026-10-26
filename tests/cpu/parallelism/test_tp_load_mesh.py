#!/usr/bin/env python
"""CPU test: an HF-native TP load shards over the toolkit's TP group, not a mesh transformers builds.

transformers ignores a ``device_mesh`` passed to ``from_pretrained`` and builds its own world-wide mesh,
so on a (dp, tp) layout every projection would come back sharded world-ways, on the wrong ranks, and
the toolkit's FSDP2 wrap over its dp mesh would run on those shards. Four gloo ranks load a tiny Qwen3
at tp=2, dp=2 through the loader's mesh hook and check each shard's size and mesh.

    python tests/cpu/parallelism/test_tp_load_mesh.py
"""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import DistributedConfig, Qwen3Config, Qwen3ForCausalLM

from src.model_loading.parallel import tp as tp_loading
from tests.common.gloo import run_gloo_ranks

WORLD, TP, DP = 4, 2, 2
HIDDEN = 32


def _toolkit_mesh_worker(rank: int, checkpoint: str) -> None:
    mesh = init_device_mesh("cpu", (DP, TP), mesh_dim_names=("dp", "tp"))
    with tp_loading._toolkit_tp_mesh(mesh, mesh["tp"]):
        model = Qwen3ForCausalLM.from_pretrained(
            checkpoint, distributed_config=DistributedConfig(tp_plan="auto", tp_size=TP), dtype=torch.float32
        )
    weight = model.model.layers[0].self_attn.q_proj.weight
    assert isinstance(weight, DTensor) and weight.device_mesh == mesh["tp"], weight.device_mesh
    assert tuple(weight.to_local().shape) == (HIDDEN // TP, HIDDEN), weight.to_local().shape
    assert "prepare_distribute_model" not in tp_loading.PreTrainedModel.__dict__, "the hook outlived the load"


def _transformers_mesh_worker(rank: int, checkpoint: str) -> None:
    mesh = init_device_mesh("cpu", (DP, TP), mesh_dim_names=("dp", "tp"))
    model = Qwen3ForCausalLM.from_pretrained(
        checkpoint, distributed_config=DistributedConfig(tp_plan="auto"), device_mesh=mesh["tp"], dtype=torch.float32
    )
    weight = model.model.layers[0].self_attn.q_proj.weight
    assert tuple(weight.to_local().shape) == (HIDDEN // WORLD, HIDDEN), weight.to_local().shape


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    path = tmp_path_factory.mktemp("qwen3")
    Qwen3ForCausalLM(config).save_pretrained(path)
    return str(path)


def test_the_load_shards_over_the_toolkit_tp_group(checkpoint):
    run_gloo_ranks(_toolkit_mesh_worker, WORLD, checkpoint)


def test_premise_transformers_alone_shards_over_a_mesh_of_its_own(checkpoint):
    """Anti-vacuity: without the hook, a passed tp mesh is ignored and the load shards world-ways."""
    run_gloo_ranks(_transformers_mesh_worker, WORLD, checkpoint)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
