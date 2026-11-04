#!/usr/bin/env python
"""``plan_shards_param`` answers exactly what HF's style objects do, with transformers' own ``shard_param`` as the
oracle. Lying either way wastes an 8-GPU run or bans a working model."""

from __future__ import annotations

import pytest
import torch.nn as nn
from torch.distributed.tensor import DTensor
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES, _get_parameter_plan
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.step3p7.configuration_step3p7 import Step3p7TextConfig

from halo.distributed.tensor_parallel.hf_plan import plan_shards_param, reject_plan_sharded_plain_params
from halo.distributed.tensor_parallel.plain_params import register_hand_sliced
from tests.common.distributed import fake_process_group_mesh

TP_SIZE = 4


def _places_a_shard(style_name: str, module: nn.Module, pname: str) -> bool:
    """``_StridedShard`` is not a ``Shard`` subclass in torch 2.11, so placement ``dim`` is read; a ``Replicate``
    DTensor is no shard."""
    with fake_process_group_mesh(rank=0, world_size=TP_SIZE) as mesh:
        ALL_PARALLEL_STYLES[style_name].shard_param(module, pname, mesh)
        param = module._parameters[pname]
        return isinstance(param.data, DTensor) and hasattr(param.data.placements[0], "dim")


@pytest.mark.parametrize("style_name", sorted(ALL_PARALLEL_STYLES.keys()))
def test_every_registered_style_matches_shard_param(style_name):
    """A style added upstream lands here automatically."""
    module = nn.Embedding(8, 12) if style_name == "embedding_rowwise" else nn.Linear(12, 8)
    expected = _places_a_shard(style_name, module, "weight")
    assert plan_shards_param("model.layers.0.blk.weight", {"model.layers.*.blk": style_name}) is expected


def test_the_registry_holds_both_answers():
    """Anti-vacuity: a predicate stuck at one answer would pass if the registry were uniform."""
    verdicts = {name: plan_shards_param("blk.weight", {"blk": name}) for name in ALL_PARALLEL_STYLES}
    assert verdicts["colwise"] and verdicts["rowwise"] and verdicts["packed_colwise"]
    assert not verdicts["moe_tp_experts"] and not verdicts["mla_kv_a_proj"] and not verdicts["sequence_parallel"]


def test_a_sharding_style_shards_its_weight_even_where_it_replicates_the_bias():
    """A row-parallel bias is replicated inside a style that does shard."""
    plan = {"model.layers.*.self_attn.o_proj": "rowwise"}
    assert plan_shards_param("model.layers.7.self_attn.o_proj.weight", plan)
    assert not _places_a_shard("rowwise", nn.Linear(12, 8), "bias")


def test_a_key_the_plan_does_not_cover_shards_nothing():
    plan = {"model.layers.*.self_attn.q_proj": "colwise"}
    assert not plan_shards_param("model.embed_tokens.weight", plan)
    assert not plan_shards_param("lm_head.weight", plan)


@pytest.mark.parametrize("name", ["model.layers.0.mlp.fc1", "model.layers.3.mlp.dense_h_to_4h"])
def test_embedded_digit_names_are_reported_as_sharded(name):
    """``fc1`` / ``dense_h_to_4h`` carry a digit that is no layer index; a digit wildcard would report them
    replicated."""
    plan = {"model.layers.*.mlp.fc1": "colwise", "model.layers.*.mlp.dense_h_to_4h": "colwise"}
    assert plan_shards_param(f"{name}.weight", plan)
    assert _get_parameter_plan(f"{name}.weight", plan) == "colwise"


def test_an_unregistered_style_raises_instead_of_answering_no():
    """Answering that it shards nothing would report the load as a wasted-replica run."""
    with pytest.raises(ValueError, match="embedding_colwise"):
        plan_shards_param("model.embed_tokens.weight", {"model.embed_tokens": "embedding_colwise"})


def test_shipped_moe_plans_resolve_end_to_end():
    """Every entry must resolve or TP loads of that architecture die in the lookup."""
    for config_cls in (Glm5NextTextConfig, Qwen3MoeConfig, Step3p7TextConfig):
        plan = config_cls().base_model_tp_plan
        assert plan, config_cls.__name__
        answers = {pattern: plan_shards_param(f"{pattern.replace('*', '0')}.weight", plan) for pattern in plan}
        assert any(answers.values()), (config_cls.__name__, "a shipped plan that shards nothing")


class _PlannedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk = nn.Linear(12, 8)


def test_plan_sharded_plain_param_rejected():
    """A bare slice the gradient sync would average across the TP group."""
    model = _PlannedBlock()
    with pytest.raises(ValueError, match="blk.weight"):
        reject_plan_sharded_plain_params(model, {"blk": "colwise"})


def test_plan_sharded_dtensor_param_accepted():
    model = _PlannedBlock()
    with fake_process_group_mesh(rank=0, world_size=TP_SIZE) as mesh:
        ALL_PARALLEL_STYLES["colwise"].shard_param(model.blk, "weight", mesh)
        ALL_PARALLEL_STYLES["colwise"].shard_param(model.blk, "bias", mesh)
        reject_plan_sharded_plain_params(model, {"blk": "colwise"})


def test_hand_sliced_registry_exempts_plain_param():
    """Params the toolkit slices by hand (GptOss sinks) are plain on purpose."""
    model = _PlannedBlock()
    register_hand_sliced(model, "blk.weight", 0)
    register_hand_sliced(model, "blk.bias", 0)
    reject_plan_sharded_plain_params(model, {"blk": "colwise"})


def test_uncovered_and_transform_only_plans_pass_plain_params():
    """A plain tensor is then the correct materialization."""
    model = _PlannedBlock()
    reject_plan_sharded_plain_params(model, {})
    reject_plan_sharded_plain_params(model, {"blk": "sequence_parallel"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
