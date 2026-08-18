#!/usr/bin/env python
"""``tp_plan_shards_params`` must answer exactly what HF's style objects do, and never guess "no".

The dense TP loader refuses a run whose plan shards nothing (``tp_plan="auto"`` resolves to an EMPTY
plan for an architecture shipping no ``base_model_tp_plan``, and the run silently becomes ``tp_size``
full replicas at ``1/tp_size`` throughput). The signal is this predicate over every parameter name,
so both lying directions waste an 8-GPU run or ban a working model:

* toward "shards nothing" — a real TP run is refused at load;
* toward "shards" on an activation-transform style (``moe_tp_experts``, ``mla_kv_a_proj``,
  ``sequence_parallel`` — named by shipped MoE plans like glm5_next's) — a plan that leaves every
  parameter whole passes the guard and the run trains as replicas anyway.

The oracle is transformers' own ``shard_param`` run per style on a fake process group, where
``distribute_tensor(..., src_data_rank=None)`` slices locally with no collective, so upstream drift
(a new style, a changed placement) fails here instead of passing silently.

    python tests/cpu/parallelism/test_tp_plan_sharding_styles.py
"""

from __future__ import annotations

import pytest
import torch.nn as nn
from torch.distributed.tensor import DTensor
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES, _get_parameter_tp_plan
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.step3p7.configuration_step3p7 import Step3p7TextConfig

from src.distributed.tensor_parallel.state_dict import reject_plan_sharded_plain_params, tp_plan_shards_params
from tests.common.distributed import fake_process_group_mesh

TP_SIZE = 4


def _places_a_shard(style_name: str, module: nn.Module, pname: str) -> bool:
    """Whether ``shard_param`` gives ``module.<pname>`` a disjoint shard, via transformers' own code.

    A style leaving the base no-op ``shard_param`` in place returns the parameter untouched; one that
    shards places a DTensor whose placement carries a ``dim`` (``Shard``/``_StridedShard``, the
    latter not a ``Shard`` subclass in torch 2.11). A ``Replicate`` DTensor is not a shard.
    """
    with fake_process_group_mesh(rank=0, world_size=TP_SIZE) as mesh:
        ALL_PARALLEL_STYLES[style_name].shard_param(module, pname, mesh)
        param = module._parameters[pname]
        return isinstance(param.data, DTensor) and hasattr(param.data.placements[0], "dim")


@pytest.mark.parametrize("style_name", sorted(ALL_PARALLEL_STYLES.keys()))
def test_every_registered_style_matches_shard_param(style_name):
    """The completeness pin: for EVERY registered style, our answer for a weight equals what the
    style object itself does to one. A style added upstream lands here automatically."""
    module = nn.Embedding(8, 12) if style_name == "embedding_rowwise" else nn.Linear(12, 8)
    expected = _places_a_shard(style_name, module, "weight")
    assert tp_plan_shards_params("model.layers.0.blk.weight", {"model.layers.*.blk": style_name}) is expected


def test_the_registry_holds_both_answers():
    """Anti-vacuity for the sweep above: a predicate stuck at True (or at False) would pass every
    parametrization if the registry were uniform. Shipped plans name styles of both kinds."""
    verdicts = {name: tp_plan_shards_params("blk.weight", {"blk": name}) for name in ALL_PARALLEL_STYLES}
    assert verdicts["colwise"] and verdicts["rowwise"] and verdicts["packed_colwise"]
    assert not verdicts["moe_tp_experts"] and not verdicts["mla_kv_a_proj"] and not verdicts["sequence_parallel"]


def test_a_sharding_style_shards_its_weight_even_where_it_replicates_the_bias():
    """The guard asks per model, so the load-bearing half is the weight: a row-parallel bias is
    replicated (added once after the row-reduce) inside a style that does shard."""
    plan = {"model.layers.*.self_attn.o_proj": "rowwise"}
    assert tp_plan_shards_params("model.layers.7.self_attn.o_proj.weight", plan)
    assert not _places_a_shard("rowwise", nn.Linear(12, 8), "bias")


def test_a_key_the_plan_does_not_cover_shards_nothing():
    plan = {"model.layers.*.self_attn.q_proj": "colwise"}
    assert not tp_plan_shards_params("model.embed_tokens.weight", plan)
    assert not tp_plan_shards_params("lm_head.weight", plan)


@pytest.mark.parametrize("name", ["model.layers.0.mlp.fc1", "model.layers.3.mlp.dense_h_to_4h"])
def test_embedded_digit_names_are_reported_as_sharded(name):
    """``fc1`` / ``dense_h_to_4h`` carry a digit that is not a layer index; HF shards them, so must
    we — a ``re.sub(r"\\d+", "*")`` wildcard would report them replicated. The lookup is
    transformers' own ``_get_parameter_tp_plan``, which is what makes that hold."""
    plan = {"model.layers.*.mlp.fc1": "colwise", "model.layers.*.mlp.dense_h_to_4h": "colwise"}
    assert tp_plan_shards_params(f"{name}.weight", plan)
    assert _get_parameter_tp_plan(f"{name}.weight", plan) == "colwise"


def test_an_unregistered_style_raises_instead_of_answering_no():
    """``embedding_colwise`` was a real 5.14 style the registry dropped — the poster child for a
    plan entry the running transformers cannot shard (its own ``_validate_tp_plan_styles`` refuses
    it too). Answering "shards nothing" would report the load as a wasted-replica run."""
    with pytest.raises(ValueError, match="embedding_colwise"):
        tp_plan_shards_params("model.embed_tokens.weight", {"model.embed_tokens": "embedding_colwise"})


def test_shipped_moe_plans_resolve_end_to_end():
    """Real shipped plans name activation-transform styles no static table ever covered; every entry
    must resolve — to sharded or not — or TP loads of that architecture die in the lookup."""
    for config_cls in (Glm5NextTextConfig, Qwen3MoeConfig, Step3p7TextConfig):
        plan = config_cls().base_model_tp_plan
        assert plan, config_cls.__name__
        answers = {pattern: tp_plan_shards_params(f"{pattern.replace('*', '0')}.weight", plan) for pattern in plan}
        assert any(answers.values()), (config_cls.__name__, "a shipped plan that shards nothing")


class _PlannedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk = nn.Linear(12, 8)


def test_plan_sharded_plain_param_rejected():
    """A param the applied plan shards that materialized PLAIN is a bare slice the gradient sync
    would average across the TP group — the load must refuse it, naming the param."""
    model = _PlannedBlock()
    with pytest.raises(ValueError, match="blk.weight"):
        reject_plan_sharded_plain_params(model, {"blk": "colwise"})


def test_plan_sharded_dtensor_param_accepted():
    """The healthy load: every plan-sharded param materialized as a DTensor passes the guard."""
    model = _PlannedBlock()
    with fake_process_group_mesh(rank=0, world_size=TP_SIZE) as mesh:
        ALL_PARALLEL_STYLES["colwise"].shard_param(model.blk, "weight", mesh)
        ALL_PARALLEL_STYLES["colwise"].shard_param(model.blk, "bias", mesh)
        reject_plan_sharded_plain_params(model, {"blk": "colwise"})


def test_hand_sliced_registry_exempts_plain_param():
    """Params the toolkit slices by hand (GptOss sinks) are plain on purpose and recorded in
    ``_tp_sharded_non_dtensor`` — the guard must not re-flag them."""
    model = _PlannedBlock()
    model._tp_sharded_non_dtensor = (("blk.weight", 0), ("blk.bias", 0))
    reject_plan_sharded_plain_params(model, {"blk": "colwise"})


def test_uncovered_and_transform_only_plans_pass_plain_params():
    """A plan that covers nothing, or only with activation-transform styles, shards no param — a
    plain tensor is then the correct materialization, not a regression."""
    model = _PlannedBlock()
    reject_plan_sharded_plain_params(model, {})
    reject_plan_sharded_plain_params(model, {"blk": "sequence_parallel"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
