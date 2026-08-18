#!/usr/bin/env python
"""Stage-aware loading: the checkpoint-key map a mis-rebase would corrupt in total silence.

The stage-aware loader claims a subset of the checkpoint's keys and RE-BASES them onto its stage's
layer numbering (``layers.5`` on disk becomes ``layers.1`` on the stage that starts at layer 4).
Every decoder layer of a model shares its shapes, so an off-by-``lo`` rebase loads layers
``0..k`` into every stage: no crash, no missing key, a plausible loss curve, and a model that is
silently the first stage's layers repeated. Nothing downstream can detect it.

Two properties pin it down:

  * ``local_parameter_name(global_parameter_name(k)) == k`` for every key of every stage — the
    naming map and its inverse are one rebase helper and must stay mutual inverses.
  * The stages' claimed DISK keys partition the checkpoint exactly: every decoder-layer key is
    claimed by exactly one stage, and what a stage claims is precisely what it later re-exports
    under global names. Non-layer keys (embeddings, final norm, head) are claimed by EVERY stage on
    purpose — they are O(vocab x hidden) and ``build_pipeline_stage`` drops the unowned ones.

Run: python tests/cpu/parallelism/test_pp_lazy_key_map.py
"""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3ForSequenceClassification

from src.distributed.pipeline_parallel.lazy_loader import PPWeightPlanner
from src.distributed.pipeline_parallel.split import resolve_layer_partition
from src.distributed.pipeline_parallel.stage import build_pipeline_stage, slice_backbone_to_stage
from tests.common.models import TINY_QWEN3_CONFIG

# 8 tiny layers: pp2 is the uniform split (as pp4 would be), pp3 the uneven one.
PP_SIZES = (2, 3)
LAYER_ROOT = "model.layers."


def _model(cls=Qwen3ForCausalLM, **overrides):
    torch.manual_seed(0)
    config = Qwen3Config(**{**TINY_QWEN3_CONFIG, **overrides})
    return cls(config)


def _checkpoint_keys(cls=Qwen3ForCausalLM, **overrides) -> list[str]:
    """The key set a real safetensors checkpoint of this model carries."""
    return sorted(_model(cls, **overrides).state_dict())


def _stages(pp_size: int, pp_split=None, cls=Qwen3ForCausalLM):
    """One built ``PipelineStageModule`` per stage, each from its own copy of the model."""
    return [build_pipeline_stage(_model(cls), rank, pp_size, pp_split=pp_split) for rank in range(pp_size)]


def _planners(pp_size: int, pp_split=None) -> list[PPWeightPlanner]:
    partition = resolve_layer_partition(_model(), pp_size, pp_split)
    return [PPWeightPlanner(lo, hi, LAYER_ROOT) for lo, hi in partition]


# The rebase helper itself


def test_stage_key_shifts_only_the_layer_index():
    planner = PPWeightPlanner(4, 8, LAYER_ROOT)
    assert planner.stage_key("model.layers.5.mlp.down_proj.weight") == "model.layers.1.mlp.down_proj.weight"
    assert planner.stage_key("model.layers.4.input_layernorm.weight") == "model.layers.0.input_layernorm.weight"
    # Non-layer keys are replicated verbatim on every stage.
    assert planner.stage_key("model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert planner.stage_key("lm_head.weight") == "lm_head.weight"
    # A double-digit index must not be truncated by the single-character partition.
    assert PPWeightPlanner(10, 20, LAYER_ROOT).stage_key("model.layers.13.q.weight") == "model.layers.3.q.weight"


def test_owns_is_half_open_on_the_layer_range():
    planner = PPWeightPlanner(4, 8, LAYER_ROOT)
    assert not planner.owns("model.layers.3.q.weight")
    assert planner.owns("model.layers.4.q.weight")
    assert planner.owns("model.layers.7.q.weight")
    assert not planner.owns("model.layers.8.q.weight")
    assert planner.owns("model.norm.weight")


# global_parameter_name <-> local_parameter_name


@pytest.mark.parametrize("pp_size", PP_SIZES)
def test_local_name_inverts_global_name_for_every_stage_key(pp_size):
    for stage in _stages(pp_size):
        keys = list(stage.state_dict()) + [name for name, _ in stage.named_modules() if name]
        assert keys
        for local in keys:
            assert stage.local_parameter_name(stage.global_parameter_name(local)) == local, (local, pp_size)


def test_local_name_inverts_global_name_for_a_non_lm_head():
    """``score`` (sequence classification) exercises the head branch of the inverse."""
    for stage in _stages(2, cls=Qwen3ForSequenceClassification):
        for local in stage.state_dict():
            assert stage.local_parameter_name(stage.global_parameter_name(local)) == local, local


def test_local_name_inverts_global_name_under_a_manual_split():
    for stage in _stages(2, pp_split=[5, 3]):
        for local in stage.state_dict():
            assert stage.local_parameter_name(stage.global_parameter_name(local)) == local, local


# The stages' claimed disk keys partition the checkpoint


@pytest.mark.parametrize("pp_size", PP_SIZES)
def test_claimed_layer_keys_partition_the_checkpoint_exactly(pp_size):
    keys = _checkpoint_keys()
    layer_keys = {k for k in keys if k.startswith(LAYER_ROOT)}
    assert layer_keys, "the tiny model must have decoder-layer keys for this test to mean anything"

    claimed = [{k for k in layer_keys if planner.owns(k)} for planner in _planners(pp_size)]

    union = set().union(*claimed)
    assert union == layer_keys, sorted(layer_keys - union)
    total = sum(len(c) for c in claimed)
    assert total == len(layer_keys), f"{total - len(layer_keys)} layer keys claimed by more than one stage"
    assert all(claimed), "a stage claimed no decoder layer"


@pytest.mark.parametrize("pp_size", PP_SIZES)
def test_non_layer_keys_are_claimed_by_every_stage(pp_size):
    keys = _checkpoint_keys()
    non_layer = {k for k in keys if not k.startswith(LAYER_ROOT)}
    assert non_layer
    for planner in _planners(pp_size):
        assert {k for k in non_layer if planner.owns(k)} == non_layer


@pytest.mark.parametrize("pp_size", PP_SIZES)
def test_claimed_keys_equal_what_the_stage_re_exports_globally(pp_size):
    """The end-to-end property: what the loader reads under a disk key is what the stage saves it as.

    Compares the loader's ownership filter against the stage module's own ``global_parameter_name``
    — the two halves of the round trip that a wrong ``lo`` would break in opposite directions and
    that no shape check can catch.
    """
    layer_keys = {k for k in _checkpoint_keys() if k.startswith(LAYER_ROOT)}
    stages = _stages(pp_size)
    for planner, stage in zip(_planners(pp_size), stages, strict=True):
        claimed = {k for k in layer_keys if planner.owns(k)}
        exported = {
            stage.global_parameter_name(local)
            for local in stage.state_dict()
            if local.startswith(stage.local_layer_root)
        }
        assert exported == claimed, sorted(exported ^ claimed)


@pytest.mark.parametrize("pp_size", PP_SIZES)
def test_rebased_keys_land_on_the_sliced_model(pp_size):
    """Every claimed disk key, re-based, must name a real parameter of the sliced backbone."""
    keys = _checkpoint_keys()
    for planner in _planners(pp_size):
        sliced = _model()
        slice_backbone_to_stage(sliced, planner.lo, planner.hi)
        available = set(sliced.state_dict())
        rebased = {planner.stage_key(k) for k in keys if planner.owns(k)}
        assert rebased <= available, sorted(rebased - available)
        assert {k for k in available if k.startswith(LAYER_ROOT)} <= rebased


def test_manual_split_partitions_the_checkpoint_exactly():
    layer_keys = {k for k in _checkpoint_keys() if k.startswith(LAYER_ROOT)}
    planners = [PPWeightPlanner(lo, hi, LAYER_ROOT) for lo, hi in resolve_layer_partition(_model(), 2, [5, 3])]
    claimed = [{k for k in layer_keys if planner.owns(k)} for planner in planners]
    assert set().union(*claimed) == layer_keys
    assert sum(len(c) for c in claimed) == len(layer_keys)
    # The split is honored, not silently uniformed: 5 layers' keys on stage 0, 3 on stage 1.
    per_layer = len(layer_keys) // 8
    assert len(claimed[0]) == 5 * per_layer and len(claimed[1]) == 3 * per_layer


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
