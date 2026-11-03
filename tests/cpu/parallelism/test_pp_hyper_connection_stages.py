#!/usr/bin/env python
"""Hyper-connection pipeline stages (DeepSeek-V4, GLM-5): the ``hc_mult``-widened boundary stream, gates and ownership.
Chained stages must equal the unsplit model bitwise, since any lossy boundary fails that."""

import pytest
import torch
import torch.nn as nn
from transformers import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
)

from halo.distributed.pipeline_parallel.families import PP_SPEC_MAP
from halo.distributed.pipeline_parallel.model_rules import reject_unsupported_pp_model
from halo.distributed.pipeline_parallel.partition import validate_partition
from halo.distributed.pipeline_parallel.stage import StageLayout, build_pipeline_stage
from tests.common.models import TINY_DSV4_CONFIG, TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG

# Keeps the metadata-only ``num_nextn_predict_layers: 1`` hub checkpoints declare: the MTP gate must accept the
# never-built tail.
_DSV4_CONFIG = dict(TINY_DSV4_CONFIG)

# A KDA and a DSA layer on each stage of the [2, 2] split.
_GLM5_TEXT_CONFIG = {**TINY_GLM5_CONFIG, "layer_types": ["linear_attention", "indexed_attention"] * 2}

_BATCH, _SEQ = 2, 16


def _dsv4_model(**overrides):
    torch.manual_seed(7)
    return DeepseekV4ForCausalLM(DeepseekV4Config(**{**_DSV4_CONFIG, **overrides}))


def _glm5_model(**text_overrides):
    torch.manual_seed(11)
    config = Glm5NextConfig(
        text_config={**_GLM5_TEXT_CONFIG, **text_overrides}, vision_config=dict(TINY_GLM5_VISION_CONFIG)
    )
    return Glm5NextForConditionalGeneration(config)


def _input_ids(vocab_size):
    g = torch.Generator().manual_seed(5)
    return torch.randint(0, vocab_size, (_BATCH, _SEQ), generator=g)


def _chained_stage_logits(build_model, pp_split, input_ids):
    """Returns (boundary stream, logits)."""
    stages = [
        build_pipeline_stage(build_model(), rank, len(pp_split), moe_balancing="none", pp_split=pp_split)
        for rank in range(len(pp_split))
    ]
    activation = input_ids
    boundary = None
    for stage in stages:
        activation = stage(activation)
        if not stage.is_last:
            boundary = activation
    return boundary, activation


@pytest.mark.parametrize(
    ("build_model", "pp_split", "hc_mult"),
    [
        pytest.param(_dsv4_model, [2, 1], _DSV4_CONFIG["hc_mult"], id="deepseek_v4"),
        pytest.param(_glm5_model, [2, 2], TINY_GLM5_CONFIG["hc_mult"], id="glm5_next"),
    ],
)
def test_chained_stages_reproduce_the_unsplit_model_bitwise(build_model, pp_split, hc_mult):
    """The split reorders no arithmetic, so any drift means the boundary lost state."""
    model = build_model()
    input_ids = _input_ids(model.config.get_text_config().vocab_size)
    with torch.no_grad():
        reference = model(input_ids=input_ids).logits
        boundary, logits = _chained_stage_logits(build_model, pp_split, input_ids)

    hidden = model.config.get_text_config().hidden_size
    assert boundary.shape == (_BATCH, _SEQ, hc_mult, hidden), tuple(boundary.shape)
    assert torch.equal(logits, reference), (logits - reference).abs().max().item()


@pytest.mark.parametrize(
    ("build_model", "pp_split"),
    [pytest.param(_dsv4_model, [2, 1], id="deepseek_v4"), pytest.param(_glm5_model, [2, 2], id="glm5_next")],
)
def test_gradients_flow_back_through_the_boundary_stream(build_model, pp_split):
    """A detached stream trains the first stage on zero gradients with nothing raised."""
    model = build_model()
    input_ids = _input_ids(model.config.get_text_config().vocab_size)
    stages = [
        build_pipeline_stage(build_model(), rank, len(pp_split), moe_balancing="none", pp_split=pp_split)
        for rank in range(len(pp_split))
    ]
    logits = stages[1](stages[0](input_ids))
    logits.float().pow(2).mean().backward()

    embed_grad = stages[0].model.embed_tokens.weight.grad if hasattr(stages[0].model, "embed_tokens") else None
    grads = [p.grad for p in stages[0].parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "no gradient reached stage 0"
    assert embed_grad is None or embed_grad.abs().sum() > 0


def test_hc_head_is_last_stage_state_and_identity_elsewhere():
    """An un-neutralized copy collapses the stream mid-network; a dropped one loses trained state."""
    s0 = build_pipeline_stage(_dsv4_model(), 0, 2, moe_balancing="none")
    s1 = build_pipeline_stage(_dsv4_model(), 1, 2, moe_balancing="none")
    first = {s0.global_parameter_name(k) for k in s0.state_dict()}
    last = {s1.global_parameter_name(k) for k in s1.state_dict()}

    assert isinstance(s0.model.hc_head, nn.Identity)
    assert not isinstance(s1.model.hc_head, nn.Identity)
    assert not any("hc_head" in k for k in first), sorted(k for k in first if "hc_head" in k)
    assert any("hc_head" in k for k in last)
    # No key is lost to the neutralization.
    reference = set(_dsv4_model().state_dict())
    assert first | last == reference and not (first & last)


def test_mid_chain_forward_is_bound_per_stage_and_rejects_a_collapsed_boundary():
    """Silently re-widening a collapsed stream trains on a lossy mean of the diverged streams."""
    s0 = build_pipeline_stage(_dsv4_model(), 0, 2, moe_balancing="none")
    s1 = build_pipeline_stage(_dsv4_model(), 1, 2, moe_balancing="none")
    assert "forward" not in s0.model.__dict__, "stage 0 must not be rebound to the mid-chain forward"
    assert "forward" in s1.model.__dict__, "a non-first stage must run the mid-chain stream forward"

    hidden = TINY_DSV4_CONFIG["hidden_size"]
    with pytest.raises(RuntimeError, match="hyper-connection stream"):
        s1(torch.randn(_BATCH, _SEQ, hidden))


def test_hash_moe_layers_are_pinned_to_stage_zero():
    """A hash router off stage 0 dereferences input_ids=None mid-schedule."""
    build = lambda: _dsv4_model(mlp_layer_types=["hash_moe", "hash_moe", "moe"])  # noqa: E731
    with pytest.raises(ValueError, match="hash_moe layer.*stage 0"):
        build_pipeline_stage(build(), 0, 2, moe_balancing="none", pp_split=[1, 2])
    # The gate reads the partition, not the family.
    build_pipeline_stage(build(), 0, 2, moe_balancing="none", pp_split=[2, 1])


def test_glm5_shared_indexer_chain_may_not_be_severed():
    """A stage beginning on a "shared" DSA layer would raise mid-schedule."""

    def shared_chain():
        return _glm5_model(layer_types=["indexed_attention"] * 4, indexer_types=["full", "full", "shared", "full"])

    with pytest.raises(ValueError, match="severs a DSA indexer chain"):
        build_pipeline_stage(shared_chain(), 0, 2, moe_balancing="none", pp_split=[2, 2])
    build_pipeline_stage(shared_chain(), 0, 2, moe_balancing="none", pp_split=[3, 1])


def test_rebase_safe_families_skip_the_layer_type_offset_gate():
    """Both select per-layer inputs by the layer's own attributes, so heterogeneous ``layer_types`` must not reject
    the offsets."""
    for model in (_dsv4_model(), _glm5_model(layer_types=["linear_attention"] * 3 + ["indexed_attention"])):
        layer_types = model.config.get_text_config().layer_types
        assert len(set(layer_types)) > 1, "premise: heterogeneous layer_types"
        assert layer_types[1:] != layer_types[:-1], "premise: offset 1 is not shift-invariant"
        validate_partition(model, [(0, 1), (1, len(layer_types))])


def test_loader_droppable_prefixes_cover_the_tail_module():
    """A per-node checkpoint has no ``hc_head`` shard on non-last nodes; droppable there, never on the last stage."""
    non_last = StageLayout.of(_dsv4_model()).dropped_prefixes(True, False)
    last = StageLayout.of(_dsv4_model()).dropped_prefixes(False, True)
    assert "model.hc_head" in non_last, non_last
    assert not any("hc_head" in p for p in last), last


def test_specs_declare_support_and_the_tail_module():
    """A dropped declaration reverts the family to the generic contract, first seen as a mid-schedule shape error."""
    for backbone_name in ("DeepseekV4Model", "Glm5NextTextModel"):
        spec = PP_SPEC_MAP[backbone_name]
        assert spec.refusal is None, backbone_name
        assert spec.tail_module_attrs == ("hc_head",), backbone_name
        assert spec.layer_types_rebase_safe, backbone_name
    reject_unsupported_pp_model(_dsv4_model(), "none")
    reject_unsupported_pp_model(_glm5_model(), "none")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
