#!/usr/bin/env python
"""Hyper-connection pipeline stages (DeepSeek-V4, GLM-5): boundary stream, gates, ownership.

These families' inter-layer activation is the ``hc_mult``-widened ``[B, S, hc_mult, D]`` stream, so
their stages ride three seams the generic split does not exercise: the spec-declared tail module
(``hc_head``) neutralized off non-last stages, the mid-chain forward bound on non-first stages (the
upstream forward re-widens whatever ``inputs_embeds`` it receives), and per-family partition gates
(hash-router layers pinned to stage 0, DSA indexer chains kept within a stage). The load-bearing
check is BITWISE chained equivalence: stage-0 forward → boundary stream → stage-1 forward must equal
the unsplit model exactly — any lossy boundary (a mean-collapse re-widened by replication) fails it.

GLM-5's chained checks run an all-DSA ``layer_types`` stack: the KDA linear-attention path dispatches
to the CUDA-only ``causal_conv1d`` package even on CPU (transformers' with-fallback kernel wrapper
prefers the installed package unconditionally), so the KDA interleave is covered by the GPU test
(tests/gpu/parallelism/pp/test_pp_glm5_vs_single_gpu.py) instead.

Run: python tests/cpu/parallelism/test_pp_hyper_connection_stages.py
"""

import pytest
import torch
import torch.nn as nn
from transformers import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
)

from src.distributed.pipeline_parallel.lazy_loader import _cross_stage_droppable_prefixes
from src.distributed.pipeline_parallel.split import PP_SPEC_MAP, validate_model_supports_pp
from src.distributed.pipeline_parallel.stage import build_pipeline_stage, reject_layer_type_rebase
from tests.common.models import TINY_DSV4_CONFIG, TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG

# Kept verbatim — including the metadata-only ``num_nextn_predict_layers: 1`` the hub checkpoints
# also declare: transformers drops the MTP weights at load, so the liveness-based MTP gate must
# accept the declared-but-never-built tail (the tree below holds exactly num_hidden_layers layers).
_DSV4_CONFIG = dict(TINY_DSV4_CONFIG)

# All-DSA stack for the CPU-runnable chained checks (see module docstring for why no KDA here).
_GLM5_TEXT_CONFIG = {**TINY_GLM5_CONFIG, "layer_types": ["deepseek_sparse_attention"] * 4}

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
    """Drive ``input_ids`` through the chained stages; returns (boundary stream, logits)."""
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
    """The stage chain must be numerically IDENTICAL to the unsplit forward, not merely close.

    The split reorders no arithmetic — stage k runs the same layer ops on the same stream — so any
    drift means the boundary lost state: a collapsed (3-D) boundary, a re-widened stream, an
    un-neutralized ``hc_head``/norm collapsing mid-network, or a re-based mask selection.
    """
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
    """The widened boundary must stay on the autograd graph: a detached (or re-created) stream
    trains the first stage on zero gradients with nothing raised."""
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
    """DSv4's ``hc_head`` carries LEARNED collapse weights: they must appear exactly once, on the
    last stage, and be neutralized (Identity) off it — an un-neutralized copy collapses the stream
    mid-network, a dropped one loses trained state from the checkpoint."""
    s0 = build_pipeline_stage(_dsv4_model(), 0, 2, moe_balancing="none")
    s1 = build_pipeline_stage(_dsv4_model(), 1, 2, moe_balancing="none")
    first = {s0.global_parameter_name(k) for k in s0.state_dict()}
    last = {s1.global_parameter_name(k) for k in s1.state_dict()}

    assert isinstance(s0.model.hc_head, nn.Identity)
    assert not isinstance(s1.model.hc_head, nn.Identity)
    assert not any("hc_head" in k for k in first), sorted(k for k in first if "hc_head" in k)
    assert any("hc_head" in k for k in last)
    # Union still partitions the unsplit model exactly (no key lost to the neutralization).
    reference = set(_dsv4_model().state_dict())
    assert first | last == reference and not (first & last)


def test_mid_chain_forward_is_bound_per_stage_and_rejects_a_collapsed_boundary():
    """Stage 0 must keep the REAL family forward (it widens from input_ids natively); a non-first
    stage must refuse a 3-D activation loudly — silently re-widening a collapsed stream trains on
    a lossy mean of the diverged streams."""
    s0 = build_pipeline_stage(_dsv4_model(), 0, 2, moe_balancing="none")
    s1 = build_pipeline_stage(_dsv4_model(), 1, 2, moe_balancing="none")
    assert "forward" not in s0.model.__dict__, "stage 0 must not be rebound to the mid-chain forward"
    assert "forward" in s1.model.__dict__, "a non-first stage must run the mid-chain stream forward"

    hidden = TINY_DSV4_CONFIG["hidden_size"]
    with pytest.raises(RuntimeError, match="hyper-connection stream"):
        s1(torch.randn(_BATCH, _SEQ, hidden))


def test_hash_moe_layers_are_pinned_to_stage_zero():
    """A hash (tid2eid) router off stage 0 dereferences input_ids=None mid-schedule; the partition
    gate must refuse at build time, on every rank, and name the fix."""
    build = lambda: _dsv4_model(mlp_layer_types=["hash_moe", "hash_moe", "moe"])  # noqa: E731
    with pytest.raises(ValueError, match="hash_moe layer.*stage 0"):
        build_pipeline_stage(build(), 0, 2, moe_balancing="none", pp_split=[1, 2])
    # The same layers on stage 0 pass — the gate reads the partition, not the family.
    build_pipeline_stage(build(), 0, 2, moe_balancing="none", pp_split=[2, 1])


def test_glm5_shared_indexer_chain_may_not_be_severed():
    """A stage beginning on a "shared" DSA layer would raise mid-schedule ("Shared DSA layers
    require top-k indices"); the partition gate must move that failure to build time."""
    shared_types = ["full", "full", "shared", "full"]
    with pytest.raises(ValueError, match="severs a DSA indexer chain"):
        build_pipeline_stage(_glm5_model(indexer_types=shared_types), 0, 2, moe_balancing="none", pp_split=[2, 2])
    # The chain kept whole on one stage passes.
    build_pipeline_stage(_glm5_model(indexer_types=shared_types), 0, 2, moe_balancing="none", pp_split=[3, 1])


def test_rebase_safe_families_skip_the_layer_type_offset_gate():
    """Both families select per-layer inputs by the layer's own attributes, so heterogeneous
    ``layer_types`` must not reject their (otherwise shift-variant) split offsets."""
    for model in (_dsv4_model(), _glm5_model(layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"])):
        layer_types = model.config.get_text_config().layer_types
        assert len(set(layer_types)) > 1, "premise: heterogeneous layer_types"
        assert layer_types[1:] != layer_types[:-1], "premise: offset 1 is not shift-invariant"
        reject_layer_type_rebase(model, 1, 1)  # must not raise


def test_loader_droppable_prefixes_cover_the_tail_module():
    """A per-node PP checkpoint carries no ``hc_head`` shard on non-last nodes; the loader must
    classify it droppable there (it is neutralized at the stage build) and NEVER on the last stage."""
    non_last = _cross_stage_droppable_prefixes(_dsv4_model(), 0, 2)
    last = _cross_stage_droppable_prefixes(_dsv4_model(), 1, 2)
    assert "model.hc_head" in non_last, non_last
    assert not any("hc_head" in p for p in last), last


def test_specs_declare_support_and_the_tail_module():
    """The registry rows behind everything above: a dropped declaration silently reverts the family
    to the generic contract, whose first symptom is a mid-schedule shape error."""
    for backbone_name in ("DeepseekV4Model", "Glm5NextTextModel"):
        spec = PP_SPEC_MAP[backbone_name]
        assert spec.SUPPORTS_PP, backbone_name
        assert spec.TAIL_MODULE_ATTRS == ("hc_head",), backbone_name
        assert spec.LAYER_TYPES_REBASE_SAFE, backbone_name
    validate_model_supports_pp(_dsv4_model(), "none")
    validate_model_supports_pp(_glm5_model(), "none")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
