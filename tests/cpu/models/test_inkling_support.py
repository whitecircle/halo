#!/usr/bin/env python
"""Inkling EP support: contracts, registration, router parity against the real transformers block. Inkling normalises
the routed top-k and the shared experts jointly, so the wrapper must reproduce that single logsumexp."""

import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingTopkRouter

from halo.distributed.context_parallel.model_rules import _UNSUPPORTED_SEQUENCE_AXIS_LAYERS
from halo.distributed.expert_parallel.families.inkling import EPInklingMoELayer
from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.pipeline_parallel.families import PP_SPEC_MAP
from halo.model_loading.parallel.lazy_plan import is_lazy_loadable
from halo.models.moe.config_fields import resolve_expert_count

SEED = 1234
N_ROUTED = 8
N_SHARED = 2
TOP_K = 2
HIDDEN = 64
ROUTE_SCALE = 8.0


def test_registration():
    assert MOE_LAYER_MAP["InklingMoE"] is EPInklingMoELayer
    # The hub has no bare "inkling": these are the composite config's and the text config's spellings.
    assert EPInklingMoELayer.FAMILY.model_types == ("inkling_text", "inkling_mm_model")
    assert EPInklingMoELayer.SUPPORTS_BIAS_BALANCING
    assert EPInklingMoELayer.SEVERS_AUX_LOSS
    # gate.weight is [n_routed + n_shared, hidden]; inferring the expert count from it over-counts.
    assert "gate.weight" not in EPInklingMoELayer._NUM_EXPERTS_ATTR_PATHS


def test_capability_contracts():
    """The lazy loaders translate the hub's Thinking Machines namespace through the conversion entry. RL weight sync
    stays refused: no sync into either pinned engine's Inkling loader has been validated."""
    assert EPInklingMoELayer.FAMILY.hub_conversion_keys == ("inkling_mm_model",)
    assert EPInklingMoELayer.FAMILY.supports_lazy_loading is True
    assert EPInklingMoELayer.FAMILY.weight_sync_refusal is not None
    # PP: the generic upstream split, cut on whole layer_types periods; ``embed_norm`` goes with the embedding module.
    spec = PP_SPEC_MAP["InklingTextModel"]
    assert spec.validated and spec.refusal is None
    assert spec.stream_forward is None and not spec.layer_types_rebase_safe
    # CP: the veto fires before the attention-class scan so the refusal names the convolutions.
    assert "InklingShortConvolution" in _UNSUPPORTED_SEQUENCE_AXIS_LAYERS


def test_lazy_loader_admits_inkling_checkpoints(tmp_path):
    """A missed spelling silently routes the 532 GB checkpoint to the from_pretrained fallback."""
    for model_type in ("inkling_mm_model", "inkling_text"):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
        assert is_lazy_loadable(str(tmp_path)) is True, model_type


def _hf_router():
    torch.manual_seed(SEED)
    router = InklingTopkRouter(
        InklingTextConfig(
            hidden_size=HIDDEN,
            n_routed_experts=N_ROUTED,
            n_shared_experts=N_SHARED,
            num_experts_per_tok=TOP_K,
            route_scale=ROUTE_SCALE,
        )
    )
    nn.init.normal_(router.weight, std=0.5)
    nn.init.normal_(router.e_score_correction_bias, std=0.1)
    with torch.no_grad():
        router.global_scale.fill_(1.3)
    return router


def _bare_ep_layer(router, **attrs) -> EPInklingMoELayer:
    layer = object.__new__(EPInklingMoELayer)
    nn.Module.__init__(layer)
    layer.gate = router
    layer.top_k = TOP_K
    layer.n_shared_experts = N_SHARED
    layer.route_scale = ROUTE_SCALE
    layer.num_experts = N_ROUTED
    layer.balancing_biases = None
    layer.expert_load_counter = None
    layer._forced_topk_indices = None
    for name, value in attrs.items():
        setattr(layer, name, value)
    return layer


def _joint_normalisation(router, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Reference weights at ``indices``, from unbiased logits."""
    routed_logits = router_logits[..., :-N_SHARED]
    shared_logits = router_logits[..., -N_SHARED:]
    topk_logits = torch.cat([routed_logits.gather(-1, indices), shared_logits], dim=-1)
    log_probs = F.logsigmoid(topk_logits)
    weights = torch.exp(log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True))
    return weights * ROUTE_SCALE * router.global_scale


def test_expert_count_resolves_from_config():
    """Config-side consumers probe `num_local_experts`, which InklingTextConfig must alias to `n_routed_experts`."""
    config = InklingTextConfig(n_routed_experts=N_ROUTED, n_shared_experts=N_SHARED)
    assert resolve_expert_count(config) == N_ROUTED


def test_route_matches_hf_router():
    router = _hf_router()
    torch.manual_seed(SEED)
    hidden = torch.randn(16, HIDDEN)

    _routed_logits, hf_weights, hf_indices, hf_gammas = router(hidden)

    layer = _bare_ep_layer(router)
    router_logits = torch.nn.functional.linear(hidden, router.weight)
    indices, weights, gammas = layer._route_joint(router_logits.float())

    assert torch.equal(indices, hf_indices)
    assert torch.allclose(weights, hf_weights.float(), atol=1e-6)
    assert torch.allclose(gammas, hf_gammas.float(), atol=1e-6)


def test_shared_gammas_come_from_joint_normalisation():
    """Routed + shared weights sum to route_scale * global_scale only if they share one logsumexp."""
    router = _hf_router()
    torch.manual_seed(SEED)
    hidden = torch.randn(16, HIDDEN)
    layer = _bare_ep_layer(router)

    router_logits = torch.nn.functional.linear(hidden, router.weight)
    _indices, weights, gammas = layer._route_joint(router_logits.float())

    total = weights.sum(dim=-1) + gammas.sum(dim=-1)
    expected = ROUTE_SCALE * router.global_scale.item()
    assert torch.allclose(total, torch.full_like(total, expected), atol=1e-5)


def test_balancing_bias_steers_selection_only():
    """A bias that leaks into the gate trains the router on perturbed weights."""
    router = _hf_router()
    torch.manual_seed(SEED)
    hidden = torch.randn(8, HIDDEN)
    router_logits = torch.nn.functional.linear(hidden, router.weight).float()

    biases = torch.zeros(N_ROUTED)
    biases[3] = 100.0
    biased = _bare_ep_layer(router, balancing_biases=biases)
    indices, weights, gammas = biased._route_joint(router_logits)

    assert (indices == 3).any(dim=-1).all(), "balancing bias must steer selection"
    expected = _joint_normalisation(router, router_logits, indices)
    assert torch.allclose(weights, expected[..., :TOP_K], atol=1e-6)
    assert torch.allclose(gammas, expected[..., -N_SHARED:], atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
