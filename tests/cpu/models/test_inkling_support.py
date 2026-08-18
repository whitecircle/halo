#!/usr/bin/env python
"""CPU tests for Inkling (Thinking Machines) EP support: contracts, registration, router parity.

The contract tests exercise the toolkit's own registries; the parity tests run the EP wrapper's
router against the real ``transformers.models.inkling`` block. Inkling normalises the routed top-k
and the shared experts JOINTLY, so the wrapper must reproduce that single logsumexp — computing the
routed weights alone changes every weight.

    python tests/cpu/models/test_inkling_support.py
"""

import json
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingTopkRouter

from src.distributed.context_parallel.validation import _UNSUPPORTED_SEQUENCE_AXIS_LAYERS
from src.distributed.expert_parallel.layers.inkling import EPInklingMoELayer
from src.distributed.expert_parallel.lazy_loader import lazy_loader_supports_checkpoint
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP
from src.distributed.pipeline_parallel.split import PP_SPEC_MAP

SEED = 1234
N_ROUTED = 8
N_SHARED = 2
TOP_K = 2
HIDDEN = 64
ROUTE_SCALE = 8.0


# Contract tests — the toolkit's own registries


def test_registration():
    assert MOE_LAYER_MAP["InklingMoE"] is EPInklingMoELayer
    # The composite hub config spells its model_type "inkling_mm_model" (there is no bare
    # "inkling"); the text config spells "inkling_text". Both feed every model_type-keyed consumer.
    assert EPInklingMoELayer.HF_MODEL_TYPES == ("inkling_mm_model", "inkling_text")
    assert EPInklingMoELayer._supports_bias_balancing
    assert EPInklingMoELayer._ep_severs_aux_loss
    # gate.weight is [n_routed + n_shared, hidden]; inferring the expert count from it over-counts.
    assert "gate.weight" not in EPInklingMoELayer._NUM_EXPERTS_ATTR_PATHS


def test_capability_contracts():
    """The hub checkpoint keeps Thinking Machines' namespace (model.llm.*, wq_du/wk_dv, interleaved
    w13_weight). The lazy loaders translate it through the declared conversion entry; the RL weight
    sync stays refused (it sends live module-tree names a hub-namespace server silently skips)."""
    assert EPInklingMoELayer._HUB_CONVERSION_KEYS == ("inkling_mm_model",)
    assert EPInklingMoELayer._supports_lazy_loading is True
    assert EPInklingMoELayer._supports_weight_sync is False
    # PP: the text backbone takes the generic split path (no spec); the composite class is refused
    # by the generic VLM gate in the trainer mixin, not by a family spec.
    assert "InklingTextModel" not in PP_SPEC_MAP
    # CP: the short convolutions run over the sequence axis; the veto fires before the
    # attention-class scan so the refusal names the mechanism.
    assert "InklingShortConvolution" in _UNSUPPORTED_SEQUENCE_AXIS_LAYERS


def test_lazy_loader_admits_inkling_checkpoints(tmp_path):
    """Both real model_type spellings must resolve to the conversion-backed lazy path — a missed
    spelling silently routes the 532 GB checkpoint to the from_pretrained fallback."""
    for model_type in ("inkling_mm_model", "inkling_text"):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
        assert lazy_loader_supports_checkpoint(str(tmp_path)) is True, model_type


# Router parity — against the real transformers.models.inkling block


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
    """An EPInklingMoELayer skeleton without EP process-group state (CPU unit tests)."""
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
    """The reference joint routed+shared normalisation at the given indices, from UNBIASED logits."""
    routed_logits = router_logits[..., :-N_SHARED]
    shared_logits = router_logits[..., -N_SHARED:]
    topk_logits = torch.cat([routed_logits.gather(-1, indices), shared_logits], dim=-1)
    log_probs = F.logsigmoid(topk_logits)
    weights = torch.exp(log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True))
    return weights * ROUTE_SCALE * router.global_scale


def test_expert_count_resolves_from_config():
    """Config-side consumers (loader dispatch, EP divisibility validation, MoE metrics) probe
    `num_local_experts`, which InklingTextConfig must alias to `n_routed_experts`."""
    from src.models.moe_balancing import ROUTER_EXPERT_COUNT_FIELDS, get_first_router_field

    config = InklingTextConfig(n_routed_experts=N_ROUTED, n_shared_experts=N_SHARED)
    assert get_first_router_field(config, ROUTER_EXPERT_COUNT_FIELDS) == N_ROUTED


def test_route_matches_hf_router():
    """Selection, routed weights, and shared gammas must all match InklingTopkRouter exactly."""
    router = _hf_router()
    torch.manual_seed(SEED)
    hidden = torch.randn(16, HIDDEN)

    _routed_logits, hf_weights, hf_indices, hf_gammas = router(hidden)

    layer = _bare_ep_layer(router)
    router_logits = torch.nn.functional.linear(hidden, router.weight)
    indices, weights, gammas = layer._route(router_logits.float())

    assert torch.equal(indices, hf_indices)
    assert torch.allclose(weights, hf_weights.float(), atol=1e-6)
    assert torch.allclose(gammas, hf_gammas.float(), atol=1e-6)


def test_shared_gammas_come_from_joint_normalisation():
    """The shared share is not an independent sigmoid: routed + shared weights sum to
    route_scale * global_scale, which only holds if they share one logsumexp."""
    router = _hf_router()
    torch.manual_seed(SEED)
    hidden = torch.randn(16, HIDDEN)
    layer = _bare_ep_layer(router)

    router_logits = torch.nn.functional.linear(hidden, router.weight)
    _indices, weights, gammas = layer._route(router_logits.float())

    total = weights.sum(dim=-1) + gammas.sum(dim=-1)
    expected = ROUTE_SCALE * router.global_scale.item()
    assert torch.allclose(total, torch.full_like(total, expected), atol=1e-5)


def test_balancing_bias_steers_selection_only():
    """A large balancing bias must move the top-k while the returned weights stay EXACTLY the
    joint normalisation of the unbiased logits at the biased indices — a bias that leaks into the
    gate trains the router on perturbed weights."""
    router = _hf_router()
    torch.manual_seed(SEED)
    hidden = torch.randn(8, HIDDEN)
    router_logits = torch.nn.functional.linear(hidden, router.weight).float()

    biases = torch.zeros(N_ROUTED)
    biases[3] = 100.0
    biased = _bare_ep_layer(router, balancing_biases=biases)
    indices, weights, gammas = biased._route(router_logits)

    assert (indices == 3).any(dim=-1).all(), "balancing bias must steer selection"
    expected = _joint_normalisation(router, router_logits, indices)
    assert torch.allclose(weights, expected[..., :TOP_K], atol=1e-6)
    assert torch.allclose(gammas, expected[..., -N_SHARED:], atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
