#!/usr/bin/env python
"""CPU parity coverage for Laguna's EP wrapper, against the REAL in-library MoE block.

The wrapper claims `transformers.models.laguna` by class name and by ``model_type``, so the block it
is checked against must be the real one: a hand-written stand-in can declare attributes the library
class does not have (``norm_topk_prob`` is exactly such an attribute — ``modular_laguna`` deletes the
field, so nothing in the real block/gate/config carries it) and would then hide a resolution bug that
fires the moment a checkpoint loads in library format.

Run: ``python tests/cpu/parallelism/test_laguna_ep.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.laguna.configuration_laguna import LagunaConfig
from transformers.models.laguna.modeling_laguna import LagunaSparseMoeBlock

from src.distributed.expert_parallel.expert_weights import resolve_ep_merge_layer_class
from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.layers.laguna import EPLagunaMoELayer
from src.distributed.expert_parallel.patching import build_moe_layer_map
from tests.common.parallelism import single_process_ep_config

E, H, M, K = 8, 16, 32, 3
SCALING = 2.5


def _library_config() -> LagunaConfig:
    return LagunaConfig(
        hidden_size=H,
        intermediate_size=4 * H,
        moe_intermediate_size=M,
        shared_expert_intermediate_size=M,
        num_experts=E,
        num_experts_per_tok=K,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        moe_routed_scaling_factor=SCALING,
    )


def _library_block() -> LagunaSparseMoeBlock:
    """The real `LagunaSparseMoeBlock`, fp64 so routing-weight scale errors cannot hide in bf16 noise."""
    torch.manual_seed(7)
    block = LagunaSparseMoeBlock(_library_config()).to(torch.float64)
    for parameter in block.parameters():
        nn.init.normal_(parameter, std=0.5)
    nn.init.normal_(block.gate.e_score_correction_bias, std=0.3)
    return block


def test_laguna_owns_its_wrapper_by_module_name_and_model_type():
    """Both registries must land on the Laguna class, not on GLM-4's — they differ in the routing
    default below, so resolving to the parent silently changes every routed weight."""
    assert build_moe_layer_map()["LagunaSparseMoeBlock"] is EPLagunaMoELayer
    assert resolve_ep_merge_layer_class("laguna") is EPLagunaMoELayer
    assert resolve_ep_merge_layer_class("glm4_moe_lite") is EPGlm4MoELayer
    assert issubclass(EPLagunaMoELayer, EPGlm4MoELayer)


def test_library_block_declares_no_norm_topk_prob_so_the_family_default_decides():
    """Anti-vacuity guard for the test below: if a future transformers release puts the attribute
    back on the block/gate/config, the resolution chain — not the default — supplies it, and the
    parity assertion would stop exercising `_NORM_TOPK_PROB_DEFAULT`."""
    block = _library_block()
    assert not any(hasattr(source, "norm_topk_prob") for source in (block, block.gate, _library_config()))
    # Laguna is the family that OPTS OUT of the knob, which is what lets its default decide; GLM-4
    # declares it on gate and config, so there it stays required and no default may stand in.
    assert "norm_topk_prob" in EPLagunaMoELayer._OPTIONAL_ROUTING_KNOBS
    assert "norm_topk_prob" not in EPGlm4MoELayer._OPTIONAL_ROUTING_KNOBS
    assert EPLagunaMoELayer._NORM_TOPK_PROB_DEFAULT is True

    layer = EPLagunaMoELayer(block, single_process_ep_config(E)).cpu()
    assert layer.norm_topk_prob is True


def test_laguna_forward_matches_the_library_block():
    """`LagunaTopKRouter.forward` normalizes the gathered sigmoid scores unconditionally. Skipping it
    scales every token's routed output by the sum of its top-k scores (~0.9-2.0x, data dependent),
    which no shape or dtype check can catch."""
    block = _library_block()
    inputs = torch.randn(2, 5, H, dtype=torch.float64)

    with torch.no_grad():
        expected = block(inputs)
        actual = EPLagunaMoELayer(block, single_process_ep_config(E)).cpu()(inputs)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_laguna_shared_expert_and_bias_follow_the_library_block():
    block = _library_block()
    expected_act_fn = block.experts.act_fn
    layer = EPLagunaMoELayer(block, single_process_ep_config(E)).cpu()
    assert layer._shared_expert_attr == "shared_experts"
    assert layer.routed_scaling_factor == SCALING
    assert layer.n_group == 1 and layer.topk_group == 1  # no group-limited selection in Laguna
    assert layer.act_fn is expected_act_fn
    assert {name for name, _ in layer.replicated_named_params()} == {
        f"shared_experts.{name}" for name, _ in layer.shared_experts.named_parameters()
    }


def test_laguna_gather_restores_the_hub_per_expert_layout():
    block = _library_block()
    gate_up = block.experts.gate_up_proj.data.clone()
    down = block.experts.down_proj.data.clone()

    state = EPLagunaMoELayer(block, single_process_ep_config(E)).cpu().gather_expert_state_dict()

    assert len(state) == E * 3
    for expert in range(E):
        assert torch.equal(state[f"experts.{expert}.gate_proj.weight"].double(), gate_up[expert, :M])
        assert torch.equal(state[f"experts.{expert}.up_proj.weight"].double(), gate_up[expert, M:])
        assert torch.equal(state[f"experts.{expert}.down_proj.weight"].double(), down[expert])


class _RemoteRouter(nn.Module):
    """The revision-pinned hub router: declares ``norm_topk_prob`` itself and ships no bias tensor."""

    def __init__(self, norm_topk_prob: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H))
        self.e_score_correction_bias = nn.Parameter(torch.empty(E, device="meta"), requires_grad=False)
        self.top_k = K
        self.norm_topk_prob = norm_topk_prob

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight).float()
        # Deliberately incorrect cached choices: the EP wrapper must route from the logits itself.
        weights = torch.ones(*logits.shape[:-1], K, device=logits.device)
        indices = torch.zeros(*logits.shape[:-1], K, dtype=torch.long, device=logits.device)
        return logits, weights, indices


class _RemoteExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = E
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * M, H))
        self.down_proj = nn.Parameter(torch.randn(E, H, M))
        self.act_fn = F.silu


class _RemoteBlock(nn.Module):
    """Hub remote-code layout: shared expert named ``shared_expert``, knobs on the router."""

    def __init__(self, norm_topk_prob: bool = True):
        super().__init__()
        self.gate = _RemoteRouter(norm_topk_prob)
        self.experts = _RemoteExperts()
        self.shared_expert = nn.Linear(H, H, bias=False)
        self.routed_scaling_factor = SCALING


@pytest.mark.parametrize("declared", [True, False])
def test_remote_code_declaration_still_wins_over_the_family_default(declared):
    """The hub revisions the shipped configs pin put the knob on the router. Hard-coding the Laguna
    default instead of keeping the resolution chain would ignore a checkpoint that says otherwise."""
    torch.manual_seed(3)
    layer = EPLagunaMoELayer(_RemoteBlock(norm_topk_prob=declared), single_process_ep_config(E)).cpu()
    assert layer.norm_topk_prob is declared
    # Remote-code blocks name the shared expert in the singular and omit the correction bias.
    assert layer._shared_expert_attr == "shared_expert"
    assert not layer.gate.e_score_correction_bias.is_meta
    assert torch.count_nonzero(layer.gate.e_score_correction_bias) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
