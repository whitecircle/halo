#!/usr/bin/env python
"""EP-wrapped Zaya carries the upstream block's cross-layer EDA router gradient, checked in fp64 against upstream. A
severed edge leaves a plausible gradient of the wrong magnitude (~1e-2), against ~1e-7 from fp32 routing weights."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers.models.zaya.configuration_zaya import ZayaConfig
from transformers.models.zaya.modeling_zaya import ZayaSparseMoeBlock

from halo.distributed.expert_parallel.families.zaya import EPZayaMoELayer
from tests.common.numerics import fro_rel_err
from tests.common.parallelism import single_process_ep_config

E, H, M, ROUTER_H = 4, 64, 32, 32
# EDA is off on layer 0, so a chain from 0 covers the dead link and the live ones.
LAYER_INDICES = (0, 1, 2)

# Whose only cross-layer gradient path is the EDA state.
_EDA_FED_PARAMS = ("down_proj.weight", "down_proj.bias", "router_states_scale")

# fp32 routing weights put the floor at ~1e-7; a severed edge sits at ~1e-2.
_TOLERANCE = 1e-5


def _config() -> ZayaConfig:
    return ZayaConfig(
        vocab_size=512,
        hidden_size=H,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        moe_intermediate_size=M,
        num_experts=E,
        num_experts_per_tok=1,
        router_hidden_size=ROUTER_H,
        tie_word_embeddings=True,
        use_cache=False,
    )


def _blocks() -> nn.ModuleList:
    torch.manual_seed(8)
    config = _config()
    blocks = nn.ModuleList(ZayaSparseMoeBlock(config, index) for index in LAYER_INDICES)
    blocks.to(torch.float64)
    for parameter in blocks.parameters():
        nn.init.normal_(parameter, std=0.3)
    return blocks


def _ep_layers() -> nn.ModuleList:
    return nn.ModuleList(EPZayaMoELayer(block, single_process_ep_config(E)).cpu() for block in _blocks())


def _run_chain(layers, inputs: torch.Tensor) -> tuple[list[torch.Tensor], dict[str, torch.Tensor | None]]:
    state = None
    hidden_states = inputs
    outputs = []
    loss = hidden_states.new_zeros(())
    for layer in layers:
        hidden_states, state = layer(hidden_states, state)
        outputs.append(hidden_states.detach().clone())
        # Position-weighted so no output element's gradient cancels another's.
        weights = torch.arange(1, hidden_states.numel() + 1, dtype=hidden_states.dtype)
        loss = loss + (hidden_states * weights.view_as(hidden_states)).sum()
    loss.backward()

    grads: dict[str, torch.Tensor | None] = {}
    for index, layer in enumerate(layers):
        for name, parameter in layer.gate.named_parameters():
            grads[f"L{index}.{name}"] = None if parameter.grad is None else parameter.grad.detach().clone()
    return outputs, grads


def test_ep_zaya_forward_matches_the_upstream_block():
    """The router's masked skip slot must contribute nothing."""
    inputs = torch.randn(2, 5, H, dtype=torch.float64)

    reference, _ = _run_chain(_blocks(), inputs)
    actual, _ = _run_chain(_ep_layers(), inputs)

    for index, (got, expected) in enumerate(zip(actual, reference, strict=True)):
        assert expected.abs().sum() > 0, f"L{index}: the reference output is all zeros — nothing is compared"
        assert fro_rel_err(got, expected) < _TOLERANCE, f"L{index}: EP output differs from the upstream block"


def test_ep_zaya_router_gradient_matches_the_upstream_block():
    """The last layer has no downstream EDA consumer, so checking it alone passes with the edge severed."""
    inputs = torch.randn(2, 5, H, dtype=torch.float64)

    _outputs, reference = _run_chain(_blocks(), inputs)
    _ep_outputs, actual = _run_chain(_ep_layers(), inputs)

    assert set(actual) == set(reference)
    for key, expected in reference.items():
        if expected is None:
            assert actual[key] is None, key
            continue
        relative = fro_rel_err(actual[key], expected)
        assert relative < _TOLERANCE, f"{key}: EP router gradient differs by {relative:.3e} relative"


def test_the_eda_state_leaves_the_ep_layer_differentiable():
    """A ``.detach()`` is invisible in the forward value and the last layer's gradients."""
    layer = _ep_layers()[-1]
    inputs = torch.randn(2, 5, H, dtype=torch.float64, requires_grad=True)

    _output, state = layer(inputs, None)

    assert state.requires_grad and state.grad_fn is not None
    # Anti-vacuity: the edge must reach the router weights.
    state.sum().backward()
    assert layer.gate.down_proj.weight.grad is not None
    assert torch.count_nonzero(layer.gate.down_proj.weight.grad) > 0


@pytest.mark.parametrize("suffix", _EDA_FED_PARAMS)
def test_upstream_zaya_block_keeps_the_state_attached(suffix):
    """If upstream ever detaches the state, the equivalence test would pass with both sides losing the gradient."""
    blocks = _blocks()
    _output, state = blocks[0](torch.randn(2, 5, H, dtype=torch.float64), None)
    assert state.grad_fn is not None

    # layer_idx > 0 owns the EDA-only ``router_states_scale`` as well.
    assert any(name.endswith(suffix) for name, _ in blocks[-1].gate.named_parameters()), (
        f"ZayaRouter no longer exposes '{suffix}' — the EDA gradient path this test pins has moved."
    )


def test_router_discard_slot_is_masked_before_dispatch():
    """A raw index of ``num_experts`` at dispatch would silently misroute under ``ep_size > 1``."""
    layers = _ep_layers()
    for layer in layers:
        # The skip slot's bias starts at -1.0, so no token would select it; biased here to keep the equivalence
        # references nonzero.
        layer.gate.balancing_biases[-1] = 0.25
    records: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record(_module, _args, output):
        records.append((output[1].detach(), output[2].detach()))  # (probs, indices)

    for layer in layers:
        layer.gate.register_forward_hook(record)
    _run_chain(layers, torch.randn(2, 5, H, dtype=torch.float64))

    assert len(records) == len(LAYER_INDICES)
    discarded = 0
    for probs, indices in records:
        assert (indices < E).all(), "a discard-slot index reached the dispatch boundary"
        skip = (probs == 0).all(dim=-1)
        discarded += int(skip.sum())
        assert (indices[skip] == 0).all(), "a discarded token was not re-pointed at expert 0"
    assert discarded > 0, "no token selected the skip slot — the masking contract went unexercised"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
