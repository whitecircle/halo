#!/usr/bin/env python
"""Bailing bias balancing through the gate's native ``expert_bias``. The shared adoption contract is pinned in
``tests/cpu/parallelism/test_native_bias_balancing_adoption.py``."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from halo.callbacks.balancing import BalancingBiasCallback
from halo.distributed.expert_parallel.families.bailing import EPBailingMoELayer
from halo.models.moe.balancing import resolve_balancing_mode

NUM_EXPERTS = 8
# The hub slot's dtype: adoption has to upcast it, so a fp32 fixture would hide that step.
SLOT_DTYPE = torch.bfloat16


class _FakeBailingGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("expert_bias", torch.zeros(NUM_EXPERTS, dtype=SLOT_DTYPE))


def _bare_layer() -> EPBailingMoELayer:
    layer = object.__new__(EPBailingMoELayer)
    nn.Module.__init__(layer)
    layer.gate = _FakeBailingGate()
    layer.num_experts = NUM_EXPERTS
    return layer


class _Shell(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer


def test_class_declarations():
    assert EPBailingMoELayer.SUPPORTS_BIAS_BALANCING
    # The wrapper returns a bare tensor, so router logits never reach the outputs and `auto` resolves to bias_update.
    assert EPBailingMoELayer.SEVERS_AUX_LOSS


def test_the_expert_load_hook_is_a_no_op_until_balancing_is_enabled():
    """Routing calls the hook unconditionally, and the slot's presence is what ``iter_balancing_routers`` keys on."""
    layer = _bare_layer()
    layer._record_expert_load(torch.zeros((4, 2), dtype=torch.int64))
    assert not hasattr(layer, "expert_load_counter")


def test_property_tracks_buffer_re_registration():
    """``.to()`` and dtype casts re-register buffers; a stale alias would update a tensor the gate does not read."""
    layer = _bare_layer()
    layer.enable_bias_balancing()
    layer.gate.expert_bias = layer.gate.expert_bias.clone()  # what any re-registration does
    assert layer.balancing_biases is layer.gate.expert_bias


def test_setter_writes_through_to_the_gate():
    layer = _bare_layer()
    layer.enable_bias_balancing()
    replacement = torch.full((NUM_EXPERTS,), 0.25)
    layer.balancing_biases = replacement
    assert layer.gate.expert_bias is replacement, "assignment must re-register on the gate, not shadow it"


def test_record_expert_load_counts_selections():
    layer = _bare_layer()
    layer.enable_bias_balancing()
    indices = torch.tensor([[0, 1], [0, 2], [0, 1]], dtype=torch.int64)
    layer._record_expert_load(indices)
    expected = torch.zeros(NUM_EXPERTS)
    expected[0], expected[1], expected[2] = 3, 2, 1
    assert torch.equal(layer.expert_load_counter, expected)


def test_callback_sign_update_lands_in_the_gate_buffer():
    layer = _bare_layer()
    layer.enable_bias_balancing()
    counts = torch.zeros(NUM_EXPERTS)
    counts[0] = 10.0
    layer.expert_load_counter = counts.clone()

    callback = BalancingBiasCallback(update_rate=1e-3)
    callback.on_step_end(args=None, state=SimpleNamespace(global_step=1), control=None, model=_Shell(layer))

    bias = layer.gate.expert_bias
    assert bias[0].item() == pytest.approx(-1e-3), "overloaded expert must be biased DOWN"
    assert torch.allclose(bias[1:], torch.full((NUM_EXPERTS - 1,), 1e-3)), "starved experts biased UP"
    assert torch.equal(layer.expert_load_counter, torch.zeros(NUM_EXPERTS)), "counters must reset per step"


def test_auto_resolves_bailing_to_bias_update():
    shell = _Shell(_bare_layer())
    assert resolve_balancing_mode("auto", shell, is_moe=True) == "bias_update"
    # No wrapper carries the bias and the forward takes no output_router_logits flag, so auto gives none.
    assert resolve_balancing_mode("auto", _Shell(nn.Linear(4, 4)), is_moe=True) == "none"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
