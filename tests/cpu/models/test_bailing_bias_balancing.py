#!/usr/bin/env python
"""CPU tests for Bailing bias-update balancing through the gate's native ``expert_bias``.

Bailing's EP wrapper delegates routing to the hub gate, so instead of the base's side-buffer it
adopts the gate's own persistent ``expert_bias`` (``topk_method: noaux_tc``) as the balancing buffer.
The adoption contract shared with every other native-slot family — identity, the fp32 upcast, the
export key, buffer-not-Parameter, idempotence — is pinned once for the whole roster in
``tests/cpu/parallelism/test_native_bias_balancing_adoption.py``. What lives here is what is
Bailing's own: the class declarations that make ``auto`` resolve the family to ``bias_update``, the
expert-load recording its routing feeds, and the round trip from a real
``RouterBiasBalancingCallback`` step into the gate buffer.

    python tests/cpu/models/test_bailing_bias_balancing.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.callbacks.router_bias_balancing import RouterBiasBalancingCallback
from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.models.moe_balancing import resolve_balancing_mode

NUM_EXPERTS = 8
# The hub slot's dtype: adoption has to upcast it, so a fp32 fixture would hide that step.
SLOT_DTYPE = torch.bfloat16


class _FakeBailingGate(nn.Module):
    """The one contract the wrapper's balancing rides on: a persistent ``expert_bias`` buffer."""

    def __init__(self):
        super().__init__()
        self.register_buffer("expert_bias", torch.zeros(NUM_EXPERTS, dtype=SLOT_DTYPE))


def _bare_layer() -> EPBailingMoELayer:
    """An EPBailingMoELayer skeleton without EP process-group state (CPU unit tests)."""
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
    assert EPBailingMoELayer._supports_bias_balancing
    # The wrapper returns a bare tensor where the hub block returns (hidden, router_logits), so
    # router logits never reach the outputs — what lets `auto` resolve the family to bias_update.
    assert EPBailingMoELayer._ep_severs_aux_loss


def test_the_expert_load_hook_is_a_no_op_until_balancing_is_enabled():
    """Bailing's routing calls the hook unconditionally, so an unbalanced run must leave no counter
    behind — the slot's presence is what ``iter_balancing_routers`` keys on."""
    layer = _bare_layer()
    layer._record_expert_load(torch.zeros((4, 2), dtype=torch.int64))
    assert not hasattr(layer, "expert_load_counter")


def test_property_tracks_buffer_re_registration():
    """``.to()``/dtype casts re-register module buffers; a stale alias would keep updating a tensor
    the gate no longer reads — balancing that silently does nothing."""
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
    counts[0] = 10.0  # expert 0 overloaded, everything else starved
    layer.expert_load_counter = counts.clone()

    callback = RouterBiasBalancingCallback(update_rate=1e-3)
    callback.on_step_end(args=None, state=SimpleNamespace(global_step=1), control=None, model=_Shell(layer))

    bias = layer.gate.expert_bias
    assert bias[0].item() == pytest.approx(-1e-3), "overloaded expert must be biased DOWN"
    assert torch.allclose(bias[1:], torch.full((NUM_EXPERTS - 1,), 1e-3)), "starved experts biased UP"
    assert torch.equal(layer.expert_load_counter, torch.zeros(NUM_EXPERTS)), "counters must reset per step"


def test_auto_resolves_bailing_to_bias_update():
    shell = _Shell(_bare_layer())
    assert resolve_balancing_mode("auto", shell, is_moe=True) == "bias_update"
    # Sanity control: without the wrapper nothing carries the bias, and the shell's forward consults
    # no output_router_logits flag either, so auto falls through to none — never to an aux_loss the
    # balancing strategy refuses (tests/cpu/models/test_moe_balancing_auto_resolution.py).
    assert resolve_balancing_mode("auto", _Shell(nn.Linear(4, 4)), is_moe=True) == "none"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
