#!/usr/bin/env python
"""Native EP expert-LoRA: the disable a KL reference pass needs and the fail-loud resume guard.
The stub layer bypasses the DeepEP ``__init__`` and overrides ``_grouped_mm`` with a plain matmul to run on CPU."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from halo.distributed.expert_parallel.moe_layer.expert_lora import (
    disable_expert_adapters,
    install_ep_aware_disable_adapter,
    load_expert_lora_adapters,
)
from tests.common.ep_stubs import StubEPLayerBase

_E, _K, _N, _R = 2, 4, 4, 2  # experts, in-dim, out-dim, lora rank


class _StubEPLayer(StubEPLayerBase):
    """Sets only what the LoRA projection and resume guard read; ``_grouped_mm`` is a plain batched matmul."""

    def __init__(self, *, with_lora: bool = True, attrs=("gate_proj",)):
        super().__init__()
        self.expert_lora_scaling = 2.0
        self.expert_lora_dropout = nn.Identity()
        self._expert_adapters_enabled = True
        self.expert_start, self.expert_end = 0, _E
        self.ep_config = SimpleNamespace(ep_size=1)
        self.expert_tp_size = 1
        self.expert_tp_rank = 0

        for attr in attrs:
            setattr(self, attr, nn.Parameter(torch.randn(_E, _K, _N)))
            if with_lora:
                # Non-zero A and B, so the delta is really present.
                a = nn.Parameter(torch.randn(_E, _K, _R))
                b = nn.Parameter(torch.randn(_E, _R, _N))
                setattr(self, f"{attr}_lora_A", a)
                setattr(self, f"{attr}_lora_B", b)
        self._expert_lora_attrs = frozenset(attrs) if with_lora else frozenset()

    def _grouped_mm(self, mat_a, mat_b, *, offs=None, lowp=True):
        return torch.bmm(mat_a, mat_b)


def _tokens():
    torch.manual_seed(0)
    return torch.randn(_E, 3, _K)  # [E, tokens_per_expert, K]


def test_disable_removes_lora_delta_not_just_zero_b():
    """Outside the context the projection must differ, proving the toggle over a coincidental B=0."""
    layer = _StubEPLayer()
    x = _tokens()
    offs = torch.tensor([3, 6])

    base_only = layer._grouped_mm(x, layer.gate_proj)
    with_delta = layer._expert_proj(x, "gate_proj", offs, torch.float32)
    assert not torch.allclose(with_delta, base_only), "LoRA delta is zero — A/B not effective; test is vacuous"

    with disable_expert_adapters(layer):
        disabled = layer._expert_proj(x, "gate_proj", offs, torch.float32)
    assert torch.allclose(disabled, base_only, atol=1e-5), "delta not removed inside disable context"

    again = layer._expert_proj(x, "gate_proj", offs, torch.float32)
    assert torch.allclose(again, with_delta, atol=1e-5), "delta not restored after context"


def test_disable_single_loop_path_also_removed():
    """The per-expert loop path (``_expert_proj_single``) respects the same flag."""
    layer = _StubEPLayer()
    x = torch.randn(3, _K)
    base = x @ layer.gate_proj[0]
    full = layer._expert_proj_single(0, x, "gate_proj")
    assert not torch.allclose(full, base)
    with disable_expert_adapters(layer):
        assert torch.allclose(layer._expert_proj_single(0, x, "gate_proj"), base, atol=1e-5)


def test_disable_restores_prior_per_layer_state_not_blanket_true():
    """A blanket re-enable would wrongly flip the pre-disabled layer to True."""
    on_layer = _StubEPLayer()
    off_layer = _StubEPLayer()
    off_layer._expert_adapters_enabled = False
    container = nn.ModuleList([on_layer, off_layer])

    with disable_expert_adapters(container):
        assert on_layer._expert_adapters_enabled is False
        assert off_layer._expert_adapters_enabled is False

    assert on_layer._expert_adapters_enabled is True
    assert off_layer._expert_adapters_enabled is False, "prior-disabled layer wrongly re-enabled"


def test_disable_reentrant_inner_exit_does_not_reenable():
    layer = _StubEPLayer()
    with disable_expert_adapters(layer):
        with disable_expert_adapters(layer):
            assert layer._expert_adapters_enabled is False
        assert layer._expert_adapters_enabled is False
    assert layer._expert_adapters_enabled is True


class _StubPeftModel(nn.Module):
    """A real ``@contextmanager disable_adapter`` flipping a flag, with EP layers reachable through ``.modules()``."""

    def __init__(self):
        super().__init__()
        self.peft_adapter_active = True
        self.ep = _StubEPLayer()

    @contextlib.contextmanager
    def disable_adapter(self):
        self.peft_adapter_active = False
        try:
            yield
        finally:
            self.peft_adapter_active = True


def test_ep_aware_disable_flips_both_flags_and_restores():
    pm = _StubPeftModel()
    install_ep_aware_disable_adapter(pm)

    assert pm.peft_adapter_active is True
    assert pm.ep._expert_adapters_enabled is True
    with pm.disable_adapter():
        assert pm.peft_adapter_active is False, "peft adapter not disabled"
        assert pm.ep._expert_adapters_enabled is False, "EP expert adapter not disabled"
    assert pm.peft_adapter_active is True
    assert pm.ep._expert_adapters_enabled is True


def test_ep_aware_is_idempotent():
    """The marker is set and the second call returns early."""
    pm = _StubPeftModel()
    install_ep_aware_disable_adapter(pm)
    wrapped_once = pm.disable_adapter
    install_ep_aware_disable_adapter(pm)
    assert pm.disable_adapter is wrapped_once, "double-wrapped — idempotence broken"
    assert pm._ep_disable_adapter_patched is True
    with pm.disable_adapter():
        assert pm.ep._expert_adapters_enabled is False


def _holding(layer: nn.Module) -> nn.Module:
    """A model holding ``layer`` at ``layer``, so its saved keys are ``layer.experts.<attr>.lora_{A,B}``."""
    root = nn.Module()
    root.layer = layer
    return root


def test_load_raises_on_key_set_mismatch():
    """The alternative is silently rebuilding zero-init adapters; nothing may be copied."""
    layer = _StubEPLayer(attrs=("gate_proj", "down_proj"))
    before = {name: param.detach().clone() for name, param in layer.named_parameters()}
    # Saved state from a checkpoint with the fused gate_up_proj naming.
    bad_state = {
        "layer.experts.gate_up_proj.lora_A": torch.randn(_E, _K, _R),
        "layer.experts.gate_up_proj.lora_B": torch.randn(_E, _R, _N),
    }
    with pytest.raises(RuntimeError, match="do not match the ones this model builds"):
        load_expert_lora_adapters(_holding(layer), bad_state)
    assert all(torch.equal(param, before[name]) for name, param in layer.named_parameters())


def test_load_roundtrips_matching_keys():
    layer = _StubEPLayer(attrs=("gate_proj",))
    new_a = torch.randn(_E, _K, _R)
    new_b = torch.randn(_E, _R, _N)
    load_expert_lora_adapters(
        _holding(layer), {"layer.experts.gate_proj.lora_A": new_a, "layer.experts.gate_proj.lora_B": new_b}
    )
    assert torch.allclose(layer.gate_proj_lora_A.data, new_a)
    assert torch.allclose(layer.gate_proj_lora_B.data, new_b)


def test_load_empty_state_raises_the_descriptive_mismatch():
    """The copy loop's bare ``KeyError`` would name neither the layer nor the remedy."""
    layer = _StubEPLayer(attrs=("gate_proj",))
    with pytest.raises(RuntimeError, match=r"missing from the file \['layer\.experts\.gate_proj\.lora_A'"):
        load_expert_lora_adapters(_holding(layer), {})


def test_load_empty_state_on_a_layer_without_adapters_is_a_no_op():
    layer = _StubEPLayer(attrs=())
    load_expert_lora_adapters(_holding(layer), {})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
