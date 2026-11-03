#!/usr/bin/env python
"""Fused-GLU gates accept what real checkpoints carry: ``ACT2FN`` instances, not ``nn.SiLU``/``nn.GELU``.
A gate against torch classes degrades every run to the eager combine while stubbed parity tests stay green."""

from functools import partial

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4Experts

from halo.distributed.expert_parallel.families.gemma4 import EPGemma4MoELayer
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.families.roster import ep_layer_classes
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPSharedExpertsMoELayerBase
from halo.kernels import fused_glu
from halo.kernels.fused_glu import GluCombine, is_gelu_tanh_activation, is_silu_activation, resolve_fused_glu_mul
from tests.common.parallelism import single_process_ep_config

E, K, H, M = 4, 2, 8, 16


def test_silu_gate_accepts_real_checkpoint_activations():
    assert is_silu_activation(ACT2FN["silu"])  # What HF MoE modules actually carry.
    assert is_silu_activation(F.silu)
    assert is_silu_activation(nn.SiLU())
    assert not is_silu_activation(ACT2FN["gelu"])
    assert not is_silu_activation(ACT2FN["gelu_pytorch_tanh"])


def test_gelu_tanh_gate_accepts_real_checkpoint_activations():
    assert is_gelu_tanh_activation(ACT2FN["gelu_pytorch_tanh"])
    assert is_gelu_tanh_activation(nn.GELU(approximate="tanh"))
    assert not is_gelu_tanh_activation(nn.GELU())  # Exact GELU is not the tanh approximation.
    assert not is_gelu_tanh_activation(ACT2FN["silu"])


def test_resolver_maps_each_activation_to_its_own_kernel():
    """A right answer wired to the wrong kernel would change the activation silently."""
    assert resolve_fused_glu_mul(ACT2FN["silu"]) == GluCombine.silu()
    assert resolve_fused_glu_mul(ACT2FN["gelu_pytorch_tanh"]) == GluCombine.gelu_tanh()
    assert resolve_fused_glu_mul(ACT2FN["gelu"]) is None  # Exact GELU: no kernel implements it.
    assert resolve_fused_glu_mul(ACT2FN["relu"]) is None
    assert resolve_fused_glu_mul("not-callable") is None


def _real_gemma4_experts() -> Gemma4TextExperts:
    """A stand-in certifies its own assumptions; the real one builds ``act_fn`` from ``config.hidden_activation``
    and stores ``down_proj`` as ``[E, H, M]``. Parameters start ``torch.empty``."""
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=H,
        intermediate_size=2 * M,
        moe_intermediate_size=M,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=E,
        num_experts_per_tok=K,
    )
    experts = Gemma4TextExperts(config)
    for parameter in experts.parameters():
        nn.init.normal_(parameter, std=0.02)
    return experts


def test_gemma4_layer_enables_fused_geglu_with_production_activation():
    """The layer must latch the kernel, not merely recognize the gate."""
    layer = EPGemma4MoELayer(_real_gemma4_experts(), single_process_ep_config(E)).cpu()
    assert layer._fused_glu_mul == GluCombine.gelu_tanh()


def test_gemma4_fused_combine_matches_the_eager_activation():
    """The CPU path is eager, so this pins the wiring; ``tests/gpu/kernels/test_fused_glu.py`` pins the Triton
    kernel."""
    layer = EPGemma4MoELayer(_real_gemma4_experts(), single_process_ep_config(E)).cpu()

    gate, up = torch.randn(7, M), torch.randn(7, M)
    expected = F.gelu(gate, approximate="tanh") * up
    torch.testing.assert_close(layer._glu_combine(gate, up), expected)


def test_packed_combine_follows_the_latch(monkeypatch):
    """Any other latch keeps the chunked seam rather than silently switching activation."""
    layer = EPGemma4MoELayer(_real_gemma4_experts(), single_process_ep_config(E)).cpu()
    gate_up = torch.randn(7, 2 * M)
    packed = []
    real_packed = GluCombine.packed
    monkeypatch.setattr(GluCombine, "packed", lambda self, gu: packed.append(self) or real_packed(self, gu))
    expected = F.gelu(gate_up[:, :M], approximate="tanh") * gate_up[:, M:]
    torch.testing.assert_close(layer._glu_combine_packed(gate_up), expected)
    assert packed == [GluCombine.gelu_tanh()]

    layer._fused_glu_mul = GluCombine.clamped_silu(7.0)
    torch.testing.assert_close(
        layer._glu_combine_packed(gate_up), fused_glu.clamped_silu_mul_eager(gate_up[:, :M], gate_up[:, M:], 7.0)
    )
    assert packed == [GluCombine.gelu_tanh(), GluCombine.clamped_silu(7.0)]

    chunked = []
    layer._fused_glu_mul = lambda gate, up: chunked.append(gate.shape) or F.silu(gate) * up
    torch.testing.assert_close(layer._glu_combine_packed(gate_up), F.silu(gate_up[:, :M]) * gate_up[:, M:])
    assert len(packed) == 2 and chunked == [(7, M)]


def test_real_mistral4_module_activation_passes_the_gate():
    config = Mistral4Config(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_routed_experts=E,
        num_experts_per_tok=K,
    )
    assert resolve_fused_glu_mul(Mistral4Experts(config).act_fn) == GluCombine.silu()


def _families_forking_the_combine() -> list[str]:
    return [
        cls.__name__
        for cls in ep_layer_classes()
        if vars(cls).get("HF_MODULE_NAMES") and cls._glu_combine is not EPMoELayerBase._glu_combine
    ]


def test_a_family_owning_its_combine_latches_it_rather_than_forking_the_seam():
    """Overriding ``_glu_combine`` alone misreports the latch, and the packed path would skip the override. GptOss
    is outside the seam entirely."""
    forked = _families_forking_the_combine()
    assert not forked, (
        f"{forked} override _glu_combine instead of latching the combine into _fused_glu_mul, so the "
        f"construction summary reports a combine the layer does not run."
    )
    assert EPGptOssMoELayer._glu_combine_name is not EPMoELayerBase._glu_combine_name


def test_a_combine_inherited_from_an_intermediate_base_is_a_fork_too(monkeypatch):
    def clamped(self, gate, up):
        return gate * up

    monkeypatch.setattr(EPSharedExpertsMoELayerBase, "_glu_combine", clamped, raising=False)
    below = {
        cls.__name__
        for cls in ep_layer_classes()
        if vars(cls).get("HF_MODULE_NAMES") and issubclass(cls, EPSharedExpertsMoELayerBase)
    }
    assert below and set(_families_forking_the_combine()) == below


def test_base_glu_combine_name_tracks_the_latch(monkeypatch):
    """Fused when armed, ``eager`` when not and under ``HALO_FUSED_GLU=0``."""
    layer = EPGemma4MoELayer(_real_gemma4_experts(), single_process_ep_config(E)).cpu()
    monkeypatch.setattr(fused_glu, "_FUSED_GLU_ENABLED", True)
    assert layer._glu_combine_name() == "gelu_tanh"
    layer._fused_glu_mul = GluCombine.clamped_silu(7.0)
    assert layer._glu_combine_name() == "clamped_silu"
    layer._fused_glu_mul = partial(fused_glu.clamped_silu_mul_eager, limit=7.0)
    assert layer._glu_combine_name() == "clamped_silu_mul_eager"

    monkeypatch.setattr(fused_glu, "_FUSED_GLU_ENABLED", False)
    assert layer._glu_combine_name() == "eager"
    layer._fused_glu_mul = GluCombine.gelu_tanh()
    assert layer._glu_combine_name() == "eager"

    monkeypatch.setattr(fused_glu, "_FUSED_GLU_ENABLED", True)
    layer._fused_glu_mul = None
    assert layer._glu_combine_name() == "eager"


def test_gptoss_glu_combine_name_follows_the_switch(monkeypatch):
    layer = object.__new__(EPGptOssMoELayer)
    layer._gptoss_glu = GluCombine.gptoss(1.702, 7.0)
    monkeypatch.setattr(fused_glu, "_FUSED_GLU_ENABLED", True)
    assert layer._glu_combine_name() == "gptoss"
    monkeypatch.setattr(fused_glu, "_FUSED_GLU_ENABLED", False)
    assert layer._glu_combine_name() == "eager"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
