#!/usr/bin/env python
"""Fused-GLU activation gates must accept the activations real checkpoints carry.

Transformers MoE modules hold ``ACT2FN`` instances (``SiLUActivation``, ``GELUTanh``) — not
``nn.SiLU``/``nn.GELU`` — so a gate written against the torch classes silently degrades every
production run to the eager combine while the parity tests (stubbed with ``F.silu``) stay green.
:func:`resolve_fused_glu_mul` therefore probes BEHAVIOUR, and what this pins is that the probe
answers correctly for the objects a real block hands it — plus the wiring that turns that verdict
into the kernel a layer actually runs.

    python tests/cpu/parallelism/test_fused_glu_activation_gates.py
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.layers.gemma4 import EPGemma4MoELayer
from src.kernels.fused_glu import (
    fused_gelu_tanh_mul,
    fused_silu_mul,
    is_gelu_tanh_activation,
    is_silu_activation,
    resolve_fused_glu_mul,
)

E, K, H, M = 4, 2, 8, 16


def test_silu_gate_accepts_real_checkpoint_activations():
    assert is_silu_activation(ACT2FN["silu"])  # what HF MoE modules actually carry
    assert is_silu_activation(F.silu)
    assert is_silu_activation(nn.SiLU())
    assert not is_silu_activation(ACT2FN["gelu"])
    assert not is_silu_activation(ACT2FN["gelu_pytorch_tanh"])


def test_gelu_tanh_gate_accepts_real_checkpoint_activations():
    assert is_gelu_tanh_activation(ACT2FN["gelu_pytorch_tanh"])
    assert is_gelu_tanh_activation(nn.GELU(approximate="tanh"))
    assert not is_gelu_tanh_activation(nn.GELU())  # exact GELU is not the tanh approximation
    assert not is_gelu_tanh_activation(ACT2FN["silu"])


def test_resolver_maps_each_activation_to_its_own_kernel():
    """The two probes feed ONE resolver, so a gate that answers right but is wired to the wrong
    kernel — tanh-GELU dispatched to the SiLU kernel — would change the activation silently."""
    assert resolve_fused_glu_mul(ACT2FN["silu"]) is fused_silu_mul
    assert resolve_fused_glu_mul(ACT2FN["gelu_pytorch_tanh"]) is fused_gelu_tanh_mul
    assert resolve_fused_glu_mul(ACT2FN["gelu"]) is None  # exact GELU: no kernel implements it
    assert resolve_fused_glu_mul(ACT2FN["relu"]) is None
    assert resolve_fused_glu_mul("not-callable") is None


def _real_gemma4_experts() -> Gemma4TextExperts:
    """The genuine ``Gemma4TextExperts`` — the object the layer meets in production.

    A stand-in declares its own ``act_fn`` and its own ``down_proj`` orientation, so it certifies the
    test's own assumptions rather than the library's: the real module builds ``act_fn`` from
    ``config.hidden_activation`` (a ``GELUTanh`` *instance*, not ``nn.GELU``) and stores ``down_proj``
    as ``[E, H, M]``. Its parameters are ``torch.empty`` — initialize them so no uninitialized NaN
    reaches the fused-GLU latch.
    """
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
    """End of the wire: the layer must latch the tanh-GeGLU kernel, not merely recognize the gate."""
    config = EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False)
    config.finalize_expert_assignment(E)
    layer = EPGemma4MoELayer(_real_gemma4_experts(), config).cpu()
    assert layer._fused_glu_mul is fused_gelu_tanh_mul


def test_gemma4_fused_combine_matches_the_eager_activation():
    """The latched kernel must compute what ``act_fn(gate) * up`` computes. The CPU path is eager,
    so this pins the WIRING; ``tests/gpu/kernels/test_fused_glu.py`` pins the Triton kernel."""
    config = EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False)
    config.finalize_expert_assignment(E)
    layer = EPGemma4MoELayer(_real_gemma4_experts(), config).cpu()

    gate, up = torch.randn(7, M), torch.randn(7, M)
    expected = F.gelu(gate, approximate="tanh") * up
    torch.testing.assert_close(layer._glu_combine(gate, up), expected)


def test_real_mistral4_module_activation_passes_the_gate():
    """The exact production object: ``Mistral4Experts.act_fn`` is a ``SiLUActivation`` instance."""
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
    from transformers.models.mistral4.modeling_mistral4 import Mistral4Experts

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
    assert resolve_fused_glu_mul(Mistral4Experts(config).act_fn) is fused_silu_mul


def test_a_family_owning_its_combine_latches_it_rather_than_forking_the_seam():
    """``_glu_combine_name()`` must name what the layer ACTUALLY runs.

    A family with a clamped SwiGLU (DeepSeek-V4, GLM-5 Next, Step-3.7) binds its own combine INTO the
    latch, so one declaration drives both the compute and the reported name. Overriding
    ``_glu_combine`` alone instead would let the summary report the latch (``eager``) for exactly the
    families that are not eager; overriding both lets the two disagree. GptOss is the one family
    outside the seam entirely — its interleaved-bias paths never call ``_glu_combine``.
    """
    from src.distributed.expert_parallel.base_layer import EPMoELayerBase
    from src.distributed.expert_parallel.expert_weights import ep_layer_classes
    from src.distributed.expert_parallel.layers.gpt_oss import EPGptOssMoELayer

    forked = [
        cls.__name__
        for cls in ep_layer_classes()
        if vars(cls).get("HF_MODULE_NAMES") and "_glu_combine" in vars(cls) and cls is not EPGptOssMoELayer
    ]
    assert not forked, (
        f"{forked} override _glu_combine instead of latching the combine into _fused_glu_mul, so the "
        f"construction summary reports a combine the layer does not run."
    )
    assert EPGptOssMoELayer._glu_combine_name is not EPMoELayerBase._glu_combine_name


def test_base_glu_combine_name_tracks_the_latch():
    """Keeps the override test above honest: for a family that DOES use the base seam, the reported
    name must follow the latch — fused when armed, ``eager`` when not."""
    config = EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False)
    config.finalize_expert_assignment(E)
    layer = EPGemma4MoELayer(_real_gemma4_experts(), config).cpu()
    assert layer._glu_combine_name() == fused_gelu_tanh_mul.__name__

    layer._fused_glu_mul = None
    assert layer._glu_combine_name() == "eager"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
