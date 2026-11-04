#!/usr/bin/env python
"""Silent-failure invariants of the EP layers against real HF blocks: expert activation, the fused SwiGLU latch,
fp32 routing weights. A restated ``F.silu`` trains the wrong non-linearity."""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.cohere2_moe.configuration_cohere2_moe import Cohere2MoeConfig
from transformers.models.cohere2_moe.modeling_cohere2_moe import Cohere2MoeSparseMoeBlock
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteMoE
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingMoE
from transformers.models.laguna.configuration_laguna import LagunaConfig
from transformers.models.laguna.modeling_laguna import LagunaSparseMoeBlock
from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig
from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeSparseMoeBlock
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from halo.distributed.expert_parallel.families.bailing import EPBailingMoELayer
from halo.distributed.expert_parallel.families.cohere2_moe import EPCohere2MoELayer
from halo.distributed.expert_parallel.families.gemma4 import EPGemma4MoELayer
from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.distributed.expert_parallel.families.inkling import EPInklingMoELayer
from halo.distributed.expert_parallel.families.laguna import EPLagunaMoELayer
from halo.distributed.expert_parallel.families.lfm2 import EPLfm2MoELayer
from halo.distributed.expert_parallel.families.mistral4 import EPMistral4MoELayer
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.kernels.fused_glu import GluCombine, is_silu_activation, silu_mul_eager
from tests.common.parallelism import single_process_ep_config

E, H, M, K = 4, 16, 32, 2  # experts, hidden, moe intermediate, top_k
ACT = "gelu"  # Deliberately not silu: no wrapper may assume it.

_COMMON = {
    "hidden_size": H,
    "intermediate_size": 4 * H,
    "moe_intermediate_size": M,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 32,
    "hidden_act": ACT,
}


class _BailingExpertMLP(nn.Module):
    """Bailing is remote-code, so `BailingMoeV2MLP` is mirrored here, including the ``act_fn`` the wrapper must
    read."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, M, bias=False)
        self.up_proj = nn.Linear(H, M, bias=False)
        self.down_proj = nn.Linear(M, H, bias=False)
        self.act_fn = ACT2FN[ACT]


class _BailingGate(nn.Module):
    """`BailingMoeV2Gate` returns ``(topk_idx, topk_weight, logits)`` with all routing internal."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H))
        self.top_k = K
        self.num_experts = E
        self.routed_scaling_factor = 1.0

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight).float()
        scores = logits.sigmoid()
        weights, indices = torch.topk(scores, K, dim=-1)
        return indices, weights.to(hidden_states.dtype), logits


class _BailingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _BailingGate()
        self.experts = nn.ModuleList(_BailingExpertMLP() for _ in range(E))
        self.shared_experts = None


def _qwen3():
    return Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(num_experts=E, num_experts_per_tok=K, **_COMMON))


def _qwen3_5():
    return Qwen3_5MoeSparseMoeBlock(
        Qwen3_5MoeConfig(num_experts=E, num_experts_per_tok=K, shared_expert_intermediate_size=M, **_COMMON)
    )


def _glm4_lite(**overrides):
    return Glm4MoeLiteMoE(
        Glm4MoeLiteConfig(
            n_routed_experts=E,
            num_local_experts=E,
            num_experts_per_tok=K,
            n_group=1,
            topk_group=1,
            n_shared_experts=1,
            **{**_COMMON, **overrides},
        )
    )


def _laguna(**overrides):
    return LagunaSparseMoeBlock(
        LagunaConfig(
            num_experts=E,
            num_experts_per_tok=K,
            shared_expert_intermediate_size=M,
            **{**_COMMON, **overrides},
        )
    )


def _lfm2():
    return Lfm2MoeSparseMoeBlock(Lfm2MoeConfig(num_experts=E, num_experts_per_tok=K, **_COMMON))


def _inkling(**overrides):
    return InklingMoE(
        InklingTextConfig(
            n_routed_experts=E,
            num_experts_per_tok=K,
            n_shared_experts=1,
            **{**_COMMON, **overrides},
        )
    )


def _mistral4():
    return Mistral4MoE(
        Mistral4Config(
            n_routed_experts=E,
            num_experts_per_tok=K,
            n_group=1,
            topk_group=1,
            n_shared_experts=1,
            **_COMMON,
        )
    )


def _cohere2_moe(**overrides):
    """Its experts are ``intermediate_size`` wide; that field carries M here."""
    config = {key: value for key, value in _COMMON.items() if key != "moe_intermediate_size"}
    return Cohere2MoeSparseMoeBlock(
        Cohere2MoeConfig(
            num_experts=E,
            num_experts_per_tok=K,
            num_shared_experts=1,
            **{**config, "intermediate_size": M, **overrides},
        )
    )


FAMILIES = [
    ("qwen3", EPQwen3MoELayer, _qwen3, "experts", True),
    ("qwen3_5", EPQwen3_5MoELayer, _qwen3_5, "experts", True),
    ("glm4_moe_lite", EPGlm4MoELayer, _glm4_lite, "experts", True),
    ("laguna", EPLagunaMoELayer, _laguna, "experts", True),
    ("lfm2", EPLfm2MoELayer, _lfm2, "experts", False),  # Lfm2MoeExperts pins F.silu on the container.
    ("mistral4", EPMistral4MoELayer, _mistral4, "experts", True),
    ("bailing", EPBailingMoELayer, _BailingBlock, "experts.0", True),
    ("inkling", EPInklingMoELayer, _inkling, "experts", True),
    ("cohere2_moe", EPCohere2MoELayer, _cohere2_moe, "experts", True),
]
_FAMILY_FIELDS = ("name", "wrapper", "factory", "act_owner", "tracks_hidden_act")
_FAMILY_IDS = [family[0] for family in FAMILIES]

FUSED_SWIGLU_FAMILIES = [
    (name, wrapper, factory)
    for name, wrapper, factory, _owner, _tracks in FAMILIES
    # Keying on a named hidden_act param would empty this list.
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(factory).parameters.values())
]
# A zero-case parametrize is a skip, which reads green.
assert FUSED_SWIGLU_FAMILIES, "no factory accepts a hidden_act override; the fused-SwiGLU matrix would not run"

# Only some HF blocks keep their config.
CONFIG_CARRYING_FAMILIES = [
    (name, wrapper, factory, act_owner)
    for name, wrapper, factory, act_owner, _tracks in FAMILIES
    if getattr(factory(), "config", None) is not None
]
assert CONFIG_CARRYING_FAMILIES, "no HF block keeps a config; the config rung of the fallback would not be exercised"


def _resolve(module: nn.Module, path: str):
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _build(factory, wrapper, dtype=torch.float32, prepare=None):
    torch.manual_seed(0)
    block = factory().to(dtype)
    for parameter in block.parameters():
        nn.init.normal_(parameter, std=0.2)
    if prepare is not None:
        prepare(block)
    return block, wrapper(block, single_process_ep_config(E)).cpu()


@pytest.mark.parametrize(_FAMILY_FIELDS, FAMILIES, ids=_FAMILY_IDS)
def test_expert_activation_is_read_from_the_wrapped_module(name, wrapper, factory, act_owner, tracks_hidden_act):
    """Identity, so a checkpoint's ``hidden_act`` can never be overridden by the wrapper's own assumption."""
    block, layer = _build(factory, wrapper)
    expected = _resolve(block, act_owner).act_fn

    assert layer.act_fn is expected, f"{name} did not adopt the wrapped module's activation"
    # Anti-vacuity: LFM2 pins F.silu; the others land on the table's non-silu.
    assert (layer.act_fn is not F.silu) is tracks_hidden_act


@pytest.mark.parametrize(_FAMILY_FIELDS, FAMILIES, ids=_FAMILY_IDS)
def test_the_activation_name_fallback_reads_the_wrapped_block(name, wrapper, factory, act_owner, tracks_hidden_act):
    """An upstream rename lands here: a wrapper that stops handing its block to ``_resolve_activation`` raises."""

    def strip_container_activation(block):
        delattr(_resolve(block, act_owner), "act_fn")
        # Distinct from ``ACT``, so a config read cannot fake this pass.
        block.hidden_act = "relu"

    _block, layer = _build(factory, wrapper, prepare=strip_container_activation)

    assert type(layer.act_fn) is type(ACT2FN["relu"]), f"{name} ignored the block's own hidden_act"
    assert layer._fused_glu_mul is None, f"{name} armed the SiLU kernel for a ReLU gate"


@pytest.mark.parametrize(
    ("name", "wrapper", "factory", "act_owner"),
    CONFIG_CARRYING_FAMILIES,
    ids=[family[0] for family in CONFIG_CARRYING_FAMILIES],
)
def test_the_activation_name_fallback_falls_through_to_the_block_config(name, wrapper, factory, act_owner):
    _block, layer = _build(factory, wrapper, prepare=lambda block: delattr(_resolve(block, act_owner), "act_fn"))

    assert type(layer.act_fn) is type(ACT2FN[ACT]), f"{name} ignored config.hidden_act={ACT!r}"


def test_gemma4_activation_name_fallback_reads_the_wrapped_experts_module():
    """Off the table above: its block is the expert container and its forward takes the router outputs."""
    experts = Gemma4TextExperts(
        Gemma4TextConfig(hidden_size=H, moe_intermediate_size=M, num_experts=E, top_k_experts=K, hidden_activation=ACT)
    )
    delattr(experts, "act_fn")
    # Distinct from the config's ``hidden_activation`` and the gelu_pytorch_tanh default, so neither can fake this
    # pass.
    experts.hidden_act = "relu"

    layer = EPGemma4MoELayer(experts, single_process_ep_config(E)).cpu()

    assert type(layer.act_fn) is type(ACT2FN["relu"]), "gemma4 ignored the wrapped block's own hidden_act"
    assert layer._fused_glu_mul is None, "gemma4 armed the SiLU kernel for a ReLU gate"


@pytest.mark.parametrize(_FAMILY_FIELDS, FAMILIES, ids=_FAMILY_IDS)
def test_every_silu_family_arms_the_fused_kernel(name, wrapper, factory, act_owner, tracks_hidden_act):
    """Numerically invisible when lost, pure lost throughput. Blocks are built at ``hidden_act: gelu``, so most
    assert disarmed and LFM2, pinned to ``F.silu``, armed."""
    _block, layer = _build(factory, wrapper)
    expected = GluCombine.silu() if is_silu_activation(layer.act_fn) else None

    assert layer._fused_glu_mul == expected, (
        f"{name}: act_fn={layer.act_fn!r} wanted {expected} but latched {layer._fused_glu_mul}"
    )


@pytest.mark.parametrize(("hidden_act", "fused"), [("silu", True), ("gelu", False)])
@pytest.mark.parametrize(
    ("name", "wrapper", "factory"), FUSED_SWIGLU_FAMILIES, ids=[f[0] for f in FUSED_SWIGLU_FAMILIES]
)
def test_the_fused_swiglu_kernel_is_armed_exactly_when_the_block_gates_with_silu(
    name, wrapper, factory, hidden_act, fused, monkeypatch
):
    """The check is behavioural: the block's `ACT2FN["silu"]` is neither `nn.SiLU` nor `F.silu`."""
    torch.manual_seed(0)
    layer = wrapper(factory(hidden_act=hidden_act), single_process_ep_config(E)).cpu()

    expected = GluCombine.silu() if fused else None
    assert layer._fused_glu_mul == expected, f"{name} at hidden_act={hidden_act!r} armed the wrong combine"
    # Why the gate cannot be nominal: this holds for the accepted row too.
    assert not isinstance(layer.act_fn, nn.SiLU) and layer.act_fn is not F.silu

    calls: list[torch.Size] = []

    def spy(gate, up):
        calls.append(gate.shape)
        return silu_mul_eager(gate, up)

    if layer._fused_glu_mul is not None:
        monkeypatch.setattr(layer, "_fused_glu_mul", spy)
    gate, up = torch.randn(6, M), torch.randn(6, M)
    out = layer._glu_combine(gate, up)

    # The flag is only worth pinning if it routes the compute.
    assert calls == ([gate.shape] if fused else [])
    silu_gated = F.silu(gate) * up
    if fused:
        assert torch.allclose(out, silu_gated)
    else:
        assert not torch.allclose(out, silu_gated)


@pytest.mark.parametrize(_FAMILY_FIELDS, FAMILIES, ids=_FAMILY_IDS)
def test_routing_weights_reach_the_dispatch_boundary_in_fp32(
    name, wrapper, factory, act_owner, tracks_hidden_act, monkeypatch
):
    """The tensor is a graph input of the DeepEP dispatch, whose backward returns fp32 grads; a bf16 input silently
    downcasts them."""
    _block, layer = _build(factory, wrapper, dtype=torch.bfloat16)
    seen: list[torch.dtype] = []

    original = EPMoELayerBase._dispatch_compute_combine

    def spy(self, flat, experts, weights, input_dtype):
        seen.append(weights.dtype)
        return original(self, flat, experts, weights, input_dtype)

    monkeypatch.setattr(EPMoELayerBase, "_dispatch_compute_combine", spy)
    layer(torch.randn(2, 5, H, dtype=torch.bfloat16))

    assert seen, f"{name} never reached the dispatch boundary"
    assert seen == [torch.float32] * len(seen), f"{name} dispatched {seen[0]} routing weights"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
