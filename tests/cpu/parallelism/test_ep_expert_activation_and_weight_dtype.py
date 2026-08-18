#!/usr/bin/env python
"""Silent-failure invariants of the EP MoE wrappers, checked against the real HF blocks.

* **The expert activation comes from the wrapped module.** Every family's expert container resolves
  ``ACT2FN[config.hidden_act]``; a wrapper that restates ``F.silu`` instead trains a checkpoint with
  ``hidden_act != "silu"`` on the wrong non-linearity — same shapes, same dtypes, no error.
* **And when the container carries none, the NAME is resolved off the wrapped block.** That fallback
  is one chain in ``_resolve_activation`` (block ``hidden_act`` → block ``config`` → the family
  default), not a per-wrapper spelling: a wrapper that stops handing its block over lands on the SiLU
  default with no error, which is the same silent wrong non-linearity one rung down.
* **The fused SwiGLU kernel is armed exactly when the block gates with SiLU.** GLM-4 Lite / Laguna swap
  one Triton kernel in for the activation and the multiply; disarming it costs only throughput, so
  nothing but a real block at ``hidden_act: silu`` catches a gate that stopped recognizing SiLU.
* **Routing weights reach the dispatch boundary in fp32.** ``_to_topk_weights`` upcasts inside
  ``DeepEPDispatchFunction.forward``, so a bf16 weight tensor still produces the right FORWARD; but
  the backward then hands DeepEP's fp32 ``grad_topk_weights`` back to a bf16 graph input and the
  autograd engine downcasts it silently, so the router-weight gradient of that family alone is
  computed in bf16. Under ETP the same tensor is the operand of ``SumGradAcrossGroup``, whose reduce
  would likewise run in bf16.

Each is checked over a table of families built from their real ``transformers`` classes, so a wrapper
that hard-codes any of these properties fails here rather than in a training run.

Run: ``python tests/cpu/parallelism/test_ep_expert_activation_and_weight_dtype.py`` (or ``pytest -m cpu``).
"""

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

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.distributed.expert_parallel.layers.cohere2_moe import EPCohere2MoELayer
from src.distributed.expert_parallel.layers.gemma4 import EPGemma4MoELayer
from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.layers.inkling import EPInklingMoELayer
from src.distributed.expert_parallel.layers.laguna import EPLagunaMoELayer
from src.distributed.expert_parallel.layers.lfm2 import EPLfm2MoELayer
from src.distributed.expert_parallel.layers.mistral4 import EPMistral4MoELayer
from src.distributed.expert_parallel.layers.qwen3 import EPQwen3MoELayer
from src.distributed.expert_parallel.layers.qwen3_5 import EPQwen3_5MoELayer
from src.kernels.fused_glu import fused_silu_mul, is_silu_activation, silu_mul_eager
from tests.common.parallelism import single_process_ep_config

E, H, M, K = 4, 16, 32, 2  # experts, hidden, moe intermediate, top_k
ACT = "gelu"  # deliberately NOT silu: no wrapper may assume it

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
    """`BailingMoeV2MLP`: Bailing is remote-code, so its per-expert module is mirrored here — including
    the ``act_fn`` attribute the wrapper must read."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, M, bias=False)
        self.up_proj = nn.Linear(H, M, bias=False)
        self.down_proj = nn.Linear(M, H, bias=False)
        self.act_fn = ACT2FN[ACT]


class _BailingGate(nn.Module):
    """`BailingMoeV2Gate`: returns ``(topk_idx, topk_weight, logits)`` with all routing internal."""

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
    """Cohere2's experts are ``intermediate_size`` wide; that field carries M here."""
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
    ("lfm2", EPLfm2MoELayer, _lfm2, "experts", False),  # Lfm2MoeExperts pins F.silu on the container
    ("mistral4", EPMistral4MoELayer, _mistral4, "experts", True),
    ("bailing", EPBailingMoELayer, _BailingBlock, "experts.0", True),
    ("inkling", EPInklingMoELayer, _inkling, "experts", True),
    ("cohere2_moe", EPCohere2MoELayer, _cohere2_moe, "experts", True),
]
_FAMILY_FIELDS = ("name", "wrapper", "factory", "act_owner", "tracks_hidden_act")
_FAMILY_IDS = [family[0] for family in FAMILIES]

# The both-ways matrix needs a factory buildable at a NON-silu hidden_act; only these two take it.
FUSED_SWIGLU_FAMILIES = [
    (name, wrapper, factory)
    for name, wrapper, factory, _owner, _tracks in FAMILIES
    # The factories take ``**overrides``: keying on a NAMED hidden_act param empties this list.
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(factory).parameters.values())
]
# An empty derivation must fail loudly: a zero-case parametrize is a SKIP, which reads green.
assert FUSED_SWIGLU_FAMILIES, "no factory accepts a hidden_act override; the fused-SwiGLU matrix would not run"

# The name fallback's second rung needs a block that KEEPS its config; only some HF blocks do.
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
    """Identity, not equality: the wrapper must hold the very activation object the HF experts use,
    so a checkpoint's ``hidden_act`` can never be overridden by the wrapper's own assumption."""
    block, layer = _build(factory, wrapper)
    expected = _resolve(block, act_owner).act_fn

    assert layer.act_fn is expected, f"{name} did not adopt the wrapped module's activation"
    # Anti-vacuity: families honouring hidden_act must land on the table's non-silu; LFM2 pins F.silu.
    assert (layer.act_fn is not F.silu) is tracks_hidden_act


@pytest.mark.parametrize(_FAMILY_FIELDS, FAMILIES, ids=_FAMILY_IDS)
def test_the_activation_name_fallback_reads_the_wrapped_block(name, wrapper, factory, act_owner, tracks_hidden_act):
    """With no live ``act_fn`` on the expert container, the activation is rebuilt from the BLOCK's own
    ``hidden_act`` — for every family, not just the one wrapper that spelled that probe itself.

    This is the rung an upstream rename lands on: the container moves or loses ``act_fn``, and the
    only thing between the run and a silent SiLU is this resolution. A wrapper that resolves the name
    itself (or stops handing its block to ``_resolve_activation``) drops to the default instead, and
    the whole expert stack trains on the wrong non-linearity with no error.
    """

    def strip_container_activation(block):
        delattr(_resolve(block, act_owner), "act_fn")
        # A third activation: distinct from the table's ``ACT`` (which every config here carries) and
        # from the ``silu`` default, so neither a config read nor the default can fake this pass.
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
    """Second rung of the same chain: a block carrying no ``hidden_act`` of its own must resolve the
    name from its ``config`` rather than dropping to the SiLU default."""
    _block, layer = _build(factory, wrapper, prepare=lambda block: delattr(_resolve(block, act_owner), "act_fn"))

    assert type(layer.act_fn) is type(ACT2FN[ACT]), f"{name} ignored config.hidden_act={ACT!r}"


def test_gemma4_activation_name_fallback_reads_the_wrapped_experts_module():
    """Gemma 4 is off the table above — its wrapped block IS the expert container and its forward
    takes the router outputs — but it runs the same chain, so the block rung must still be handed the
    block. A positional-only source leaves ``original_layer=None`` and drops straight to the default.
    """
    experts = Gemma4TextExperts(
        Gemma4TextConfig(hidden_size=H, moe_intermediate_size=M, num_experts=E, top_k_experts=K, hidden_activation=ACT)
    )
    delattr(experts, "act_fn")
    # A third activation, as above: distinct from the config's ``hidden_activation`` and from the
    # gelu_pytorch_tanh default, so neither can fake this pass.
    experts.hidden_act = "relu"

    layer = EPGemma4MoELayer(experts, single_process_ep_config(E)).cpu()

    assert type(layer.act_fn) is type(ACT2FN["relu"]), "gemma4 ignored the wrapped block's own hidden_act"
    assert layer._fused_glu_mul is None, "gemma4 armed the SiLU kernel for a ReLU gate"


@pytest.mark.parametrize(_FAMILY_FIELDS, FAMILIES, ids=_FAMILY_IDS)
def test_every_silu_family_arms_the_fused_kernel(name, wrapper, factory, act_owner, tracks_hidden_act):
    """A SiLU-gated family must get the fused kernel — ALL of them, not the one that wired it.

    A fused branch living on a single family's ``_glu_combine`` override leaves the other six SiLU
    families silently running the eager activation plus a separate multiply (and the separate
    autograd chain) on every base compute path, per MoE layer, per step. It is numerically invisible
    — pure lost throughput — which is exactly why it survives unnoticed.

    The latch is DERIVED in ``_resolve_activation`` from the resolved activation, so what this pins
    across the whole roster is that correspondence, not a hand-kept list. The blocks here are
    built at ``hidden_act: gelu`` on purpose (see ``ACT``), so most families assert the disarmed
    direction and LFM2 — whose upstream pins ``F.silu`` regardless — asserts the armed one.
    """
    _block, layer = _build(factory, wrapper)
    expected = fused_silu_mul if is_silu_activation(layer.act_fn) else None

    assert layer._fused_glu_mul is expected, (
        f"{name}: act_fn={layer.act_fn!r} wanted {expected} but latched {layer._fused_glu_mul}"
    )


@pytest.mark.parametrize(("hidden_act", "fused"), [("silu", True), ("gelu", False)])
@pytest.mark.parametrize(
    ("name", "wrapper", "factory"), FUSED_SWIGLU_FAMILIES, ids=[f[0] for f in FUSED_SWIGLU_FAMILIES]
)
def test_the_fused_swiglu_kernel_is_armed_exactly_when_the_block_gates_with_silu(
    name, wrapper, factory, hidden_act, fused, monkeypatch
):
    """`_fused_glu_mul` decides whether one Triton kernel replaces the activation and the multiply on
    every base compute path. Getting it wrong the safe way is numerically invisible — pure lost
    throughput — so nothing but a real block built at ``hidden_act: silu`` can hold the line, and the
    check has to be behavioural: the block's `ACT2FN["silu"]` is neither `nn.SiLU` nor `F.silu`."""
    torch.manual_seed(0)
    layer = wrapper(factory(hidden_act=hidden_act), single_process_ep_config(E)).cpu()

    expected = fused_silu_mul if fused else None
    assert layer._fused_glu_mul is expected, f"{name} at hidden_act={hidden_act!r} armed the wrong combine"
    # Why the gate cannot be nominal: this holds for the accepted row too.
    assert not isinstance(layer.act_fn, nn.SiLU) and layer.act_fn is not F.silu

    calls: list[torch.Size] = []

    def spy(gate, up):
        calls.append(gate.shape)
        return silu_mul_eager(gate, up)

    # The fused branch dispatches through the latched attribute on the BASE seam, one call site.
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
    """Every family must hand ``_dispatch_compute_combine`` fp32 weights even when the block runs in
    bf16 — the tensor is a graph input of the DeepEP dispatch, whose backward returns fp32 grads."""
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
