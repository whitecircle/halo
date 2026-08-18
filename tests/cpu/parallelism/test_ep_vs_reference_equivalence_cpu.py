#!/usr/bin/env python
"""At ``ep_size=1`` every EP wrapper must compute exactly what the upstream block computes.

``ep_size=1`` leaves the DeepEP dispatcher inert, so the wrapper's forward is its own re-derivation
of the family's MoE: it re-reads the gate, re-applies the family's normalization / routed scaling /
activation, permutes tokens, runs the fused expert GEMMs and scatters the result back. Every step
there is a place a family can silently disagree with its checkpoint — a dropped ``norm_topk_prob``,
a routed-scaling factor resolved to its default, a shared expert added on the wrong side, an
activation restated instead of adopted. None of those changes a shape, a dtype or a key; they change
the numbers, and a run trains happily on the wrong function.

So the gate is numeric equivalence against the REAL ``transformers`` block (never a stand-in, which
would only certify the test's own assumptions), over the roster of families whose wrapper owns a
re-derived forward. ``EPGemma4MoELayer`` wraps ``Gemma4TextExperts`` rather than a block — Gemma4
inlines MoE in its decoder layer — so it is compared at that seam, with the router decision supplied.

That roster is derived from ``ep_layer_classes()`` rather than kept by hand
(``test_every_ep_family_with_a_re_derived_forward_is_compared``), so a family cannot join the toolkit
with its forward uncompared; the two families this file cannot host are named in ``_UNCOMPARABLE``
with the reason.

The band is RELATIVE to the reference's own magnitude, because both floors are: fp32 bottoms out at
float epsilon times the output scale (the two paths accumulate the same terms in a different order),
and fp64 bottoms out around 1e-7 relative for the families whose router hands out fp64 routing
weights — the wrapper casts those to fp32 by contract at the dispatch boundary, so fp64 is the
LOOSER of the two, not the tighter. Probed worst case across families, dtypes and seeds is ~3.1e-7
relative; a real disagreement (a dropped normalization, a defaulted scaling factor) moves the output
by ~1e-1 relative, and even swapping tanh-GELU for exact GELU moves it ~80x the band. ``test_the_comparison_is_sensitive_to_the_expert_weights`` pins that the
band really does sit far below a weight-level change rather than merely being wide.

Run: ``python tests/cpu/parallelism/test_ep_vs_reference_equivalence_cpu.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from transformers.models.cohere2_moe.configuration_cohere2_moe import Cohere2MoeConfig
from transformers.models.cohere2_moe.modeling_cohere2_moe import Cohere2MoeSparseMoeBlock
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteMoE
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
from transformers.models.inkling.configuration_inkling import InklingTextConfig
from transformers.models.inkling.modeling_inkling import InklingMoE
from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig
from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeSparseMoeBlock
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from transformers.models.step3p7.configuration_step3p7 import Step3p7TextConfig
from transformers.models.step3p7.modeling_step3p7 import Step3p7SparseMoeBlock
from transformers.models.zaya.configuration_zaya import ZayaConfig
from transformers.models.zaya.modeling_zaya import ZayaSparseMoeBlock

from src.distributed.expert_parallel.expert_weights import ep_layer_classes
from src.distributed.expert_parallel.layers.cohere2_moe import EPCohere2MoELayer
from src.distributed.expert_parallel.layers.deepseek_v4 import EPDeepseekV4MoELayer
from src.distributed.expert_parallel.layers.gemma4 import EPGemma4MoELayer
from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.layers.glm5_next import EPGlm5NextMoELayer
from src.distributed.expert_parallel.layers.gpt_oss import EPGptOssMoELayer
from src.distributed.expert_parallel.layers.inkling import EPInklingMoELayer
from src.distributed.expert_parallel.layers.lfm2 import EPLfm2MoELayer
from src.distributed.expert_parallel.layers.mistral4 import EPMistral4MoELayer
from src.distributed.expert_parallel.layers.qwen3 import EPQwen3MoELayer
from src.distributed.expert_parallel.layers.qwen3_5 import EPQwen3_5MoELayer
from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from src.distributed.expert_parallel.layers.zaya import EPZayaMoELayer
from src.models.moe_balancing import resolve_balancing_slot
from tests.common.parallelism import single_process_ep_config

E, H, M, K = 8, 16, 32, 3  # experts, hidden, moe intermediate, top_k
TOKENS = (2, 5)  # batch, sequence
# ZayaConfig refuses anything else at construction, so Zaya is compared at its own top_k.
ZAYA_K = 1

# Non-neutral on purpose: at the defaults a wrapper that resolved either knob to its default
# would still match the block, and the whole file would be moot.
_ROUTING_KNOBS = {"norm_topk_prob": True, "routed_scaling_factor": 2.5}

# GLM-5 Next's clamped-SwiGLU bound, set well INSIDE the fixture's activation scale (~0.8 std at
# these shapes; the family default 10.0 never clamps here) so a wrapper that dropped the clamp
# diverges instead of matching the block on in-range activations.
_GLM5_SWIGLU_LIMIT = 0.5

# Step-3.7's per-layer clamp bounds, both inside the activation scale for the same reason. The
# routed and shared bounds differ so a wrapper that read the wrong list still diverges.
_STEP3P7_SWIGLU_LIMIT = 0.5
_STEP3P7_SHARED_SWIGLU_LIMIT = 0.4

_COMMON = {
    "hidden_size": H,
    "intermediate_size": 4 * H,
    "moe_intermediate_size": M,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 32,
    "hidden_act": "silu",
}

# Fraction of the reference's own max magnitude the two paths may differ by: ~6x the measured
# worst case (3.1e-7) and ~80x below the smallest real disagreement probed here.
_RELATIVE_BAND = 2e-6
# A 1% move on the expert weights must clear the band by orders of magnitude, or the band is vacuous.
_SENSITIVITY_SCALE = 1.01
_SENSITIVITY_FLOOR = 1e-3


def _qwen3_moe() -> Qwen3MoeSparseMoeBlock:
    return Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(num_experts=E, num_experts_per_tok=K, **_COMMON))


def _qwen3_5_moe() -> Qwen3_5MoeSparseMoeBlock:
    return Qwen3_5MoeSparseMoeBlock(
        Qwen3_5MoeConfig(num_experts=E, num_experts_per_tok=K, shared_expert_intermediate_size=M, **_COMMON)
    )


def _glm4_moe_lite() -> Glm4MoeLiteMoE:
    return Glm4MoeLiteMoE(
        Glm4MoeLiteConfig(
            n_routed_experts=E,
            num_local_experts=E,
            num_experts_per_tok=K,
            n_group=1,
            topk_group=1,
            n_shared_experts=1,
            **_ROUTING_KNOBS,
            **_COMMON,
        )
    )


def _glm5_next() -> Glm5NextTextMoE:
    """``num_key_value_heads`` must equal ``num_attention_heads`` (the config's own architecture
    validation), and the default ``mlp_layer_types`` would leave a model this shallow all-dense —
    neither touches the standalone MoE block built here, but the config refuses to build otherwise."""
    return Glm5NextTextMoE(
        Glm5NextTextConfig(
            n_routed_experts=E,
            num_experts_per_tok=K,
            n_group=1,
            topk_group=1,
            n_shared_experts=1,
            swiglu_limit=_GLM5_SWIGLU_LIMIT,
            **_ROUTING_KNOBS,
            **{**_COMMON, "num_key_value_heads": _COMMON["num_attention_heads"]},
        )
    )


def _lfm2_moe() -> Lfm2MoeSparseMoeBlock:
    return Lfm2MoeSparseMoeBlock(
        Lfm2MoeConfig(num_experts=E, num_experts_per_tok=K, use_expert_bias=True, **_ROUTING_KNOBS, **_COMMON)
    )


def _mistral4() -> Mistral4MoE:
    return Mistral4MoE(
        Mistral4Config(
            n_routed_experts=E,
            num_experts_per_tok=K,
            n_group=1,
            topk_group=1,
            n_shared_experts=1,
            **_ROUTING_KNOBS,
            **_COMMON,
        )
    )


def _gemma4_experts() -> Gemma4TextExperts:
    text_config = {key: value for key, value in _COMMON.items() if key != "hidden_act"}
    return Gemma4TextExperts(Gemma4TextConfig(num_experts=E, num_experts_per_tok=K, **text_config))


def _gpt_oss() -> GptOssMLP:
    """GptOss has no ``moe_intermediate_size``: its experts are ``intermediate_size`` wide, so that
    field carries M here rather than the dense width every other family puts in it."""
    config = {key: value for key, value in _COMMON.items() if key != "moe_intermediate_size"}
    return GptOssMLP(
        GptOssConfig(**{**config, "intermediate_size": M, "num_local_experts": E, "num_experts_per_tok": K})
    )


def _zaya() -> ZayaSparseMoeBlock:
    """``router_hidden_size`` is the width of the cross-layer EDA state the gate threads along; it is
    small here for the same reason everything else is. ``layer_idx=0`` takes the no-incoming-state
    branch, which is the shape ``_block_inputs`` feeds."""
    config = {key: value for key, value in _COMMON.items() if key != "intermediate_size"}
    return ZayaSparseMoeBlock(
        ZayaConfig(num_experts=E, num_experts_per_tok=ZAYA_K, router_hidden_size=8, **config), layer_idx=0
    )


def _deepseek_v4() -> DeepseekV4SparseMoeBlock:
    """``mlp_layer_types=["moe"]`` selects the top-k router. The hash variant is out of scope here: it
    routes off ``tid2eid[input_ids]``, which the block only receives from its decoder layer."""
    return DeepseekV4SparseMoeBlock(
        DeepseekV4Config(
            n_routed_experts=E,
            num_experts_per_tok=K,
            n_shared_experts=1,
            mlp_layer_types=["moe"],
            **_ROUTING_KNOBS,
            **_COMMON,
        ),
        layer_idx=0,
    )


def _cohere2_moe() -> Cohere2MoeSparseMoeBlock:
    """Cohere2's experts are ``intermediate_size`` wide (no ``moe_intermediate_size``), so that field
    carries M here. Sigmoid selection with top-k renorm plus the AVERAGED shared expert (0.5 output
    scale) exercise the wrapper's own gating and combination — a wrapper defaulting the strategy to
    ``sum`` would diverge by 2x."""
    config = {key: value for key, value in _COMMON.items() if key != "moe_intermediate_size"}
    return Cohere2MoeSparseMoeBlock(
        Cohere2MoeConfig(
            **{**config, "intermediate_size": M},
            num_experts=E,
            num_experts_per_tok=K,
            num_shared_experts=2,
            shared_expert_combination_strategy="average",
            expert_selection_fn="sigmoid",
            norm_topk_prob=True,
        )
    )


def _inkling() -> InklingMoE:
    """``n_shared_experts >= 1`` is required by the wrapper (its ``_route`` slices the last shared
    logit columns), and > 1 is what the joint routed+shared normalisation is actually about."""
    return InklingMoE(
        InklingTextConfig(n_routed_experts=E, num_experts_per_tok=K, n_shared_experts=2, route_scale=8.0, **_COMMON)
    )


def _step3p7() -> Step3p7SparseMoeBlock:
    """The block reads its per-layer clamp bounds (``swiglu_limits`` / ``swiglu_limits_shared``) and
    ``moe_router_scaling_factor`` off the config at ``layer_idx`` — all non-neutral here so a wrapper
    that dropped any of them diverges. The family has no ``norm_topk_prob`` knob (renormalization is
    unconditional) and no group limiting."""
    return Step3p7SparseMoeBlock(
        Step3p7TextConfig(
            n_routed_experts=E,
            num_experts_per_tok=K,
            share_expert_dim=M,
            moe_router_scaling_factor=_ROUTING_KNOBS["routed_scaling_factor"],
            swiglu_limits=[_STEP3P7_SWIGLU_LIMIT],
            swiglu_limits_shared=[_STEP3P7_SHARED_SWIGLU_LIMIT],
            mlp_layer_types=["sparse"],
            **_COMMON,
        ),
        layer_idx=0,
    )


def _block_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Whole-block families take 3-D hidden states and route internally."""
    generator = torch.Generator().manual_seed(11)
    return (torch.randn(*TOKENS, H, dtype=dtype, generator=generator),)


def _experts_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """``Gemma4TextExperts`` takes flat tokens plus the sibling router's decision."""
    generator = torch.Generator().manual_seed(11)
    count = TOKENS[0] * TOKENS[1]
    weights, index = torch.rand(count, E, generator=generator).topk(K, dim=-1)
    return (torch.randn(count, H, dtype=dtype, generator=generator), index, weights.to(dtype))


FAMILIES = [
    ("glm4_moe_lite", EPGlm4MoELayer, _glm4_moe_lite, _block_inputs),
    ("glm5_next", EPGlm5NextMoELayer, _glm5_next, _block_inputs),
    ("lfm2_moe", EPLfm2MoELayer, _lfm2_moe, _block_inputs),
    ("mistral4", EPMistral4MoELayer, _mistral4, _block_inputs),
    ("gemma4", EPGemma4MoELayer, _gemma4_experts, _experts_inputs),
    ("qwen3_moe", EPQwen3MoELayer, _qwen3_moe, _block_inputs),
    ("qwen3_5_moe", EPQwen3_5MoELayer, _qwen3_5_moe, _block_inputs),
    ("gpt_oss", EPGptOssMoELayer, _gpt_oss, _block_inputs),
    ("zaya", EPZayaMoELayer, _zaya, _block_inputs),
    ("deepseek_v4", EPDeepseekV4MoELayer, _deepseek_v4, _block_inputs),
    ("inkling", EPInklingMoELayer, _inkling, _block_inputs),
    ("cohere2_moe", EPCohere2MoELayer, _cohere2_moe, _block_inputs),
    ("step3p7", EPStep3p7MoELayer, _step3p7, _block_inputs),
]
_FIELDS = ("family", "wrapper", "factory", "inputs")
_IDS = [family[0] for family in FAMILIES]
_DTYPE_IDS = {torch.float32: "fp32", torch.float64: "fp64"}

# The families whose router carries a selection bias the wrapper must read — their own declaration,
# so the roster follows the class hierarchy rather than a second list to maintain.
_BIASED_FAMILIES = [family for family in FAMILIES if family[1]._NATIVE_BALANCING_BIAS_ATTR]
_BIASED_IDS = [family[0] for family in _BIASED_FAMILIES]

# Multiplier applied to a wrapper's own selection bias in the mutation control below: large enough to
# reorder a near-boundary expert, small enough that catching it is not trivial.
_BIAS_PERTURBATION = 1.5

_UNCOMPARABLE = {
    "EPBailingMoELayer": (
        "remote-code modeling — transformers 5.16 ships no bailing_moe module, and a stand-in "
        "reference would only certify this file's own assumptions"
    ),
    "EPLagunaMoELayer": (
        "compared against the real LagunaSparseMoeBlock in tests/cpu/parallelism/test_laguna_ep.py, "
        "which needs that family's own fp64 + e_score_correction_bias fixture"
    ),
}


def _build(factory, wrapper, dtype: torch.dtype) -> tuple[nn.Module, nn.Module]:
    """Return ``(untouched reference, EP wrapper around an identical copy)``.

    The reference is deep-copied BEFORE wrapping: the wrapper copies the expert weights into its own
    layout and adopts the gate by reference, so a shared module would make the comparison partly
    self-referential. Float BUFFERS are randomized alongside the parameters — the routing correction
    biases (``gate.e_score_correction_bias``, LFM2's ``expert_bias``) live there, and leaving them at
    zero would silence the one term the wrapper has to re-read from the block.
    """
    torch.manual_seed(0)
    module = factory().to(dtype)
    for parameter in module.parameters():
        nn.init.normal_(parameter, std=0.2)
    for _name, buffer in module.named_buffers():
        if buffer.is_floating_point():
            nn.init.normal_(buffer, std=0.3)
    return copy.deepcopy(module), wrapper(module, single_process_ep_config(E)).cpu()


def _band(reference: torch.Tensor) -> float:
    return _RELATIVE_BAND * reference.abs().max().item()


def _primary_output(output) -> torch.Tensor:
    """The expert output, for the families whose block returns more than it.

    GptOss returns ``(hidden, router_scores)`` and Zaya ``(hidden, carried EDA router state)``; in
    both the tail comes straight off a router the wrapper adopts and calls unchanged, so element 0
    is the whole of what the wrapper re-derives — and the only part a comparison can attribute to it.
    """
    return output[0] if isinstance(output, tuple) else output


def test_every_ep_family_with_a_re_derived_forward_is_compared():
    """Derived from the class hierarchy, not a hand-kept list: a family added without a numeric
    reference lands here rather than training silently on a wrapper nobody ever compared. The
    ``HF_MODULE_NAMES`` filter is what separates real families from intermediate bases."""
    declaring = {cls.__name__ for cls in ep_layer_classes() if vars(cls).get("HF_MODULE_NAMES")}
    compared = {wrapper.__name__ for _family, wrapper, _factory, _inputs in FAMILIES}

    assert not compared & set(_UNCOMPARABLE), "a family cannot be both compared here and excused from it"
    missing = declaring - compared - set(_UNCOMPARABLE)
    assert not missing, (
        f"{sorted(missing)} own a re-derived ep1 forward that nothing compares to its upstream "
        f"module — add a fixture to FAMILIES, or list it in _UNCOMPARABLE with the reason"
    )
    stale = set(_UNCOMPARABLE) - declaring
    assert not stale, f"_UNCOMPARABLE still excuses {sorted(stale)}, which is no longer a registered family"


@pytest.mark.parametrize("dtype", list(_DTYPE_IDS), ids=lambda d: _DTYPE_IDS[d])
@pytest.mark.parametrize(_FIELDS, FAMILIES, ids=_IDS)
def test_ep1_forward_matches_the_upstream_module(family, wrapper, factory, inputs, dtype):
    reference, layer = _build(factory, wrapper, dtype)
    args = inputs(dtype)

    with torch.no_grad():
        expected = _primary_output(reference(*args))
        actual = _primary_output(layer(*args))

    # Anti-vacuity: an all-zero reference would make any tolerance pass.
    assert expected.abs().max().item() > 1e-2, f"{family}: reference output is ~zero — nothing was compared"
    assert actual.shape == expected.shape
    difference = (actual - expected).abs().max().item()
    assert difference <= _band(expected), (
        f"{family} at {dtype}: EP wrapper diverges from {type(reference).__name__} by {difference:.3e} "
        f"(band {_band(expected):.3e}) — the wrapper's re-derived routing/scaling/activation no "
        f"longer reproduces the upstream block"
    )


@pytest.mark.parametrize("dtype", list(_DTYPE_IDS), ids=lambda d: _DTYPE_IDS[d])
@pytest.mark.parametrize(_FIELDS, FAMILIES, ids=_IDS)
def test_ep1_input_gradient_matches_the_upstream_module(family, wrapper, factory, inputs, dtype):
    """Forward parity alone leaves the backward unchecked, and the wrapper does NOT inherit it: the
    dispatch/permute/scatter chain is a hand-written autograd path, and the routed-weight gradient
    crosses a dtype boundary on the way back."""
    reference, layer = _build(factory, wrapper, dtype)
    args = inputs(dtype)
    reference_tokens = args[0].clone().requires_grad_(True)
    layer_tokens = args[0].clone().requires_grad_(True)

    expected = _primary_output(reference(reference_tokens, *args[1:]))
    actual = _primary_output(layer(layer_tokens, *args[1:]))
    cotangent = torch.randn(expected.shape, dtype=dtype, generator=torch.Generator().manual_seed(5))
    expected.backward(cotangent)
    actual.backward(cotangent)

    assert reference_tokens.grad.abs().max().item() > 1e-2, f"{family}: reference input grad is ~zero"
    difference = (layer_tokens.grad - reference_tokens.grad).abs().max().item()
    assert difference <= _band(reference_tokens.grad), (
        f"{family} at {dtype}: input gradient diverges by {difference:.3e} (band {_band(reference_tokens.grad):.3e})"
    )


@pytest.mark.parametrize(_FIELDS, _BIASED_FAMILIES, ids=_BIASED_IDS)
def test_the_wrapper_reads_its_own_selection_bias(family, wrapper, factory, inputs):
    """The control for the equivalence above, on the one term that decides WHICH experts run.

    A wrapper can read its family's selection bias wrongly in ways nothing else notices: after the
    top-k instead of before it, added to the gate weights as well, or dropped because the family
    renamed the slot. None moves a shape, a dtype or a key. So scale the wrapper's OWN slot and
    compare it against ITSELF: a wrapper whose output does not move never read the tensor.

    Self-comparison, not wrapper-vs-reference — the latter passes for a wrapper that ignores its bias
    entirely (it then simply disagrees with the block, which is what the equivalence test above is
    for). Verified by severing ``_selection_scores``: this assertion fails, that one already did.
    """
    _reference, layer = _build(factory, wrapper, torch.float64)
    slot = resolve_balancing_slot(layer, wrapper._NATIVE_BALANCING_BIAS_ATTR)
    # A declared slot that does not resolve is the upstream rename the declaration exists to catch,
    # not a reason to skip: the wrapper would then route unbiased against a block that does not.
    assert slot is not None, (
        f"{family} declares _NATIVE_BALANCING_BIAS_ATTR='{wrapper._NATIVE_BALANCING_BIAS_ATTR}' "
        f"but the built layer carries no such tensor"
    )
    assert getattr(*slot).abs().max().item() > 1e-3, f"{family}: the fixture's selection bias is ~zero"

    args = inputs(torch.float64)
    with torch.no_grad():
        before = _primary_output(layer(*args))
        getattr(*slot).mul_(_BIAS_PERTURBATION)
        after = _primary_output(layer(*args))

    difference = (after - before).abs().max().item()
    assert difference > _band(before), (
        f"{family}: scaling '{wrapper._NATIVE_BALANCING_BIAS_ATTR}' by {_BIAS_PERTURBATION} moved this "
        f"wrapper's own output by only {difference:.3e} (band {_band(before):.3e}) — its forward does "
        f"not read the selection bias its family declares"
    )


def test_the_fixtures_carry_non_neutral_routing_knobs():
    """Anti-vacuity for the fixtures themselves: the families that HAVE a normalization and a routed
    scaling factor must be built with both switched away from their neutral values, or a wrapper that
    dropped either would still reproduce the block exactly."""
    for factory in (_glm4_moe_lite, _glm5_next, _mistral4, _lfm2_moe):
        gate = factory().gate  # 5.14 moved both knobs onto the *TopkRouter
        assert gate.norm_topk_prob is True, f"{factory.__name__}: normalization is off"
        assert gate.routed_scaling_factor == _ROUTING_KNOBS["routed_scaling_factor"], (
            f"{factory.__name__}: routed scaling is neutral"
        )
    # GLM-5 Next's clamp must sit inside the fixture's activation range, or a wrapper that dropped
    # it would still match the block (the family default 10.0 is neutral at these scales).
    assert _glm5_next().experts.swiglu_limit == _GLM5_SWIGLU_LIMIT < 1.0
    # DeepSeek-V4's own default (1.5) is non-neutral, so the fixture must move off it to catch a
    # defaulted resolve.
    assert _deepseek_v4().gate.routed_scaling_factor == _ROUTING_KNOBS["routed_scaling_factor"]
    # LFM2's correction bias is a buffer the wrapper has to re-read; the block must actually use it.
    assert _lfm2_moe().use_expert_bias is True
    # Step-3.7's clamps are per-layer: the routed bound must sit inside the activation range (the
    # roster default 0 = unclamped is neutral), the block-level scaling must be non-neutral, and the
    # SHARED bound must be active too — the wrapper adopts the shared expert by reference, so this
    # is what makes a rebuilt/unclamped shared expert diverge.
    step3p7 = _step3p7()
    assert step3p7.experts.limit == _STEP3P7_SWIGLU_LIMIT < 1.0
    assert step3p7.shared_experts.limit == _STEP3P7_SHARED_SWIGLU_LIMIT < 1.0
    assert step3p7.routed_scaling_factor == _ROUTING_KNOBS["routed_scaling_factor"]


@pytest.mark.parametrize(_FIELDS, FAMILIES, ids=_IDS)
def test_the_comparison_is_sensitive_to_the_expert_weights(family, wrapper, factory, inputs):
    """The bands above are only meaningful if they sit far below a real change. Nudging the wrapper's
    OWN expert weights by 1% must move its output by orders of magnitude more than the band — which
    also proves the compared forward actually consumes those weights rather than a stale copy."""
    reference, layer = _build(factory, wrapper, torch.float32)
    args = inputs(torch.float32)

    with torch.no_grad():
        expected = _primary_output(reference(*args))
        for _name, parameter in layer.expert_named_params():
            parameter.mul_(_SENSITIVITY_SCALE)
        perturbed = _primary_output(layer(*args))

    shift = (perturbed - expected).abs().max().item()
    assert shift > _SENSITIVITY_FLOOR, (
        f"{family}: a {_SENSITIVITY_SCALE:g}x change on every expert weight moved the output by only "
        f"{shift:.3e} — the equivalence assertions above are not measuring the expert compute"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
