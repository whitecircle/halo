#!/usr/bin/env python
"""At ``ep_size=1`` every EP layer computes what the real transformers block computes. The band is relative: fp32
floors at epsilon times the output scale, and fp64 at ~1e-7 where the wrapper casts routing weights to fp32."""

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
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMoE
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextSparseMoeBlock
from transformers.models.step3p7.configuration_step3p7 import Step3p7TextConfig
from transformers.models.step3p7.modeling_step3p7 import Step3p7SparseMoeBlock
from transformers.models.zaya.configuration_zaya import ZayaConfig
from transformers.models.zaya.modeling_zaya import ZayaSparseMoeBlock

from halo.distributed.expert_parallel.families.cohere2_moe import EPCohere2MoELayer
from halo.distributed.expert_parallel.families.deepseek_v4 import EPDeepseekV4MoELayer
from halo.distributed.expert_parallel.families.gemma4 import EPGemma4MoELayer
from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.distributed.expert_parallel.families.glm5_next import EPGlm5NextMoELayer
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.families.inkling import EPInklingMoELayer
from halo.distributed.expert_parallel.families.lfm2 import EPLfm2MoELayer
from halo.distributed.expert_parallel.families.mistral4 import EPMistral4MoELayer
from halo.distributed.expert_parallel.families.nemotron_h import EPNemotronHMoELayer
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from halo.distributed.expert_parallel.families.qwen3_next import EPQwen3NextMoELayer
from halo.distributed.expert_parallel.families.qwen4_exp import EPQwen4ExpMoELayer
from halo.distributed.expert_parallel.families.roster import ep_layer_classes
from halo.distributed.expert_parallel.families.step3p7 import EPStep3p7MoELayer
from halo.distributed.expert_parallel.families.zaya import EPZayaMoELayer
from halo.models.moe.balancing import resolve_balancing_slot
from tests.common.parallelism import single_process_ep_config

E, H, M, K = 8, 16, 32, 3  # experts, hidden, moe intermediate, top_k
TOKENS = (2, 5)
# ZayaConfig refuses anything else at construction.
ZAYA_K = 1

# Non-neutral: at the defaults, a wrapper that resolved either knob to its default would still match.
_ROUTING_KNOBS = {"norm_topk_prob": True, "routed_scaling_factor": 2.5}

# Well inside the fixture's activation scale (~0.8 std; the family default 10.0 never clamps here), so a dropped clamp
# diverges.
_GLM5_SWIGLU_LIMIT = 0.5

# The routed and shared bounds differ so a wrapper that read the wrong list still diverges.
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

# ~6x the measured worst case (3.1e-7) and ~80x below the smallest real disagreement probed here.
_RELATIVE_BAND = 2e-6
# A 1% move must clear the band by orders of magnitude, or the band is vacuous.
_SENSITIVITY_SCALE = 1.01
_SENSITIVITY_FLOOR = 1e-3


def _qwen3_moe() -> Qwen3MoeSparseMoeBlock:
    return Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(num_experts=E, num_experts_per_tok=K, **_COMMON))


def _qwen3_5_moe() -> Qwen3_5MoeSparseMoeBlock:
    return Qwen3_5MoeSparseMoeBlock(
        Qwen3_5MoeConfig(num_experts=E, num_experts_per_tok=K, shared_expert_intermediate_size=M, **_COMMON)
    )


def _qwen3_next() -> Qwen3NextSparseMoeBlock:
    """Renormalization off sets the family apart from Qwen3.5-MoE."""
    return Qwen3NextSparseMoeBlock(
        Qwen3NextConfig(
            num_experts=E, num_experts_per_tok=K, shared_expert_intermediate_size=M, norm_topk_prob=False, **_COMMON
        )
    )


def _qwen4_exp() -> Qwen4ExpTextSparseMoeBlock:
    return Qwen4ExpTextSparseMoeBlock(
        Qwen4ExpTextConfig(
            num_experts=E, num_experts_per_tok=K, shared_expert_intermediate_size=M, norm_topk_prob=False, **_COMMON
        )
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
    """The config's own validation requires ``num_key_value_heads`` to equal ``num_attention_heads``, and
    ``mlp_layer_types`` to be set for a shallow model."""
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
    """Its experts are ``intermediate_size`` wide, so that field carries M here."""
    config = {key: value for key, value in _COMMON.items() if key != "moe_intermediate_size"}
    return GptOssMLP(
        GptOssConfig(**{**config, "intermediate_size": M, "num_local_experts": E, "num_experts_per_tok": K})
    )


def _zaya() -> ZayaSparseMoeBlock:
    """``layer_idx=0`` takes the no-incoming-state branch, the shape ``_block_inputs`` feeds."""
    config = {key: value for key, value in _COMMON.items() if key != "intermediate_size"}
    return ZayaSparseMoeBlock(
        ZayaConfig(num_experts=E, num_experts_per_tok=ZAYA_K, router_hidden_size=8, **config), layer_idx=0
    )


def _deepseek_v4() -> DeepseekV4SparseMoeBlock:
    """The hash variant routes off ``tid2eid[input_ids]``, which the block only receives from its decoder layer."""
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
    """Its experts are ``intermediate_size`` wide. The averaged shared expert means a wrapper defaulting the
    strategy to ``sum`` diverges by 2x."""
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
    """The wrapper's ``_route_joint`` slices the last shared logit columns, so ``n_shared_experts >= 1``."""
    return InklingMoE(
        InklingTextConfig(n_routed_experts=E, num_experts_per_tok=K, n_shared_experts=2, route_scale=8.0, **_COMMON)
    )


def _step3p7() -> Step3p7SparseMoeBlock:
    """Bounds and scaling are non-neutral so a wrapper dropping any diverges; the family has no ``norm_topk_prob``
    knob and no group limiting."""
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


def _nemotron_h() -> NemotronHMoE:
    """A latent narrower than the hidden state, so a wrapper running its experts on the hidden state cannot build."""
    config = {key: value for key, value in _COMMON.items() if key != "hidden_act"}
    return NemotronHMoE(
        NemotronHConfig(
            n_routed_experts=E,
            num_experts_per_tok=K,
            moe_latent_size=H // 2,
            moe_shared_expert_intermediate_size=M,
            n_group=1,
            topk_group=1,
            mlp_hidden_act="relu2",
            layers_block_type=["moe"],
            **_ROUTING_KNOBS,
            **config,
        ),
        layer_idx=0,
    )


def _block_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(11)
    return (torch.randn(*TOKENS, H, dtype=dtype, generator=generator),)


def _experts_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(11)
    count = TOKENS[0] * TOKENS[1]
    weights, index = torch.rand(count, E, generator=generator).topk(K, dim=-1)
    return (torch.randn(count, H, dtype=dtype, generator=generator), index, weights.to(dtype))


FAMILIES = [
    ("glm4_moe_lite", EPGlm4MoELayer, _glm4_moe_lite, _block_inputs),
    ("glm5_next", EPGlm5NextMoELayer, _glm5_next, _block_inputs),
    ("lfm2_moe", EPLfm2MoELayer, _lfm2_moe, _block_inputs),
    ("mistral4", EPMistral4MoELayer, _mistral4, _block_inputs),
    ("nemotron_h", EPNemotronHMoELayer, _nemotron_h, _block_inputs),
    ("gemma4", EPGemma4MoELayer, _gemma4_experts, _experts_inputs),
    ("qwen3_moe", EPQwen3MoELayer, _qwen3_moe, _block_inputs),
    ("qwen3_5_moe", EPQwen3_5MoELayer, _qwen3_5_moe, _block_inputs),
    ("qwen3_next", EPQwen3NextMoELayer, _qwen3_next, _block_inputs),
    ("qwen4_exp", EPQwen4ExpMoELayer, _qwen4_exp, _block_inputs),
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

# Their spec's own slot, so the roster follows the declarations.
_BIASED_FAMILIES = [family for family in FAMILIES if family[1].FAMILY.balancing_slot]
_BIASED_IDS = [family[0] for family in _BIASED_FAMILIES]

# Large enough to reorder a near-boundary expert, small enough that catching it is not trivial.
_BIAS_PERTURBATION = 1.5

_UNCOMPARABLE = {
    "EPBailingMoELayer": (
        "remote-code modeling — transformers 5.20 ships no bailing_moe module, and a stand-in "
        "reference would only certify this file's own assumptions"
    ),
    "EPLagunaMoELayer": (
        "compared against the real LagunaSparseMoeBlock in tests/cpu/parallelism/test_laguna_ep.py, "
        "which needs that family's own fp64 + e_score_correction_bias fixture"
    ),
}


def _build(factory, wrapper, dtype: torch.dtype) -> tuple[nn.Module, nn.Module]:
    """Deep-copied before wrapping, since the wrapper adopts the gate by reference; float buffers are randomized
    because the routing biases live there."""
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
    """Element 0 is all the wrapper re-derives; the tail comes straight off a router it adopts unchanged."""
    return output[0] if isinstance(output, tuple) else output


def test_every_ep_family_with_a_re_derived_forward_is_compared():
    """A family added without a numeric reference lands here; ``HF_MODULE_NAMES`` separates real families from
    intermediate bases."""
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
        f"{family} at {dtype}: EP layer diverges from {type(reference).__name__} by {difference:.3e} "
        f"(band {_band(expected):.3e}) — the wrapper's re-derived routing/scaling/activation no "
        f"longer reproduces the upstream block"
    )


@pytest.mark.parametrize("dtype", list(_DTYPE_IDS), ids=lambda d: _DTYPE_IDS[d])
@pytest.mark.parametrize(_FIELDS, FAMILIES, ids=_IDS)
def test_ep1_input_gradient_matches_the_upstream_module(family, wrapper, factory, inputs, dtype):
    """The dispatch/permute/scatter chain is a hand-written autograd path, and the routed-weight gradient crosses a
    dtype boundary."""
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
    """Self-comparison, since wrapper-vs-reference passes for a wrapper ignoring its bias; an output that does not
    move never read the tensor."""
    _reference, layer = _build(factory, wrapper, torch.float64)
    slot = resolve_balancing_slot(layer, layer._native_balancing_attr)
    # A slot that does not resolve is the upstream rename the declaration exists to catch.
    assert slot is not None, (
        f"{family} declares the balancing slot '{layer._native_balancing_attr}' "
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
        f"{family}: scaling '{layer._native_balancing_attr}' by {_BIAS_PERTURBATION} moved this "
        f"wrapper's own output by only {difference:.3e} (band {_band(before):.3e}) — its forward does "
        f"not read the selection bias its family declares"
    )


def test_the_fixtures_carry_non_neutral_routing_knobs():
    """Anti-vacuity: at neutral values a wrapper that dropped either would still reproduce the block."""
    for factory in (_glm4_moe_lite, _glm5_next, _mistral4, _lfm2_moe, _nemotron_h):
        gate = factory().gate  # 5.14 moved both knobs onto the *TopkRouter.
        assert gate.norm_topk_prob is True, f"{factory.__name__}: normalization is off"
        assert gate.routed_scaling_factor == _ROUTING_KNOBS["routed_scaling_factor"], (
            f"{factory.__name__}: routed scaling is neutral"
        )
    assert _glm5_next().experts.swiglu_limit == _GLM5_SWIGLU_LIMIT < 1.0
    # DeepSeek-V4's own default (1.5) is non-neutral, so the fixture moves off it to catch a defaulted resolve.
    assert _deepseek_v4().gate.routed_scaling_factor == _ROUTING_KNOBS["routed_scaling_factor"]
    assert _lfm2_moe().use_expert_bias is True
    # The wrapper adopts the shared expert by reference, so an active shared bound is what makes a rebuilt shared
    # expert diverge.
    step3p7 = _step3p7()
    assert step3p7.experts.limit == _STEP3P7_SWIGLU_LIMIT < 1.0
    assert step3p7.shared_experts.limit == _STEP3P7_SHARED_SWIGLU_LIMIT < 1.0
    assert step3p7.routed_scaling_factor == _ROUTING_KNOBS["routed_scaling_factor"]


@pytest.mark.parametrize(_FIELDS, FAMILIES, ids=_IDS)
def test_the_comparison_is_sensitive_to_the_expert_weights(family, wrapper, factory, inputs):
    """Also proves the compared forward consumes those weights rather than a stale copy."""
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
