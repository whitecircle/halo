#!/usr/bin/env python
"""The DeepSeek-V3 group-limited routing contract: one shared body and one writer of its six attributes. These fail
if a family forks the body or a knob goes neutral against the real block."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import Mistral4Config
from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteMoE
from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.distributed.expert_parallel.families.mistral4 import EPMistral4MoELayer
from halo.distributed.expert_parallel.families.roster import ep_layer_classes
from halo.distributed.expert_parallel.families.step3p7 import EPStep3p7MoELayer
from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from halo.models.moe.family_specs import MoEFamilySpec
from tests.common.parallelism import single_process_ep_config

E, H, M, K = 4, 8, 16, 2  # experts, hidden, intermediate, top_k

# Keep in lockstep with _group_limited_topk's body.
_ROUTING_KNOBS = (
    "n_routed_experts",
    "n_group",
    "topk_group",
    "norm_topk_prob",
    "routed_scaling_factor",
    "top_k",
)


class _Config:
    hidden_act = "silu"


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H))
        self.top_k = K


class _FusedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * M, H))
        self.down_proj = nn.Parameter(torch.randn(E, H, M))
        self.num_experts = E


class _GroupRoutedBlock(nn.Module):
    def __init__(self, **overrides):
        super().__init__()
        self.config = _Config()
        self.gate = _Gate()
        self.routed_experts = _FusedExperts()
        self.experts = self.routed_experts
        self.shared_experts = None
        self.n_routed_experts = E
        self.n_group = 2
        self.topk_group = 1
        self.routed_scaling_factor = 2.5
        for key, value in overrides.items():
            setattr(self, key, value)


class _BareBlock(_GroupRoutedBlock):
    """``norm_topk_prob`` absent, so the family default applies."""

    def __init__(self):
        super().__init__()
        assert not hasattr(self, "norm_topk_prob")


class _Step3p7Block(_GroupRoutedBlock):
    """What Step-3.7's own hooks read: an always-built shared expert and the per-layer clamp bound."""

    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.shared_experts = nn.Identity()
        self.routed_experts.limit = float("inf")


def _build(cls, block):
    return cls(block, single_process_ep_config(E)).cpu()


def _group_limited_families() -> list[type]:
    return [
        cls
        for cls in ep_layer_classes()
        if issubclass(cls, EPGroupLimitedMoELayerBase) and vars(cls).get("HF_MODULE_NAMES")
    ]


# An extension that skipped super() would leave all six knobs neutral with no attribute missing, so only the values
# catch it.
_KNOB_WRITER_FAMILIES = sorted(
    {EPGlm4MoELayer, EPMistral4MoELayer, *(cls for cls in _group_limited_families() if "_adopt_routing" in vars(cls))},
    key=lambda cls: cls.__name__,
)


def test_the_roster_routing_this_way_is_exactly_the_five_families_that_do():
    """Equality, since a superset check passes when a family is reparented off this base, dropping its fork check."""
    assert {cls.__name__ for cls in _group_limited_families()} == {
        "EPGlm4MoELayer",
        "EPLagunaMoELayer",
        "EPMistral4MoELayer",
        "EPGlm5NextMoELayer",
        "EPStep3p7MoELayer",
    }


@pytest.mark.parametrize("cls", _group_limited_families(), ids=lambda c: c.__name__)
def test_no_family_forks_the_shared_routing_body(cls):
    """Every way copies can disagree is silent: bias order, which scores the weights come from, whether replay runs."""
    forked = [name for name in ("route_tokens_to_experts", "_group_limited_topk") if name in vars(cls)]
    assert not forked, (
        f"{cls.__name__} re-implements {forked} instead of inheriting the shared group-limited "
        f"routing. Express the family's difference as _routing_scores / _NORM_TOPK_PROB_DEFAULT / "
        f"_TOPK_WEIGHT_NORM_EPS, or subclass EPSharedExpertsMoELayerBase if the routing is genuinely "
        f"a different one."
    )


@pytest.mark.parametrize("cls", _KNOB_WRITER_FAMILIES, ids=lambda c: c.__name__)
def test_every_consumed_knob_is_populated(cls):
    """Also pins that a family extending ``_adopt_routing`` still calls ``super()``."""
    torch.manual_seed(0)
    layer = _build(cls, _GroupRoutedBlock(norm_topk_prob=True))
    for knob in _ROUTING_KNOBS:
        assert hasattr(layer, knob), f"{cls.__name__} never set '{knob}' (read by _group_limited_topk)"
    assert layer.n_routed_experts == E
    assert layer.n_group == 2
    assert layer.topk_group == 1
    assert layer.routed_scaling_factor == 2.5
    assert int(layer.top_k) == K


def test_the_family_default_decides_only_where_the_family_opted_the_knob_out():
    """A fallback would silently change routed-weight scale on a block that simply renamed the attribute."""
    torch.manual_seed(0)
    assert "norm_topk_prob" in EPStep3p7MoELayer._OPTIONAL_ROUTING_KNOBS
    assert _build(EPStep3p7MoELayer, _Step3p7Block()).norm_topk_prob is EPStep3p7MoELayer._NORM_TOPK_PROB_DEFAULT

    assert "norm_topk_prob" not in EPGlm4MoELayer._OPTIONAL_ROUTING_KNOBS
    with pytest.raises(AttributeError, match="norm_topk_prob"):
        _build(EPGlm4MoELayer, _BareBlock())

    for value in (True, False):
        assert _build(EPGlm4MoELayer, _GroupRoutedBlock(norm_topk_prob=value)).norm_topk_prob is value


_REAL_KNOBS = {
    "n_routed_experts": E,
    "num_experts_per_tok": K,
    "n_group": 2,
    "topk_group": 1,
    "routed_scaling_factor": 2.5,
}


def _real_mistral4_block():
    config = Mistral4Config(
        vocab_size=64,
        hidden_size=H,
        intermediate_size=M,
        moe_intermediate_size=M,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=8,
        q_lora_rank=None,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        norm_topk_prob=False,
        **_REAL_KNOBS,
    )
    return Mistral4MoE(config)


def _real_glm4_block():
    config = Glm4MoeLiteConfig(
        vocab_size=64,
        hidden_size=H,
        intermediate_size=M,
        moe_intermediate_size=M,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        **_REAL_KNOBS,
    )
    return Glm4MoeLiteMoE(config)


@pytest.mark.parametrize(
    ("cls", "build"),
    [(EPMistral4MoELayer, _real_mistral4_block), (EPGlm4MoELayer, _real_glm4_block)],
)
def test_real_upstream_blocks_deliver_non_neutral_knobs(cls, build):
    """The installed ``*TopkRouter`` modules carry the knobs on the gate (spelled ``num_group``), which only a real
    block exercises."""
    torch.manual_seed(0)
    layer = _build(cls, build())
    assert layer.n_routed_experts == E
    assert layer.n_group == 2, "group count fell back to 1 — group-limited routing silently disabled"
    assert layer.topk_group == 1
    assert layer.routed_scaling_factor == 2.5, "scaling fell back to 1.0 — routed weights silently unscaled"
    assert int(layer.top_k) == K


@pytest.mark.parametrize("cls", [EPStep3p7MoELayer, EPGlm4MoELayer])
def test_a_routing_knob_that_vanished_raises_instead_of_defaulting(cls):
    """The neutral 1.0 under-scales routed weights by 2.5x on Step-3.7-Flash and 1.8x on GLM-4 with no attribute
    missing."""
    torch.manual_seed(0)
    block = _Step3p7Block() if cls is EPStep3p7MoELayer else _GroupRoutedBlock(norm_topk_prob=True)
    del block.routed_scaling_factor
    with pytest.raises(AttributeError, match="routed_scaling_factor"):
        _build(cls, block)


def test_an_opted_out_knob_keeps_its_neutral_default():
    """GLM-4 also serves revisions declaring no group limiting."""
    torch.manual_seed(0)
    block = _GroupRoutedBlock(norm_topk_prob=True)
    del block.n_group
    del block.topk_group
    layer = _build(EPGlm4MoELayer, block)
    assert (layer.n_group, layer.topk_group) == (1, 1)


def test_a_knob_can_be_opted_out_under_either_of_its_spellings():
    """Matching one canonical spelling would leave a family whose router carries only the alias unable to construct."""

    class _AliasSpelled(EPGlm4MoELayer):
        # A test-local subclass inheriting GLM-4's registrations would shadow the real family in every registry walk.
        HF_MODULE_NAMES = ()
        FAMILY = MoEFamilySpec(model_types=())
        _OPTIONAL_ROUTING_KNOBS = ("num_group", "topk_group")

    torch.manual_seed(0)
    block = _GroupRoutedBlock(norm_topk_prob=True)
    del block.n_group
    del block.topk_group
    assert _build(_AliasSpelled, block).n_group == 1


def test_the_weight_norm_floor_is_the_familys_own():
    """DeepSeek-V3 floors at 1e-20, Step-3.7 declares 0.0; they agree where the sum is O(1), so this feeds a sum
    small enough for the floor to dominate."""
    torch.manual_seed(0)
    floored = _build(EPGlm4MoELayer, _GroupRoutedBlock(norm_topk_prob=True))
    unfloored = _build(EPStep3p7MoELayer, _Step3p7Block(norm_topk_prob=True))
    assert (floored._TOPK_WEIGHT_NORM_EPS, unfloored._TOPK_WEIGHT_NORM_EPS) == (1e-20, 0.0)

    scores = torch.full((1, E), 1e-30)
    scores[0, 0] = 2e-30  # A strict top-2 order, so both wrappers select the same experts.
    _floored_idx, floored_weights = floored._group_limited_topk(scores, scores)
    _unfloored_idx, unfloored_weights = unfloored._group_limited_topk(scores, scores)

    # Without a floor the tiny sum normalizes exactly; with one, 1e-20 swamps it.
    assert torch.allclose(unfloored_weights.sum(), torch.tensor(1.0) * unfloored.routed_scaling_factor)
    assert floored_weights.abs().max() < 1e-9, floored_weights


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
