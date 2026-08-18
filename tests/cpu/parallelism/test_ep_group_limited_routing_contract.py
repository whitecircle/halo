#!/usr/bin/env python
"""CPU tests for the DeepSeek-V3 group-limited routing contract on EP layers.

``EPGroupLimitedMoELayerBase`` owns one routing body — score, selection-only bias, group-limited
top-k, unbiased gate weights — for every family that routes this way, plus the single writer of the
six attributes it reads (``n_routed_experts``, ``n_group``, ``topk_group``, ``norm_topk_prob``,
``routed_scaling_factor``, ``top_k``). Writer and reader in one class is what stops a family from
half-populating the contract; the shared body is what stops four near-identical copies from
drifting, which is a SILENT routing change (a bias added in the wrong order, a knob defaulted)
rather than a crash.

These tests fail if a family forks the body, or if a knob resolves to its neutral default against
the real upstream block.

Run: ``python tests/cpu/parallelism/test_ep_group_limited_routing_contract.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPGroupLimitedMoELayerBase
from src.distributed.expert_parallel.expert_weights import ep_layer_classes
from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.layers.mistral4 import EPMistral4MoELayer
from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from tests.common.parallelism import single_process_ep_config

E, H, M, K = 4, 8, 16, 2  # experts, hidden, intermediate, top_k

# The knobs _group_limited_topk consumes — keep in lockstep with the method body.
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
    """HF block carrying the DeepSeek-V3 group-routing knobs explicitly."""

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
    """Same block with ``norm_topk_prob`` ABSENT, so the family default applies."""

    def __init__(self):
        super().__init__()
        assert not hasattr(self, "norm_topk_prob")


class _Step3p7Block(_GroupRoutedBlock):
    """The same block carrying what Step-3.7's own hooks read: an always-built shared expert and the
    per-layer clamp bound on the expert container."""

    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.shared_experts = nn.Identity()
        self.routed_experts.limit = float("inf")


def _build(cls, block):
    return cls(block, single_process_ep_config(E)).cpu()


def _group_limited_families() -> list[type]:
    """Every registered family routing through the shared body — from the class hierarchy, so a
    family joining it is covered here without editing a list."""
    return [
        cls
        for cls in ep_layer_classes()
        if issubclass(cls, EPGroupLimitedMoELayerBase) and vars(cls).get("HF_MODULE_NAMES")
    ]


# Whose knob writer is exercised against a real block: the two families that build from the shared
# one, plus every family EXTENDING _init_routing. An extension that skipped super() would leave all
# six knobs at their neutral defaults with no attribute missing, so only the values catch it.
_KNOB_WRITER_FAMILIES = sorted(
    {EPGlm4MoELayer, EPMistral4MoELayer, *(cls for cls in _group_limited_families() if "_init_routing" in vars(cls))},
    key=lambda cls: cls.__name__,
)


def test_the_roster_routing_this_way_is_exactly_the_five_families_that_do():
    """Anti-vacuity for the derived roster, as an EQUALITY: a superset check passes just as happily
    when a family is reparented off this base, which silently drops its fork check below (and, with
    it, the only thing pinning that it still routes the shared way)."""
    assert {cls.__name__ for cls in _group_limited_families()} == {
        "EPGlm4MoELayer",
        "EPLagunaMoELayer",
        "EPMistral4MoELayer",
        "EPGlm5NextMoELayer",
        "EPStep3p7MoELayer",
    }


@pytest.mark.parametrize("cls", _group_limited_families(), ids=lambda c: c.__name__)
def test_no_family_forks_the_shared_routing_body(cls):
    """A family may declare how it SCORES its logits and where its knobs live; re-implementing the
    selection itself is how four copies drifted apart before, and every way they can disagree —
    the order the two biases are added in, which scores the weights come from, whether replay runs
    — is silent."""
    forked = [name for name in ("route_tokens_to_experts", "_group_limited_topk") if name in vars(cls)]
    assert not forked, (
        f"{cls.__name__} re-implements {forked} instead of inheriting the shared group-limited "
        f"routing. Express the family's difference as _routing_scores / _NORM_TOPK_PROB_DEFAULT / "
        f"_TOPK_WEIGHT_NORM_EPS, or subclass EPSharedExpertsMoELayerBase if the routing is genuinely "
        f"a different one."
    )


@pytest.mark.parametrize("cls", _KNOB_WRITER_FAMILIES, ids=lambda c: c.__name__)
def test_every_consumed_knob_is_populated(cls):
    """Every knob the shared top-k reads, at the value the real block carries — which is also what
    pins that a family extending ``_init_routing`` (GLM-4 materializes a meta correction bias first)
    still calls ``super()``."""
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
    """Step-3.7's router renormalizes unconditionally and nothing in its chain declares the knob, so
    it opts out and its default decides. A family that does NOT opt out may not fall back at all: the
    default would silently change routed-weight scale on a block that simply renamed the attribute."""
    torch.manual_seed(0)
    assert "norm_topk_prob" in EPStep3p7MoELayer._OPTIONAL_ROUTING_KNOBS
    assert _build(EPStep3p7MoELayer, _Step3p7Block()).norm_topk_prob is EPStep3p7MoELayer._NORM_TOPK_PROB_DEFAULT

    assert "norm_topk_prob" not in EPGlm4MoELayer._OPTIONAL_ROUTING_KNOBS
    with pytest.raises(AttributeError, match="norm_topk_prob"):
        _build(EPGlm4MoELayer, _BareBlock())

    # An explicit block value is what every family reads.
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
    from transformers import Mistral4Config
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

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
    from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteMoE

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
    """Against the INSTALLED transformers modules, with knobs at non-neutral values — the synthetic
    block above carries a pre-5.14 layout and passed while Mistral4 silently resolved every knob to
    its default (the 5.14 *TopkRouter refactor moved them onto the gate, spelled ``num_group``)."""
    torch.manual_seed(0)
    layer = _build(cls, build())
    assert layer.n_routed_experts == E
    assert layer.n_group == 2, "group count fell back to 1 — group-limited routing silently disabled"
    assert layer.topk_group == 1
    assert layer.routed_scaling_factor == 2.5, "scaling fell back to 1.0 — routed weights silently unscaled"
    assert int(layer.top_k) == K


@pytest.mark.parametrize("cls", [EPStep3p7MoELayer, EPGlm4MoELayer])
def test_a_routing_knob_that_vanished_raises_instead_of_defaulting(cls):
    """The neutral 1.0 is not a safe fallback for ``routed_scaling_factor``: it under-scales every
    routed weight by 2.5x on Step-3.7-Flash and 1.8x on GLM-4, with no attribute missing anywhere. No
    family opts that knob out, so an upstream rename raises on all of them."""
    torch.manual_seed(0)
    block = _Step3p7Block() if cls is EPStep3p7MoELayer else _GroupRoutedBlock(norm_topk_prob=True)
    del block.routed_scaling_factor
    with pytest.raises(AttributeError, match="routed_scaling_factor"):
        _build(cls, block)


def test_an_opted_out_knob_keeps_its_neutral_default():
    """The escape hatch itself: GLM-4 also serves revisions that declare no group limiting, so those
    two knobs fall through to the neutral 1 rather than refusing to construct."""
    torch.manual_seed(0)
    block = _GroupRoutedBlock(norm_topk_prob=True)
    del block.n_group
    del block.topk_group
    layer = _build(EPGlm4MoELayer, block)
    assert (layer.n_group, layer.topk_group) == (1, 1)


def test_a_knob_can_be_opted_out_under_either_of_its_spellings():
    """``n_group`` is ``num_group`` on the ``*TopkRouter`` modules. Matching the opt-out against one
    canonical spelling would leave a family whose router carries only the alias unable to declare it —
    the same trap in reverse: it could never construct."""

    class _AliasSpelled(EPGlm4MoELayer):
        # Claims nothing: a test-local subclass that inherited GLM-4's registrations would shadow the
        # real family in every registry walked by module name or model_type.
        HF_MODULE_NAMES = ()
        HF_MODEL_TYPES = ()
        _OPTIONAL_ROUTING_KNOBS = ("num_group", "topk_group")

    torch.manual_seed(0)
    block = _GroupRoutedBlock(norm_topk_prob=True)
    del block.n_group
    del block.topk_group
    assert _build(_AliasSpelled, block).n_group == 1


def test_the_weight_norm_floor_is_the_familys_own():
    """``_TOPK_WEIGHT_NORM_EPS`` is only meaningful if it reaches the division.

    DeepSeek-V3 floors the top-k weight sum at 1e-20; Step-3.7 renormalizes with no floor at all, so
    it declares 0.0. The two agree everywhere the sum is O(1) — which is why this feeds a sum small
    enough for the floor to dominate, the regime a diverged router reaches. A knob nothing can
    distinguish would be a knob nobody can trust."""
    torch.manual_seed(0)
    floored = _build(EPGlm4MoELayer, _GroupRoutedBlock(norm_topk_prob=True))
    unfloored = _build(EPStep3p7MoELayer, _Step3p7Block(norm_topk_prob=True))
    assert (floored._TOPK_WEIGHT_NORM_EPS, unfloored._TOPK_WEIGHT_NORM_EPS) == (1e-20, 0.0)

    scores = torch.full((1, E), 1e-30)
    scores[0, 0] = 2e-30  # a strict top-2 order, so both wrappers select the same experts
    _floored_idx, floored_weights = floored._group_limited_topk(scores, scores)
    _unfloored_idx, unfloored_weights = unfloored._group_limited_topk(scores, scores)

    # Without a floor the tiny sum normalizes exactly; with one, 1e-20 swamps it and the weights collapse.
    assert torch.allclose(unfloored_weights.sum(), torch.tensor(1.0) * unfloored.routed_scaling_factor)
    assert floored_weights.abs().max() < 1e-9, floored_weights


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
