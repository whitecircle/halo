#!/usr/bin/env python
"""CPU contract tests for the implicit-reference-model guard and the live-sinks predicate it reads.

TRL builds its own reference model whenever ``beta != 0`` and the policy is not PEFT-wrapped. On a
live-sinks policy (``reset_sinks: false``, the GptOss on-policy RL flow) that reference is NOT
restricted to sink-carrying attention, so the two compute different log-probs for identical tokens
and the KL is biased on every token. That is a property of the loaded weights, so the guard must fire
in EVERY parallelism mode — a run at ``ep_size == 1`` (pure FSDP2 data-parallel beside the rollout
servers, the shipped ``*-full-ep1.yaml`` shape) is exactly as wrong as one under EP.

The liveness predicate has to read the policy stamp rather than the tensors: the reset only yields
``sinks is None`` under flash_attention_2, and fills with ``dtype.min`` on every other backend (FA4,
flex, eager), so a presence test would report RESET sinks as live on the production Blackwell path.

Run: python tests/cpu/trainers/test_implicit_reference_model_gate.py  (or pytest)
"""

import types

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState

from src.models.patches.gpt_oss_sinks import SinksPolicy, apply_sinks_policy, has_live_attention_sinks
from src.trainers.mixins.validation import ParallelismValidationMixin

# The sinks patches log through accelerate's logger, which refuses to emit before the state exists.
PartialState()

NUM_HEADS = 4


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sinks = nn.Parameter(torch.randn(NUM_HEADS))
        self.q_proj = nn.Linear(8, 8, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])


class _SinksModel(nn.Module):
    """Minimal ``*ForCausalLM`` layout that ``_gpt_oss_sink_attentions`` walks."""

    def __init__(self, model_type="gpt_oss"):
        super().__init__()
        self.model = _Backbone()
        self.config = types.SimpleNamespace(model_type=model_type, num_attention_heads=NUM_HEADS)


def _guard(model, *, ref_model, ep_size):
    """Run the mixin's guard against a stand-in trainer carrying just what it reads."""
    stub = types.SimpleNamespace(
        model=model,
        ref_model=ref_model,
        parallelism_config=types.SimpleNamespace(is_ep_mode=ep_size > 1, ep_size=ep_size),
    )
    return ParallelismValidationMixin._validate_implicit_reference_model(stub)


def _live_model(attn_implementation="flash_attention_4"):
    model = _SinksModel()
    apply_sinks_policy(model, model.config, policy=SinksPolicy.LIVE, attn_implementation=attn_implementation)
    return model


def _reset_model(attn_implementation="flash_attention_4"):
    model = _SinksModel()
    apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED, attn_implementation=attn_implementation)
    return model


@pytest.mark.parametrize("attn", ["flash_attention_4", "flash_attention_2", "eager"])
def test_sinks_policy_stamp_tracks_the_decision_not_the_tensor(attn):
    assert has_live_attention_sinks(_live_model(attn)) is True
    assert has_live_attention_sinks(_reset_model(attn)) is False


def test_reset_leaves_a_non_none_tensor_on_non_fa2_backends():
    """The reason a presence test cannot stand in for the stamp."""
    fa4_reset = _reset_model("flash_attention_4")
    assert fa4_reset.model.layers[0].self_attn.sinks is not None
    assert has_live_attention_sinks(fa4_reset) is False

    fa2_reset = _reset_model("flash_attention_2")
    assert fa2_reset.model.layers[0].self_attn.sinks is None


def test_model_without_sinks_is_never_live():
    """``reset_sinks: false`` on a non-sinks family must not stamp the flag."""
    model = _SinksModel(model_type="qwen3")
    apply_sinks_policy(model, model.config, policy=SinksPolicy.LIVE, attn_implementation="flash_attention_2")
    assert has_live_attention_sinks(model) is False


@pytest.mark.parametrize("ep_size", [1, 2, 8])
def test_live_sinks_reference_is_rejected_in_every_parallelism_mode(ep_size):
    """The guard must reach the sinks in every mode: at ep_size == 1 an early return never sees them."""
    with pytest.raises(ValueError, match="LIVE attention sinks"):
        _guard(_live_model(), ref_model=object(), ep_size=ep_size)


@pytest.mark.parametrize("ep_size", [1, 8])
def test_reset_sinks_reference_is_allowed(ep_size):
    """The over-fire guard: a dtype-min-filled sink is not a live sink."""
    _guard(_reset_model(), ref_model=object(), ep_size=ep_size)


@pytest.mark.parametrize("ep_size", [1, 8])
def test_no_reference_model_is_never_gated(ep_size):
    """beta == 0 / PEFT: TRL nulls ref_model, so there is no second model to disagree."""
    _guard(_live_model(), ref_model=None, ep_size=ep_size)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
