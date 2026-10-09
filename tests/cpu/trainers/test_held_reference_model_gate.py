#!/usr/bin/env python
"""CPU contract tests for the held-reference-model guard and the live-sinks predicate it reads.

A reference whose sinks differ from the policy's computes different log-probs for identical tokens, and
the KL is biased on every token: a live-sinks policy (``reset_sinks: false``, the GptOss on-policy RL
flow) beside a reference loaded with the sinks reset, or a neutralized policy (``reset_sinks: true``,
the default) beside a reference TRL built itself, whose default attention (eager) applies the pretrained
sinks no loader neutralized. That is a property of the loaded weights, so the guard must fire in EVERY
parallelism mode — a run at ``ep_size == 1`` (pure FSDP2 data-parallel beside the rollout servers, the
shipped ``*-full-ep1.yaml`` shape) is exactly as wrong as one under EP. A reference loaded with the
policy's own ``reset_sinks`` carries the same sinks and passes.

The liveness predicate has to read the policy stamp rather than the tensors: the reset only yields
``sinks is None`` under flash_attention_2, and fills with ``dtype.min`` on every other backend (FA4,
flex, eager), so a presence test would report RESET sinks as live on the production Blackwell path. A
sinks model nothing stamped kept its pretrained sinks, so it reads as live.

Run: python tests/cpu/trainers/test_held_reference_model_gate.py  (or pytest)
"""

import types

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from transformers import GptOssConfig, GptOssForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssPreTrainedModel
from trl.trainer.utils import create_model_from_path

from src.distributed.loading.frozen_models import PREFERENCE_REFERENCE_ALTERNATIVES
from src.distributed.parallelism_config import ParallelismConfig
from src.models.patches.gpt_oss_sinks import SinksPolicy, apply_sinks_policy, has_live_attention_sinks
from src.trainers.mixins.validation import ParallelismValidationMixin
from tests.common.models import TINY_GPTOSS_CONFIG
from tests.common.parallelism import make_parallelism_config

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
        parallelism_config=make_parallelism_config(ep_size=ep_size) if ep_size > 1 else ParallelismConfig(),
        _reference_alternatives=PREFERENCE_REFERENCE_ALTERNATIVES,
    )
    return ParallelismValidationMixin._validate_held_reference_model(stub)


def _trl_built_model():
    """What TRL's own reference build leaves: the family's config, its pretrained sinks, no policy stamped."""
    return _SinksModel()


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
    """``reset_sinks: false`` on a non-sinks family must not stamp the flag, and an unstamped one is not live."""
    model = _SinksModel(model_type="qwen3")
    apply_sinks_policy(model, model.config, policy=SinksPolicy.LIVE, attn_implementation="flash_attention_2")
    assert has_live_attention_sinks(model) is False
    assert has_live_attention_sinks(_SinksModel(model_type="qwen3")) is False


def test_an_unstamped_sinks_model_keeps_its_pretrained_sinks_live():
    """No loader neutralized them, and GptOss's default attention applies them."""
    model = _trl_built_model()
    assert torch.isfinite(model.model.layers[0].self_attn.sinks).all()
    assert has_live_attention_sinks(model) is True


def test_the_reference_trl_builds_itself_runs_the_pretrained_sinks(tmp_path, monkeypatch):
    """The premise, on the installed TRL and transformers: TRL's own build applies no sinks policy and lands
    on eager attention, which adds the sink column to every softmax."""
    # A reset-sinks SDPA load earlier in this process opens the class's SDPA gate for the whole process
    # (``_enable_sink_model_sdpa``); the premise is transformers' own default, so it reads the declared flag.
    monkeypatch.setattr(GptOssPreTrainedModel, "_supports_sdpa", False)
    config = {key: value for key, value in TINY_GPTOSS_CONFIG.items() if key != "attn_implementation"}
    GptOssForCausalLM(GptOssConfig(**{**config, "num_hidden_layers": 2})).save_pretrained(tmp_path)
    reference = create_model_from_path(str(tmp_path))
    assert reference.config._attn_implementation == "eager"
    assert has_live_attention_sinks(reference) is True


@pytest.mark.parametrize("ep_size", [1, 2, 8])
def test_a_reference_without_the_live_sinks_is_rejected_in_every_parallelism_mode(ep_size):
    """The guard must reach the sinks in every mode: at ep_size == 1 an early return never sees them."""
    with pytest.raises(ValueError, match=r"disagree on attention sinks \(policy: live, reference: not live\)"):
        _guard(_live_model(), ref_model=_reset_model(), ep_size=ep_size)


@pytest.mark.parametrize("ep_size", [1, 8])
@pytest.mark.parametrize("reference", [_live_model, _trl_built_model], ids=["live", "trl-built"])
def test_a_live_reference_beside_a_reset_policy_is_rejected(ep_size, reference):
    """A reference TRL built itself runs the pretrained sinks the neutralized policy dropped."""
    with pytest.raises(ValueError, match=r"\(policy: not live, reference: live\)"):
        _guard(_reset_model(), ref_model=reference(), ep_size=ep_size)


@pytest.mark.parametrize("ep_size", [1, 8])
def test_a_reference_carrying_the_policys_sinks_is_allowed(ep_size):
    """The over-fire guard: a reference loaded with the policy's reset_sinks scores under the same sinks, live
    or neutralized, a dtype-min-filled sink is not a live sink, and a reference TRL built itself carries the
    live policy's pretrained sinks."""
    _guard(_live_model(), ref_model=_live_model(), ep_size=ep_size)
    _guard(_reset_model(), ref_model=_reset_model(), ep_size=ep_size)
    _guard(_live_model(), ref_model=_trl_built_model(), ep_size=ep_size)


@pytest.mark.parametrize("ep_size", [1, 8])
def test_no_reference_model_is_never_gated(ep_size):
    """beta == 0 / PEFT: TRL nulls ref_model, so there is no second model to disagree."""
    _guard(_live_model(), ref_model=None, ep_size=ep_size)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
