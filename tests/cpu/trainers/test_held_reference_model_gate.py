#!/usr/bin/env python
"""The held-reference guard and its live-sinks predicate: a reference with other sinks biases the KL on every token.
The predicate reads the policy stamp: the reset yields ``sinks is None`` only under FA2, ``dtype.min`` elsewhere."""

import types

import pytest
import torch
import torch.nn as nn
from transformers import GptOssConfig, GptOssForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssPreTrainedModel
from trl.trainer.utils import create_model_from_path

from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.frozen_models import PREFERENCE_REFERENCE_ALTERNATIVES
from halo.models.attention.gpt_oss_sinks import SinksPolicy, apply_sinks_policy, has_live_attention_sinks
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.models import TINY_GPTOSS_CONFIG
from tests.common.parallelism import make_parallelism_config

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
    stub = types.SimpleNamespace(
        model=model,
        ref_model=ref_model,
        parallelism_config=make_parallelism_config(ep_size=ep_size) if ep_size > 1 else ParallelismConfig(),
        REFERENCE_ALTERNATIVES=PREFERENCE_REFERENCE_ALTERNATIVES,
    )
    return DistributedTrainerMixin._validate_held_reference_model(stub)


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
    """A non-sinks family must not stamp the flag, and an unstamped one is not live."""
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
    """TRL's build applies no sinks policy and lands on eager attention, which adds the sink column to each softmax."""
    # An earlier reset-sinks SDPA load opens the class's SDPA gate for the process (``install_sdpa_for_reset_sinks``).
    monkeypatch.setattr(GptOssPreTrainedModel, "_supports_sdpa", False)
    config = {key: value for key, value in TINY_GPTOSS_CONFIG.items() if key != "attn_implementation"}
    GptOssForCausalLM(GptOssConfig(**{**config, "num_hidden_layers": 2})).save_pretrained(tmp_path)
    reference = create_model_from_path(str(tmp_path))
    assert reference.config._attn_implementation == "eager"
    assert has_live_attention_sinks(reference) is True


@pytest.mark.parametrize("ep_size", [1, 2, 8])
def test_a_reference_without_the_live_sinks_is_rejected_in_every_parallelism_mode(ep_size):
    """At ep_size == 1 an early return would never see the sinks."""
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
    """A dtype-min-filled sink is not live, and a TRL-built reference carries the live policy's pretrained sinks."""
    _guard(_live_model(), ref_model=_live_model(), ep_size=ep_size)
    _guard(_reset_model(), ref_model=_reset_model(), ep_size=ep_size)
    _guard(_live_model(), ref_model=_trl_built_model(), ep_size=ep_size)


@pytest.mark.parametrize("ep_size", [1, 8])
def test_no_reference_model_is_never_gated(ep_size):
    """beta == 0 / PEFT: TRL nulls ref_model, so there is no second model to disagree."""
    _guard(_live_model(), ref_model=None, ep_size=ep_size)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
