#!/usr/bin/env python
"""The GptOss sink policies and their gates.

Pins: the two user knobs resolve to one ``SinksPolicy`` (the contradiction raises); every policy
leaves the sink values untouched except the reset, and sets ``requires_grad`` the way its name says;
TRAINABLE counts as live for the trainer gates and is refused under an implementation without a sink
gradient, under an adapter run, and under weight-sync RL.

Fails if: the trainable branch mutates sink values or leaves them frozen, the reset stops freezing,
the live branch stops freezing, any of the refusals disappears, or TRAINABLE stops counting as live.
"""

from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from transformers import GptOssConfig, GptOssForCausalLM

PartialState()  # the sinks policy logs through accelerate's logger, which requires live state

from src.distributed.loading.peft_setup import setup_peft_model
from src.models.patches.gpt_oss_sinks import (
    SinksPolicy,
    apply_sinks_policy,
    has_live_attention_sinks,
    stamped_sinks_policy,
)
from src.trainers.grpo.rollout.weight_sync import validate_weight_sync_support


def _tiny_gpt_oss():
    config = GptOssConfig(
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
    )
    torch.manual_seed(0)
    model = GptOssForCausalLM(config)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.sinks.normal_()
    return model, config


def _sinks(model):
    return [layer.self_attn.sinks for layer in model.model.layers]


def test_from_flags_resolves_the_two_knobs():
    assert SinksPolicy.from_flags(reset_sinks=True) is SinksPolicy.NEUTRALIZED
    assert SinksPolicy.from_flags(reset_sinks=False) is SinksPolicy.LIVE
    assert SinksPolicy.from_flags(reset_sinks=False, train_sinks=True) is SinksPolicy.TRAINABLE
    with pytest.raises(ValueError, match="contradicts"):
        SinksPolicy.from_flags(reset_sinks=True, train_sinks=True)
    assert not SinksPolicy.NEUTRALIZED.live and SinksPolicy.LIVE.live and SinksPolicy.TRAINABLE.live


def test_trainable_branch_keeps_values_and_grads():
    model, config = _tiny_gpt_oss()
    before = [s.detach().clone() for s in _sinks(model)]
    apply_sinks_policy(model, config, policy=SinksPolicy.TRAINABLE, attn_implementation="eager")
    for sinks, prior in zip(_sinks(model), before, strict=True):
        assert torch.equal(sinks.detach(), prior), "trainable branch must not mutate sink values"
        assert sinks.requires_grad, "trainable branch must leave sinks trainable"
    assert stamped_sinks_policy(model) is SinksPolicy.TRAINABLE
    assert has_live_attention_sinks(model), "TRAINABLE must count as live for the trainer gates"


def test_live_branch_freezes_and_reset_branch_neutralizes_frozen():
    model, config = _tiny_gpt_oss()
    apply_sinks_policy(model, config, policy=SinksPolicy.LIVE, attn_implementation="eager")
    assert all(not s.requires_grad for s in _sinks(model)), "live (non-trainable) branch must freeze"
    assert stamped_sinks_policy(model) is SinksPolicy.LIVE

    model, config = _tiny_gpt_oss()
    apply_sinks_policy(model, config, policy=SinksPolicy.NEUTRALIZED, attn_implementation="eager")
    for sinks in _sinks(model):
        assert torch.all(sinks == torch.finfo(sinks.dtype).min), "reset must fill dtype min"
        assert not sinks.requires_grad, "a neutralized sink has zero gradient — it must not stay in the optimizer"
    assert not has_live_attention_sinks(model)


@pytest.mark.parametrize("attn_implementation", ["flash_attention_2", "eager"])
def test_an_attention_without_sinks_is_refused_on_every_branch(attn_implementation):
    """A sink-less attention (an out-of-tree or remote-code GptOss variant) must fail the same way on
    both branches: the FA2 branch's ``attn.sinks = None`` would otherwise CREATE the attribute and
    stamp NEUTRALIZED over a model whose attention never had a sink to disable."""
    model, config = _tiny_gpt_oss()
    for layer in model.model.layers:
        del layer.self_attn._parameters["sinks"]

    with pytest.raises(AttributeError, match="sinks"):
        apply_sinks_policy(model, config, policy=SinksPolicy.NEUTRALIZED, attn_implementation=attn_implementation)
    assert stamped_sinks_policy(model) is None, "nothing was neutralized, so nothing may be stamped"


@pytest.mark.parametrize("attn_implementation", ["flex_attention", "flash_attention_3", "sdpa", None])
def test_trainable_refuses_implementations_without_a_sink_gradient(attn_implementation):
    model, config = _tiny_gpt_oss()
    with pytest.raises(ValueError, match="sink gradient"):
        apply_sinks_policy(model, config, policy=SinksPolicy.TRAINABLE, attn_implementation=attn_implementation)
    assert stamped_sinks_policy(model) is None, "a refused policy must not be stamped"


def test_adapter_run_refuses_trainable_sinks():
    model, config = _tiny_gpt_oss()
    apply_sinks_policy(model, config, policy=SinksPolicy.TRAINABLE, attn_implementation="eager")
    args = SimpleNamespace(unfreeze_layers_patterns=None, freeze_layers_patterns=None)
    model_config = SimpleNamespace(
        use_peft=True,
        lora_target_modules=["q_proj"],
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_modules_to_save=None,
    )
    with pytest.raises(ValueError, match="full fine-tuning"):
        setup_peft_model(args, model, model_config)


def test_weight_sync_refuses_trainable_sinks():
    model, config = _tiny_gpt_oss()
    apply_sinks_policy(model, config, policy=SinksPolicy.TRAINABLE, attn_implementation="eager")
    with pytest.raises(ValueError, match="SFT-only"):
        validate_weight_sync_support(model)
    # The frozen live policy — the shipped RL shape — passes the same gate.
    model, config = _tiny_gpt_oss()
    apply_sinks_policy(model, config, policy=SinksPolicy.LIVE, attn_implementation="eager")
    validate_weight_sync_support(model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
