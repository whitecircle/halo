#!/usr/bin/env python
"""Training-provenance sidecar: recorded at adapter save, re-applied at merge. A PEFT merge rebuilds the base from the
hub, whose sinks are live, and would silently serve attention the adapter never trained under."""

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from transformers import CONFIG_MAPPING

from halo.checkpoint.format import TRAINING_PROVENANCE_FILE
from halo.checkpoint.sidecars import PROVENANCE_GPT_OSS_SINKS, apply_training_provenance, write_training_provenance
from halo.models.attention.gpt_oss_sinks import SinksPolicy, apply_sinks_policy, stamped_sinks_policy

NUM_HEADS = 4


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.sinks = nn.Parameter(torch.ones(NUM_HEADS))


class _GptOssStub(nn.Module):
    """Real family config, one decoder layer with live sinks."""

    def __init__(self):
        super().__init__()
        self.config = CONFIG_MAPPING["gpt_oss"](
            num_hidden_layers=1, num_attention_heads=NUM_HEADS, num_local_experts=4, num_experts_per_tok=2
        )
        layer = nn.Module()
        layer.self_attn = _Attention()
        backbone = nn.Module()
        backbone.layers = nn.ModuleList([layer])
        self.model = backbone


def _read(d: str) -> dict:
    with open(os.path.join(d, TRAINING_PROVENANCE_FILE)) as fh:
        return json.load(fh)


def test_writer_records_the_neutralized_policy():
    model = _GptOssStub()
    apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED)
    with tempfile.TemporaryDirectory() as d:
        write_training_provenance(model, d)
        assert _read(d) == {PROVENANCE_GPT_OSS_SINKS: SinksPolicy.NEUTRALIZED}


def test_writer_records_the_live_policy():
    model = _GptOssStub()
    apply_sinks_policy(model, model.config, policy=SinksPolicy.LIVE)
    with tempfile.TemporaryDirectory() as d:
        write_training_provenance(model, d)
        assert _read(d) == {PROVENANCE_GPT_OSS_SINKS: SinksPolicy.LIVE}


def test_writer_records_the_trainable_policy():
    model = _GptOssStub()
    apply_sinks_policy(model, model.config, policy=SinksPolicy.TRAINABLE, attn_implementation="eager")
    with tempfile.TemporaryDirectory() as d:
        write_training_provenance(model, d)
        assert _read(d) == {PROVENANCE_GPT_OSS_SINKS: SinksPolicy.TRAINABLE}


def test_stamp_readable_through_a_non_delegating_wrapper():
    """A wrapper not delegating plain attributes must not read as no sinks, silencing the RL sink gate."""
    model = _GptOssStub()
    apply_sinks_policy(model, model.config, policy=SinksPolicy.LIVE)
    wrapper = nn.Module()
    wrapper.module = model
    assert stamped_sinks_policy(wrapper) is SinksPolicy.LIVE


@pytest.mark.parametrize("policy", list(SinksPolicy))
def test_failed_layer_walk_raises_instead_of_stamping(policy):
    """Stamping from a zero count records ``neutralized`` on live sinks and misleads the RL gate and a later merge."""
    model = _GptOssStub()
    del model.model  # No walkable backbone, so _gpt_oss_sink_attentions yields nothing.
    with pytest.raises(RuntimeError, match="touched no attention layers"):
        apply_sinks_policy(model, model.config, policy=policy, attn_implementation="eager")
    assert stamped_sinks_policy(model) is None


def test_writer_records_nothing_when_no_policy_ran():
    """An absent stamp means nothing to record; guessing would neutralize a live-sink artifact on the next merge."""
    model = _GptOssStub()
    with tempfile.TemporaryDirectory() as d:
        write_training_provenance(model, d)
        assert not os.path.isfile(os.path.join(d, TRAINING_PROVENANCE_FILE))


def test_writer_skips_families_without_sinks():
    """The policy stamps only sink-carrying models."""
    model = nn.Module()
    model.config = CONFIG_MAPPING["qwen3"](num_hidden_layers=1)
    apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED)
    with tempfile.TemporaryDirectory() as d:
        write_training_provenance(model, d)
        assert not os.path.isfile(os.path.join(d, TRAINING_PROVENANCE_FILE))


def test_apply_neutralizes_merged_sinks():
    """Back to dtype.min, the state the adapter trained under."""
    model = _GptOssStub()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, TRAINING_PROVENANCE_FILE), "w") as fh:
            json.dump({"gpt_oss_attention_sinks": "neutralized"}, fh)
        actions = apply_training_provenance(model, d)
    assert actions, "the neutralization must be reported to the tool's output"
    sinks = model.model.layers[0].self_attn.sinks
    assert torch.equal(sinks.data, torch.full((NUM_HEADS,), torch.finfo(sinks.dtype).min))


def test_apply_keeps_live_sinks_untouched():
    model = _GptOssStub()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, TRAINING_PROVENANCE_FILE), "w") as fh:
            json.dump({"gpt_oss_attention_sinks": "live"}, fh)
        assert apply_training_provenance(model, d) == []
    assert torch.equal(model.model.layers[0].self_attn.sinks.data, torch.ones(NUM_HEADS))


def test_apply_is_a_noop_without_the_record():
    model = _GptOssStub()
    with tempfile.TemporaryDirectory() as d:
        assert apply_training_provenance(model, d) == []
    assert torch.equal(model.model.layers[0].self_attn.sinks.data, torch.ones(NUM_HEADS))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
