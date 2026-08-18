#!/usr/bin/env python
"""Training-provenance sidecar: record at adapter save, re-apply at merge.

A GptOss adapter trains against live OR neutralized attention sinks (``reset_sinks``), but a PEFT
merge rebuilds the base from the hub, whose sinks are always live — so without the record the merge
silently serves attention the adapter never trained under. ``PeftAdapterSaver`` writes
``training_provenance.json`` next to the adapter, and the merge tools re-apply it via
``_apply_training_provenance``.

    python tests/cpu/checkpoint/test_training_provenance.py
"""

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from transformers import CONFIG_MAPPING

from src.checkpoint.format import PROVENANCE_GPT_OSS_SINKS, TRAINING_PROVENANCE_FILE
from src.checkpoint.tool_io import _apply_training_provenance
from src.distributed.checkpoint.peft import PeftAdapterSaver
from src.models.patches.gpt_oss_sinks import SinksPolicy, apply_sinks_policy, stamped_sinks_policy

# The sinks policy logs through accelerate's logger, which requires an initialized state.
PartialState()

NUM_HEADS = 4


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.sinks = nn.Parameter(torch.ones(NUM_HEADS))


class _GptOssStub(nn.Module):
    """Minimal GptOss-shaped tree: real family config + one decoder layer with live sinks."""

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
        PeftAdapterSaver._write_training_provenance(model, d)
        assert _read(d) == {PROVENANCE_GPT_OSS_SINKS: SinksPolicy.NEUTRALIZED}


def test_writer_records_the_live_policy():
    model = _GptOssStub()
    apply_sinks_policy(model, model.config, policy=SinksPolicy.LIVE)
    with tempfile.TemporaryDirectory() as d:
        PeftAdapterSaver._write_training_provenance(model, d)
        assert _read(d) == {PROVENANCE_GPT_OSS_SINKS: SinksPolicy.LIVE}


def test_writer_records_the_trainable_policy():
    model = _GptOssStub()
    apply_sinks_policy(model, model.config, policy=SinksPolicy.TRAINABLE, attn_implementation="eager")
    with tempfile.TemporaryDirectory() as d:
        PeftAdapterSaver._write_training_provenance(model, d)
        assert _read(d) == {PROVENANCE_GPT_OSS_SINKS: SinksPolicy.TRAINABLE}


def test_stamp_readable_through_a_non_delegating_wrapper():
    """The stamp lives on the LOADED model; a wrapper whose ``__getattr__`` does not delegate plain
    attributes (bare ``nn.Module`` holding ``.module``) must not read as "no sinks" — that would
    silence the RL sink gate on a wrapped GptOss model."""
    model = _GptOssStub()
    apply_sinks_policy(model, model.config, policy=SinksPolicy.LIVE)
    wrapper = nn.Module()
    wrapper.module = model
    assert stamped_sinks_policy(wrapper) is SinksPolicy.LIVE


@pytest.mark.parametrize("policy", list(SinksPolicy))
def test_failed_layer_walk_raises_instead_of_stamping(policy):
    """A sinks-carrying config whose layer walk finds nothing (unrecognized layout) must raise for
    EVERY policy. Stamping from the zero count recorded ``neutralized`` on a model whose sinks are
    live — silencing the RL sink gate and steering a later merge to neutralize a live artifact."""
    model = _GptOssStub()
    del model.model  # no walkable backbone -> _gpt_oss_sink_attentions yields nothing
    with pytest.raises(RuntimeError, match="touched no attention layers"):
        apply_sinks_policy(model, model.config, policy=policy, attn_implementation="eager")
    assert stamped_sinks_policy(model) is None


def test_writer_records_nothing_when_no_policy_ran():
    """An absent stamp means "nothing to record", never "neutralized" — guessing would neutralize a
    live-sink artifact on the next merge."""
    model = _GptOssStub()
    with tempfile.TemporaryDirectory() as d:
        PeftAdapterSaver._write_training_provenance(model, d)
        assert not os.path.isfile(os.path.join(d, TRAINING_PROVENANCE_FILE))


def test_writer_skips_families_without_sinks():
    """The family signal derives from the stamp: the policy stamps only sink-carrying models, so a
    sink-less family records nothing even after the policy ran."""
    model = nn.Module()
    model.config = CONFIG_MAPPING["qwen3"](num_hidden_layers=1)
    apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED)
    with tempfile.TemporaryDirectory() as d:
        PeftAdapterSaver._write_training_provenance(model, d)
        assert not os.path.isfile(os.path.join(d, TRAINING_PROVENANCE_FILE))


def test_apply_neutralizes_merged_sinks():
    """The merge loads the hub base (live sinks); a neutralized-sinks record must reset them to
    dtype.min — the state the adapter actually trained under and every trainer checkpoint emits."""
    model = _GptOssStub()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, TRAINING_PROVENANCE_FILE), "w") as fh:
            json.dump({"gpt_oss_attention_sinks": "neutralized"}, fh)
        actions = _apply_training_provenance(model, d)
    assert actions, "the neutralization must be reported to the tool's output"
    sinks = model.model.layers[0].self_attn.sinks
    assert torch.equal(sinks.data, torch.full((NUM_HEADS,), torch.finfo(sinks.dtype).min))


def test_apply_keeps_live_sinks_untouched():
    model = _GptOssStub()
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, TRAINING_PROVENANCE_FILE), "w") as fh:
            json.dump({"gpt_oss_attention_sinks": "live"}, fh)
        assert _apply_training_provenance(model, d) == []
    assert torch.equal(model.model.layers[0].self_attn.sinks.data, torch.ones(NUM_HEADS))


def test_apply_is_a_noop_without_the_record():
    model = _GptOssStub()
    with tempfile.TemporaryDirectory() as d:
        assert _apply_training_provenance(model, d) == []
    assert torch.equal(model.model.layers[0].self_attn.sinks.data, torch.ones(NUM_HEADS))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
