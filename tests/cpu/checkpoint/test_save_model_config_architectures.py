#!/usr/bin/env python
"""``config.json``'s ``architectures`` must name the head that was actually trained.

Every parallel save writes the config by hand instead of calling ``save_pretrained`` (which would
also re-write the weights it just gathered), so it has to reproduce what ``save_pretrained`` stamps —
including ``architectures``, which it takes from the LIVE class. The config a run loads still carries
the hub's value: a reward model built as ``Qwen3ForSequenceClassification`` from a ``Qwen3ForCausalLM``
checkpoint keeps ``["Qwen3ForCausalLM"]`` in memory. Writing that through means every
``architectures[0]``-keyed consumer — vLLM, TGI, ``AutoModel`` resolution — silently serves an LM head
off a reward checkpoint. Nothing raises; the artifact simply is not the model.

A real ``PreTrainedModel`` gets its stale value REPLACED here (through the FSDP2 in-place
``__class__`` swap, whose dynamic ``FSDP<Name>`` subclass must not be mistaken for the architecture).
A carrier with no ``PreTrainedModel`` in its MRO cannot be read that way, so its config's value is
written verbatim — which is why the PIPELINE STAGE, the carrier the toolkit actually ships, stamps
the live class at build time instead: it wraps the model away behind a plain ``nn.Module``, so by the
time the writer sees it there is nothing left to derive from.

    python tests/cpu/checkpoint/test_save_model_config_architectures.py
"""

import json
import os

import pytest
import torch.nn as nn
from accelerate import PartialState
from transformers import CONFIG_MAPPING, AutoModelForSequenceClassification

PartialState()  # save_model_config logs through accelerate's logger

from src.checkpoint.config_export import hf_architecture_name, save_model_config
from src.distributed.pipeline_parallel.stage import build_pipeline_stage

HUB_ARCHITECTURE = "Qwen3ForCausalLM"  # what the base checkpoint's config.json carries
LIVE_ARCHITECTURE = "Qwen3ForSequenceClassification"  # what the run actually built


def _tiny_qwen3_config(num_hidden_layers: int = 1):
    """Every field goes through the constructor: 5.14's strict validator derives ``layer_types`` in
    ``__init__``, so shrinking ``num_hidden_layers`` afterwards fails at write time."""
    return CONFIG_MAPPING["qwen3"](
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=64,
        vocab_size=128,
        tie_word_embeddings=False,
        num_labels=1,
        architectures=[HUB_ARCHITECTURE],
    )


def _written_architectures(output_dir: str) -> list[str]:
    with open(os.path.join(output_dir, "config.json")) as f:
        return json.load(f)["architectures"]


def test_the_live_task_class_replaces_the_stale_hub_value(tmp_path):
    model = AutoModelForSequenceClassification.from_config(_tiny_qwen3_config())
    # Anti-vacuity: the planted value must be the WRONG one, or the assertion below is an identity.
    assert type(model).__name__ == LIVE_ARCHITECTURE
    assert model.config.architectures == [HUB_ARCHITECTURE]

    save_model_config(model, str(tmp_path))

    assert _written_architectures(str(tmp_path)) == [LIVE_ARCHITECTURE]


def test_the_fsdp2_class_swap_is_not_mistaken_for_the_architecture(tmp_path):
    """FSDP2 rewrites ``model.__class__`` in place to a dynamic ``FSDP<Name>`` subclass defined in
    torch's namespace. Reading ``type(model).__name__`` would stamp that name into every sharded
    run's config.json — a class no loader can resolve."""
    model = AutoModelForSequenceClassification.from_config(_tiny_qwen3_config())
    swapped = type(f"FSDP{type(model).__name__}", (type(model),), {})
    swapped.__module__ = "torch.distributed.fsdp._fully_shard._fsdp_init"
    model.__class__ = swapped

    save_model_config(model, str(tmp_path))

    assert _written_architectures(str(tmp_path)) == [LIVE_ARCHITECTURE]


def test_a_non_hf_carrier_writes_the_value_its_config_carries(tmp_path):
    """A carrier with no ``PreTrainedModel`` in its MRO — the pipeline stage, a sentence-transformers
    shell — has no live class to read, so the writer must pass its config's value through verbatim
    rather than invent one from the carrier's own class. Whatever stamped that value upstream is the
    only answer available here."""
    carrier = nn.Module()
    carrier.config = _tiny_qwen3_config()
    carrier.config.architectures = [LIVE_ARCHITECTURE]  # what build_pipeline_stage stamps
    assert hf_architecture_name(carrier) is None, "premise: nothing in this carrier names an architecture"

    save_model_config(carrier, str(tmp_path))

    assert _written_architectures(str(tmp_path)) == [LIVE_ARCHITECTURE]


def test_a_pipeline_stage_carries_the_live_task_class_not_the_hubs(tmp_path):
    """The stage module is the carrier the writers actually receive under PP, and it is a plain
    ``nn.Module`` — so the value has to be stamped where the real class is still in hand.

    A reward/classification run builds ``Qwen3ForSequenceClassification`` from a ``Qwen3ForCausalLM``
    checkpoint; without the stamp every PP stage ships ``["Qwen3ForCausalLM"]`` and an
    ``architectures[0]``-keyed consumer serves an LM head off a reward checkpoint.
    """
    model = AutoModelForSequenceClassification.from_config(_tiny_qwen3_config(num_hidden_layers=2))
    assert model.config.architectures == [HUB_ARCHITECTURE], "premise: the loaded config is stale"

    stage = build_pipeline_stage(model, pp_rank=1, pp_size=2)

    assert hf_architecture_name(stage) is None, "premise: the stage hides the HF class from the writer"
    assert stage.config.architectures == [LIVE_ARCHITECTURE]

    save_model_config(stage, str(tmp_path))

    assert _written_architectures(str(tmp_path)) == [LIVE_ARCHITECTURE]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
