#!/usr/bin/env python
"""``config.json``'s ``architectures`` names the head that was trained, or vLLM, TGI and ``AutoModel`` serve an LM
head off a reward checkpoint. Parallel saves write the config by hand."""

import json
import os

import pytest
import torch.nn as nn
from transformers import CONFIG_MAPPING, AutoModelForSequenceClassification

from halo.checkpoint.config_export import hf_architecture_name, save_model_config
from halo.distributed.pipeline_parallel.stage import build_pipeline_stage

HUB_ARCHITECTURE = "Qwen3ForCausalLM"  # What the base checkpoint's config.json carries.
LIVE_ARCHITECTURE = "Qwen3ForSequenceClassification"  # What the run actually built.


def _tiny_qwen3_config(num_hidden_layers: int = 1):
    """5.14's strict validator derives ``layer_types`` in ``__init__``, so shrinking ``num_hidden_layers`` later
    fails at write time."""
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
    # Anti-vacuity: the planted value must be the wrong one.
    assert type(model).__name__ == LIVE_ARCHITECTURE
    assert model.config.architectures == [HUB_ARCHITECTURE]

    save_model_config(model, str(tmp_path))

    assert _written_architectures(str(tmp_path)) == [LIVE_ARCHITECTURE]


def test_the_fsdp2_class_swap_is_not_mistaken_for_the_architecture(tmp_path):
    """Reading ``type(model).__name__`` would stamp ``FSDP<Name>`` into every sharded run's config.json, a class no
    loader can resolve."""
    model = AutoModelForSequenceClassification.from_config(_tiny_qwen3_config())
    swapped = type(f"FSDP{type(model).__name__}", (type(model),), {})
    swapped.__module__ = "torch.distributed.fsdp._fully_shard._fsdp_init"
    model.__class__ = swapped

    save_model_config(model, str(tmp_path))

    assert _written_architectures(str(tmp_path)) == [LIVE_ARCHITECTURE]


def test_a_non_hf_carrier_writes_the_value_its_config_carries(tmp_path):
    """No live class to read, so the value passes through verbatim rather than from the carrier's class."""
    carrier = nn.Module()
    carrier.config = _tiny_qwen3_config()
    carrier.config.architectures = [LIVE_ARCHITECTURE]  # What build_pipeline_stage stamps.
    assert hf_architecture_name(carrier) is None, "premise: nothing in this carrier names an architecture"

    save_model_config(carrier, str(tmp_path))

    assert _written_architectures(str(tmp_path)) == [LIVE_ARCHITECTURE]


def test_a_pipeline_stage_carries_the_live_task_class_not_the_hubs(tmp_path):
    """The stage is a plain ``nn.Module``, so the value is stamped where the real class is still in hand."""
    model = AutoModelForSequenceClassification.from_config(_tiny_qwen3_config(num_hidden_layers=2))
    assert model.config.architectures == [HUB_ARCHITECTURE], "premise: the loaded config is stale"

    stage = build_pipeline_stage(model, pp_rank=1, pp_size=2)

    assert hf_architecture_name(stage) is None, "premise: the stage hides the HF class from the writer"
    assert stage.config.architectures == [LIVE_ARCHITECTURE]

    save_model_config(stage, str(tmp_path))

    assert _written_architectures(str(tmp_path)) == [LIVE_ARCHITECTURE]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
