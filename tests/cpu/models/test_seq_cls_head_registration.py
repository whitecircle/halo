#!/usr/bin/env python
"""Each toolkit-registered classification head resolves through ``AutoModel``: transformers ships none for Gemma 4 or
MoE Qwen3.5/3.6 (composite and text configs), so a missing spelling fails at load on a checkpoint that trained fine."""

from __future__ import annotations

import ast
import subprocess
import sys

import pytest
import torch
from transformers import AutoModelForSequenceClassification
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
)
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig
from transformers.models.qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

# Importing is the registration under test.
from halo.models.upstream.seq_cls_heads import (
    Gemma4ForSequenceClassification,
    Gemma4TextForSequenceClassification,
    Qwen3_5MoeForSequenceClassification,
    Qwen3_5MoeTextForSequenceClassification,
)
from tests.common.repo import REPO_ROOT

_GEMMA4_TEXT_KWARGS = {
    "vocab_size": 128,
    "vocab_size_per_layer_input": 128,  # defaults to the production 262144-row per-layer table
    "hidden_size": 32,
    "hidden_size_per_layer_input": 16,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "global_head_dim": 8,
    "max_position_embeddings": 128,
    "sliding_window": 16,
    "pad_token_id": 0,
}

_QWEN3_5_MOE_TEXT_KWARGS = {
    "vocab_size": 128,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 128,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 16,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    # The family is hybrid; one of each layer type keeps the tiny stack representative.
    "layer_types": ["linear_attention", "full_attention"],
    "pad_token_id": 0,
}

# (model_type, config factory, expected head class); the id is the ``model_type`` a checkpoint carries.
_REGISTRATIONS = [
    (
        "gemma4",
        lambda: Gemma4Config(text_config=dict(_GEMMA4_TEXT_KWARGS), vision_config=None, audio_config=None),
        Gemma4ForSequenceClassification,
    ),
    ("gemma4_text", lambda: Gemma4TextConfig(**_GEMMA4_TEXT_KWARGS), Gemma4TextForSequenceClassification),
    (
        "qwen3_5_moe",
        lambda: Qwen3_5MoeConfig(text_config=dict(_QWEN3_5_MOE_TEXT_KWARGS)),
        Qwen3_5MoeForSequenceClassification,
    ),
    (
        "qwen3_5_moe_text",
        lambda: Qwen3_5MoeTextConfig(**_QWEN3_5_MOE_TEXT_KWARGS),
        Qwen3_5MoeTextForSequenceClassification,
    ),
]
_IDS = [entry[0] for entry in _REGISTRATIONS]

_PARAMETRIZE = pytest.mark.parametrize(("model_type", "build_config", "expected"), _REGISTRATIONS, ids=_IDS)


def test_the_production_import_path_pulls_in_every_registration():
    """Asserted in a fresh interpreter because this file imports the heads itself: in-process the chain could be
    severed and every other test here would still pass."""
    probe = (
        "import halo.model_loading.parallel.loader\n"
        "from transformers import AutoModelForSequenceClassification as A\n"
        "print(sorted(c.model_type for c in A._model_mapping._extra_content))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    registered = set(ast.literal_eval(result.stdout.strip().splitlines()[-1]))

    assert set(_IDS) <= registered, f"only {sorted(registered)} reachable from the production import"


@pytest.mark.parametrize("model_type", _IDS)
def test_transformers_still_ships_no_native_head(model_type):
    """Once upstream registers its own head, the registration helper no-ops and the module must be retired."""
    assert model_type not in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES


@_PARAMETRIZE
def test_auto_resolves_each_config_class_to_its_own_head(model_type, build_config, expected):
    config = build_config()
    assert config.model_type == model_type, "the registration must cover the spelling checkpoints carry"
    config.num_labels = 3

    model = AutoModelForSequenceClassification.from_config(config)

    assert type(model) is expected
    assert model.score.out_features == 3


@_PARAMETRIZE
def test_each_head_claims_the_config_class_it_is_registered_under(model_type, build_config, expected):
    """The two must agree, or ``from_pretrained`` builds the head and then rejects the config that selected it."""
    assert expected.config_class is type(build_config())


@_PARAMETRIZE
def test_each_head_runs(model_type, build_config, expected):
    """A head that cannot forward would satisfy every mapping assertion, and the text towers have no other coverage."""
    config = build_config()
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_config(config).eval()

    input_ids = torch.randint(1, 127, (2, 5))
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits

    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


@_PARAMETRIZE
def test_resolution_fails_without_the_registration(model_type, build_config, expected, monkeypatch):
    """Without this, the tests above would pass on a transformers release that ships the head itself, and the shim
    could be deleted unnoticed."""
    config_class = expected.config_class
    assert MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING._extra_content[config_class] is expected
    monkeypatch.delitem(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING._extra_content, config_class)

    with pytest.raises(ValueError, match="Unrecognized configuration class"):
        AutoModelForSequenceClassification.from_config(build_config())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
