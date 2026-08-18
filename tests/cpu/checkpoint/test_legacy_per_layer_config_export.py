#!/usr/bin/env python
"""Every exported ``config.json`` must carry the flat legacy attention keys the rollout server reads.

transformers 5.16 folds Gemma 4's full-attention geometry (``global_head_dim``,
``num_global_key_value_heads``) into ``per_layer_config`` and serializes only that folded form. The
pinned vLLM server parses checkpoints with the transformers 5.14 line, which refuses a config that
carries ``per_layer_config`` at all (``AmbiguousGlobalPerLayerAttributeError`` at parse) and reads
the two flat keys instead. The family declares the mapping on its EP layer class
(``_LEGACY_PER_LAYER_CONFIG_KEYS``) and every config writer the toolkit owns flattens through it —
``save_model_config`` (the parallel saves), ``save_full_checkpoint`` (the after-training tools, the
PEFT merge output) and the EP gathered save on top of them.

Nothing raises on the export side without the rewrite: the artifact loads on 5.16 and only the
server refuses it, so the gate is the written JSON — the flat keys present at the values the
full-attention layers carry, ``per_layer_config`` gone, and a 5.16 reload rebuilding the same
per-layer geometry from the flat keys.

    python tests/cpu/checkpoint/test_legacy_per_layer_config_export.py
"""

import json
import os

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from transformers import CONFIG_MAPPING
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

PartialState()  # save_ep_model logs through accelerate's logger

from src.checkpoint.config_export import (
    export_legacy_per_layer_config,
    flatten_per_layer_config,
    save_model_config,
)
from src.checkpoint.tool_io import save_full_checkpoint
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.distributed.expert_parallel.saving import save_ep_model
from tests.common.models import TINY_GEMMA4_MOE_CONFIG, TINY_QWEN3_MOE_CONFIG

GLOBAL_HEAD_DIM = TINY_GEMMA4_MOE_CONFIG["global_head_dim"]
GLOBAL_KV_HEADS = TINY_GEMMA4_MOE_CONFIG["num_global_key_value_heads"]
LEGACY_KEYS = ("global_head_dim", "num_global_key_value_heads")


def _carrier(config) -> nn.Module:
    """What the parallel writers hand ``save_model_config``: a module carrying the config."""
    carrier = nn.Module()
    carrier.config = config
    return carrier


def _config_json(output_dir: str) -> dict:
    with open(os.path.join(output_dir, "config.json")) as f:
        return json.load(f)


def _per_layer_geometry(text_config) -> list[tuple[int, int]]:
    return [(layer.head_dim, layer.num_key_value_heads) for layer in text_config.per_layer_config]


def _assert_flat_form(section: dict) -> None:
    assert "per_layer_config" not in section, section.get("per_layer_config")
    assert section["global_head_dim"] == GLOBAL_HEAD_DIM
    assert section["num_global_key_value_heads"] == GLOBAL_KV_HEADS
    # The sliding-layer values stay on the plain keys — the flat keys carry ONLY the full layers.
    assert section["head_dim"] == TINY_GEMMA4_MOE_CONFIG["head_dim"]
    assert section["num_key_value_heads"] == TINY_GEMMA4_MOE_CONFIG["num_key_value_heads"]


def test_premise_a_plain_5_16_write_folds_the_keys(tmp_path):
    """Anti-vacuity: on this transformers a plain ``save_pretrained`` writes the folded form and none
    of the flat keys — otherwise every assertion below passes without the rewrite."""
    Gemma4Config(text_config=TINY_GEMMA4_MOE_CONFIG, vision_config=None, audio_config=None).save_pretrained(tmp_path)
    text = _config_json(str(tmp_path))["text_config"]
    assert text["per_layer_config"], "5.16 no longer folds Gemma 4's geometry — the rewrite may be retired"
    assert not any(key in text for key in LEGACY_KEYS)
    # And the geometry is genuinely heterogeneous, so a flatten has something to express.
    assert TINY_GEMMA4_MOE_CONFIG["head_dim"] != GLOBAL_HEAD_DIM
    assert TINY_GEMMA4_MOE_CONFIG["num_key_value_heads"] != GLOBAL_KV_HEADS


def test_a_composite_export_writes_the_flat_keys_on_the_text_section(tmp_path):
    config = Gemma4Config(text_config=TINY_GEMMA4_MOE_CONFIG, vision_config=None, audio_config=None)
    save_model_config(_carrier(config), str(tmp_path))
    payload = _config_json(str(tmp_path))
    _assert_flat_form(payload["text_config"])
    assert "per_layer_config" not in payload  # the wrapper section never carried one


def test_a_text_only_export_writes_the_flat_keys_on_the_root_section(tmp_path):
    save_model_config(_carrier(Gemma4TextConfig(**TINY_GEMMA4_MOE_CONFIG)), str(tmp_path))
    _assert_flat_form(_config_json(str(tmp_path)))


def test_the_flat_form_reloads_to_the_same_per_layer_geometry(tmp_path):
    """The flat keys are the whole of what a reload rebuilds, so the rewrite is lossless only if a
    5.16 reload of the flattened file yields the per-layer geometry the source carried."""
    source = Gemma4Config(text_config=TINY_GEMMA4_MOE_CONFIG, vision_config=None, audio_config=None)
    save_model_config(_carrier(source), str(tmp_path))
    reloaded = Gemma4Config.from_pretrained(str(tmp_path))
    assert _per_layer_geometry(reloaded.text_config) == _per_layer_geometry(source.text_config)
    assert reloaded.text_config.layer_types == source.text_config.layer_types
    assert len(set(_per_layer_geometry(source.text_config))) == 2, "premise: two distinct layer geometries"


def test_save_full_checkpoint_writes_the_flat_form(tmp_path):
    """The after-training tools' writer (the PEFT merge output among them) goes through
    ``model.save_pretrained``, not ``save_model_config`` — it owes the same rewrite."""
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(Gemma4TextConfig(**TINY_GEMMA4_MOE_CONFIG)).to(torch.bfloat16)
    save_full_checkpoint(model, str(tmp_path))
    _assert_flat_form(_config_json(str(tmp_path)))
    assert Gemma4ForCausalLM.from_pretrained(str(tmp_path), dtype=torch.bfloat16) is not None


def test_the_ep_gathered_save_writes_the_flat_form(tmp_path):
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(Gemma4TextConfig(**TINY_GEMMA4_MOE_CONFIG)).to(torch.bfloat16)
    patch_moe_model_for_ep(model, EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False))
    save_ep_model(model, str(tmp_path))
    _assert_flat_form(_config_json(str(tmp_path)))


def test_a_family_declaring_no_legacy_keys_is_written_verbatim(tmp_path):
    config = CONFIG_MAPPING["qwen3_moe"](**TINY_QWEN3_MOE_CONFIG)
    reference_dir, export_dir = tmp_path / "reference", tmp_path / "export"
    config.save_pretrained(reference_dir)
    save_model_config(_carrier(config), str(export_dir))
    assert _config_json(str(export_dir)) == _config_json(str(reference_dir))


def test_a_config_already_in_the_flat_form_is_left_untouched(tmp_path):
    """A hub-format file (5.14-written, or one this rewrite already produced) has nothing to fold."""
    config_file = tmp_path / "config.json"
    flat = {
        **TINY_GEMMA4_MOE_CONFIG,
        "model_type": "gemma4_text",
        "layer_types": ["sliding_attention", "full_attention"],
    }
    config_file.write_text(json.dumps(flat))
    before = config_file.read_bytes()
    export_legacy_per_layer_config(str(tmp_path))
    assert config_file.read_bytes() == before


def test_an_adapter_only_directory_has_nothing_to_rewrite(tmp_path):
    export_legacy_per_layer_config(str(tmp_path))
    assert os.listdir(tmp_path) == []


GEMMA4_KEYS = {
    "global_head_dim": ("full_attention", "head_dim"),
    "num_global_key_value_heads": ("full_attention", "num_key_value_heads"),
}


def _section(per_layer_config: dict) -> dict:
    return {
        "layer_types": ["sliding_attention", "full_attention", "full_attention"],
        "head_dim": 8,
        "num_key_value_heads": 4,
        "sliding_window": 16,
        "per_layer_config": per_layer_config,
    }


def test_flatten_takes_the_full_attention_values_and_drops_the_fold():
    section = _section(
        {"1": {"head_dim": 16, "num_key_value_heads": 2}, "2": {"head_dim": 16, "num_key_value_heads": 2}}
    )
    assert flatten_per_layer_config(section, GEMMA4_KEYS) is True
    assert section["global_head_dim"] == 16
    assert section["num_global_key_value_heads"] == 2
    assert "per_layer_config" not in section
    assert section["head_dim"] == 8 and section["num_key_value_heads"] == 4


def test_flatten_falls_back_to_the_global_value_for_an_unoverridden_field():
    """``attention_k_eq_v: false`` leaves the KV count unoverridden on the full layers; the flat key
    then carries the global value (what a 5.14 reader falls back to as well)."""
    section = _section({"1": {"head_dim": 16}, "2": {"head_dim": 16}})
    flatten_per_layer_config(section, GEMMA4_KEYS)
    assert section["global_head_dim"] == 16
    assert section["num_global_key_value_heads"] == 4


def test_flatten_refuses_a_declared_field_that_differs_between_layers_of_one_type():
    section = _section({"1": {"head_dim": 16}, "2": {"head_dim": 32}})
    with pytest.raises(ValueError, match="global_head_dim"):
        flatten_per_layer_config(section, GEMMA4_KEYS)


def test_flatten_refuses_an_override_no_flat_key_expresses():
    section = _section({"0": {"head_dim": 16}, "1": {"head_dim": 16}, "2": {"head_dim": 16}})
    with pytest.raises(ValueError, match="layer 0"):
        flatten_per_layer_config(section, GEMMA4_KEYS)


def test_flatten_tolerates_an_undeclared_override_equal_to_the_global():
    section = _section({"1": {"head_dim": 16, "sliding_window": 16}, "2": {"head_dim": 16}})
    flatten_per_layer_config(section, GEMMA4_KEYS)
    assert section["global_head_dim"] == 16


def test_flatten_is_a_no_op_without_a_fold():
    section = {"layer_types": ["full_attention"], "head_dim": 8}
    assert flatten_per_layer_config(section, GEMMA4_KEYS) is False
    assert section == {"layer_types": ["full_attention"], "head_dim": 8}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
