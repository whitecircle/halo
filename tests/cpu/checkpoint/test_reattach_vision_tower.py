"""reattach_vision_tower rebuilds the servable wrapper layout from a text-only export.

End-to-end on a tiny real Qwen3.5-MoE composite: the trained text weights must land back under
``model.language_model.*`` (values from the EXPORT, not the base), the base's vision tower must
ride along, and the config must be the composite the serving engines register. A wrapper-layout
input must be refused. These tests fail if the prefixing, the base-tensor carryover, or the
config graft breaks.
"""

import json
import os

import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)

from scripts.after_training.reattach_vision_tower import reattach_vision_tower


def _tiny_composite_config() -> Qwen3_5MoeConfig:
    text = {
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "full_attention_interval": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "vocab_size": 64,
        "max_position_embeddings": 64,
    }
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2, "out_hidden_size": 32}
    return Qwen3_5MoeConfig(text_config=text, vision_config=vision)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("reattach")
    base_dir, export_dir = str(root / "base"), str(root / "export")
    composite = _tiny_composite_config()
    torch.manual_seed(0)
    Qwen3_5MoeForConditionalGeneration(composite).save_pretrained(base_dir)

    export = Qwen3_5MoeForCausalLM(composite.text_config)
    # Marker: a value the base cannot contain, proving the output's text weights come from the export.
    with torch.no_grad():
        export.model.embed_tokens.weight.fill_(3.5)
    export.save_pretrained(export_dir)
    return base_dir, export_dir


def _weight_map(directory: str) -> dict[str, str]:
    """key -> shard filename, for indexed and single-file checkpoints alike."""
    index = os.path.join(directory, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            return json.load(f)["weight_map"]
    from safetensors import safe_open

    with safe_open(os.path.join(directory, "model.safetensors"), framework="pt") as f:
        return dict.fromkeys(f.keys(), "model.safetensors")


def test_reattach_rebuilds_wrapper_layout(artifacts, tmp_path):
    base_dir, export_dir = artifacts
    out = str(tmp_path / "out")
    reattach_vision_tower(export_dir, base_dir, out, trust_remote_code=False)

    weight_map = _weight_map(out)
    keys = set(weight_map)
    assert any(k.startswith("model.language_model.") for k in keys), sorted(keys)[:5]
    assert any(k.startswith("model.visual.") for k in keys), "vision tower not carried over"
    assert not any(k.startswith("model.embed_tokens") for k in keys), "text keys left un-prefixed"

    config = AutoConfig.from_pretrained(out)
    assert config.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert getattr(config, "vision_config", None) is not None
    assert config.text_config.vocab_size == 64

    from safetensors import safe_open

    embed_key = "model.language_model.embed_tokens.weight"
    with safe_open(os.path.join(out, weight_map[embed_key]), framework="pt") as f:
        assert torch.all(f.get_tensor(embed_key) == 3.5), "trained export weights were not the ones written"


def test_wrapper_layout_input_is_refused(artifacts, tmp_path):
    base_dir, _ = artifacts
    with pytest.raises(ValueError, match="not a text-only export"):
        reattach_vision_tower(base_dir, base_dir, str(tmp_path / "out2"), trust_remote_code=False)


def test_a_base_storing_its_text_tower_under_another_prefix_is_refused(artifacts, tmp_path):
    """The export supersedes the base's text tower by key prefix. A base keeping a vendor namespace
    (renamed only inside ``from_pretrained``) has nothing under ``model.language_model.``, so every
    one of its text tensors was carried over beside the trained ones — two text towers colliding on
    load — and the run reported success. Refused before the output directory exists."""
    from safetensors.torch import load_file, save_file

    base_dir, export_dir = artifacts
    vendor_base = tmp_path / "vendor_base"
    vendor_base.mkdir()
    tensors = load_file(os.path.join(base_dir, "model.safetensors"))
    save_file(
        {key.replace("model.language_model.", "model.llm.", 1): value for key, value in tensors.items()},
        str(vendor_base / "model.safetensors"),
        metadata={"format": "pt"},
    )
    AutoConfig.from_pretrained(base_dir).save_pretrained(vendor_base)
    out = tmp_path / "out4"
    with pytest.raises(ValueError, match="stores no text tower under model.language_model"):
        reattach_vision_tower(export_dir, str(vendor_base), str(out), trust_remote_code=False)
    assert not out.exists(), "a refused re-attachment must not create its output directory"


def test_an_output_aimed_at_the_base_is_refused(artifacts):
    """The base is streamed from while the writer's close sweeps every ``model*.safetensors`` it did
    not write, so ``--output_dir`` equal to a local ``--model_id`` replaced the base checkpoint with
    the wrapper artifact and reported success."""
    base_dir, export_dir = artifacts
    before = sorted(os.listdir(base_dir))
    with pytest.raises(ValueError, match="same path"):
        reattach_vision_tower(export_dir, base_dir, base_dir, trust_remote_code=False)
    assert sorted(os.listdir(base_dir)) == before, "a refused re-attachment touched the base directory"


def test_a_base_of_another_family_is_refused(artifacts, tmp_path):
    """The graft is a plain attribute assignment, so a Qwen3.5 text tower would slot into a GLM-5
    wrapper's config without complaint — a checkpoint that loads and serves garbage."""
    from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration

    from tests.common.models import TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG

    _, export_dir = artifacts
    other_base = str(tmp_path / "glm5_base")
    Glm5NextForConditionalGeneration(
        Glm5NextConfig(text_config=dict(TINY_GLM5_CONFIG), vision_config=dict(TINY_GLM5_VISION_CONFIG))
    ).save_pretrained(other_base)
    out = tmp_path / "out3"
    with pytest.raises(ValueError, match="wraps a 'glm5_next_text' text tower"):
        reattach_vision_tower(export_dir, other_base, str(out), trust_remote_code=False)
    assert not out.exists(), "a refused re-attachment must not create its output directory"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
