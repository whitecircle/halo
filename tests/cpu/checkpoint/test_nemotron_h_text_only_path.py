#!/usr/bin/env python
"""The whole text-only path of Nemotron-H against the release's spelling: a load off the composite checkpoint, an export
in the tower's own spelling, and the reattach that restores the composite with everything no run trains, its multi-token
head included, byte for byte from the base."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForImageTextToText, NemotronHForCausalLM

from halo.checkpoint.tools import adapter_merge
from halo.checkpoint.tools.offline_export import save_full_checkpoint
from halo.checkpoint.tools.vision_reattach import reattach_vision_tower
from halo.model_loading import model_preparation
from halo.model_loading.model_preparation import auto_load_model
from tests.common.nemotron_h import HUB_VISION_NORM, NEMOTRON_H_COMPOSITE, write_tiny_hub_checkpoint
from tests.common.tiny_models import tiny_family_model, word_level_tokenizer

_MTP = "language_model.mtp."
_TRAINED = ("language_model.backbone.", "language_model.lm_head.")
_HEAD_MARKER = 3.5
_ADAPTER = LoraConfig(r=2, lora_alpha=4, target_modules=["in_proj", "q_proj"], task_type="CAUSAL_LM")


def _safetensors(directory: Path) -> dict[str, torch.Tensor]:
    return load_file(directory / "model.safetensors")


def _text_export(base: Path, export: Path, **load_kwargs) -> NemotronHForCausalLM:
    model = auto_load_model(str(base), text_only=True, dtype=torch.float32, **load_kwargs)
    assert type(model) is NemotronHForCausalLM
    with torch.no_grad():
        model.lm_head.weight.fill_(_HEAD_MARKER)
    save_full_checkpoint(model, str(export), source_dir=str(base))
    return model


@pytest.fixture
def base(tmp_path) -> Path:
    directory = tmp_path / "base"
    write_tiny_hub_checkpoint(directory)
    return directory


def test_a_text_export_stores_the_towers_spelling_with_no_composite_level(base, tmp_path):
    _text_export(base, tmp_path / "export")
    written = _safetensors(tmp_path / "export")
    assert {key.split(".")[0] for key in written} == {"backbone", "lm_head"}, sorted(written)[:6]
    expected = {key.removeprefix("language_model.") for key in _safetensors(base) if key.startswith(_TRAINED)}
    assert set(written) == expected, sorted(set(written) ^ expected)[:6]
    assert AutoConfig.from_pretrained(tmp_path / "export").model_type == "nemotron_h"


def test_the_reattach_restores_every_key_the_release_stores(base, tmp_path):
    export, out = tmp_path / "export", tmp_path / "out"
    _text_export(base, export)
    reattach_vision_tower(str(export), str(base), str(out), trust_remote_code=False)
    assert set(_safetensors(out)) == set(_safetensors(base))


def test_what_no_run_trains_streams_back_byte_for_byte(base, tmp_path):
    """The multi-token head, the vision tower, its norm and the projector; a trained head shows the export is the source
    of the rest."""
    export, out = tmp_path / "export", tmp_path / "out"
    _text_export(base, export)
    reattach_vision_tower(str(export), str(base), str(out), trust_remote_code=False)
    written, stored = _safetensors(out), _safetensors(base)
    carried = [key for key in stored if not key.startswith(_TRAINED)]
    assert any(key.startswith(_MTP) for key in carried), "the fixture holds a multi-token head"
    for key in carried:
        assert torch.equal(written[key], stored[key]), key
    assert torch.all(written["language_model.lm_head.weight"] == _HEAD_MARKER)
    assert not torch.equal(written["language_model.lm_head.weight"], stored["language_model.lm_head.weight"])


def test_the_reattached_composite_loads_natively_with_the_trained_head_and_the_release_norm(base, tmp_path):
    export, out = tmp_path / "export", tmp_path / "out"
    _text_export(base, export)
    reattach_vision_tower(str(export), str(base), str(out), trust_remote_code=False)
    reloaded = AutoModelForImageTextToText.from_pretrained(out, dtype=torch.float32)
    assert torch.all(reloaded.language_model.lm_head.weight == _HEAD_MARKER)
    assert torch.equal(reloaded.vision_final_layernorm.weight, _safetensors(base)[f"{HUB_VISION_NORM}.weight"])


def _model_of_another_width(directory: Path) -> None:
    """The family's other sizes share its model types, so only the tensors tell them apart."""
    torch.manual_seed(1)
    model = tiny_family_model(NEMOTRON_H_COMPOSITE, overrides={"hidden_size": 128})
    model.save_pretrained(directory)


def test_a_base_of_another_size_in_the_family_is_refused(base, tmp_path):
    export, other, out = tmp_path / "export", tmp_path / "other", tmp_path / "out"
    _text_export(base, export)
    _model_of_another_width(other)
    with pytest.raises(ValueError, match="another model of this family"):
        reattach_vision_tower(str(export), str(other), str(out), trust_remote_code=False)
    assert not out.exists(), "a refused re-attachment must not create its output directory"


def test_a_grown_vocabulary_reattaches(base, tmp_path):
    export, out = tmp_path / "export", tmp_path / "out"
    model = auto_load_model(str(base), text_only=True, dtype=torch.float32)
    grown = model.config.vocab_size + 8
    model.resize_token_embeddings(grown)
    save_full_checkpoint(model, str(export), source_dir=str(base))
    reattach_vision_tower(str(export), str(base), str(out), trust_remote_code=False)
    written = _safetensors(out)
    assert written["language_model.lm_head.weight"].shape[0] == grown
    assert written["language_model.backbone.embeddings.weight"].shape[0] == grown
    assert AutoConfig.from_pretrained(out).get_text_config().vocab_size == grown


def test_a_text_only_adapter_merges_through_the_text_class_and_reattaches(base, tmp_path):
    """Keys on the text class's tree name no module of the composite, so the merge finds the class through the spec."""
    PartialState()
    adapter, merged, served = tmp_path / "adapter", tmp_path / "merged", tmp_path / "served"
    text_model = auto_load_model(str(base), text_only=True, dtype=torch.float32)
    peft_model = get_peft_model(text_model, _ADAPTER)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.normal_()
    peft_model.save_pretrained(adapter)
    word_level_tokenizer(("hello", "world")).save_pretrained(adapter)

    ids = torch.arange(30, 46).unsqueeze(0)
    with torch.no_grad():
        adapted = auto_load_model(str(base), text_only=True, dtype=torch.float32)
        reference = PeftModel.from_pretrained(adapted, adapter).merge_and_unload().eval()(input_ids=ids).logits
        stock = auto_load_model(str(base), text_only=True, dtype=torch.float32).eval()(input_ids=ids).logits
    assert not torch.allclose(reference, stock), "the adapter moves nothing, so the merge could be a no-op"

    adapter_merge.merge_peft_adapter(
        adapter_dir=str(adapter), output_dir=str(merged), dtype=torch.float32, verbose=False
    )
    assert {key.split(".")[0] for key in _safetensors(merged)} == {"backbone", "lm_head"}
    with torch.no_grad():
        merged_logits = auto_load_model(str(merged), dtype=torch.float32).eval()(input_ids=ids).logits
    torch.testing.assert_close(merged_logits, reference, rtol=1e-5, atol=1e-5)

    reattach_vision_tower(str(merged), str(base), str(served), trust_remote_code=False)
    assert set(_safetensors(served)) == set(_safetensors(base))
    with torch.no_grad():
        composite = AutoModelForImageTextToText.from_pretrained(served, dtype=torch.float32).eval()
        served_logits = composite(input_ids=ids).logits
    torch.testing.assert_close(served_logits, reference, rtol=1e-5, atol=1e-5)


def test_without_the_declared_layout_the_text_load_has_no_class(base, monkeypatch):
    monkeypatch.setattr(model_preparation, "find_moe_family_spec", lambda _model_type: None)
    with pytest.raises(ValueError, match="no text-only"):
        auto_load_model(str(base), text_only=True, dtype=torch.float32)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
