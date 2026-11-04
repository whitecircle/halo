#!/usr/bin/env python
"""``save_full_checkpoint`` sweeps stale source tokenizer files, which transformers 5 reads over the fresh save, and
never hands a dequantized load to the fp8 revert."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.integrations.finegrained_fp8 import Fp8Dequantize

from halo.checkpoint.hub_layout import conversions_to_revert
from halo.checkpoint.tools.offline_export import save_full_checkpoint
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN3_MOE_CONFIG
from tests.common.tiny_models import word_level_tokenizer

_STALE = {"special_tokens_map.json": {"pad_token": "<stale>"}, "chat_template.json": {"chat_template": "STALE"}}


def _source_with_stale_sidecars(path) -> tuple[Qwen3ForCausalLM, str]:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    model.save_pretrained(path)
    for name, payload in _STALE.items():
        with open(os.path.join(path, name), "w") as f:
            json.dump(payload, f)
    return model, str(path)


def test_stale_legacy_tokenizer_files_do_not_survive_a_fresh_processing_class_save(tmp_path):
    model, source = _source_with_stale_sidecars(tmp_path / "src")
    out = tmp_path / "out"
    save_full_checkpoint(model, str(out), processing_class=word_level_tokenizer(("hello",)), source_dir=source)

    assert os.path.isfile(out / "tokenizer.json"), "the fresh tokenizer was not written"
    for name in _STALE:
        assert not os.path.exists(out / name), f"{name} carried over from the source would override the fresh save"


def test_a_processing_class_that_still_writes_a_legacy_file_keeps_it(tmp_path):
    """Keyed on the file being untouched by the fresh save, not on its name."""
    model, source = _source_with_stale_sidecars(tmp_path / "src")
    out = tmp_path / "out"

    class _WritesLegacyMap:
        def save_pretrained(self, directory):
            word_level_tokenizer(("hello",)).save_pretrained(directory)
            with open(os.path.join(directory, "special_tokens_map.json"), "w") as f:
                json.dump({"pad_token": "<fresh>"}, f)

    save_full_checkpoint(model, str(out), processing_class=_WritesLegacyMap(), source_dir=source)
    with open(out / "special_tokens_map.json") as f:
        assert json.load(f) == {"pad_token": "<fresh>"}
    assert not os.path.exists(out / "chat_template.json")


def test_without_a_processing_class_the_sources_tokenizer_files_stay_consistent(tmp_path):
    """Without a fresh save the copied files are the tokenizer."""
    model, source = _source_with_stale_sidecars(tmp_path / "src")
    out = tmp_path / "out"
    save_full_checkpoint(model, str(out), source_dir=source)
    for name in _STALE:
        assert os.path.isfile(out / name)


def test_a_dequantizing_load_reverts_the_family_registry_mapping():
    """A consumed ``Fp8Dequantize`` swaps in the family's registry mapping; a plain record reverts as recorded."""
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG))
    registry = conversions_to_revert(model)  # No record: the registry mapping.
    assert registry, "premise: the family registers conversions, so the swap is observable"

    model._weight_conversions = []
    assert conversions_to_revert(model) == []
    model._weight_conversions = [SimpleNamespace(operations=[object()])]
    assert len(conversions_to_revert(model)) == 1
    model._weight_conversions = [SimpleNamespace(operations=[Fp8Dequantize(hf_quantizer=None)])]
    assert [c.source_patterns for c in conversions_to_revert(model)] == [c.source_patterns for c in registry]


def test_a_dequantized_load_is_saved_through_the_registry_mapping(tmp_path, monkeypatch):
    """No ``save_original_format`` override, and the load's own record is left as it was."""
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    recorded = [SimpleNamespace(operations=[Fp8Dequantize(hf_quantizer=None)])]
    model._weight_conversions = recorded
    seen = []
    real_save = type(model).save_pretrained
    monkeypatch.setattr(
        type(model),
        "save_pretrained",
        lambda self, d, **kw: (seen.append((kw, self._weight_conversions)), real_save(self, d))[1],
    )

    save_full_checkpoint(model, str(tmp_path / "a"))
    kwargs, conversions_during_save = seen[-1]
    assert "save_original_format" not in kwargs
    assert conversions_during_save is not recorded
    assert model._weight_conversions is recorded
    assert os.path.isfile(tmp_path / "a" / "model.safetensors")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
