#!/usr/bin/env python
"""``save_full_checkpoint`` must not let the source's stale tokenizer files outlive the fresh save,
and must never hand a dequantized load back to the fp8 revert.

transformers 5 reads three tokenizer files it does not write — ``special_tokens_map.json``,
``added_tokens.json`` and ``chat_template.json`` — and each OVERRIDES the fresh
``tokenizer_config.json`` / ``tokenizer.json`` / ``chat_template.jinja`` beside it. The aux copy
carries them over from the source before the processing class is saved, so without the sweep a
``patch_vocab.py --chat_template`` on a base shipping ``chat_template.json`` serves the OLD
template through ``AutoProcessor``. A processing class that still writes one of those files
keeps it. And a model whose load dequantized an fp8 source must be reverted through its registry
conversion mapping (``revert_conversions_for``, the rule every writer shares), not the
quantizer-rewritten list the load recorded (which ``save_pretrained`` cannot invert) and not
``save_original_format=False`` (the internal layout no loader reads).

    python tests/cpu/checkpoint/test_save_full_checkpoint_sidecars.py
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.integrations.finegrained_fp8 import Fp8Dequantize

from src.checkpoint.format import revert_conversions_for
from src.checkpoint.tool_io import save_full_checkpoint
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN3_MOE_CONFIG

_STALE = {"special_tokens_map.json": {"pad_token": "<stale>"}, "chat_template.json": {"chat_template": "STALE"}}


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")


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
    save_full_checkpoint(model, str(out), processing_class=_tiny_tokenizer(), source_dir=source)

    assert os.path.isfile(out / "tokenizer.json"), "the fresh tokenizer was not written"
    for name in _STALE:
        assert not os.path.exists(out / name), f"{name} carried over from the source would override the fresh save"


def test_a_processing_class_that_still_writes_a_legacy_file_keeps_it(tmp_path):
    """The sweep is keyed on the file being untouched by the fresh save, not on its name."""
    model, source = _source_with_stale_sidecars(tmp_path / "src")
    out = tmp_path / "out"

    class _WritesLegacyMap:
        def save_pretrained(self, directory):
            _tiny_tokenizer().save_pretrained(directory)
            with open(os.path.join(directory, "special_tokens_map.json"), "w") as f:
                json.dump({"pad_token": "<fresh>"}, f)

    save_full_checkpoint(model, str(out), processing_class=_WritesLegacyMap(), source_dir=source)
    with open(out / "special_tokens_map.json") as f:
        assert json.load(f) == {"pad_token": "<fresh>"}
    assert not os.path.exists(out / "chat_template.json")


def test_without_a_processing_class_the_sources_tokenizer_files_stay_consistent(tmp_path):
    """No fresh tokenizer save means the copied files ARE the tokenizer — nothing is swept."""
    model, source = _source_with_stale_sidecars(tmp_path / "src")
    out = tmp_path / "out"
    save_full_checkpoint(model, str(out), source_dir=source)
    for name in _STALE:
        assert os.path.isfile(out / name)


def test_a_dequantizing_load_reverts_the_family_registry_mapping():
    """The revert reads what the load recorded, except a record carrying a quantizer's op: a plain
    record (empty, or ops of the family's own) is reverted as recorded, a consumed ``Fp8Dequantize``
    swaps in the family's registry mapping."""
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG))
    registry = revert_conversions_for(model)  # no record: the registry mapping
    assert registry, "premise: the family registers conversions, so the swap is observable"

    model._weight_conversions = []
    assert revert_conversions_for(model) == []
    model._weight_conversions = [SimpleNamespace(operations=[object()])]
    assert len(revert_conversions_for(model)) == 1
    model._weight_conversions = [SimpleNamespace(operations=[Fp8Dequantize(hf_quantizer=None)])]
    assert [c.source_patterns for c in revert_conversions_for(model)] == [c.source_patterns for c in registry]


def test_a_dequantized_load_is_saved_through_the_registry_mapping(tmp_path, monkeypatch):
    """``save_pretrained`` runs its default revert over the registry mapping — no
    ``save_original_format`` override — and the load's own record is left as it was."""
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
