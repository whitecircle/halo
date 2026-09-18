#!/usr/bin/env python
"""An adapter trained through a multimodal checkpoint's text-only class must merge — or refuse loudly.

``text_only_model: true`` trains a VLM checkpoint as its ``*ForCausalLM`` sibling, whose module tree
spells the decoder ``model.layers``; the merge tool resolves the widest class for the same
checkpoint, which spells it ``model.language_model.layers``. PEFT resolves no key across that gap,
warns, and merges nothing, so the tool would write the untouched base as a finished model. The merge
reads the adapter's keys, meets them through the class they address, and refuses a base the keys do
not belong to.

    python tests/cpu/checkpoint/test_merge_text_only_adapter_e2e.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

from src.checkpoint.format import ADAPTER_SAFETENSORS_FILE
from tests.common.models import TINY_QWEN35_CONFIG
from tests.common.utils import load_script_module

_VISION = {
    "depth": 1,
    "hidden_size": 16,
    "intermediate_size": 16,
    "num_heads": 2,
    "out_hidden_size": TINY_QWEN35_CONFIG["hidden_size"],
}
_TARGET, _CONTROL = "q_proj", "o_proj"


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")


def _build_multimodal_base(base_dir: Path) -> None:
    torch.manual_seed(0)
    config = Qwen3_5Config(text_config={**TINY_QWEN35_CONFIG, "tie_word_embeddings": False}, vision_config=_VISION)
    Qwen3_5ForConditionalGeneration(config).save_pretrained(base_dir)
    _tiny_tokenizer().save_pretrained(base_dir)


def _train_text_only_adapter(base_dir: Path, adapter_dir: Path) -> dict[str, torch.Tensor]:
    """The adapter a ``text_only_model`` run saves: keys spelled on the text-only class's tree."""
    model = Qwen3_5ForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=[_TARGET], task_type="CAUSAL_LM"))
    deltas: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, module in peft_model.named_modules():
            if name.endswith(_TARGET) and hasattr(module, "lora_B"):
                module.lora_B["default"].weight.normal_()
                delta = (module.lora_B["default"].weight @ module.lora_A["default"].weight) * module.scaling["default"]
                deltas[name.removeprefix("base_model.model.") + ".weight"] = delta.clone()
    peft_model.save_pretrained(adapter_dir)
    _tiny_tokenizer().save_pretrained(adapter_dir)
    return deltas


def _merge_script():
    return load_script_module("scripts/after_training/merge_peft_adapters.py", "merge_peft_adapters_text_only")


def test_a_text_only_adapter_merges_into_the_class_it_was_trained_through():
    PartialState()
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter, out = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "out"
        _build_multimodal_base(base)
        deltas = _train_text_only_adapter(base, adapter)

        _merge_script().merge_peft_adapter(
            adapter_dir=str(adapter), output_dir=str(out), dtype=torch.float32, verbose=False
        )

        merged = Qwen3_5ForCausalLM.from_pretrained(out, dtype=torch.float32).state_dict()
        original = Qwen3_5ForCausalLM.from_pretrained(base, dtype=torch.float32).state_dict()

    assert deltas and all(delta.abs().max() > 1e-3 for delta in deltas.values()), "premise: the adapter is a no-op"
    for key, delta in deltas.items():
        assert torch.allclose(merged[key], original[key] + delta, atol=1e-5), (
            f"{key} did not receive its delta: the merge met the adapter through the wrong class and PEFT merged nothing"
        )
    controls = [key for key in original if key.endswith(f"{_CONTROL}.weight")]
    assert controls and all(torch.equal(merged[key], original[key]) for key in controls)


def test_an_adapter_whose_keys_address_no_module_of_the_base_is_refused():
    """A relabelled key set that fits neither class must not produce a merged checkpoint at all."""
    PartialState()
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter, out = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "out"
        _build_multimodal_base(base)
        _train_text_only_adapter(base, adapter)
        weights = adapter / ADAPTER_SAFETENSORS_FILE
        save_file({key.replace(".layers.", ".blocks."): tensor for key, tensor in load_file(weights).items()}, weights)

        with pytest.raises(ValueError, match="does not have"):
            _merge_script().merge_peft_adapter(
                adapter_dir=str(adapter), output_dir=str(out), dtype=torch.float32, verbose=False
            )
        assert not out.exists(), "a refused merge left an output directory behind"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
