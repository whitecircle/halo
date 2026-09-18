#!/usr/bin/env python
"""The module paths a saved adapter addresses, and the class the merge meets them through.

``adapter_module_paths`` reads the base model's module paths off an adapter file's tensor keys: below
PEFT's ``base_model.model.`` prefix, up to the segment PEFT appends (``.lora_A``, ``.modules_to_save``,
``.base_layer``), or minus the parameter leaf of a key ``save_embedding_layers`` writes whole.
``load_base_for_adapter`` retries the text-only class on any absent path — a
``modules_to_save`` lm_head resolves on both classes, so "every path absent" would never fire for it —
and refuses a base neither class fits, chaining a loader that has no text-only class to offer.

Run: pytest tests/cpu/checkpoint/test_adapter_module_paths.py
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from src.checkpoint.adapters import adapter_module_paths, load_base_for_adapter
from src.checkpoint.format import ADAPTER_SAFETENSORS_FILE

_Q, _EMBED, _HEAD = "model.layers.0.self_attn.q_proj", "model.embed_tokens", "lm_head"
_PREFIX = "base_model.model."
# Every key spelling PEFT 0.18.1 writes into an adapter file, and the module each addresses: LoRA, DoRA,
# a ``modules_to_save`` head (written as the wrapper's clone, its original and a bare copy) and the
# embeddings ``save_embedding_layers`` adds.
_KEY_TO_MODULE = {
    f"{_PREFIX}{_Q}.lora_A.weight": _Q,
    f"{_PREFIX}{_Q}.lora_B.weight": _Q,
    f"{_PREFIX}{_Q}.lora_magnitude_vector": _Q,
    f"{_PREFIX}{_HEAD}.modules_to_save.weight": _HEAD,
    f"{_PREFIX}{_HEAD}.original_module.weight": _HEAD,
    f"{_PREFIX}{_HEAD}.weight": _HEAD,
    f"{_PREFIX}{_EMBED}.weight": _EMBED,
    f"{_PREFIX}{_EMBED}.base_layer.weight": _EMBED,  # an adapted embedding, saved whole
}


def _adapter(tmp_path: Path, keys) -> str:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    save_file({key: torch.zeros(1) for key in keys}, str(adapter / ADAPTER_SAFETENSORS_FILE))
    return str(adapter)


def _tree(cls, *module_paths: str) -> nn.Module:
    root = cls()
    for path in module_paths:
        node = root
        for atom in path.split("."):
            if not hasattr(node, atom):
                node.add_module(atom, nn.Module())
            node = getattr(node, atom)
    return root


class Multimodal(nn.Module):
    """The widest class: the decoder under ``language_model``, a vision tower, the head on top."""


class TextOnly(nn.Module):
    """The ``*ForCausalLM`` sibling: the decoder directly under ``model``."""


def _multimodal():
    return _tree(Multimodal, "model.language_model.layers.0.self_attn.q_proj", "model.visual.blocks.0.attn.qkv", _HEAD)


def _text_only():
    return _tree(TextOnly, _Q, _EMBED, _HEAD)


def _loader(text_only_model=None):
    """A base loader: the multimodal class by default, ``text_only_model`` on request, or a refusal."""
    requests = []

    def load(base_model_path, *, excuse_task_head, text_only=False):
        requests.append(text_only)
        if not text_only:
            return _multimodal()
        if text_only_model is None:
            raise ValueError("A classification base has one class; text_only applies to causal-LM bases only.")
        return text_only_model

    return load, requests


def test_every_peft_key_spelling_resolves_to_its_module(tmp_path):
    adapter = _adapter(tmp_path, _KEY_TO_MODULE)

    assert adapter_module_paths(adapter) == set(_KEY_TO_MODULE.values())


def test_an_adapter_the_wrapper_carries_loads_once(tmp_path):
    adapter = _adapter(tmp_path, [f"{_PREFIX}model.language_model.layers.0.self_attn.q_proj.lora_A.weight"])
    load, requests = _loader(_text_only())

    model = load_base_for_adapter(adapter, "/base", load, excuse_task_head=False, log=lambda _: None)

    assert isinstance(model, Multimodal) and requests == [False]


def test_a_shared_lm_head_does_not_hide_the_text_only_class(tmp_path):
    """``lm_head`` resolves on both classes and the decoder paths only on the text-only one; retrying
    only when every path is absent would refuse this adapter."""
    adapter = _adapter(tmp_path, [f"{_PREFIX}{_Q}.lora_A.weight", f"{_PREFIX}{_HEAD}.modules_to_save.default.weight"])
    load, requests = _loader(_text_only())
    logged = []

    model = load_base_for_adapter(adapter, "/base", load, excuse_task_head=True, log=logged.append)

    assert isinstance(model, TextOnly) and requests == [False, True]
    assert logged and "text-only class of Multimodal" in logged[0]


def test_a_key_set_neither_class_carries_is_refused(tmp_path):
    adapter = _adapter(tmp_path, [f"{_PREFIX}model.blocks.0.attn.lora_A.weight", f"{_PREFIX}{_HEAD}.weight"])
    load, _requests = _loader(_text_only())

    with pytest.raises(ValueError, match=r"Multimodal .* does not have .*model\.blocks\.0\.attn"):
        load_base_for_adapter(adapter, "/base", load, excuse_task_head=False, log=lambda _: None)


def test_a_base_with_no_text_only_class_is_refused_for_the_mismatch(tmp_path):
    """A one-class loader refuses ``text_only``; the user still reads the mismatch, that refusal chained."""
    adapter = _adapter(tmp_path, [f"{_PREFIX}model.blocks.0.attn.lora_A.weight"])
    load, requests = _loader(text_only_model=None)

    with pytest.raises(ValueError, match="does not have") as caught:
        load_base_for_adapter(adapter, "/base", load, excuse_task_head=False, log=lambda _: None)

    assert requests == [False, True]
    assert "one class" in str(caught.value.__cause__)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
