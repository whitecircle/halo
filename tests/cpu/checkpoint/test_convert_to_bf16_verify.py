#!/usr/bin/env python
"""CPU tests for ``convert_to_bf16.py --verify``: it must read the STORED dtypes, by parameter share.

Two ways for this check to be useless:

* Reloading the saved directory through ``from_pretrained`` casts every tensor to ``config.dtype`` on
  the way in, so a checkpoint whose tensors are all fp32 comes back 100% bfloat16 and the check can
  never FAIL.
* Counting TENSORS rather than parameters can never PASS: the conversion deliberately keeps every norm
  leaf in fp32, and on a dense model those are ~36% of the tensor COUNT (never above 43% on the
  roster, 1-2% on MoE) while being under 0.01% of the parameters.

So the fixtures here are built from real configs and a real PEFT adapter — a hand-picked tensor mix is
exactly what hides the second failure mode, and the third: PEFT restores LoRA A/B to fp32 as it saves,
so an unmerged adapter is 100% fp32 by design and must not be failed for it.

Run: ``python tests/cpu/checkpoint/test_convert_to_bf16_verify.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM

import scripts.after_training.convert_to_bf16 as convert_module
from scripts.after_training.convert_to_bf16 import (
    _BF16_VERIFY_MIN_FRACTION,
    convert_to_bf16,
    verify_model_conversion,
)

# Small enough to build on CPU in a test, wide enough that the norm-to-weight ratio is a real model's.
_TINY = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 64,
    "tie_word_embeddings": False,
}


def _fp32_checkpoint(path, model_type="qwen3", **overrides):
    """A real (randomly initialized) fp32 checkpoint of ``model_type`` — the conversion's input."""
    config = AutoConfig.for_model(model_type, **{**_TINY, **overrides})
    AutoModelForCausalLM.from_config(config).save_pretrained(str(path))
    return str(path)


def _stored_dtype_counts(path):
    """(tensor count, parameter count) per stored dtype of ``path``'s model.safetensors."""
    return _stored_dtype_counts_of(os.path.join(path, "model.safetensors"))


def _stored_dtype_counts_of(safetensors_path):
    """(tensor count, parameter count) per stored dtype — the two weightings the check can use."""
    tensors, params = {}, {}
    with safe_open(str(safetensors_path), framework="pt") as reader:
        for key in reader.keys():  # noqa: SIM118
            header = reader.get_slice(key)
            dtype = header.get_dtype()
            tensors[dtype] = tensors.get(dtype, 0) + 1
            params[dtype] = params.get(dtype, 0) + int(torch.tensor(header.get_shape()).prod())
    return tensors, params


@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_verify_accepts_what_the_conversion_itself_produces(tmp_path, model_type):
    """The check must pass on the tool's own output. Counting TENSORS instead of parameters fails here
    on every dense model, because the fp32 norms it deliberately keeps outnumber the weights."""
    src = _fp32_checkpoint(tmp_path / f"{model_type}_src", model_type=model_type)
    out = tmp_path / f"{model_type}_out"
    convert_to_bf16(src, str(out), "causal_lm", verify=True)  # raises if verification fails


@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_converted_checkpoint_is_mostly_fp32_tensors_but_mostly_bf16_parameters(tmp_path, model_type):
    """Anti-vacuity for the test above: it only means something if the two weightings really disagree
    on a real architecture — a fixture whose norms are a small share of the tensor count would pass
    either way."""
    src = _fp32_checkpoint(tmp_path / f"{model_type}_src", model_type=model_type)
    out = tmp_path / f"{model_type}_out"
    convert_to_bf16(src, str(out), "causal_lm")

    tensors, params = _stored_dtype_counts(str(out))
    by_tensor = tensors["BF16"] / (tensors["BF16"] + tensors.get("F32", 0))
    by_param = params["BF16"] / (params["BF16"] + params.get("F32", 0))
    assert by_tensor < _BF16_VERIFY_MIN_FRACTION, f"fixture has too few fp32 norms to be realistic: {tensors}"
    assert by_param > _BF16_VERIFY_MIN_FRACTION, params


def test_verify_fails_on_a_checkpoint_still_stored_in_fp32(tmp_path):
    """The artifact --verify exists to catch: nothing is cast, while config.json claims bfloat16. A
    check that reloads through ``from_pretrained`` reads it back as 100% BF16."""
    src = _fp32_checkpoint(tmp_path / "src")
    config = json.loads((tmp_path / "src" / "config.json").read_text())
    config["dtype"] = "bfloat16"
    (tmp_path / "src" / "config.json").write_text(json.dumps(config))

    assert verify_model_conversion(src) is False


def test_verify_fails_a_conversion_that_left_the_embedding_in_fp32(tmp_path):
    """A partial conversion, not a missing one — and the shape that decides how tight the bar has to
    be. Embedding + lm_head are single-digit percent of the parameters (less on a larger model), so a
    loose bar accepts them and what passes ends up depending on model size rather than on the bug."""
    src = _fp32_checkpoint(tmp_path / "src")
    out = tmp_path / "out"
    convert_to_bf16(src, str(out), "causal_lm")

    state = load_file(os.path.join(out, "model.safetensors"))
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        state[key] = state[key].float()
    save_file(state, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})

    _, params = _stored_dtype_counts(str(out))
    bf16_share = params["BF16"] / (params["BF16"] + params["F32"])
    assert 0.9 < bf16_share < _BF16_VERIFY_MIN_FRACTION, (
        f"fixture no longer lands in the band a 0.9 bar accepts: {bf16_share}"
    )
    assert verify_model_conversion(str(out)) is False


def test_verify_ignores_non_float_tensors(tmp_path):
    """An integer buffer can never be bf16; counting it would fail a correct conversion once the
    checkpoint carries enough of them."""
    src = _fp32_checkpoint(tmp_path / "src")
    out = tmp_path / "out"
    convert_to_bf16(src, str(out), "causal_lm")

    state = load_file(os.path.join(out, "model.safetensors"))
    state.update({f"model.layers.{i}.expert_counts": torch.zeros(4096, dtype=torch.int64) for i in range(8)})
    save_file(state, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})

    assert verify_model_conversion(str(out)) is True


def _lora_adapter(tmp_path, base_dir):
    """A real unmerged PEFT save over ``base_dir`` — the artifact ``--peft`` without ``--merge_adapter``
    produces, whose LoRA A/B PEFT deliberately restores to fp32."""
    model = AutoModelForCausalLM.from_pretrained(base_dir)
    adapter_dir = tmp_path / "adapter"
    get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"])).save_pretrained(str(adapter_dir))
    return str(adapter_dir)


def test_verify_accepts_an_unmerged_peft_save(tmp_path):
    """PEFT writes LoRA A/B in fp32 whatever dtype the model was cast to, so a bf16 assertion on an
    unmerged adapter fails the tool's own correct output — after it has already written it."""
    adapter = _lora_adapter(tmp_path, _fp32_checkpoint(tmp_path / "base"))
    out = tmp_path / "out"
    convert_to_bf16(adapter, str(out), "causal_lm", is_peft=True, verify=True)  # raises if verification fails

    stored = set(_stored_dtype_counts_of(out / "adapter_model.safetensors")[0])
    assert stored == {"F32"}, f"fixture no longer reproduces PEFT's fp32 restore: {stored}"


def test_verify_still_applies_to_a_merged_peft_save(tmp_path, monkeypatch):
    """Anti-over-exemption: --merge_adapter writes a real model, so the tool's own ``verify`` GATE
    must still judge it. Asserted by forcing the checker to fail and requiring the gate to raise —
    calling ``verify_model_conversion`` directly would pass even if the gate exempted PEFT."""
    adapter = _lora_adapter(tmp_path, _fp32_checkpoint(tmp_path / "base"))
    out = tmp_path / "out"
    monkeypatch.setattr(convert_module, "verify_model_conversion", lambda path: False)
    with pytest.raises(RuntimeError, match="BF16 verification failed"):
        convert_to_bf16(adapter, str(out), "causal_lm", is_peft=True, merge_adapter=True, verify=True)


def test_verify_rejects_an_empty_adapter(tmp_path):
    """An adapter with no tensors loads as a silent no-op, so "nothing to assert" is not a pass."""
    adapter_dir = tmp_path / "empty_adapter"
    os.makedirs(adapter_dir)
    save_file({}, os.path.join(adapter_dir, "adapter_model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="nothing in it"):
        verify_model_conversion(str(adapter_dir))


def test_verify_threshold_is_strict_at_the_boundary(tmp_path):
    """Exactly at the bar is a FAIL, so the comparison cannot be relaxed to >= unnoticed — and the bar
    itself has to be tight enough that a wholly fp32 embedding (a few % of parameters) is caught."""
    ckpt = tmp_path / "boundary"
    os.makedirs(ckpt)
    at_threshold = round(_BF16_VERIFY_MIN_FRACTION * 100)
    save_file(
        {
            "model.a": torch.zeros(at_threshold, dtype=torch.bfloat16),
            "model.b": torch.zeros(100 - at_threshold, dtype=torch.float32),
        },
        os.path.join(ckpt, "model.safetensors"),
        metadata={"format": "pt"},
    )
    assert verify_model_conversion(str(ckpt)) is False


def test_verify_raises_when_nothing_float_is_stored(tmp_path):
    """No float tensors means the share is undefined; returning True there would be a silent pass."""
    ckpt = tmp_path / "ints"
    os.makedirs(ckpt)
    save_file({"model.counts": torch.zeros(4, dtype=torch.int64)}, os.path.join(ckpt, "model.safetensors"))
    with pytest.raises(RuntimeError, match="no floating-point tensors"):
        verify_model_conversion(str(ckpt))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
