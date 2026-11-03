#!/usr/bin/env python
"""``convert_to_bf16.py --verify`` reads stored dtypes, by parameter share. ``from_pretrained`` would cast everything
to ``config.dtype``, and fp32 norms are ~36% of tensors on a dense model but under 0.01% of parameters."""

from __future__ import annotations

import json
import os

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM

from halo.checkpoint.tools import offline_export
from halo.checkpoint.tools.bf16_conversion import (
    _BF16_VERIFY_MIN_FRACTION,
    convert_to_bf16,
    verify_adapter_conversion,
    verify_model_conversion,
)

# Wide enough that the norm-to-weight ratio is a real model's.
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
    config = AutoConfig.for_model(model_type, **{**_TINY, **overrides})
    AutoModelForCausalLM.from_config(config).save_pretrained(str(path))
    return str(path)


def _stored_dtype_counts(path):
    return _stored_dtype_counts_of(os.path.join(path, "model.safetensors"))


def _stored_dtype_counts_of(safetensors_path):
    """(tensor count, parameter count) per stored dtype."""
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
    """Counting tensors fails here on every dense model."""
    src = _fp32_checkpoint(tmp_path / f"{model_type}_src", model_type=model_type)
    out = tmp_path / f"{model_type}_out"
    convert_to_bf16(src, str(out), "causal_lm", verify=True)  # Raises if verification fails.


@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_converted_checkpoint_is_mostly_fp32_tensors_but_mostly_bf16_parameters(tmp_path, model_type):
    """Anti-vacuity: the two weightings must really disagree on a real architecture."""
    src = _fp32_checkpoint(tmp_path / f"{model_type}_src", model_type=model_type)
    out = tmp_path / f"{model_type}_out"
    convert_to_bf16(src, str(out), "causal_lm")

    tensors, params = _stored_dtype_counts(str(out))
    by_tensor = tensors["BF16"] / (tensors["BF16"] + tensors.get("F32", 0))
    by_param = params["BF16"] / (params["BF16"] + params.get("F32", 0))
    assert by_tensor < _BF16_VERIFY_MIN_FRACTION, f"fixture has too few fp32 norms to be realistic: {tensors}"
    assert by_param > _BF16_VERIFY_MIN_FRACTION, params


def test_verify_fails_on_a_checkpoint_still_stored_in_fp32(tmp_path):
    """Nothing is cast while config.json claims bfloat16; a check reloading through ``from_pretrained`` reads 100%
    BF16."""
    src = _fp32_checkpoint(tmp_path / "src")
    config = json.loads((tmp_path / "src" / "config.json").read_text())
    config["dtype"] = "bfloat16"
    (tmp_path / "src" / "config.json").write_text(json.dumps(config))

    assert verify_model_conversion(src) is False


def test_verify_fails_a_conversion_that_left_the_embedding_in_fp32(tmp_path):
    """Embedding + lm_head are single-digit percent of parameters, so a loose bar makes the verdict depend on model
    size."""
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
    """An integer buffer can never be bf16."""
    src = _fp32_checkpoint(tmp_path / "src")
    out = tmp_path / "out"
    convert_to_bf16(src, str(out), "causal_lm")

    state = load_file(os.path.join(out, "model.safetensors"))
    state.update({f"model.layers.{i}.expert_counts": torch.zeros(4096, dtype=torch.int64) for i in range(8)})
    save_file(state, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})

    assert verify_model_conversion(str(out)) is True


def _lora_adapter(tmp_path, base_dir):
    """PEFT writes LoRA A/B in fp32."""
    model = AutoModelForCausalLM.from_pretrained(base_dir)
    adapter_dir = tmp_path / "adapter"
    get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"])).save_pretrained(str(adapter_dir))
    return str(adapter_dir)


def _refuse_model_loads(monkeypatch):
    """An adapter converts from its weights file alone."""

    def _refused(*_args, **_kwargs):
        raise AssertionError("the adapter conversion loaded a model")

    monkeypatch.setattr(offline_export, "auto_load_model", _refused)
    monkeypatch.setattr(offline_export, "from_pretrained_verified", _refused)


def test_a_peft_conversion_writes_the_adapter_at_bf16_without_loading_the_base(tmp_path, monkeypatch):
    """Loading the base to re-save an adapter PEFT wrote back in fp32 left a bf16 input upcast."""
    adapter = _lora_adapter(tmp_path, _fp32_checkpoint(tmp_path / "base"))
    assert set(_stored_dtype_counts_of(os.path.join(adapter, "adapter_model.safetensors"))[0]) == {"F32"}
    _refuse_model_loads(monkeypatch)
    out = tmp_path / "out"

    convert_to_bf16(adapter, str(out), "causal_lm", is_peft=True, verify=True)  # Raises if verification fails.

    assert set(_stored_dtype_counts_of(out / "adapter_model.safetensors")[0]) == {"BF16"}
    assert (out / "adapter_config.json").is_file(), "the adapter's config must travel with its weights"


def test_a_bf16_adapter_keeps_its_weights_bit_for_bit(tmp_path, monkeypatch):
    adapter = _lora_adapter(tmp_path, _fp32_checkpoint(tmp_path / "base"))
    weights = os.path.join(adapter, "adapter_model.safetensors")
    bf16 = {key: tensor.to(torch.bfloat16) for key, tensor in load_file(weights).items()}
    save_file(bf16, weights, metadata={"format": "pt"})
    _refuse_model_loads(monkeypatch)
    out = tmp_path / "out"

    convert_to_bf16(adapter, str(out), "causal_lm", is_peft=True)

    written = load_file(str(out / "adapter_model.safetensors"))
    assert written.keys() == bf16.keys()
    assert all(torch.equal(written[key], bf16[key]) and written[key].dtype == torch.bfloat16 for key in bf16)


def test_an_unmerged_peft_save_verifies_its_adapter_not_a_stale_model(tmp_path):
    """A stale model.safetensors in --output_dir would fail the adapter just written, or pass it unchecked."""
    adapter = _lora_adapter(tmp_path, _fp32_checkpoint(tmp_path / "base"))
    out = _fp32_checkpoint(tmp_path / "out")
    convert_to_bf16(adapter, out, "causal_lm", is_peft=True, verify=True)

    assert os.path.isfile(os.path.join(out, "adapter_model.safetensors"))
    assert set(_stored_dtype_counts(out)[0]) == {"F32"}, "premise: the stale fp32 model is still there"


def test_verify_rejects_an_empty_adapter(tmp_path):
    """An adapter with no tensors loads as a silent no-op."""
    adapter_dir = tmp_path / "empty_adapter"
    os.makedirs(adapter_dir)
    save_file({}, os.path.join(adapter_dir, "adapter_model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="nothing in it"):
        verify_adapter_conversion(os.path.join(adapter_dir, "adapter_model.safetensors"))


def test_verify_threshold_is_strict_at_the_boundary(tmp_path):
    """Exactly at the bar is a fail, so the comparison cannot be relaxed to >= unnoticed."""
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
    """Returning True there would be a silent pass."""
    ckpt = tmp_path / "ints"
    os.makedirs(ckpt)
    save_file({"model.counts": torch.zeros(4, dtype=torch.int64)}, os.path.join(ckpt, "model.safetensors"))
    with pytest.raises(RuntimeError, match="no floating-point tensors"):
        verify_model_conversion(str(ckpt))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
