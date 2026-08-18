#!/usr/bin/env python
"""``convert_deepseek_v4_bf16.py`` must dequantize a block-fp8 DeepSeek-V4 checkpoint exactly on the
installed transformers, and export the layout a training run reads.

End-to-end on a tiny random-init DeepSeek-V4 written in the hub's fp8 layout (128x128 block
scales beside every quantized Linear, per-expert hub keys, the release's ``quantization_config``):
the tool's output must hold every quantized weight as the block-wise ``weight * scale_inv``
product an independent loop computes, stored bf16 with no scale sidecar left, under a
``config.json`` that lost the quantization scheme and gained nothing else — the parallel plans
the loader workaround blanks must not leak into it. The two load-time hazards this pins are
transformers 5.16's ``update_tp_plan`` crash on a config with a class-level EP plan and
``save_pretrained``'s revert through the quantizer-rewritten conversions, which cannot reach the
hub layout (the export goes through the registry mapping instead).

Run: ``python tests/cpu/checkpoint/test_convert_deepseek_v4_bf16.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM, PreTrainedTokenizerFast

from scripts.before_training.convert_deepseek_v4_bf16 import main, reject_uneven_fp8_blocks
from tests.common.models import TINY_DSV4_CONFIG
from tests.cpu.models.test_deepseek_v4_support import randomize_tid2eid

_BLOCK = (128, 128)
_FP8_MAX = 448.0
_HUB_QUANT_CONFIG = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


def _quantized_key(key: str) -> bool:
    """The Linear weights the hub quantizes: projections and experts, never the router, norms or embeddings."""
    return key.endswith(".weight") and "layers." in key and "norm" not in key and ".gate." not in key


def _block_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[rows, cols]`` → fp8 e4m3 + fp32 scale grid ``ceil(rows/128) x ceil(cols/128)``; ``scale_inv[i, j]``
    MULTIPLIES block ``[i, j]`` on dequant (the DeepSeek-V3/V4 convention)."""
    rows, cols = weight.shape
    grid_rows, grid_cols = math.ceil(rows / _BLOCK[0]), math.ceil(cols / _BLOCK[1])
    padded = torch.zeros(grid_rows * _BLOCK[0], grid_cols * _BLOCK[1])
    padded[:rows, :cols] = weight.float()
    amax = padded.view(grid_rows, _BLOCK[0], grid_cols, _BLOCK[1]).abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = amax / _FP8_MAX
    expanded = scale.repeat_interleave(_BLOCK[0], 0)[:rows].repeat_interleave(_BLOCK[1], 1)[:, :cols]
    return (weight.float() / expanded).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn), scale.contiguous()


def _reference_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Independent per-block loop — the oracle the tool's output is judged against."""
    rows, cols = weight.shape
    out = torch.empty(rows, cols, dtype=torch.float32)
    for i in range(scale.shape[0]):
        for j in range(scale.shape[1]):
            row_slice = slice(i * _BLOCK[0], min((i + 1) * _BLOCK[0], rows))
            col_slice = slice(j * _BLOCK[1], min((j + 1) * _BLOCK[1], cols))
            out[row_slice, col_slice] = weight[row_slice, col_slice].float() * scale[i, j]
    return out.to(torch.bfloat16)


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")


def _write_fp8_checkpoint(path) -> tuple[str, dict[str, torch.Tensor], dict]:
    """A tiny DeepSeek-V4 saved by transformers (hub layout), then re-written with its Linear weights
    block-quantized to fp8. Returns the source dir, the dequant oracle keyed by hub name, and the
    pristine config dict the export must reproduce minus the quantization scheme."""
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(DeepseekV4Config(**TINY_DSV4_CONFIG)).to(torch.bfloat16)
    randomize_tid2eid(model)
    model.save_pretrained(path)
    _tiny_tokenizer().save_pretrained(path)
    with open(os.path.join(path, "config.json")) as f:
        pristine = json.load(f)

    hub = load_file(os.path.join(path, "model.safetensors"))
    tensors, oracle = {}, {}
    for key, value in hub.items():
        if _quantized_key(key) and value.dim() == 2:
            quantized, scale = _block_quantize(value)
            tensors[key], tensors[key + "_scale_inv"] = quantized, scale
            oracle[key] = _reference_dequant(quantized, scale)
        else:
            tensors[key] = value
    assert len(oracle) > 8, "fixture quantized too little to be a test of the dequant path"
    save_file(tensors, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({**pristine, "quantization_config": _HUB_QUANT_CONFIG}, f, indent=2)
    return str(path), oracle, pristine


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("dsv4_convert")
    source, oracle, pristine = _write_fp8_checkpoint(root / "src")
    output = str(root / "out")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sys, "argv", ["convert_deepseek_v4_bf16.py", "--model_id", source, "--output_dir", output])
    try:
        main()
    finally:
        monkeypatch.undo()
    return output, oracle, pristine


def _stored(output: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors, dtypes = {}, {}
    with safe_open(os.path.join(output, "model.safetensors"), framework="pt") as reader:
        for key in reader.keys():  # noqa: SIM118 - safe_open has .keys() but is not a mapping
            tensors[key] = reader.get_tensor(key)
            dtypes[key] = reader.get_slice(key).get_dtype()
    return tensors, dtypes


def test_quantized_weights_match_the_blockwise_oracle(converted):
    """Exact equality per tensor: the loader's block dequant, the tool's bf16 flattening and the
    dequantized-layout save must together reproduce the reference product bit for bit."""
    output, oracle, _ = converted
    tensors, _ = _stored(output)
    missing = sorted(set(oracle) - set(tensors))
    assert not missing, f"quantized weights absent from the export: {missing[:5]}"
    for key, want in oracle.items():
        assert tensors[key].dtype == torch.bfloat16, key
        assert torch.equal(tensors[key], want), f"{key} diverges from the block-wise oracle"


def test_the_export_is_bf16_with_no_scale_sidecars(converted):
    output, _, _ = converted
    tensors, dtypes = _stored(output)
    assert not any(key.endswith("_scale_inv") for key in tensors), sorted(tensors)[:5]
    floats = {key: dtype for key, dtype in dtypes.items() if dtype.startswith(("F", "BF"))}
    assert floats and set(floats.values()) == {"BF16"}, set(floats.values())


def test_config_loses_the_quantization_scheme_and_nothing_else_leaks(converted):
    """The workaround blanks the instance-level parallel plans for the load; the exported config
    must be the pristine one (no ``base_model_ep_plan``/``base_model_tp_plan`` keys, no scheme)."""
    output, _, pristine = converted
    with open(os.path.join(output, "config.json")) as f:
        exported = json.load(f)
    assert "quantization_config" not in exported
    assert not {key for key in exported if key.endswith("_plan")}, sorted(exported)
    assert exported["model_type"] == "deepseek_v4"
    assert set(pristine) - {"quantization_config"} <= set(exported)
    assert os.path.isfile(os.path.join(output, "tokenizer.json"))


def test_an_uneven_fp8_block_grid_is_refused_before_the_load(tmp_path):
    """transformers' dequantizer splits a dimension into EQUAL pieces, so a 192-row matrix with a
    (2, 1) grid gets scale[0] on rows 0-95 instead of 0-127 — silently. Refuse it up front."""
    weight, scale = _block_quantize(torch.randn(192, 128))
    assert tuple(scale.shape) == (2, 1)
    save_file(
        {"w": weight, "w_scale_inv": scale}, os.path.join(tmp_path, "model.safetensors"), metadata={"format": "pt"}
    )
    with pytest.raises(ValueError, match="neither one block nor a multiple"):
        reject_uneven_fp8_blocks(str(tmp_path), _BLOCK)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
