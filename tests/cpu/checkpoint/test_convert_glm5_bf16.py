#!/usr/bin/env python
"""``convert_glm5_bf16.py`` must dequantize fp8 blocks exactly and strip only the quant sidecars.

End-to-end on a tiny fake fp8-block checkpoint: every fp8 tensor must come out as the block-wise
``weight * scale_inv`` product an independent per-block loop computes — including a tail block on a
non-multiple-of-block dimension — while unquantized tensors keep their STORED dtype (bf16 stays
bf16, the fp32-strict ones stay fp32). The ``*_scale_inv`` sidecars must be dropped, but the
hyper-connection ``hc_*_scale`` tensors — real weights whose names also end in "scale" — must
survive. The emitted ``config.json`` loses exactly its ``quantization_config``. Structural faults
(an fp8 tensor without its sidecar, a wrong-shaped scale grid, a checkpoint that was never
quantized) must be refused before the output directory exists.

Run: ``python tests/cpu/checkpoint/test_convert_glm5_bf16.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.before_training.convert_glm5_bf16 import dequantize_block_fp8, main

_BLOCK = (4, 4)
# Rows are NOT a multiple of the block (10 = 2 full blocks + a 2-row tail), so the tail block must
# get its own scale row; columns are an exact multiple.
_FP8_KEY = "model.language_model.layers.3.mlp.experts.0.gate_proj.weight"
_FP8_SHAPE = (10, 8)
_BF16_KEY = "model.language_model.layers.0.input_layernorm.weight"
# The family's fp32 pins are transformers' _keep_in_fp32_modules_strict — e_score_correction_bias,
# conv1d, dt_bias, A_log — so the release stores them fp32 and the conversion must not flatten them.
_FP32_KEY = "model.language_model.layers.3.mlp.gate.e_score_correction_bias"
_FP32_CONV1D_KEY = "model.language_model.layers.0.self_attn.k_conv1d.weight"
# The trap: a real hyper-connection weight whose name ends in "scale" — not a quant sidecar.
_HC_SCALE_KEY = "model.language_model.layers.3.hc_attn_scale"


def _fp8_weight() -> torch.Tensor:
    """A deterministic fp8 tensor whose values are exact in e4m3 (halves in [-4, 4))."""
    values = (torch.arange(math.prod(_FP8_SHAPE), dtype=torch.float32).reshape(_FP8_SHAPE) % 16 - 8) / 2.0
    return values.to(torch.float8_e4m3fn)


def _scale_grid() -> torch.Tensor:
    """A (3, 2) grid of distinct powers of two; the tail-block row differs from its neighbor, so a
    dequant that mis-indexes (or never reaches) the tail block cannot pass by accident."""
    grid = torch.tensor([[1.0, 2.0], [0.5, 4.0], [8.0, 0.25]], dtype=torch.float32)
    assert not torch.equal(grid[2], grid[1])
    return grid


def _reference_dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Independent per-block loop — the oracle the vectorized implementation is judged against."""
    rows, cols = weight.shape
    out = torch.empty(rows, cols, dtype=torch.float32)
    for i in range(scale.shape[0]):
        for j in range(scale.shape[1]):
            row_slice = slice(i * _BLOCK[0], min((i + 1) * _BLOCK[0], rows))
            col_slice = slice(j * _BLOCK[1], min((j + 1) * _BLOCK[1], cols))
            out[row_slice, col_slice] = weight[row_slice, col_slice].float() * scale[i, j]
    return out.to(torch.bfloat16)


def _write_checkpoint(path, tensors: dict[str, torch.Tensor], *, quantization_config=...) -> str:
    os.makedirs(path)
    save_file(tensors, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    if quantization_config is ...:
        quantization_config = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": list(_BLOCK)}
    config: dict = {
        "model_type": "glm5_next",
        "text_config": {"model_type": "glm5_next_text", "dtype": "bfloat16"},
    }
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast"}, f)
    return str(path)


def _default_tensors() -> dict[str, torch.Tensor]:
    """Deterministic, so a test can rebuild the dict and compare against the converted output."""
    return {
        _FP8_KEY: _fp8_weight(),
        _FP8_KEY + "_scale_inv": _scale_grid(),
        _BF16_KEY: (torch.arange(8, dtype=torch.float32) / 8).to(torch.bfloat16),
        _FP32_KEY: torch.arange(16, dtype=torch.float32) / 3,
        _FP32_CONV1D_KEY: torch.arange(24, dtype=torch.float32).reshape(6, 1, 4) / 8,
        _HC_SCALE_KEY: torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32),
    }


def _run(source: str, output_dir: str, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["convert_glm5_bf16.py", "--model_id", source, "--output_dir", output_dir])
    main()


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """One conversion of the tiny fake checkpoint, shared by the read-only assertions."""
    root = tmp_path_factory.mktemp("glm5_convert")
    source = _write_checkpoint(root / "src", _default_tensors())
    output = str(root / "out")
    monkeypatch = pytest.MonkeyPatch()
    try:
        _run(source, output, monkeypatch)
    finally:
        monkeypatch.undo()
    return source, output


def _stored(output: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors, dtypes = {}, {}
    with safe_open(os.path.join(output, "model.safetensors"), framework="pt") as reader:
        for key in reader.keys():  # noqa: SIM118 - safe_open has .keys() but is not a mapping
            tensors[key] = reader.get_tensor(key)
            dtypes[key] = reader.get_slice(key).get_dtype()
    return tensors, dtypes


def test_block_dequant_matches_the_blockwise_reference(converted):
    """Exact equality against the independent per-block loop, tail block included — a transposed
    grid, a swapped expansion axis, or a truncated tail all land off this oracle."""
    _, output = converted
    tensors, dtypes = _stored(output)
    expected = _reference_dequant(_fp8_weight(), _scale_grid())
    assert dtypes[_FP8_KEY] == "BF16"
    assert torch.equal(tensors[_FP8_KEY], expected)
    # Anti-vacuity: the tail block really is scaled differently from the block above it.
    assert not torch.equal(expected[8:10], _reference_dequant(_fp8_weight(), _scale_grid()[[0, 1, 1]])[8:10])


def test_scale_sidecars_are_dropped_and_hc_scale_tensors_kept(converted):
    _, output = converted
    tensors, dtypes = _stored(output)
    assert not any(key.endswith("_scale_inv") for key in tensors), sorted(tensors)
    assert torch.equal(tensors[_HC_SCALE_KEY], _default_tensors()[_HC_SCALE_KEY])
    assert dtypes[_HC_SCALE_KEY] == "F32"


def test_unquantized_tensors_keep_their_stored_dtype(converted):
    """bf16 stays bf16 and the fp32-strict tensors stay fp32 — no uniform-bf16 flattening."""
    _, output = converted
    tensors, dtypes = _stored(output)
    source_tensors = _default_tensors()
    assert dtypes[_BF16_KEY] == "BF16" and torch.equal(tensors[_BF16_KEY], source_tensors[_BF16_KEY])
    for key in (_FP32_KEY, _FP32_CONV1D_KEY):
        assert dtypes[key] == "F32" and torch.equal(tensors[key], source_tensors[key])


def test_config_loses_quantization_config_and_aux_files_ride_along(converted):
    _, output = converted
    with open(os.path.join(output, "config.json")) as f:
        config = json.load(f)
    assert "quantization_config" not in config
    assert config["model_type"] == "glm5_next"
    assert config["text_config"]["dtype"] == "bfloat16"
    assert os.path.isfile(os.path.join(output, "tokenizer_config.json"))


def test_fp8_without_a_sidecar_is_refused_before_any_write(tmp_path, monkeypatch):
    tensors = _default_tensors()
    del tensors[_FP8_KEY + "_scale_inv"]
    source = _write_checkpoint(tmp_path / "src", tensors)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="no co-located"):
        _run(source, str(out), monkeypatch)
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_a_mismatched_scale_grid_is_refused_before_any_write(tmp_path, monkeypatch):
    tensors = _default_tensors()
    tensors[_FP8_KEY + "_scale_inv"] = torch.ones(2, 2)
    source = _write_checkpoint(tmp_path / "src", tensors)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="scale grid"):
        _run(source, str(out), monkeypatch)
    assert not out.exists()


def test_an_unquantized_checkpoint_is_refused(tmp_path, monkeypatch):
    """Pointing the tool at an already-BF16 checkpoint must refuse, not silently re-shard it."""
    source = _write_checkpoint(tmp_path / "src", {_BF16_KEY: torch.zeros(4, 4)}, quantization_config=None)
    with pytest.raises(ValueError, match="quantization_config"):
        _run(source, str(tmp_path / "out"), monkeypatch)


def test_a_foreign_quant_scheme_is_refused(tmp_path, monkeypatch):
    source = _write_checkpoint(
        tmp_path / "src",
        _default_tensors(),
        quantization_config={"quant_method": "mxfp4", "weight_block_size": [1, 32]},
    )
    with pytest.raises(ValueError, match="Unsupported quantization scheme"):
        _run(source, str(tmp_path / "out"), monkeypatch)


def test_an_in_place_conversion_is_refused(tmp_path, monkeypatch):
    source = _write_checkpoint(tmp_path / "src", _default_tensors())
    before = sorted(os.listdir(source))
    with pytest.raises(ValueError, match="input and output directory are the same"):
        _run(source, source, monkeypatch)
    assert sorted(os.listdir(source)) == before, "the refused input directory must be left untouched"


def test_dequantize_rejects_a_non_2d_weight():
    with pytest.raises(ValueError, match="2D"):
        dequantize_block_fp8(torch.zeros(4, dtype=torch.float8_e4m3fn), torch.ones(1), _BLOCK, "w")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
