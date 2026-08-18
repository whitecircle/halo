#!/usr/bin/env python
"""``convert_mistral4_bf16.py`` must dequantize the FP8 release exactly and drop only the quant sidecars.

End-to-end on a tiny fake checkpoint in the release's layout: a 2-D fp8 matrix with a scalar
``_scale_inv``, a 3-D fused expert bank with one scale per expert, a static activation scale, and
unquantized tensors of every dtype the release carries. Every fp8 tensor must come out as the
``weight * scale`` product an independent loop computes, the other floats must land in bf16 (the
output is uniform), integers must pass through, the sidecars must be gone, and ``config.json``
must lose exactly its ``quantization_config`` while gaining ``dtype: bfloat16``. Structural faults
(an fp8 tensor without its sidecar, a scale layout this tool does not dequantize) must be refused
before the output directory exists.

Run: ``python tests/cpu/checkpoint/test_convert_mistral4_bf16.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.before_training.convert_mistral4_bf16 import main

_DENSE_KEY = "model.language_model.layers.0.self_attn.q_proj.weight"
_EXPERT_KEY = "model.language_model.layers.1.mlp.experts.gate_up_proj"
_ACTIVATION_SCALE_KEY = "model.language_model.layers.0.self_attn.q_proj.weight_activation_scale"
_FP32_KEY = "model.language_model.layers.1.mlp.gate.weight"
_BF16_KEY = "model.language_model.norm.weight"
_INT_KEY = "model.language_model.layers.1.mlp.expert_index"
_NUM_EXPERTS = 4


def _fp8(shape: tuple[int, ...]) -> torch.Tensor:
    """Deterministic values exact in e4m3 (halves in [-4, 4))."""
    return ((torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape) % 16 - 8) / 2.0).to(
        torch.float8_e4m3fn
    )


def _default_tensors() -> dict[str, torch.Tensor]:
    return {
        _DENSE_KEY: _fp8((8, 16)),
        _DENSE_KEY + "_scale_inv": torch.tensor([0.25], dtype=torch.float32),
        _ACTIVATION_SCALE_KEY: torch.tensor([1.5], dtype=torch.float32),
        _EXPERT_KEY: _fp8((_NUM_EXPERTS, 6, 8)),
        _EXPERT_KEY + "_scale_inv": torch.tensor([1.0, 2.0, 0.5, 4.0], dtype=torch.float32).view(_NUM_EXPERTS, 1, 1),
        _FP32_KEY: torch.arange(32, dtype=torch.float32).reshape(4, 8) / 3,
        _BF16_KEY: (torch.arange(8, dtype=torch.float32) / 8).to(torch.bfloat16),
        _INT_KEY: torch.arange(_NUM_EXPERTS, dtype=torch.int64),
    }


def _expected(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Independent per-tensor oracle: scalar scale broadcast, per-expert scale over the leading axis."""
    dense = (tensors[_DENSE_KEY].float() * tensors[_DENSE_KEY + "_scale_inv"].item()).to(torch.bfloat16)
    bank = tensors[_EXPERT_KEY].float()
    per_expert = torch.stack(
        [bank[e] * tensors[_EXPERT_KEY + "_scale_inv"].flatten()[e] for e in range(_NUM_EXPERTS)]
    ).to(torch.bfloat16)
    return {
        _DENSE_KEY: dense,
        _EXPERT_KEY: per_expert,
        _FP32_KEY: tensors[_FP32_KEY].to(torch.bfloat16),
        _BF16_KEY: tensors[_BF16_KEY],
        _INT_KEY: tensors[_INT_KEY],
    }


def _write_checkpoint(path, tensors: dict[str, torch.Tensor]) -> str:
    os.makedirs(path)
    save_file(tensors, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    config = {
        "model_type": "mistral3",
        "text_config": {"model_type": "mistral4"},
        "quantization_config": {"quant_method": "fp8", "activation_scheme": "static", "weight_block_size": None},
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast"}, f)
    return str(path)


def _run(source: str, output_dir: str, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["convert_mistral4_bf16.py", "--model_id", source, "--output_dir", output_dir])
    main()


def _stored(output: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    tensors, dtypes = {}, {}
    with safe_open(os.path.join(output, "model.safetensors"), framework="pt") as reader:
        for key in reader.keys():  # noqa: SIM118 - safe_open has .keys() but is not a mapping
            tensors[key] = reader.get_tensor(key)
            dtypes[key] = reader.get_slice(key).get_dtype()
    return tensors, dtypes


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    root = tmp_path_factory.mktemp("mistral4_convert")
    source = _write_checkpoint(root / "src", _default_tensors())
    output = str(root / "out")
    monkeypatch = pytest.MonkeyPatch()
    try:
        _run(source, output, monkeypatch)
    finally:
        monkeypatch.undo()
    return source, output


def test_every_tensor_matches_the_oracle_and_the_output_is_uniform_bf16(converted):
    _, output = converted
    tensors, dtypes = _stored(output)
    expected = _expected(_default_tensors())
    assert set(tensors) == set(expected), sorted(tensors)
    for key, want in expected.items():
        assert tensors[key].dtype == want.dtype, key
        assert torch.equal(tensors[key], want), key
    assert dtypes[_FP32_KEY] == "BF16" and dtypes[_INT_KEY] == "I64"
    # Anti-vacuity: the per-expert scales differ, so a scalar broadcast of any one of them cannot pass.
    bank = _default_tensors()[_EXPERT_KEY].float()
    assert not torch.equal(tensors[_EXPERT_KEY], bank.to(torch.bfloat16))


def test_config_loses_quantization_config_and_aux_files_ride_along(converted):
    _, output = converted
    with open(os.path.join(output, "config.json")) as f:
        config = json.load(f)
    assert "quantization_config" not in config
    assert config["dtype"] == "bfloat16"
    assert config["model_type"] == "mistral3"
    assert os.path.isfile(os.path.join(output, "tokenizer_config.json"))


def test_fp8_without_a_sidecar_is_refused_before_any_write(tmp_path, monkeypatch):
    tensors = _default_tensors()
    del tensors[_DENSE_KEY + "_scale_inv"]
    source = _write_checkpoint(tmp_path / "src", tensors)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="no co-located"):
        _run(source, str(out), monkeypatch)
    assert not out.exists(), "a refused conversion must not create its output directory"


def test_an_unsupported_scale_layout_is_refused_before_any_write(tmp_path, monkeypatch):
    """A block grid on a 2-D matrix is the GLM-5 scheme, not this one — dequantizing it as a scalar
    would scale every block by the first entry."""
    tensors = _default_tensors()
    tensors[_DENSE_KEY + "_scale_inv"] = torch.ones(2, 2)
    source = _write_checkpoint(tmp_path / "src", tensors)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Unsupported FP8 scale layout"):
        _run(source, str(out), monkeypatch)
    assert not out.exists()


def test_an_in_place_conversion_is_refused(tmp_path, monkeypatch):
    source = _write_checkpoint(tmp_path / "src", _default_tensors())
    before = sorted(os.listdir(source))
    with pytest.raises(ValueError, match="input and output directory are the same"):
        _run(source, source, monkeypatch)
    assert sorted(os.listdir(source)) == before, "the refused input directory must be left untouched"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
