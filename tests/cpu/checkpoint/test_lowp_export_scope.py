#!/usr/bin/env python
"""The low-precision export honours the run's lowp scope knobs, which no checkpoint records, so
``quantize_to_lowp.py`` takes them as flags. The window is ``halo.kernels.lowp.spec.kept_block_indices``."""

import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import load_file, save_file

from halo.checkpoint.tools.lowp_export import quantize_checkpoint
from halo.kernels.lowp.spec import kept_block_indices
from halo.models.module_tree import decoder_layer_index

_BLOCKS = 3


def _make_checkpoint(directory: str) -> None:
    """``model_type: qwen3_moe`` so fused 3-D experts resolve a contraction axis."""
    torch.manual_seed(0)
    ckpt = {}
    for block in range(_BLOCKS):
        prefix = f"model.layers.{block}"
        ckpt[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(128, 64, dtype=torch.bfloat16)
        ckpt[f"{prefix}.mlp.down_proj.weight"] = torch.randn(64, 128, dtype=torch.bfloat16)
        ckpt[f"{prefix}.mlp.experts.gate_up_proj"] = torch.randn(2, 64, 256, dtype=torch.bfloat16)
        ckpt[f"{prefix}.mlp.experts.down_proj"] = torch.randn(2, 128, 64, dtype=torch.bfloat16)
        ckpt[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(64, 64, dtype=torch.bfloat16)
    save_file(ckpt, os.path.join(directory, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3_moe"}, f)


def _export(tmp: str, **scope) -> tuple[dict, dict]:
    src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
    os.makedirs(src)
    _make_checkpoint(src)
    quantize_checkpoint(src, out, "mxfp8", **scope)
    with open(os.path.join(out, "quantization_config.json")) as f:
        return load_file(os.path.join(out, "model.safetensors")), json.load(f)


def _packed_bases(tensors: dict) -> set[str]:
    """With the ``.weight_packed`` suffix removed."""
    return {name[: -len(".weight_packed")] for name in tensors if name.endswith(".weight_packed")}


def _quantized_blocks(tensors: dict) -> set[int | None]:
    return {decoder_layer_index(name) for name in _packed_bases(tensors)}


def test_keep_first_block_leaves_block_zero_in_high_precision():
    with tempfile.TemporaryDirectory() as tmp:
        tensors, manifest = _export(tmp, keep_first_blocks=1)
    assert _quantized_blocks(tensors) == {1, 2}
    # The original key survives with its original dtype, a stronger check than a missing packed key.
    assert tensors["model.layers.0.mlp.gate_proj.weight"].dtype == torch.bfloat16
    assert tensors["model.layers.0.mlp.experts.gate_up_proj"].dtype == torch.bfloat16
    assert "model.layers.0.mlp.gate_proj.weight_packed" not in tensors
    assert manifest["scope"]["kept_blocks"] == [0]


def test_keep_last_block_window_is_measured_against_the_checkpoints_own_depth():
    """Depth is read from the keys; an off-by-one silently quantizes the most precision-sensitive block."""
    with tempfile.TemporaryDirectory() as tmp:
        tensors, manifest = _export(tmp, keep_last_blocks=1)
    assert _quantized_blocks(tensors) == {0, 1}
    assert manifest["scope"]["kept_blocks"] == [2]


def test_no_scope_flags_quantizes_every_block():
    with tempfile.TemporaryDirectory() as tmp:
        tensors, manifest = _export(tmp)
    assert _quantized_blocks(tensors) == set(range(_BLOCKS))
    assert manifest["scope"]["kept_blocks"] == []


def test_dense_mlp_can_be_excluded_without_touching_the_experts():
    with tempfile.TemporaryDirectory() as tmp:
        tensors, manifest = _export(tmp, apply_dense_mlp=False)
    expected = {f"model.layers.{b}.mlp.experts.{k}" for b in range(_BLOCKS) for k in ("gate_up_proj", "down_proj")}
    assert _packed_bases(tensors) == expected
    assert tensors["model.layers.0.mlp.gate_proj.weight"].dtype == torch.bfloat16
    assert manifest["scope"]["apply_dense_mlp"] is False


def test_moe_experts_can_be_excluded_without_touching_the_dense_mlp():
    with tempfile.TemporaryDirectory() as tmp:
        tensors, manifest = _export(tmp, apply_moe_experts=False)
    expected = {f"model.layers.{b}.mlp.{k}" for b in range(_BLOCKS) for k in ("gate_proj", "down_proj")}
    assert _packed_bases(tensors) == expected
    assert tensors["model.layers.0.mlp.experts.gate_up_proj"].dtype == torch.bfloat16
    assert manifest["scope"]["apply_moe_experts"] is False


def test_kept_block_indices_is_the_one_home_for_the_window():
    """The export and apply_lowp_compute read this function, so a drift is not expressible."""
    assert kept_block_indices(8, 2, 1) == {0, 1, 7}
    assert kept_block_indices(8, 0, 0) == set()
    assert kept_block_indices(3, 9, 9) == {0, 1, 2}, "asking to keep more blocks than exist keeps all"
    assert kept_block_indices(0, 2, 2) == set(), "no blocks resolved -> nothing to keep"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
