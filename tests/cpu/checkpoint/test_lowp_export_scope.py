#!/usr/bin/env python
"""The low-precision export must honour the training run's lowp SCOPE knobs.

``apply_mixed_precision_compute`` quantizes a subset of the model: ``lowp_apply_dense_mlp`` /
``lowp_apply_moe_experts`` choose which halves, and ``lowp_keep_first_blocks`` /
``lowp_keep_last_blocks`` hold the precision-sensitive blocks at the two ends in bf16. None of that
is recoverable from a checkpoint, so ``quantize_to_lowp.py`` takes the four values back as flags:
quantizing by name-regex alone ships protected blocks quantized while its docstring promises an
exact reproduction of the QAT forward.

The block window itself lives in one place (``src.kernels.lowp.mixed_precision.kept_block_indices``),
read by the training conversion and the export alike, so the two cannot drift.

Run: python tests/cpu/checkpoint/test_lowp_export_scope.py
"""

import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import load_file, save_file

from src.kernels.lowp.mixed_precision import block_index, kept_block_indices
from tests.common.utils import load_script_module

quantize_checkpoint = load_script_module("scripts/after_training/quantize_to_lowp.py").quantize_checkpoint

_BLOCKS = 3


def _make_checkpoint(directory: str) -> None:
    """A 3-block model with a dense MLP and a fused-expert MoE FFN in every block.

    ``model_type: qwen3_moe`` so the fused 3-D expert tensors resolve a contraction axis (the export
    refuses an unknown family rather than guessing one).
    """
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
    """Run the export with ``scope``; return ``(output tensors, quantization_config.json)``."""
    src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
    os.makedirs(src)
    _make_checkpoint(src)
    quantize_checkpoint(src, out, "mxfp8", **scope)
    with open(os.path.join(out, "quantization_config.json")) as f:
        return load_file(os.path.join(out, "model.safetensors")), json.load(f)


def _packed_bases(tensors: dict) -> set[str]:
    """Names that were quantized, with the ``.weight_packed`` suffix the export appends removed."""
    return {name[: -len(".weight_packed")] for name in tensors if name.endswith(".weight_packed")}


def _quantized_blocks(tensors: dict) -> set[int | None]:
    return {block_index(name) for name in _packed_bases(tensors)}


# The block window


def test_keep_first_block_leaves_block_zero_in_high_precision():
    """keep_first_blocks=1: block 0 ships as-is, every later block is quantized."""
    with tempfile.TemporaryDirectory() as tmp:
        tensors, manifest = _export(tmp, keep_first_blocks=1)
    assert _quantized_blocks(tensors) == {1, 2}
    # Untouched means the original key survives with its original dtype — not merely "no packed key".
    assert tensors["model.layers.0.mlp.gate_proj.weight"].dtype == torch.bfloat16
    assert tensors["model.layers.0.mlp.experts.gate_up_proj"].dtype == torch.bfloat16
    assert "model.layers.0.mlp.gate_proj.weight_packed" not in tensors
    assert manifest["scope"]["kept_blocks"] == [0]


def test_keep_last_block_window_is_measured_against_the_checkpoints_own_depth():
    """keep_last_blocks=1 must protect block 2 of 3 — proving the depth is read from the keys, not
    assumed. An off-by-one here silently quantizes the most precision-sensitive block."""
    with tempfile.TemporaryDirectory() as tmp:
        tensors, manifest = _export(tmp, keep_last_blocks=1)
    assert _quantized_blocks(tensors) == {0, 1}
    assert manifest["scope"]["kept_blocks"] == [2]


def test_no_scope_flags_quantizes_every_block():
    """The defaults are the training defaults, so an unscoped run is unchanged by this feature."""
    with tempfile.TemporaryDirectory() as tmp:
        tensors, manifest = _export(tmp)
    assert _quantized_blocks(tensors) == set(range(_BLOCKS))
    assert manifest["scope"]["kept_blocks"] == []


# The dense / expert halves


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


# The shared window helper


def test_kept_block_indices_is_the_one_home_for_the_window():
    """Both ends, clamped: the export and apply_mixed_precision_compute read this same function, so a
    drift between what training kept and what the export skips is not expressible."""
    assert kept_block_indices(8, 2, 1) == {0, 1, 7}
    assert kept_block_indices(8, 0, 0) == set()
    assert kept_block_indices(3, 9, 9) == {0, 1, 2}, "asking to keep more blocks than exist keeps all"
    assert kept_block_indices(0, 2, 2) == set(), "no blocks resolved -> nothing to keep"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
