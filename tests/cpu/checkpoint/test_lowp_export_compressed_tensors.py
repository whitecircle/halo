#!/usr/bin/env python
"""``quantize_to_lowp.py --export_format compressed-tensors``: the layout vLLM's compressed-tensors loader reads, in
place of Halo's ``block_scaled`` manifest, which it refuses.
- Each format writes the tensors its compressed-tensors compressor names: ``weight`` + ``weight_scale`` for mxfp8,
  ``weight_packed`` + ``weight_scale`` for mxfp4, plus ``weight_global_scale`` for nvfp4, and no ``weight_shape``.
- Decoding them by compressed-tensors' rules reproduces the QAT forward's weight bit for bit, and so does decoding an
  nvfp4 ``gate_proj`` + ``up_proj`` pair the way vLLM decodes its merged layer, under one global scale.
- ``config.json`` carries the ``quantization_config`` compressed-tensors itself serializes for the format, with
  targets that match exactly the quantized modules under the engine's matching rule.
- What an engine cannot load quantized is refused before any write.
The decoders and the expected config mirror compressed-tensors 0.17.0 and vLLM 0.32.0, the releases in the
``vllm-0.32.0`` image, which the training image does not install."""

import json
import os
import re
import sys
from unittest import mock

import pytest
import torch
from safetensors.torch import load_file, save_file

from halo.checkpoint.tools import lowp_export
from halo.checkpoint.tools.lowp_export import BLOCK_SCALED, COMPRESSED_TENSORS, quantize_checkpoint
from halo.kernels.lowp.quantization import fake_quant, quantize_nvfp4
from tests.common.repo import load_script_module

FORMATS = ("mxfp8", "mxfp4", "nvfp4")
_BLOCKS = 3
_EXPERTS = 2
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# compressed_tensors.compressors.nvfp4.helpers.kE2M1ToFloat
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# ``QuantizationScheme(targets=[...], format=..., **PRESET_SCHEMES[name]).model_dump_json()`` for MXFP8, MXFP4 and
# NVFP4A16, as compressed-tensors 0.17.0 prints it.
_WEIGHT_ARGS = {
    "mxfp8": {"num_bits": 8, "group_size": 32, "strategy": "group", "scale_dtype": "torch.uint8"},
    "mxfp4": {"num_bits": 4, "group_size": 32, "strategy": "group", "scale_dtype": "torch.uint8"},
    "nvfp4": {"num_bits": 4, "group_size": 16, "strategy": "tensor_group", "scale_dtype": "torch.float8_e4m3fn"},
}
_COMPRESSION_FORMAT = {
    "mxfp8": "mxfp8-quantized",
    "mxfp4": "mxfp4-pack-quantized",
    "nvfp4": "nvfp4-pack-quantized",
}


def _expected_args(fmt: str, *, dynamic: bool) -> dict:
    return {
        "type": "float",
        "symmetric": True,
        "block_structure": None,
        "dynamic": dynamic,
        "actorder": None,
        "zp_dtype": None,
        "observer": None if dynamic else "memoryless_minmax",
        "observer_kwargs": {},
        **_WEIGHT_ARGS[fmt],
    }


def _weights(blocks: int = _BLOCKS) -> dict[str, torch.Tensor]:
    """A per-expert MoE with a dense MLP beside the experts, so both seams export, plus what must stay bf16. ``up_proj``
    is three times ``gate_proj``, so each pair's own nvfp4 global scales differ by a power of two or more."""
    torch.manual_seed(0)
    ckpt = {
        "model.embed_tokens.weight": torch.randn(100, 64, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(100, 64, dtype=torch.bfloat16),
    }
    for block in range(blocks):
        prefix = f"model.layers.{block}"
        ckpt[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(64, 64, dtype=torch.bfloat16)
        ckpt[f"{prefix}.input_layernorm.weight"] = torch.randn(64, dtype=torch.bfloat16)
        ckpt[f"{prefix}.mlp.gate.weight"] = torch.randn(_EXPERTS, 64, dtype=torch.bfloat16)
        for owner in ("mlp", *(f"mlp.experts.{expert}" for expert in range(_EXPERTS))):
            ckpt[f"{prefix}.{owner}.gate_proj.weight"] = torch.randn(128, 64, dtype=torch.bfloat16) * 0.1
            ckpt[f"{prefix}.{owner}.up_proj.weight"] = torch.randn(128, 64, dtype=torch.bfloat16) * 0.3
            ckpt[f"{prefix}.{owner}.down_proj.weight"] = torch.randn(64, 128, dtype=torch.bfloat16) * 0.1
    return ckpt


def _write(directory, ckpt: dict[str, torch.Tensor]) -> str:
    os.makedirs(directory, exist_ok=True)
    save_file(ckpt, os.path.join(directory, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3_moe"}, f)
    return str(directory)


def _export(tmp_path, fmt: str, ckpt: dict | None = None, **kwargs) -> tuple[dict, dict, dict, dict]:
    """``(source weights, exported tensors, config.json's quantization_config, sidecar manifest)``."""
    ckpt = _weights() if ckpt is None else ckpt
    src, out = _write(tmp_path / "src", ckpt), str(tmp_path / "out")
    quantize_checkpoint(src, out, fmt, export_format=COMPRESSED_TENSORS, **kwargs)
    with open(os.path.join(out, "config.json")) as f:
        config = json.load(f)["quantization_config"]
    with open(os.path.join(out, "quantization_config.json")) as f:
        manifest = json.load(f)
    return ckpt, load_file(os.path.join(out, "model.safetensors")), config, manifest


def _quantized_modules(first_block: int = 0) -> list[str]:
    owners = ("mlp", *(f"mlp.experts.{expert}" for expert in range(_EXPERTS)))
    blocks = range(first_block, _BLOCKS)
    return [f"model.layers.{b}.{owner}.{proj}" for b in blocks for owner in owners for proj in _PROJECTIONS]


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """``unpack_fp4_from_uint8``: the even element in the low nibble, a sign bit over a three-bit level index."""
    codes = torch.stack([packed & 0x0F, packed >> 4], dim=-1).flatten(-2).long()
    return _E2M1[codes & 0x7] * torch.where((codes & 0x8).bool(), -1.0, 1.0)


def _mx_scale(scale: torch.Tensor) -> torch.Tensor:
    """``decompress_mx_scale``: a biased power-of-two exponent."""
    return torch.exp2((scale.to(torch.int32) - 127).float())


def _compressed_tensors_dequant(saved: dict, module: str, fmt: str) -> torch.Tensor:
    """The weight compressed-tensors' ``decompress`` rebuilds: MXFP8's naive ``weight x scale``, the packed
    compressors' unpack, and NVFP4's ``scale / global_scale``."""
    if fmt == "mxfp8":
        values, scale = saved[f"{module}.weight"].float(), _mx_scale(saved[f"{module}.weight_scale"])
    elif fmt == "mxfp4":
        values, scale = _unpack_fp4(saved[f"{module}.weight_packed"]), _mx_scale(saved[f"{module}.weight_scale"])
    else:
        values = _unpack_fp4(saved[f"{module}.weight_packed"])
        scale = saved[f"{module}.weight_scale"].float() / saved[f"{module}.weight_global_scale"]
    group = _WEIGHT_ARGS[fmt]["group_size"]
    return (values * scale.repeat_interleave(group, dim=-1)).to(torch.bfloat16)


def _engine_matches(layer_name: str, targets: list[str]) -> bool:
    """vLLM's ``is_equal_or_regex_match`` for a layer name: ``re.match`` on a ``re:`` target, else equality."""
    return any(
        re.match(target[3:], layer_name) is not None if target.startswith("re:") else target == layer_name
        for target in targets
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_each_format_writes_the_tensors_its_compressor_names(tmp_path, fmt):
    source, saved, _config, _manifest = _export(tmp_path, fmt)
    group = _WEIGHT_ARGS[fmt]["group_size"]
    for module in _quantized_modules():
        out_features, in_features = source[f"{module}.weight"].shape
        expected = {f"{module}.weight_scale"}
        if fmt == "mxfp8":
            expected.add(f"{module}.weight")
            assert saved[f"{module}.weight"].dtype == torch.float8_e4m3fn
            assert saved[f"{module}.weight"].shape == (out_features, in_features)
        else:
            expected.add(f"{module}.weight_packed")
            assert saved[f"{module}.weight_packed"].dtype == torch.uint8
            assert saved[f"{module}.weight_packed"].shape == (out_features, in_features // 2)
        if fmt == "nvfp4":
            expected.add(f"{module}.weight_global_scale")
            assert saved[f"{module}.weight_global_scale"].shape == (1,)
        assert {key for key in saved if key.startswith(f"{module}.")} == expected
        scale = saved[f"{module}.weight_scale"]
        assert scale.shape == (out_features, in_features // group)
        assert scale.dtype == (torch.float8_e4m3fn if fmt == "nvfp4" else torch.uint8)
    untouched = set(source) - {f"{module}.weight" for module in _quantized_modules()}
    assert untouched and all(torch.equal(saved[key], source[key]) for key in untouched)
    assert not any(key.endswith(".weight_shape") for key in saved)


@pytest.mark.parametrize("fmt", FORMATS)
def test_decoding_by_compressed_tensors_rules_reproduces_the_qat_forward(tmp_path, fmt):
    source, saved, _config, _manifest = _export(tmp_path, fmt)
    for module in _quantized_modules():
        torch.testing.assert_close(
            _compressed_tensors_dequant(saved, module, fmt),
            fake_quant(source[f"{module}.weight"], fmt),
            rtol=0,
            atol=0,
            msg=f"{fmt}: {module} decodes to another weight than the QAT forward used",
        )


@pytest.mark.parametrize("fmt", FORMATS)
def test_config_json_carries_what_compressed_tensors_serializes_for_the_format(tmp_path, fmt):
    _source, _saved, config, manifest = _export(tmp_path, fmt)
    (group,) = config.pop("config_groups").values()
    assert config == {
        "quant_method": "compressed-tensors",
        "kv_cache_scheme": None,
        "format": _COMPRESSION_FORMAT[fmt],
        "quantization_status": "compressed",
        "global_compression_ratio": None,
        "ignore": [],
    }
    targets = group.pop("targets")
    assert targets and all(target.startswith("re:") for target in targets)
    assert group == {
        "weights": _expected_args(fmt, dynamic=False),
        # MX trained with both GEMM operands quantized. NVFP4 exports weight-only (NVFP4A16).
        "input_activations": None if fmt == "nvfp4" else _expected_args(fmt, dynamic=True),
        "output_activations": None,
        "format": _COMPRESSION_FORMAT[fmt],
    }
    # Halo's own description stays in the sidecar, for its tools and a re-export.
    assert manifest["format"] == fmt and manifest["scope"]["apply_moe_experts"] is True
    assert sorted(manifest["quantized_weights"]) == sorted(f"{module}.weight" for module in _quantized_modules())
    assert "compressed-tensors" in manifest["layout"]


def test_nvfp4_gate_and_up_decode_under_the_merged_layers_one_global_scale(tmp_path):
    """vLLM loads ``gate_proj`` and ``up_proj`` into one layer and decodes it under one nvfp4 global scale: a dense
    ``gate_up_proj`` the largest stored divisor, a MoE expert's ``w13`` the gate's. Each pair is written with one."""
    source, saved, _config, _manifest = _export(tmp_path, "nvfp4")
    owners = {module.rsplit(".", 1)[0] for module in _quantized_modules()}
    for owner in sorted(owners):
        gate, up = f"{owner}.gate_proj", f"{owner}.up_proj"
        own = {quantize_nvfp4(source[f"{m}.weight"].float()).global_scale.item() for m in (gate, up)}
        assert len(own) == 2, f"{owner}: the fixture should give gate and up different global scales"
        divisors = [saved[f"{m}.weight_global_scale"] for m in (gate, up)]
        assert torch.equal(*divisors), f"{owner}: gate and up written with different global scales {divisors}"
        engine_divisor = divisors[0] if ".experts." in owner else torch.maximum(*divisors)
        for module in (gate, up):
            merged = {**saved, f"{module}.weight_global_scale": engine_divisor}
            torch.testing.assert_close(
                _compressed_tensors_dequant(merged, module, "nvfp4"),
                fake_quant(source[f"{module}.weight"], "nvfp4"),
                rtol=0,
                atol=0,
                msg=f"{module} decodes under its merged layer's global scale to another weight than QAT used",
            )


def test_a_pair_that_cannot_share_one_nvfp4_global_scale_exactly_is_refused(tmp_path):
    """``up_proj`` 2^-16 of ``gate_proj``: under the gate's global scale its block scales fall to e4m3 subnormals and
    round, which would serve another weight than QAT trained."""
    ckpt = _weights(blocks=1)
    gate = ckpt["model.layers.0.mlp.gate_proj.weight"]
    ckpt["model.layers.0.mlp.up_proj.weight"] = (gate.float() * 2**-16).to(torch.bfloat16)
    src, out = _write(tmp_path / "src", ckpt), tmp_path / "out"
    with pytest.raises(ValueError, match=r"mlp\.up_proj\.weight.*one nvfp4 global scale.*rounds \d+ of its"):
        quantize_checkpoint(src, str(out), "nvfp4", export_format=COMPRESSED_TENSORS)
    assert not (out / "config.json").exists()
    quantize_checkpoint(src, str(out), "nvfp4", export_format=BLOCK_SCALED)  # Halo's own layout keeps each scale


def test_targets_match_exactly_the_quantized_modules(tmp_path):
    """Under the engine's rule, against the checkpoint's names and the names an engine gives the same layers."""
    source, _saved, config, _manifest = _export(tmp_path, "mxfp4", keep_first_blocks=1)
    (group,) = config["config_groups"].values()
    targets = group["targets"]
    quantized = set(_quantized_modules(first_block=1))
    weight_modules = {key[: -len(".weight")] for key, tensor in source.items() if tensor.dim() >= 2}
    assert {module for module in weight_modules if _engine_matches(module, targets)} == quantized
    # vLLM roots a composite's text backbone elsewhere, and asks for a MoE layer by its first expert's projections.
    assert _engine_matches("language_model.model.layers.2.mlp.experts.0.gate_proj", targets)
    assert _engine_matches("language_model.model.layers.1.mlp.down_proj", targets)
    assert not _engine_matches("language_model.model.layers.0.mlp.experts.0.gate_proj", targets)  # kept block
    assert not _engine_matches("model.layers.1.self_attn.q_proj", targets)
    assert not _engine_matches("model.xlayers.1.mlp.gate_proj", targets)
    assert not _engine_matches("model.layers.1.mlp.gate_proj_extra", targets)
    assert not _engine_matches("model.layers.11.mlp.gate_proj", targets)


def test_a_fused_expert_bank_is_refused_before_any_write(tmp_path):
    ckpt = _weights(blocks=1)
    ckpt["model.layers.0.mlp.experts.gate_up_proj"] = torch.randn(_EXPERTS, 64, 256, dtype=torch.bfloat16)
    src, out = _write(tmp_path / "src", ckpt), tmp_path / "out"
    with pytest.raises(ValueError, match=r"fused 3-D expert bank.*unfuse_moe_experts\.py"):
        quantize_checkpoint(src, str(out), "mxfp4", export_format=COMPRESSED_TENSORS)
    assert not out.exists()
    quantize_checkpoint(src, str(out), "mxfp4", export_format=BLOCK_SCALED)  # Halo's own layout still takes it


def test_a_scoped_weight_that_cannot_be_blocked_is_refused_before_any_write(tmp_path):
    """Left in high precision beside quantized siblings, it would give one fused layer two schemes."""
    ckpt = _weights(blocks=1)
    ckpt["model.layers.0.mlp.up_proj.weight"] = torch.randn(128, 72, dtype=torch.bfloat16)
    src, out = _write(tmp_path / "src", ckpt), tmp_path / "out"
    with pytest.raises(ValueError, match=r"up_proj\.weight.*not divisible by the mxfp8 block size 32"):
        quantize_checkpoint(src, str(out), "mxfp8", export_format=COMPRESSED_TENSORS)
    assert not out.exists()


def test_targets_that_reach_an_unquantized_module_are_refused_before_any_write(tmp_path):
    """A vision tower spelling its MLP like the text backbone's stays bf16, and the targets must not name it."""
    ckpt = _weights(blocks=1)
    ckpt["model.vision_tower.encoder.layers.0.mlp.gate_proj.weight"] = torch.randn(128, 64, dtype=torch.bfloat16)
    src, out = _write(tmp_path / "src", ckpt), tmp_path / "out"
    with pytest.raises(ValueError, match=r"also match 1 module.*vision_tower\.encoder\.layers\.0\.mlp\.gate_proj"):
        quantize_checkpoint(src, str(out), "nvfp4", export_format=COMPRESSED_TENSORS)
    assert not out.exists()


def test_an_unknown_export_format_is_refused(tmp_path):
    src = _write(tmp_path / "src", _weights(blocks=1))
    with pytest.raises(ValueError, match="unknown export_format 'gguf'"):
        quantize_checkpoint(src, str(tmp_path / "out"), "mxfp8", export_format="gguf")


def test_the_default_layout_stays_halos_own(tmp_path):
    src, out = _write(tmp_path / "src", _weights(blocks=1)), tmp_path / "out"
    quantize_checkpoint(src, str(out), "mxfp8")
    saved = load_file(str(out / "model.safetensors"))
    module = "model.layers.0.mlp.gate_proj"
    assert {f"{module}.weight_packed", f"{module}.weight_scale", f"{module}.weight_shape"} <= set(saved)
    with open(out / "config.json") as f:
        assert json.load(f)["quantization_config"]["quant_method"] == BLOCK_SCALED


def test_the_script_passes_export_format_through(tmp_path):
    script = load_script_module("scripts/after_training/quantize_to_lowp.py")
    src, out = _write(tmp_path / "src", _weights(blocks=1)), tmp_path / "out"
    argv = ["quantize_to_lowp.py", "--input_dir", src, "--output_dir", str(out), "--format", "mxfp4"]
    with mock.patch.object(sys, "argv", [*argv, "--export_format", "compressed-tensors"]):
        script.main()
    with open(out / "config.json") as f:
        assert json.load(f)["quantization_config"]["quant_method"] == "compressed-tensors"
    with mock.patch.object(sys, "argv", argv):
        assert script.parse_args().export_format == lowp_export.DEFAULT_EXPORT_FORMAT


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
