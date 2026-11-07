#!/usr/bin/env python
"""The served block-FP8 checkpoint equals what the weight push sends, tensor for tensor, and declares what it keeps in
bf16 by exact module path."""

import json

import pytest
import torch
from safetensors import safe_open

from halo.checkpoint.tools.block_fp8_export import export_block_fp8_checkpoint
from halo.trainers.grpo.rollout.weight_push import gather_and_send_weights
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, tiny_family_model
from tests.common.weight_sync import RecordingSender

_FAMILIES = {
    "qwen3_5_moe": TINY_MOE_VLM_FAMILIES["qwen3_5_moe"],
    "qwen3_moe": TINY_MOE_FAMILIES["qwen3_moe"],
    "qwen3": TINY_DENSE_FAMILIES["qwen3"],
    "qwen3_5": TINY_DENSE_FAMILIES["qwen3_5"],
}


def _read(directory) -> dict[str, torch.Tensor]:
    tensors = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(str(shard), "pt") as reader:
            tensors.update({key: reader.get_tensor(key) for key in reader.keys()})  # noqa: SIM118 - not a mapping
    return tensors


@pytest.mark.parametrize("family_name", list(_FAMILIES))
def test_the_checkpoint_holds_what_the_push_sends_in_name_dtype_shape_and_bytes(family_name, tmp_path):
    torch.manual_seed(0)
    model = tiny_family_model(_FAMILIES[family_name]).to(torch.bfloat16)
    model.save_pretrained(tmp_path / "bf16")
    export_block_fp8_checkpoint(str(tmp_path / "bf16"), str(tmp_path / "fp8"))

    sender = RecordingSender(keep_values=True)
    gather_and_send_weights(model, sender, precision="fp8_block")
    pushed = {param.name: param.value for param in sender.params}
    stored = _read(tmp_path / "fp8")

    assert pushed.keys() == stored.keys()
    assert any(value.dtype == torch.float8_e4m3fn for value in stored.values())
    for name, value in pushed.items():
        assert stored[name].dtype == value.dtype and stored[name].shape == value.shape, name
        assert torch.equal(stored[name].view(torch.uint8), value.contiguous().view(torch.uint8)), name


def test_the_config_names_every_bf16_linear_by_exact_path_and_never_a_quantized_one(tmp_path):
    torch.manual_seed(0)
    model = tiny_family_model(_FAMILIES["qwen3_5_moe"]).to(torch.bfloat16)
    model.save_pretrained(tmp_path / "bf16")
    export_block_fp8_checkpoint(str(tmp_path / "bf16"), str(tmp_path / "fp8"))

    config = json.loads((tmp_path / "fp8" / "config.json").read_text())
    quantization = config["quantization_config"]
    assert {key: quantization[key] for key in ("quant_method", "fmt", "activation_scheme", "weight_block_size")} == {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    kept = set(quantization["modules_to_not_convert"])
    stored = _read(tmp_path / "fp8")
    quantized = {name.removesuffix(".weight") for name, value in stored.items() if value.dtype == torch.float8_e4m3fn}
    linear_weights = {
        name.removesuffix(".weight") for name, value in stored.items() if name.endswith(".weight") and value.ndim == 2
    }
    assert not kept & quantized
    assert linear_weights - quantized == kept
    assert "model.language_model.layers.0.linear_attn.in_proj_qkv" in kept
    assert "model.language_model.layers.0.mlp.gate" in kept
    assert "model.visual.blocks.0.attn.qkv" in kept


def test_a_checkpoint_that_is_already_quantized_is_refused(tmp_path):
    torch.manual_seed(0)
    tiny_family_model(_FAMILIES["qwen3"]).to(torch.bfloat16).save_pretrained(tmp_path / "bf16")
    export_block_fp8_checkpoint(str(tmp_path / "bf16"), str(tmp_path / "fp8"))
    with pytest.raises(ValueError, match="already declares a quantization_config"):
        export_block_fp8_checkpoint(str(tmp_path / "fp8"), str(tmp_path / "again"))


def test_an_unvalidated_family_is_refused_before_anything_is_written(tmp_path):
    tiny_family_model(TINY_MOE_FAMILIES["gpt_oss"]).to(torch.bfloat16).save_pretrained(tmp_path / "bf16")
    with pytest.raises(ValueError, match="block-FP8 rollouts are refused"):
        export_block_fp8_checkpoint(str(tmp_path / "bf16"), str(tmp_path / "fp8"))
    assert not (tmp_path / "fp8").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
