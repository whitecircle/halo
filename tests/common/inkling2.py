"""Tiny Inkling 2 checkpoints in the release's own layout: NVFP4 routed experts on the early layers and MXFP8 on the
last ones (``thinkingmachines/Inkling-2-Small`` ships 40 + 8), ``model.llm.*`` names, interleaved ``w13``. Beside each
release-format checkpoint, its exact bf16 dequantization, which transformers loads as the reference."""

from __future__ import annotations

import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.inkling2.configuration_inkling2 import Inkling2Config
from transformers.models.inkling2.modeling_inkling2 import Inkling2ForConditionalGeneration

from halo.kernels.lowp.frozen_bank import FrozenExpertBank
from halo.kernels.lowp.quantization import quantize_mxfp8, quantize_nvfp4

# K stays a multiple of both block sizes (16, 32) on both projections; 6 layers at 3 per attention-residual block
# cover two blocks, and the default layer types put the one full-attention layer last.
TINY_INKLING2_TEXT = {
    "hidden_size": 128,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 32,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "vocab_size": 512,
    "unpadded_vocab_size": 500,
    "sliding_window_size": 16,
    "rel_extent": 16,
    "d_rel": 4,
    "n_layers_per_attn_res_block": 3,
    "eos_token_id": 2,
}
TINY_INKLING2_VISION = {
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "text_hidden_size": TINY_INKLING2_TEXT["hidden_size"],
    "projection_hidden_dim": 32,
}
# The release's split, scaled down: the last layers' experts are MXFP8.
TINY_INKLING2_MXFP8_LAYERS = frozenset({4, 5})
NVFP4_OUTER_SCALE_DENOMINATOR = 448.0 * 6.0

_EXPERT_KEY = "model.llm.layers.{layer}.mlp.experts.{proj}"


def tiny_inkling2_config() -> Inkling2Config:
    return Inkling2Config(text_config=dict(TINY_INKLING2_TEXT), vision_config=dict(TINY_INKLING2_VISION))


def quantize_release_bank(weight: torch.Tensor, quant_format: str) -> tuple[FrozenExpertBank, dict[str, torch.Tensor]]:
    """One ``[E, N, K]`` expert tensor in a release format: the bank, and the tensors the release stores keyed by their
    suffix on the weight's name."""
    if quant_format == "mxfp8":
        per_expert = [quantize_mxfp8(expert.float(), axis=-1) for expert in weight]
        data = torch.stack([q.data for q in per_expert])
        scales = torch.stack([q.scales for q in per_expert])
        return FrozenExpertBank("mxfp8", data, scales), {"": data, "_scale": scales}
    per_expert = [quantize_nvfp4(expert.float(), axis=-1) for expert in weight]
    data = torch.stack([q.data for q in per_expert])
    scales = torch.stack([q.scales for q in per_expert])
    global_scale = torch.stack([q.global_scale.reshape(()) for q in per_expert]).float()
    stored = {
        "": data,
        "_scale": scales,
        "_amax": global_scale * NVFP4_OUTER_SCALE_DENOMINATOR,
        "_outer_scale_denominator": torch.tensor([NVFP4_OUTER_SCALE_DENOMINATOR]),
    }
    return FrozenExpertBank("nvfp4", data, scales, global_scale), stored


def _quant_config(num_layers: int) -> dict:
    """The release's ``hf_quant_config.json`` groups for the expert tensors; the rest of its manifest is omitted."""

    def group(fmt: str, block: int) -> dict:
        layers = [i for i in range(num_layers) if (i in TINY_INKLING2_MXFP8_LAYERS) == (fmt == "mxfp8")]
        return {
            "compute_dtype": fmt,
            "weight_block_shape": [1, block] if fmt == "nvfp4" else [32, 32],
            "tensors": [_EXPERT_KEY.format(layer=i, proj=p) for i in layers for p in ("w13_weight", "w2_weight")],
            **({"weight_scale_format": "amax_denominator"} if fmt == "nvfp4" else {}),
        }

    return {"quant_method": "inkling2", "model": {"parameter_groups": [group("nvfp4", 16), group("mxfp8", 32)]}}


def write_tiny_inkling2_release(release_dir: str, reference_dir: str, seed: int = 0) -> None:
    """``release_dir``: the release layout (packed experts, modelopt ``quantization_config``, no
    ``transformers_version``). ``reference_dir``: the same weights with the experts dequantized to bf16, loadable by
    ``from_pretrained``."""
    torch.manual_seed(seed)
    config = tiny_inkling2_config()
    model = Inkling2ForConditionalGeneration(config).to(torch.bfloat16)
    model.save_pretrained(reference_dir)
    weights_path = os.path.join(reference_dir, "model.safetensors")
    with safe_open(weights_path, framework="pt") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118 - not a mapping

    release, reference = dict(tensors), dict(tensors)
    num_layers = config.text_config.num_hidden_layers
    for layer in range(num_layers):
        for proj in ("w13_weight", "w2_weight"):
            key = _EXPERT_KEY.format(layer=layer, proj=proj)
            quant_format = "mxfp8" if layer in TINY_INKLING2_MXFP8_LAYERS else "nvfp4"
            bank, stored = quantize_release_bank(tensors[key], quant_format)
            del release[key]
            release.update({f"{key}{suffix}": tensor.contiguous() for suffix, tensor in stored.items()})
            reference[key] = bank.dequantize_linear()

    save_file(reference, weights_path, metadata={"format": "pt"})
    os.makedirs(release_dir, exist_ok=True)
    save_file(release, os.path.join(release_dir, "model.safetensors"), metadata={"format": "pt"})
    for name in os.listdir(reference_dir):
        if name not in ("model.safetensors", "config.json"):
            shutil.copy(os.path.join(reference_dir, name), os.path.join(release_dir, name))

    with open(os.path.join(reference_dir, "config.json")) as handle:
        release_config = json.load(handle)
    release_config.pop("transformers_version", None)
    release_config["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
    with open(os.path.join(release_dir, "config.json"), "w") as handle:
        json.dump(release_config, handle, indent=2)
    with open(os.path.join(release_dir, "hf_quant_config.json"), "w") as handle:
        json.dump(_quant_config(num_layers), handle, indent=2)
