"""Native MXFP4 gpt-oss checkpoints in the ``openai/gpt-oss-*`` layout, written from a bf16 save, for the lazy-decode
tests: the experts as ``_blocks``/``_scales`` and the release's ``quantization_config``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from halo.kernels.lowp.quantization import FORMAT_BLOCK_SIZE, quantize_mxfp4
from halo.model_loading.lazy_safetensors.conversion import MXFP4_PROJECTIONS

# As the openai/gpt-oss-* releases declare it.
MXFP4_QUANTIZATION_CONFIG = {
    "modules_to_not_convert": [
        "model.layers.*.self_attn",
        "model.layers.*.mlp.router",
        "model.embed_tokens",
        "lm_head",
    ],
    "quant_method": "mxfp4",
}
_INDEX = "model.safetensors.index.json"


def _pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A bf16 expert bank ``[E, A, B]`` → the release's ``_blocks`` ``[E, B, A/32, 16]`` and ``_scales``
    ``[E, B, A/32]``: blocks of 32 along ``A``, two e2m1 codes per byte, low nibble first, as ``quantize_mxfp4`` packs
    them and transformers' ``convert_moe_packed_tensors`` reads them."""
    experts, a, b = weight.shape
    block = FORMAT_BLOCK_SIZE["mxfp4"]
    q = quantize_mxfp4(weight.transpose(1, 2).contiguous(), axis=-1)
    return q.data.reshape(experts, b, a // block, block // 2), q.scales.reshape(experts, b, a // block)


def write_native_mxfp4(path: str | os.PathLike) -> None:
    """Rewrite a bf16 gpt-oss save in place in the release layout. Its expert banks quantize to MXFP4, so a lazy load
    decodes the rounded values, which ``from_pretrained(quantization_config=Mxfp4Config(dequantize=True))`` reproduces."""
    path = Path(path)
    index_file = path / _INDEX
    index = json.loads(index_file.read_text()) if index_file.exists() else None
    for weights_file in sorted(path.glob("*.safetensors")):
        state = load_file(weights_file)
        banks = [key for key in state if key.rpartition(".")[2] in MXFP4_PROJECTIONS]
        if not banks:
            continue
        for key in banks:
            state[f"{key}_blocks"], state[f"{key}_scales"] = _pack(state.pop(key))
            if index is not None:
                del index["weight_map"][key]
                index["weight_map"] |= {f"{key}_blocks": weights_file.name, f"{key}_scales": weights_file.name}
        save_file(state, weights_file, metadata={"format": "pt"})
    if index is not None:
        index_file.write_text(json.dumps(index, indent=2))
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "gpt_oss":
        raise ValueError(f"{path}: native MXFP4 is gpt-oss's layout, not {config.get('model_type')!r}'s")
    config["quantization_config"] = MXFP4_QUANTIZATION_CONFIG
    (path / "config.json").write_text(json.dumps(config, indent=2))
