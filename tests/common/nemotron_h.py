"""The tiny Nemotron-H composite as the release stores it, for the tests reading its checkpoint."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import ProcessorMixin

from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, tiny_family_model

NEMOTRON_H_COMPOSITE = TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]
VISION_NORM = "vision_final_layernorm"
HUB_VISION_NORM = f"vision_projector.{VISION_NORM}"
BIAS_SLOT = "e_score_correction_bias"
# The release's ``config.json`` verbatim, at hub revision fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe.
_RELEASE_CONFIG_FILE = Path(__file__).with_name("nemotron_h_release_config.json")


def nemotron_h_release_config() -> dict:
    return json.loads(_RELEASE_CONFIG_FILE.read_text(encoding="utf-8"))


def write_tiny_hub_checkpoint(
    directory: Path,
    *,
    weights_dtype: torch.dtype = torch.float32,
    norm_weight: float = 2.5,
    norm_bias: float = 0.25,
    processor: ProcessorMixin | None = None,
) -> dict[str, torch.Tensor]:
    """Per-expert keys, the vision norm non-identity under its projector, the multi-token head present, no layer scale,
    and every float at ``weights_dtype`` but the router bias, which the release stores in fp32. ``processor`` is saved
    beside it, with the model built on its vocabulary and image token. Returns what it wrote."""
    torch.manual_seed(0)
    model = tiny_family_model(NEMOTRON_H_COMPOSITE, processor.tokenizer if processor is not None else None)
    if processor is not None:
        model.config.image_token_id = processor.image_token_id
        processor.save_pretrained(directory)
    with torch.no_grad():
        getattr(model, VISION_NORM).weight.fill_(norm_weight)
        getattr(model, VISION_NORM).bias.fill_(norm_bias)
    model.save_pretrained(directory)
    tensors = load_file(directory / "model.safetensors")
    assert {f"{HUB_VISION_NORM}.weight", f"{HUB_VISION_NORM}.bias"} <= set(tensors), "not in the hub's spelling"
    tensors = {key: value for key, value in tensors.items() if "lambda1" not in key}
    tensors["language_model.mtp.layers.0.norm.weight"] = torch.ones(4)
    tensors["language_model.mtp.layers.1.mixer.experts.0.up_proj.weight"] = torch.ones(4, 4)
    tensors["language_model.mtp.layers.1.mixer.experts.1.up_proj.weight"] = torch.ones(4, 4)
    tensors = {
        key: value.to(weights_dtype) if value.is_floating_point() and not key.endswith(BIAS_SLOT) else value
        for key, value in tensors.items()
    }
    save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})
    return tensors
