"""The Nemotron-H releases as tests address them: each one's hub id and knob, its tiny checkpoint as stored, and the
width an EP row holds its wire to."""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, ProcessorMixin

from halo.env import env_str
from tests.common.models import NEMOTRON_3_5_LIGHTNING, NEMOTRON_3_5_SUPER_VL, TINY_NEMOTRON_H_CONFIG
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, tiny_family_model

# Hub id and checkpoint knob per release. One knob for both would point Lightning's rows at Super VL's weights, where they
# pass.
NEMOTRON_RELEASES = {
    "super": (NEMOTRON_3_5_SUPER_VL, "HALO_TEST_NEMOTRON_MODEL"),
    "lightning": (NEMOTRON_3_5_LIGHTNING, "HALO_TEST_NEMOTRON_LIGHTNING_MODEL"),
}
NEMOTRON_H_COMPOSITE = TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]
COMPOSITE_TEXT_PREFIX = "language_model."
VISION_NORM = "vision_final_layernorm"
HUB_VISION_NORM = f"vision_projector.{VISION_NORM}"
BIAS_SLOT = "e_score_correction_bias"
# The release's ``config.json`` verbatim, at hub revision fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe.
_RELEASE_CONFIG_FILE = Path(__file__).with_name("nemotron_h_release_config.json")
# The Lightning release's ``config.json`` verbatim, at hub revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9.
_LIGHTNING_CONFIG_FILE = Path(__file__).with_name("nemotron_h_lightning_config.json")
_INPUT_CONDITIONER = ".input_conditioner."
_LAYER_INDEX = re.compile(r"\.layers\.(\d+)\.")
# What shrinks Lightning to the roster's widths; every other field stays as the release spells it.
_TINY_LIGHTNING_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "layers_block_type",
    "mamba_num_heads",
    "mamba_head_dim",
    "ssm_state_size",
    "n_groups",
    "chunk_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "moe_shared_expert_intermediate_size",
)


def release_model_id(release: str) -> str:
    hub_id, knob = NEMOTRON_RELEASES[release]
    return env_str(knob, hub_id)


def wire_width_checks(layers, *, hidden_size: int, latent: int | None) -> dict[str, bool]:
    """Each EP layer against the width its row declares. The model's own ``moe_latent_size`` cannot say: a row that lost
    its override would hold the latent to itself and pass."""
    if latent is None:
        return {
            "hidden_state_on_the_wire": all(
                layer.dispatcher.hidden_dim == hidden_size
                and isinstance(layer.fc1_latent_proj, nn.Identity)
                and isinstance(layer.fc2_latent_proj, nn.Identity)
                for layer in layers
            )
        }
    return {"latent_on_the_wire": all(layer.dispatcher.hidden_dim == latent for layer in layers)}


def nemotron_h_release_config() -> dict:
    return json.loads(_RELEASE_CONFIG_FILE.read_text(encoding="utf-8"))


def nemotron_h_lightning_config() -> dict:
    return json.loads(_LIGHTNING_CONFIG_FILE.read_text(encoding="utf-8"))


def tiny_lightning_config() -> dict:
    tiny = {field: TINY_NEMOTRON_H_CONFIG[field] for field in _TINY_LIGHTNING_FIELDS}
    return {**nemotron_h_lightning_config(), **tiny, "num_hidden_layers": len(tiny["layers_block_type"])}


def write_tiny_lightning_checkpoint(directory: Path) -> dict[str, torch.Tensor]:
    """Lightning's layout at the roster's widths: its config as the release spells it, no latent, keys at the root, per
    expert, the unbuilt multi-token head stored beside them, every float bf16 but the router bias. Returns what it
    wrote."""
    directory.mkdir(parents=True, exist_ok=True)
    source_config = json.dumps(tiny_lightning_config(), indent=2)
    (directory / "config.json").write_text(source_config, encoding="utf-8")
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(directory)).save_pretrained(directory)
    (directory / "config.json").write_text(source_config, encoding="utf-8")
    tensors = load_file(directory / "model.safetensors")
    assert {"backbone.layers.1.mixer.experts.0.up_proj.weight", "lm_head.weight"} <= set(tensors), "not the release's"
    tensors["mtp.layers.0.norm.weight"] = torch.ones(4)
    tensors["mtp.layers.1.mixer.experts.0.up_proj.weight"] = torch.ones(4, 4)
    for key, value in tensors.items():
        if value.is_floating_point():
            tensors[key] = value.to(torch.float32 if key.endswith(BIAS_SLOT) else torch.bfloat16)
    save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})
    return tensors


def hub_text_prefix(released_keys) -> str:
    """Where a release stores its text tower: under the composite's level, or at the root of a text-only release."""
    return COMPOSITE_TEXT_PREFIX if any(key.startswith(COMPOSITE_TEXT_PREFIX) for key in released_keys) else ""


def expected_hub_keys(released_keys, layers: int | None, text_only: bool) -> set[str]:
    """What an export of this route and depth stores of a release. No class builds the multi-token head or the
    conditioner the processor applies."""
    prefix = hub_text_prefix(released_keys)
    kept = set()
    for key in released_keys:
        if key.startswith(f"{prefix}mtp.") or _INPUT_CONDITIONER in key or (text_only and not key.startswith(prefix)):
            continue
        layer = _LAYER_INDEX.search(key)
        if layers is None or layer is None or int(layer.group(1)) < layers:
            kept.add(key)
    return kept


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
