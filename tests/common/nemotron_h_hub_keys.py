"""The keys of the Nemotron 3.5 Super VL release's ``model.safetensors.index.json``, expanded from the index's own
patterns and pinned to it by digest, so a test plans the release's checkpoint without fetching any of it."""

import hashlib
from collections.abc import Iterator

from tests.common.models import NEMOTRON_H_RELEASE_BLOCKS, NEMOTRON_H_RELEASE_LAYOUT

NEMOTRON_H_HUB_KEY_COUNT = 43_079
NEMOTRON_H_HUB_INDEX_SHA256 = "6d8a2121099eabfdcd1237bb703d2fc23079404696b8e2a426342ba83c57cc4f"

_TEXT = "language_model.backbone"
_MTP = "language_model.mtp.layers"
_VISION = "vision_model.radio_model"
_ROUTED_EXPERTS = 512
_VISION_BLOCKS = 32

_MAMBA_LEAVES = (
    "A_log",
    "D",
    "conv1d.bias",
    "conv1d.weight",
    "dt_bias",
    "in_proj.weight",
    "norm.weight",
    "out_proj.weight",
)
_ATTENTION_LEAVES = ("k_proj.weight", "o_proj.weight", "q_proj.weight", "v_proj.weight")
_MOE_LEAVES = (
    "fc1_latent_proj.weight",
    "fc2_latent_proj.weight",
    "gate.e_score_correction_bias",
    "gate.weight",
    "shared_experts.down_proj.weight",
    "shared_experts.up_proj.weight",
)
_VISION_BLOCK_LEAVES = (
    "attn.proj.bias",
    "attn.proj.weight",
    "attn.qkv.bias",
    "attn.qkv.weight",
    "mlp.fc1.bias",
    "mlp.fc1.weight",
    "mlp.fc2.bias",
    "mlp.fc2.weight",
    "norm1.bias",
    "norm1.weight",
    "norm2.bias",
    "norm2.weight",
)
_OUTSIDE_THE_LAYERS = (
    f"{_TEXT}.embeddings.weight",
    f"{_TEXT}.norm_f.weight",
    "language_model.lm_head.weight",
    "mlp1.0.weight",
    "mlp1.1.weight",
    "mlp1.3.weight",
    f"{_VISION}.input_conditioner.norm_mean",
    f"{_VISION}.input_conditioner.norm_std",
    f"{_VISION}.model.patch_generator.cls_token.token",
    f"{_VISION}.model.patch_generator.embedder.weight",
    f"{_VISION}.model.patch_generator.pos_embed",
    f"{_VISION}.model.patch_generator.video_embedder.weight",
    "vision_model.summary_idxs",
    "vision_projector.vision_final_layernorm.bias",
    "vision_projector.vision_final_layernorm.weight",
)
_MTP_ATTENTION_LAYER = ("eh_proj.weight", "enorm.weight", "hnorm.weight", "norm.weight")
_MTP_MOE_LAYER = ("final_layernorm.weight", "norm.weight")


def _experts(prefix: str) -> Iterator[str]:
    for expert in range(_ROUTED_EXPERTS):
        yield f"{prefix}.experts.{expert}.up_proj.weight"
        yield f"{prefix}.experts.{expert}.down_proj.weight"


def _text_layers() -> Iterator[str]:
    for index, block in enumerate(NEMOTRON_H_RELEASE_LAYOUT):
        kind = NEMOTRON_H_RELEASE_BLOCKS[block]
        yield f"{_TEXT}.layers.{index}.norm.weight"
        mixer = f"{_TEXT}.layers.{index}.mixer"
        leaves = {"mamba": _MAMBA_LEAVES, "attention": _ATTENTION_LEAVES, "moe": _MOE_LEAVES}[kind]
        yield from (f"{mixer}.{leaf}" for leaf in leaves)
        if kind == "moe":
            yield from _experts(mixer)


def _mtp_layers() -> Iterator[str]:
    yield from (f"{_MTP}.0.{leaf}" for leaf in _MTP_ATTENTION_LAYER)
    yield from (f"{_MTP}.0.mixer.{leaf}" for leaf in _ATTENTION_LEAVES)
    yield from (f"{_MTP}.1.{leaf}" for leaf in _MTP_MOE_LAYER)
    yield from (f"{_MTP}.1.mixer.{leaf}" for leaf in _MOE_LEAVES)
    yield from _experts(f"{_MTP}.1.mixer")


def _vision_blocks() -> Iterator[str]:
    for index in range(_VISION_BLOCKS):
        yield from (f"{_VISION}.model.blocks.{index}.{leaf}" for leaf in _VISION_BLOCK_LEAVES)


def nemotron_h_hub_keys() -> list[str]:
    """Sorted, as the digest reads them."""
    return sorted((*_OUTSIDE_THE_LAYERS, *_text_layers(), *_mtp_layers(), *_vision_blocks()))


def hub_keys_digest(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
