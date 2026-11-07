"""What a block-FP8 rollout server stores quantized: the attention and MLP projections of the text decoder, expert
banks split per expert. Routers, gates, linear-attention and Mamba blocks, embeddings, the head, norms, towers and MTP
heads stay bf16. Names are the hub's, as the weight push and the served-checkpoint tool both see them."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch

from halo.kernels.lowp.block_fp8 import BLOCK, SCALE_SUFFIX, quantize_block_fp8
from halo.kernels.lowp.spec import ATTENTION_PROJECTIONS, MLP_PROJECTIONS
from halo.models.config_access import config_model_types, text_config
from halo.models.module_tree import decoder_layer_index
from halo.models.moe.config_fields import has_routed_experts
from halo.models.moe.family_specs import experts_container_attrs, find_moe_family_spec_for_config
from halo.models.mtp.spec import ROOT_HEAD_ATTR

RolloutWeightPrecision = Literal["bf16", "fp8_block"]

# Dense families a GPU row validated (an MoE family declares its own on its spec).
DENSE_FP8_ROLLOUT_MODEL_TYPES = ("qwen3", "qwen3_5", "qwen3_5_text")
_FUSED_BANKS = ("gate_up_proj", "down_proj")
# Elements one batched quantization call works on, about 1 GiB of its fp32 copy.
_QUANTIZE_ELEMENTS = 1 << 28


def fp8_rollout_refusal(config) -> str | None:
    """Why block-FP8 rollouts are not validated for ``config``'s family, else ``None``."""
    spec = find_moe_family_spec_for_config(config)
    if spec is not None:
        return spec.fp8_rollout_refusal
    if has_routed_experts(config):
        return "it is an MoE with no family spec, so its expert layout is not declared"
    if set(DENSE_FP8_ROLLOUT_MODEL_TYPES) & set(config_model_types(config)):
        return None
    return "no GPU row has validated block-FP8 rollouts for the family"


@dataclass(frozen=True)
class Fp8RolloutPlan:
    """``expert_keys`` is the ``(gate, up, down)`` spelling the family's served experts are stored under, ``None`` for
    a dense model."""

    num_layers: int
    expert_keys: tuple[str, str, str] | None = None

    @classmethod
    def for_config(cls, config) -> Fp8RolloutPlan:
        """Raises ``ValueError`` naming the reason when block-FP8 rollouts are not validated for the family."""
        if (reason := fp8_rollout_refusal(config)) is not None:
            model_types = list(config_model_types(config))
            raise ValueError(f"block-FP8 rollouts are refused for model_type {model_types}: {reason}.")
        spec = find_moe_family_spec_for_config(config)
        return cls(text_config(config).num_hidden_layers, spec.expert_layout.per_expert_keys if spec else None)

    def expand(self, name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        """``(name, tensor)`` as the server stores it: a selected projection as its fp8 weight and ``_scale_inv``, a
        fused expert bank as those pairs per expert, anything else as it came."""
        if self._is_fused_bank(name, tensor):
            prefix, _, bank_name = name.rpartition(".")
            gate, up, down = self.expert_keys
            names = (gate, up) if bank_name == "gate_up_proj" else (down,)
            yield from self._expand_bank(prefix, tensor, names)
        elif self._is_projection(name, tensor):
            yield from self._quantized(name, tensor)
        else:
            yield name, tensor

    @staticmethod
    def _expand_bank(prefix: str, bank: torch.Tensor, names: tuple[str, ...]) -> Iterator[tuple[str, torch.Tensor]]:
        """``bank`` ``[E, N, K]`` as per-expert fp8 views and scale views of batched calls, so a layer's launches do not
        grow with its expert count. A bank is one call, or a few slices of experts past :data:`_QUANTIZE_ELEMENTS`,
        which bound the fp32 working copy. A gate/up pair splits into per-expert views of one result when the half
        width is a block multiple, and is quantized per half otherwise, where one grid would span the seam."""
        part_rows = bank.shape[1] // len(names)
        joint = part_rows % BLOCK == 0
        groups = (
            [(bank, names)] if joint else [(half, (name,)) for half, name in zip(bank.chunk(len(names), 1), names, strict=False)]
        )
        step = max(1, _QUANTIZE_ELEMENTS // groups[0][0][0].numel())
        for start in range(0, bank.shape[0], step):
            quantized = [(quantize_block_fp8(source[start : start + step]), parts) for source, parts in groups]
            for offset in range(quantized[0][0][0].shape[0]):
                for (weights, scales), parts in quantized:
                    for index, name in enumerate(parts):
                        stem = f"{prefix}.{start + offset}.{name}.weight"
                        rows = slice(index * part_rows, (index + 1) * part_rows)
                        grid = slice(index * part_rows // BLOCK, (index + 1) * part_rows // BLOCK)
                        yield stem, weights[offset, rows] if len(parts) > 1 else weights[offset]
                        yield stem + SCALE_SUFFIX, scales[offset, grid] if len(parts) > 1 else scales[offset]

    @staticmethod
    def _quantized(name: str, weight: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        quantized, scale_inv = quantize_block_fp8(weight)
        yield name, quantized
        yield name + SCALE_SUFFIX, scale_inv

    def _in_text_decoder(self, name: str) -> bool:
        index = decoder_layer_index(name)
        return index is not None and index < self.num_layers and ROOT_HEAD_ATTR not in name.split(".")

    def _is_projection(self, name: str, tensor: torch.Tensor) -> bool:
        if tensor.ndim != 2 or not name.endswith(".weight") or not self._in_text_decoder(name):
            return False
        projections = (*ATTENTION_PROJECTIONS, *MLP_PROJECTIONS, *(self.expert_keys or ()))
        return name.rpartition(".")[0].rpartition(".")[2] in projections

    def _is_fused_bank(self, name: str, tensor: torch.Tensor) -> bool:
        prefix, _, bank_name = name.rpartition(".")
        return (
            self.expert_keys is not None
            and tensor.ndim == 3
            and bank_name in _FUSED_BANKS
            and prefix.rpartition(".")[2] in experts_container_attrs()
            and self._in_text_decoder(name)
        )
