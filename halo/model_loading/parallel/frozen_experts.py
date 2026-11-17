"""``frozen_quantized_experts`` on the lazy EP load: before planning, each fused expert container on the meta shell
trades its bf16 expert weights for buffers in the release's packed layout (the family spec's
``frozen_expert_slots``), so each rank streams its expert slice of the packed bytes and scales unchanged, a buffer
keeping its stored dtype."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from safetensors import safe_open

from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.distributed.expert_parallel.families.roster import frozen_bank_families
from halo.model_loading.lazy_safetensors.conversion_selection import family_key_mapping
from halo.models.moe.family_specs import MoEFamilySpec, find_moe_family_spec_for_config

logger = logging.getLogger(__name__)

_HEADER_FORMATS = {"U8": "nvfp4", "F8_E4M3": "mxfp8"}
_SAFETENSORS_DTYPES = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}


def supports_frozen_experts(spec: MoEFamilySpec | None) -> bool:
    """Whether ``spec``'s EP layer holds frozen banks, which the family's ``frozen_expert_slots`` lay out."""
    return spec in frozen_bank_families()


def reject_frozen_experts_family(config) -> None:
    """Refuse ``frozen_quantized_experts`` on a model whose family's EP layer holds no frozen banks."""
    if not supports_frozen_experts(find_moe_family_spec_for_config(config)):
        raise ValueError(
            f"frozen_quantized_experts keeps a release's NVFP4/MXFP8 expert layout, which is implemented for "
            f"{[spec.model_types[0] for spec in frozen_bank_families()]}; this checkpoint is "
            f"{getattr(config, 'model_type', None)!r}."
        )


class _HeaderReader:
    def __init__(self, model_path: str):
        self._weights = resolve_checkpoint_weights(model_path)
        self._handles: dict[str, object] = {}

    def __call__(self, key: str) -> tuple[str, tuple[int, ...]]:
        shard = self._weights.weight_map.get(key)
        if shard is None:
            raise ValueError(f"frozen_quantized_experts: the checkpoint has no {key!r}")
        handle = self._handles.get(shard)
        if handle is None:
            handle = self._handles[shard] = safe_open(self._weights.path(shard), framework="pt")
        view = handle.get_slice(key)
        return view.get_dtype(), tuple(view.get_shape())


def _meta_buffer(owner: nn.Module, name: str, header: tuple[str, tuple[int, ...]], key: str) -> None:
    dtype_name, shape = header
    dtype = _SAFETENSORS_DTYPES.get(dtype_name)
    if dtype is None:
        raise ValueError(f"frozen_quantized_experts: {key!r} is stored as {dtype_name}, which no slot holds")
    owner.register_buffer(name, torch.empty(shape, dtype=dtype, device="meta"))


def install_frozen_bank_slots(model: nn.Module, model_path: str) -> int:
    """Swap every fused expert container's bf16 weights on the meta ``model`` for release-layout buffers sized from the
    checkpoint's headers; returns the containers converted. A bf16 expert tensor is refused, as there is nothing to
    keep frozen."""
    reject_frozen_experts_family(model.config)
    slots = find_moe_family_spec_for_config(model.config).frozen_expert_slots
    weights = resolve_checkpoint_weights(model_path)
    disk_to_model, _ = family_key_mapping(model, list(weights.weight_map))
    model_to_disk = {target: disk for disk, target in disk_to_model.items()}
    header = _HeaderReader(model_path)
    converted = 0
    for name, module in model.named_modules():
        if not all(isinstance(module._parameters.get(slot.weight), nn.Parameter) for slot in slots):
            continue
        for slot in slots:
            disk_key = model_to_disk.get(f"{name}.{slot.weight}")
            if disk_key is None:
                raise ValueError(f"frozen_quantized_experts: no checkpoint key loads {name}.{slot.weight}")
            weight_header = header(disk_key)
            quant_format = _HEADER_FORMATS.get(weight_header[0])
            if quant_format is None:
                raise ValueError(
                    f"frozen_quantized_experts: {disk_key!r} is stored as {weight_header[0]}, not a packed NVFP4 or "
                    f"MXFP8 tensor. A bf16 checkpoint trains without frozen_quantized_experts."
                )
            del module._parameters[slot.weight]
            _meta_buffer(module, slot.weight, weight_header, disk_key)
            _meta_buffer(module, slot.scale, header(f"{disk_key}_scale"), f"{disk_key}_scale")
            if quant_format == "nvfp4":
                _meta_buffer(module, slot.amax, header(f"{disk_key}_amax"), f"{disk_key}_amax")
                denominator_key = f"{disk_key}_outer_scale_denominator"
                _meta_buffer(module, slot.denominator, header(denominator_key), denominator_key)
        converted += 1
    if not converted:
        raise ValueError("frozen_quantized_experts: the model has no fused expert container to keep frozen")
    logger.info(f"frozen_quantized_experts: {converted} expert containers keep the release's packed layout")
    return converted
