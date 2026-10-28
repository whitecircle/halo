"""Streaming fp8 to bf16 conversion of a released checkpoint, and the per-release :class:`DequantRules`: the
block-wise grid of GLM-5 (:func:`block_fp8_rules`) and the static per-matrix scales of Mistral 3/4
(:func:`static_fp8_rules`). The source is walked shard by shard, so a multi-hundred-GB release converts in a
bounded working set.
"""

import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from safetensors import safe_open

from halo.checkpoint.config_export import write_config_json
from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from halo.checkpoint.reading import (
    FP8_HEADER_DTYPE,
    SAFETENSORS_FLOAT_DTYPES,
    checkpoint_shard_files,
    header_nbytes,
    header_numel,
    iter_checkpoint_shards,
    read_checkpoint_config,
)
from halo.checkpoint.tools.offline_export import (
    copy_checkpoint_aux_files,
    preflight_resource_warning,
    reject_in_place_conversion,
    resolve_checkpoint_source,
)
from halo.checkpoint.writing import StreamingShardWriter, cast_to_save_dtype

logger = logging.getLogger(__name__)

# The per-weight dequant scale, matched exactly: GLM-5 also stores hyper-connection tensors named
# ``hc_attn_scale`` / ``hc_ffn_scale``, real weights that a looser "scale" match would drop.
_FP8_SCALE_SUFFIX = "_scale_inv"
# Mistral's quant sidecars, matched exactly: the per-weight dequant scale and the static activation scales.
_STATIC_SIDECAR_SUFFIXES = (_FP8_SCALE_SUFFIX, "_activation_scale", ".activation_scale")


@dataclass(frozen=True)
class DequantRules:
    """One fp8 release's dequantization rules: scale spelling and layouts, the dequantization, and the
    size of an unquantized tensor for the disk preflight."""

    scale_suffix: str
    is_sidecar: Callable[[str], bool]
    validate_scale: Callable[[str, tuple[int, ...], tuple[int, ...]], None]
    convert: Callable[[str, torch.Tensor, torch.Tensor | None], torch.Tensor]
    passthrough_nbytes: Callable[[Any], int] = header_nbytes


def _preflight_dequant_shards(source: str, output_dir: str, *, tool: str, rules: DequantRules) -> int:
    """Header-only validation of every shard before anything is written, so a structural fault fails
    here rather than hours in; returns the output byte count."""
    scale_suffix = rules.scale_suffix
    output_bytes = 0
    for shard, reader in iter_checkpoint_shards(source):
        headers = {name: reader.get_slice(name) for name in reader.keys()}  # noqa: SIM118 - not a mapping
        for name in sorted(headers):
            if rules.is_sidecar(name):
                if name.endswith(scale_suffix) and name[: -len(scale_suffix)] not in headers:
                    raise ValueError(f"{name} has no co-located weight in {os.path.basename(shard)}")
                continue
            header = headers[name]
            if header.get_dtype() != FP8_HEADER_DTYPE:
                output_bytes += rules.passthrough_nbytes(header)
                continue
            scale = headers.get(f"{name}{scale_suffix}")
            if scale is None:
                raise ValueError(f"FP8 tensor {name} has no co-located {name}{scale_suffix}")
            rules.validate_scale(name, tuple(header.get_shape()), tuple(scale.get_shape()))
            output_bytes += header_numel(header) * torch.bfloat16.itemsize
    preflight_resource_warning(tool, output_dir, disk_bytes=output_bytes, ram_bytes=None)
    return output_bytes


def _stream_dequantized_checkpoint(
    input_shards: list[str], output_dir: str, *, rules: DequantRules, max_shard_size: str
) -> int:
    """Stream every weight of ``input_shards`` through ``rules.convert`` into a new HF-standard checkpoint;
    returns the bytes written. Keys go in sorted order, so reruns lay the output out identically."""
    os.makedirs(output_dir, exist_ok=False)
    writer = StreamingShardWriter(output_dir, max_shard_size, enabled=True)
    for index, shard_path in enumerate(input_shards, start=1):
        logger.info("Converting input shard %d/%d: %s", index, len(input_shards), os.path.basename(shard_path))
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            keys = set(shard.keys())
            for name in sorted(keys):
                if rules.is_sidecar(name):
                    continue
                tensor = shard.get_tensor(name)
                scale_name = f"{name}{rules.scale_suffix}"
                scale = shard.get_tensor(scale_name) if scale_name in keys else None
                if scale is None and tensor.dtype == torch.float8_e4m3fn:
                    raise ValueError(f"FP8 tensor {name} has no co-located {scale_name}")
                writer.add(name, rules.convert(name, tensor, scale))
    return writer.close_as_hf_checkpoint()


def _copy_dequantized_assets(source: str, output_dir: str) -> None:
    """Carry the non-weight files across and drop the ``quantization_config`` the bf16 weights no longer
    answer to."""
    copy_checkpoint_aux_files(source, output_dir)
    config = read_checkpoint_config(output_dir)
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    write_config_json(output_dir, config)


def run_dequant_conversion(
    model_id: str,
    output_dir: str,
    *,
    tool: str,
    rules: Callable[[str], DequantRules],
    revision: str | None = None,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
) -> None:
    """Stream an fp8 release into its bf16 checkpoint; ``rules`` is called with the resolved source, so
    rules read off its own config come from there."""
    source = resolve_checkpoint_source(model_id, revision)
    destination = os.path.abspath(output_dir)

    input_shards = checkpoint_shard_files(source)
    reject_in_place_conversion(source, destination)

    conversion = rules(source)
    output_bytes = _preflight_dequant_shards(source, destination, tool=tool, rules=conversion)
    logger.info("Validated %d input shards; BF16 output will be %.1f GiB", len(input_shards), output_bytes / 1024**3)

    total_bytes = _stream_dequantized_checkpoint(
        input_shards, destination, rules=conversion, max_shard_size=max_shard_size
    )
    _copy_dequantized_assets(source, destination)
    logger.info("Conversion complete: %.2f GiB written to %s", total_bytes / 1024**3, destination)


def _read_block_shape(source: str) -> tuple[int, int]:
    """The fp8 block shape the checkpoint's own config declares; also the gate on the quant scheme."""
    quant = read_checkpoint_config(source).get("quantization_config")
    if quant is None:
        raise ValueError(
            f"{source}/config.json declares no quantization_config — the checkpoint is not the fp8 "
            f"release this tool dequantizes. An already-BF16 checkpoint trains directly; point "
            f"model_name_or_path at it instead."
        )
    block = quant.get("weight_block_size")
    if quant.get("quant_method") != "fp8" or quant.get("fmt", "e4m3") != "e4m3" or not block or len(block) != 2:
        raise ValueError(
            f"Unsupported quantization scheme {quant} in {source} — this tool implements fp8-e4m3 "
            f"block-wise dequantization only."
        )
    return int(block[0]), int(block[1])


def dequantize_block_fp8(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_shape: tuple[int, int], name: str
) -> torch.Tensor:
    """One fp8 tensor times its per-block scale grid, in fp32, rounded once to bf16."""
    if weight.dim() != 2:
        raise ValueError(f"{name}: fp8 block dequantization expects a 2D weight, got {tuple(weight.shape)}")
    rows, cols = weight.shape
    block_rows, block_cols = block_shape
    grid = (math.ceil(rows / block_rows), math.ceil(cols / block_cols))
    if tuple(scale_inv.shape) != grid:
        raise ValueError(
            f"{name}: scale grid {tuple(scale_inv.shape)} does not match "
            f"ceil({rows}/{block_rows}) x ceil({cols}/{block_cols}) = {grid}"
        )
    # Expand then crop, so a tail block short of the block size (a non-multiple dimension) still
    # gets its own scale entry.
    scales = scale_inv.float().repeat_interleave(block_rows, dim=0)[:rows]
    scales = scales.repeat_interleave(block_cols, dim=1)[:, :cols]
    return (weight.float() * scales).to(torch.bfloat16)


def block_fp8_rules(source: str) -> DequantRules:
    """This release's rules: block-wise fp8, with the grid the source's own config declares.

    A 2-D fp8 weight's scale grid must be ``ceil(rows/block) x ceil(cols/block)``; everything else
    passes through at its stored size.
    """
    block_shape = _read_block_shape(source)
    logger.info("FP8 block grid from %s/config.json: %dx%d", source, *block_shape)

    def validate_scale(name: str, shape: tuple[int, ...], scale_shape: tuple[int, ...]) -> None:
        grid = tuple(math.ceil(dim / block) for dim, block in zip(shape, block_shape, strict=False))
        if len(shape) != 2 or scale_shape != grid:
            raise ValueError(
                f"{name}: weight {list(shape)} with scale grid {list(scale_shape)} does not fit blocks of "
                f"{block_shape} (expected grid {list(grid)})"
            )

    def convert(name: str, tensor: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
        return tensor if scale is None else dequantize_block_fp8(tensor, scale, block_shape, name)

    return DequantRules(
        scale_suffix=_FP8_SCALE_SUFFIX,
        is_sidecar=lambda name: name.endswith(_FP8_SCALE_SUFFIX),
        validate_scale=validate_scale,
        convert=convert,
    )


def _is_static_fp8_sidecar(name: str) -> bool:
    return name.endswith(_STATIC_SIDECAR_SUFFIXES)


def _scale_layout_ok(weight_shape: tuple[int, ...], scale_shape: tuple[int, ...]) -> bool:
    """One scalar per matrix, or one scale per expert for a 3-D bank ``[E, ...]``."""
    if math.prod(scale_shape) == 1:
        return True
    squeezed = tuple(dim for dim in scale_shape if dim != 1)
    return len(weight_shape) == 3 and squeezed == (weight_shape[0],)


def _dequantize_static_fp8(weight: torch.Tensor, scale: torch.Tensor, name: str) -> torch.Tensor:
    scale = scale.squeeze().float()
    if scale.numel() == 1:
        return (weight.float() * scale).to(torch.bfloat16)
    if scale.dim() == 1 and weight.dim() == 3 and scale.shape[0] == weight.shape[0]:
        return (weight.float() * scale.view(-1, 1, 1)).to(torch.bfloat16)
    raise ValueError(
        f"Unsupported FP8 scale layout for {name}: weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
    )


def _convert_static_fp8(name: str, tensor: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    return cast_to_save_dtype(tensor) if scale is None else _dequantize_static_fp8(tensor, scale, name)


def static_fp8_rules(_source: str) -> DequantRules:
    """This release's rules: per-matrix (or per-expert) static scales, no block grid.

    The accepted scale layouts are :func:`_scale_layout_ok`'s; every unquantized float lands as bf16,
    so the disk preflight sizes it that way, and every other tensor at its stored width.
    """

    def validate_scale(name: str, shape: tuple[int, ...], scale_shape: tuple[int, ...]) -> None:
        if not _scale_layout_ok(shape, scale_shape):
            raise ValueError(f"Unsupported FP8 scale layout for {name}: weight={shape}, scale={scale_shape}")

    def passthrough_nbytes(header) -> int:
        if header.get_dtype() in SAFETENSORS_FLOAT_DTYPES:
            return header_numel(header) * torch.bfloat16.itemsize
        return header_nbytes(header)

    return DequantRules(
        scale_suffix=_FP8_SCALE_SUFFIX,
        is_sidecar=_is_static_fp8_sidecar,
        validate_scale=validate_scale,
        convert=_convert_static_fp8,
        passthrough_nbytes=passthrough_nbytes,
    )
