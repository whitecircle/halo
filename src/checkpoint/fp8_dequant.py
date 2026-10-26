"""Streaming fp8 to bf16 conversion of a released checkpoint, shared by the ``*_bf16.py`` converters.

The per-release scale sidecar layout is :class:`DequantRules`'; the source is walked shard by shard, so a
multi-hundred-GB release converts in a bounded working set.
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from safetensors import safe_open

from src.checkpoint.config_export import write_config_json
from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from src.checkpoint.offline_export import (
    copy_checkpoint_aux_files,
    preflight_resource_warning,
    reject_in_place_conversion,
    resolve_checkpoint_source,
)
from src.checkpoint.reading import (
    FP8_HEADER_DTYPE,
    checkpoint_shard_files,
    header_nbytes,
    header_numel,
    iter_checkpoint_shards,
    read_checkpoint_config,
)
from src.checkpoint.writing import StreamingShardWriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DequantRules:
    """One fp8 release's dequantization rules: scale spelling and layouts, the dequantization, and the
    size of an unquantized tensor for the disk preflight."""

    scale_suffix: str
    is_sidecar: Callable[[str], bool]
    validate_scale: Callable[[str, tuple[int, ...], tuple[int, ...]], None]
    convert: Callable[[str, torch.Tensor, torch.Tensor | None], torch.Tensor]
    passthrough_nbytes: Callable[[Any], int] = header_nbytes


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
