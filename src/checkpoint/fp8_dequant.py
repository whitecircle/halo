"""Streaming fp8 → bf16 conversion of a released checkpoint, shared by the ``*_bf16.py`` converters.

An fp8 release stores each quantized weight beside a scale sidecar whose layout differs per release
(block grid vs per-channel vector, ``_scale_inv`` vs ``_scale``): :class:`DequantRules` carries that
part, :func:`run_dequant_conversion` everything else. The model is never materialized — the source
is walked shard by shard — so a multi-hundred-GB release converts in a bounded working set.
"""

import json
import logging
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from safetensors import safe_open
from transformers.utils import CONFIG_NAME

from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE, HF_STREAM_PART_PREFIX, copy_checkpoint_aux_files
from src.checkpoint.shard_writer import StageShardWriter
from src.checkpoint.tool_io import (
    FP8_HEADER_DTYPE,
    checkpoint_shard_files,
    header_nbytes,
    iter_checkpoint_shards,
    preflight_resource_warning,
    reject_in_place_conversion,
    resolve_checkpoint_source,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DequantRules:
    """One fp8 release's dequantization rules: how its scale sidecars are spelled, which scale
    layouts it may carry, how a weight is dequantized, and how an unquantized tensor is sized for
    the disk preflight."""

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
    """Stream an fp8 release into its BF16 checkpoint under one release's :class:`DequantRules`.

    The source is resolved (Hub id or local directory), its shards listed and an in-place run
    refused BEFORE the output directory exists. ``rules`` is called with the RESOLVED source, so a
    converter whose rules come out of the checkpoint's own config reads them there.
    """
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
    """Header-only pass over every shard of an fp8 checkpoint BEFORE anything is written.

    Every sidecar must sit beside its weight, every fp8 tensor must carry one, and
    ``rules.validate_scale`` judges the layout against the weight's shape — so a structural fault
    fails here rather than hours into the streamed write. Returns the output byte count, sized for
    the disk preflight (fp8 lands as bf16, everything else at ``rules.passthrough_nbytes``).
    """
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
            output_bytes += math.prod(header.get_shape()) * torch.bfloat16.itemsize
    preflight_resource_warning(tool, output_dir, disk_bytes=output_bytes, ram_bytes=None)
    return output_bytes


def _stream_dequantized_checkpoint(
    input_shards: list[str], output_dir: str, *, rules: DequantRules, max_shard_size: str
) -> int:
    """Stream every weight of ``input_shards`` through ``rules.convert`` into an HF-standard checkpoint.

    Shards are read in order and keys in sorted order — a set walk relayouts a multi-hundred-GB
    output on every run — with never more than the writer's pending part resident. A sidecar key is
    consumed only through the weight it scales, and an fp8 tensor with none is refused rather than
    written raw. ``output_dir`` must not exist yet. Returns the bytes written.
    """
    os.makedirs(output_dir, exist_ok=False)
    writer = StageShardWriter(output_dir, HF_STREAM_PART_PREFIX, max_shard_size, enabled=True)
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
    """Carry a dequantized checkpoint's non-weight files across and re-stamp its ``config.json``.

    The weights no longer answer to the source's ``quantization_config``, and they are bf16 now —
    a config still declaring the scheme would send ``from_pretrained`` looking for scale tensors
    that are gone.
    """
    copy_checkpoint_aux_files(source, output_dir)
    config_path = os.path.join(output_dir, CONFIG_NAME)
    with open(config_path) as f:
        config = json.load(f)
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
