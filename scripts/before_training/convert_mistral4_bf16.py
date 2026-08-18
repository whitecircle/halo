#!/usr/bin/env python
"""Stream an FP8 Mistral 3/4 checkpoint into its fused-layout BF16 equivalent.

The public FP8 releases (``mistralai/Mistral-Small-4-119B-2603``: ``quant_method: "fp8"``, static
activation scales, no block grid) store each quantized weight as fp8-e4m3 with a co-located
``<name>_scale_inv``: one scalar per 2-D matrix, one scale per expert for the fused 3-D expert banks.
This tool dequantizes shard-by-shard and re-saves sharded BF16 with no full-model RAM footprint;
every other floating tensor is cast to bf16 too, so the output is uniform. The activation scales
(``*_activation_scale``) describe the fp8 forward and are dropped with the ``quantization_config``.

Usage (inside the training image; point HF_HOME and the output at a large volume):
    HF_HOME=/mnt/hf python scripts/before_training/convert_mistral4_bf16.py \
        --model_id mistralai/Mistral-Small-4-119B-2603 \
        --output_dir /mnt/models/Mistral-Small-4-119B-BF16
"""

from __future__ import annotations

import argparse
import logging
import math

import torch

from scripts._common import add_hub_source_args, add_max_shard_size_arg
from src.checkpoint.fp8_dequant import DequantRules, run_dequant_conversion
from src.checkpoint.tool_io import SAFETENSORS_FLOAT_DTYPES
from src.log import configure_cli_logging

configure_cli_logging()
logger = logging.getLogger(__name__)

_SCALE_SUFFIX = "_scale_inv"
# The quant sidecars, matched exactly: the per-weight dequant scale and the static activation scales.
_SIDECAR_SUFFIXES = (_SCALE_SUFFIX, "_activation_scale", ".activation_scale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_hub_source_args(parser, source="The FP8 release to dequantize")
    parser.add_argument("--output_dir", required=True, help="Destination for the BF16 checkpoint")
    add_max_shard_size_arg(parser)
    return parser.parse_args()


def is_sidecar(name: str) -> bool:
    return name.endswith(_SIDECAR_SUFFIXES)


def _scale_layout_ok(weight_shape: tuple[int, ...], scale_shape: tuple[int, ...]) -> bool:
    """One scalar per matrix, or one scale per expert for a 3-D bank ``[E, ...]``."""
    if math.prod(scale_shape) == 1:
        return True
    squeezed = tuple(dim for dim in scale_shape if dim != 1)
    return len(weight_shape) == 3 and squeezed == (weight_shape[0],)


def dequantize(weight: torch.Tensor, scale: torch.Tensor, name: str) -> torch.Tensor:
    scale = scale.squeeze().float()
    if scale.numel() == 1:
        return (weight.float() * scale).to(torch.bfloat16)
    if scale.dim() == 1 and weight.dim() == 3 and scale.shape[0] == weight.shape[0]:
        return (weight.float() * scale.view(-1, 1, 1)).to(torch.bfloat16)
    raise ValueError(
        f"Unsupported FP8 scale layout for {name}: weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
    )


def convert(name: str, tensor: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    if scale is not None:
        return dequantize(tensor, scale, name)
    if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
        return tensor.to(torch.bfloat16)
    return tensor


def static_fp8_rules(_source: str) -> DequantRules:
    """This release's rules: per-matrix (or per-expert) static scales, no block grid.

    The accepted scale layouts are :func:`_scale_layout_ok`'s; every unquantized float lands as bf16,
    so the disk preflight sizes it that way rather than at its stored width.
    """

    def validate_scale(name: str, shape: tuple[int, ...], scale_shape: tuple[int, ...]) -> None:
        if not _scale_layout_ok(shape, scale_shape):
            raise ValueError(f"Unsupported FP8 scale layout for {name}: weight={shape}, scale={scale_shape}")

    def passthrough_nbytes(header) -> int:
        itemsize = torch.bfloat16.itemsize if header.get_dtype() in SAFETENSORS_FLOAT_DTYPES else torch.int8.itemsize
        return math.prod(header.get_shape()) * itemsize

    return DequantRules(
        scale_suffix=_SCALE_SUFFIX,
        is_sidecar=is_sidecar,
        validate_scale=validate_scale,
        convert=convert,
        passthrough_nbytes=passthrough_nbytes,
    )


def main() -> None:
    args = parse_args()
    run_dequant_conversion(
        args.model_id,
        args.output_dir,
        tool="convert_mistral4_bf16",
        rules=static_fp8_rules,
        revision=args.revision,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
