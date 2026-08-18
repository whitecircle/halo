#!/usr/bin/env python
"""Stream the FP8 block-quantized GLM-5.3-Flash release into its BF16 training checkpoint.

The hub release (``zai-org/GLM-5.3-Flash``) ships fp8-e4m3 weights quantized in 128x128 blocks
(``quantization_config``: ``quant_method: "fp8"``, ``fmt: "e4m3"``, ``weight_block_size:
[128, 128]``), each with a co-located fp32 ``<name>_scale_inv`` holding one scale per block
(``ceil(rows/128) x ceil(cols/128)``, with ``scale_inv[i, j]`` multiplying block ``[i, j]``, the same
convention as transformers' ``Fp8Dequantize`` and vLLM's block dequant). EP training needs plain BF16
expert tensors, so this tool dequantizes block-wise and re-saves sharded BF16, streaming
shard-by-shard with no full-model RAM footprint. Every tensor the release keeps unquantized (the KDA
linear-attention stack, norms, router, hyper-connection tensors and vision tower, per
``modules_to_not_convert``) passes through at its stored dtype, which keeps the family's fp32 pins
(transformers' ``_keep_in_fp32_modules_strict``: ``e_score_correction_bias``, ``conv1d``,
``dt_bias``, ``A_log``) fp32. The ``hc_*`` tensors are not among them (the mHC mapping upcasts its
input in the forward) and ride through at whatever dtype the release stores. The MTP layer
(checkpoint layer 45) is carried across dequantized; transformers drops it on load, so exports ship
without it. The emitted ``config.json`` loses its ``quantization_config``.

Disk budget for GLM-5.3-Flash: ~330 GB for the fp8 download cache (``HF_HOME``) plus ~650 GB for
the BF16 output; host RAM stays bounded by ``--max_shard_size``.

Usage (inside the training image; point HF_HOME and the output at a large volume):
    HF_HOME=/mnt/hf python scripts/before_training/convert_glm5_bf16.py \
        --model_id zai-org/GLM-5.3-Flash \
        --output_dir /mnt/models/GLM-5.3-Flash-BF16
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os

import torch

from scripts._common import add_hub_source_args, add_max_shard_size_arg
from src.checkpoint.fp8_dequant import DequantRules, run_dequant_conversion
from src.log import configure_cli_logging

configure_cli_logging()
logger = logging.getLogger(__name__)

# The quant sidecar suffix, matched exactly. GLM-5 also stores hyper-connection tensors named
# ``hc_attn_scale`` / ``hc_ffn_scale``, real weights that a looser "scale" match would drop.
_SCALE_SUFFIX = "_scale_inv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_hub_source_args(parser, source="The FP8 release to dequantize", default="zai-org/GLM-5.3-Flash")
    parser.add_argument("--output_dir", required=True, help="Destination for the BF16 checkpoint")
    add_max_shard_size_arg(parser)
    return parser.parse_args()


def read_block_shape(source: str) -> tuple[int, int]:
    """The fp8 block shape the checkpoint's own config declares; also the gate on the quant scheme."""
    with open(os.path.join(source, "config.json")) as f:
        config = json.load(f)
    quant = config.get("quantization_config")
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
    block_shape = read_block_shape(source)
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
        scale_suffix=_SCALE_SUFFIX,
        is_sidecar=lambda name: name.endswith(_SCALE_SUFFIX),
        validate_scale=validate_scale,
        convert=convert,
    )


def main() -> None:
    args = parse_args()
    run_dequant_conversion(
        args.model_id,
        args.output_dir,
        tool="convert_glm5_bf16",
        rules=block_fp8_rules,
        revision=args.revision,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
