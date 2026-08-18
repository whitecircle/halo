#!/usr/bin/env python
"""Convert a DeepSeek-V4 FP8/FP4 hub checkpoint to a plain BF16 checkpoint for training.

The hub checkpoints (``deepseek-ai/DeepSeek-V4-Flash[-Base]``) ship fine-grained FP8 dense
weights (e4m3, 128x128 blocks, ue8m0 scales) with FP4-packed routed experts
(``config.expert_dtype = "fp4"``: two e2m1 nibbles per int8 byte, [1, 32] scale grid). EP
training needs plain BF16 expert tensors (``patch_moe_model_for_ep`` rejects non-float expert
storage), so this script routes the checkpoint through transformers' dequantizing loader
(``FineGrainedFP8Config(dequantize=True)`` attaches ``Fp8Dequantize`` to the V4 weight-conversion
chain — verified exact against a manual e2m1 LUT dequant) and re-saves it sharded in BF16.

Disk budget: ~750 GB total — ~330 GB for the FP8/FP4 download cache (HF_HOME) plus ~420 GB for
the BF16 output. RAM: the model is materialized on CPU (~420 GB); use a high-memory host.

The saved checkpoint is UNIFORM BF16: transformers' ``_keep_in_fp32_modules_strict`` would keep
the HC/norm modules fp32, whose fp32 outputs crash the (eager-only) bf16 forward on a dtype
mismatch — and those modules upcast to fp32 internally anyway, so bf16 storage is safe.

Usage (inside the training image; point HF_HOME and the output at a large volume):
    HF_HOME=/mnt/hf python scripts/before_training/convert_deepseek_v4_bf16.py \
        --model_id deepseek-ai/DeepSeek-V4-Flash \
        --output_dir /mnt/models/DeepSeek-V4-Flash-BF16
"""

from __future__ import annotations

import argparse
import logging
import math
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

import src.distributed.expert_parallel.layers.roster  # noqa: F401 — registers the EP export roster the config finalizer requires
from scripts._common import add_hub_source_args, add_max_shard_size_arg, add_trust_remote_code_arg
from src.checkpoint.tool_io import (
    FP8_HEADER_DTYPE,
    checkpoint_shard_files,
    iter_checkpoint_shard_entries,
    preflight_resource_warning,
    reject_in_place_conversion,
    reject_sharded_checkpoint,
    resolve_checkpoint_source,
    save_full_checkpoint,
)
from src.log import configure_cli_logging
from src.models.loading.checkpoint_coverage import from_pretrained_verified

configure_cli_logging()
logger = logging.getLogger(__name__)

# The instance-level parallel plans blanked for the dequantizing load and dropped again before the
# save, so the exported config.json shows the class plan exactly as the hub config does.
_PARALLEL_PLAN_ATTRS = ("base_model_ep_plan", "base_model_tp_plan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_hub_source_args(parser, source="The FP8 release to dequantize", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--output_dir", required=True, help="Destination for the BF16 checkpoint (needs ~420 GB)")
    # This tool loads through from_pretrained, so it owes the same flag as every sibling that
    # executes a checkpoint's own code — opt-in, because --model_id defaults to a Hub repo (and the
    # hub V4 checkpoints are transformers-native, so nothing needs it).
    add_trust_remote_code_arg(parser, default=False)
    add_max_shard_size_arg(parser)
    return parser.parse_args()


def source_config(source: str, *, trust_remote_code: bool):
    """The source's config, prepared for transformers 5.16's dequantizing loader.

    ``FineGrainedFP8HfQuantizer.update_tp_plan`` rewrites a config's parallel plans through the
    expert implementation's override table, and dereferences ``None`` for every implementation
    but ``deepgemm_megamoe`` — which is none of them at load time — so a config with a class-level
    ``base_model_ep_plan`` (DeepSeek-V4's) crashes the load. Blanking the INSTANCE plans skips the
    rewrite; :func:`_restore_parallel_plans` drops the blanks before the save so the class plan
    shows through in ``config.json`` unchanged.
    """
    config = AutoConfig.from_pretrained(source, trust_remote_code=trust_remote_code)
    for attr in _PARALLEL_PLAN_ATTRS:
        setattr(config, attr, None)
    return config


def _restore_parallel_plans(config) -> None:
    for attr in _PARALLEL_PLAN_ATTRS:
        config.__dict__.pop(attr, None)


def reject_uneven_fp8_blocks(source: str, block_shape: tuple[int, int]) -> None:
    """Refuse an fp8 tensor whose scale grid transformers would apply to the wrong rows.

    ``Fp8Dequantize`` splits a dimension into ``rows // grid`` EQUAL pieces, not into block-sized
    pieces plus a tail, so a dimension that is neither a single block nor a multiple of the block
    is dequantized with every scale shifted off its block — silently. The hub V4 checkpoints are
    block-multiples throughout; this guards a variant that is not. Headers only.
    """
    for _shard, reader, key in iter_checkpoint_shard_entries(source):
        header = reader.get_slice(key)
        if header.get_dtype() != FP8_HEADER_DTYPE:
            continue
        for dim, block in zip(header.get_shape()[-2:], block_shape, strict=True):
            if math.ceil(dim / block) > 1 and dim % block:
                raise ValueError(
                    f"{key}: dimension {dim} is neither one block nor a multiple of the {block}-wide fp8 "
                    f"block, and transformers' dequantizer would scale its blocks off by the tail. "
                    f"Convert this checkpoint with a block-wise streaming dequantizer instead."
                )


def main() -> None:
    args = parse_args()
    source = resolve_checkpoint_source(args.model_id, args.revision)

    # A per-rank EP save reuses the ordinary index filename while each tensor is one rank's partial
    # slice, so from_pretrained would report the real expert keys as MISSING and randomly
    # initialize them — warning only, never raising.
    reject_sharded_checkpoint(source)
    # save_pretrained clears the model*.safetensors it does not overwrite, so an in-place run
    # destroys the FP8 source it is still reading the tokenizer from — 100+ GiB, unrecoverable.
    reject_in_place_conversion(source, args.output_dir)

    config = source_config(source, trust_remote_code=args.trust_remote_code)
    quantization = getattr(config, "quantization_config", None) or {}
    block = tuple(quantization.get("weight_block_size") or FineGrainedFP8Config().weight_block_size)
    reject_uneven_fp8_blocks(source, block)
    # The dequantized model lands in host RAM at bf16: twice the fp8 bytes, and about as much again on disk.
    fp8_bytes = sum(os.path.getsize(shard) for shard in checkpoint_shard_files(source))
    preflight_resource_warning(
        "convert_deepseek_v4_bf16", args.output_dir, disk_bytes=2 * fp8_bytes, ram_bytes=2 * fp8_bytes
    )

    logger.info(f"Loading {source} with dequantize=True (FP8 dense + FP4 experts → BF16)...")
    # Gated load: a truncated/partial download would otherwise re-save randomly initialized tensors
    # as a complete-looking BF16 checkpoint (from_pretrained only warns on missing keys).
    model = from_pretrained_verified(
        AutoModelForCausalLM,
        source,
        config=config,
        dtype=torch.bfloat16,
        quantization_config=FineGrainedFP8Config(dequantize=True),
        attn_implementation="eager",  # V4 is eager-only (head_dim=512, compressor KV concat)
        trust_remote_code=args.trust_remote_code,
    )
    _restore_parallel_plans(model.config)

    non_bf16 = {p.dtype for p in model.parameters() if p.dtype != torch.bfloat16}
    if non_bf16:
        logger.info(f"Flattening residual dtypes {non_bf16} to uniform BF16 (fp32-kept norms/HC modules)")
        model = model.to(torch.bfloat16)

    remaining_quantized = [n for n, p in model.named_parameters() if not p.is_floating_point()]
    if remaining_quantized:
        raise RuntimeError(
            f"Dequantization left {len(remaining_quantized)} non-float parameters "
            f"(e.g. {remaining_quantized[:3]}) — the FP4 expert path did not dequantize. "
            f"Check transformers' FineGrainedFP8 dequantize support for expert_dtype=fp4."
        )

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=args.trust_remote_code)

    logger.info(f"Saving BF16 checkpoint to {args.output_dir} (max_shard_size={args.max_shard_size})...")
    # save_full_checkpoint reverts a dequantized load through the registry mapping (the hub layout,
    # bf16, no scale sidecars) and carries the source's aux files across.
    save_full_checkpoint(
        model,
        args.output_dir,
        processing_class=tokenizer,
        source_dir=source,
        max_shard_size=args.max_shard_size,
    )
    logger.info("Done. Point model_name_or_path at the output directory for EP training.")


if __name__ == "__main__":
    main()
