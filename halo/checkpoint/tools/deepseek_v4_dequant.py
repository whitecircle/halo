"""Convert a DeepSeek-V4 FP8/FP4 hub release to a uniform BF16 checkpoint through transformers' dequantizing loader."""

from __future__ import annotations

import logging
import math
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from halo.checkpoint.reading import (
    FP8_HEADER_DTYPE,
    checkpoint_shard_files,
    iter_checkpoint_shard_entries,
    reject_partial_checkpoint,
)
from halo.checkpoint.tools.offline_export import (
    preflight_resource_warning,
    reject_in_place_conversion,
    reject_source_without_fp8,
    resolve_checkpoint_source,
    save_full_checkpoint,
)
from halo.model_loading.checkpoint_coverage import from_pretrained_verified

logger = logging.getLogger(__name__)

# Blanked for the dequantizing load and dropped before the save, so config.json shows the class plan as the hub does.
_PARALLEL_PLAN_ATTRS = ("base_model_ep_plan", "base_model_tp_plan")


def source_config(source: str, *, trust_remote_code: bool):
    """The source's config with instance parallel plans blanked, since ``FineGrainedFP8HfQuantizer.update_tp_plan``
    copies the class plan onto the instance with ``*_scale_inv`` entries the save would write into ``config.json``."""
    config = AutoConfig.from_pretrained(source, trust_remote_code=trust_remote_code)
    for attr in _PARALLEL_PLAN_ATTRS:
        setattr(config, attr, None)
    return config


def _restore_parallel_plans(config) -> None:
    for attr in _PARALLEL_PLAN_ATTRS:
        config.__dict__.pop(attr, None)


def reject_uneven_fp8_blocks(source: str, block_shape: tuple[int, int]) -> None:
    """Refuse an fp8 tensor whose scale grid transformers would misapply: ``Fp8Dequantize`` splits into
    ``rows // grid`` equal pieces, so a dimension neither one block nor a block multiple shifts every scale. Headers
    only."""
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


def convert_deepseek_v4_checkpoint(
    model_id: str,
    output_dir: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
) -> None:
    """Dequantize ``model_id`` (FP8 dense + FP4 experts) through transformers' loader and save it uniform BF16."""
    source = resolve_checkpoint_source(model_id, revision)

    # A per-rank EP save reuses the ordinary index filename with partial slices, so from_pretrained would randomly
    # initialize the real expert keys and only warn.
    reject_partial_checkpoint(source)
    # save_pretrained clears the model*.safetensors it does not overwrite, so an in-place run would
    # destroy the 100+ GiB FP8 source it is still reading the tokenizer from.
    reject_in_place_conversion(source, output_dir)

    config = source_config(source, trust_remote_code=trust_remote_code)
    reject_source_without_fp8(source, tool="convert_deepseek_v4_bf16")
    quantization = getattr(config, "quantization_config", None) or {}
    block = tuple(quantization.get("weight_block_size") or FineGrainedFP8Config().weight_block_size)
    reject_uneven_fp8_blocks(source, block)
    # The dequantized model lands in host RAM at bf16: twice the fp8 bytes, and about as much again on disk.
    fp8_bytes = sum(os.path.getsize(shard) for shard in checkpoint_shard_files(source))
    preflight_resource_warning(
        "convert_deepseek_v4_bf16", output_dir, disk_bytes=2 * fp8_bytes, ram_bytes=2 * fp8_bytes
    )

    logger.info(f"Loading {source} with dequantize=True (FP8 dense + FP4 experts → BF16)...")
    # A truncated download would otherwise re-save random tensors as a complete-looking BF16 checkpoint.
    model = from_pretrained_verified(
        AutoModelForCausalLM,
        source,
        config=config,
        dtype=torch.bfloat16,
        quantization_config=FineGrainedFP8Config(dequantize=True),
        attn_implementation="eager",  # V4 is eager-only (head_dim=512, compressor KV concat)
        trust_remote_code=trust_remote_code,
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

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)

    logger.info(f"Saving BF16 checkpoint to {output_dir} (max_shard_size={max_shard_size})...")
    # Reverts a dequantized load through the registry mapping (hub layout, bf16, no scale sidecars).
    save_full_checkpoint(
        model,
        output_dir,
        processing_class=tokenizer,
        source_dir=source,
        max_shard_size=max_shard_size,
    )
    logger.info("Done. Point model_name_or_path at the output directory for EP training.")
