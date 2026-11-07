"""Write the checkpoint a block-FP8 rollout server boots from: the bf16 base with its attention and MLP projections
quantized by the same plan and quantizer the weight push applies, and the ``quantization_config`` the engines read."""

from __future__ import annotations

import logging
import os

import torch
from transformers import AutoConfig

from halo.checkpoint.config_export import write_config_json
from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from halo.checkpoint.reading import checkpoint_shard_files, iter_checkpoint_tensors, read_checkpoint_config
from halo.checkpoint.tools.offline_export import (
    copy_checkpoint_aux_files,
    preflight_resource_warning,
    reject_in_place_conversion,
)
from halo.checkpoint.writing import StreamingShardWriter
from halo.kernels.lowp.block_fp8 import block_fp8_quantization_config
from halo.models.fp8_rollout import Fp8RolloutPlan

logger = logging.getLogger(__name__)


def export_block_fp8_checkpoint(input_dir: str, output_dir: str, max_shard_size: str = DEFAULT_MAX_SHARD_SIZE) -> None:
    """Stream ``input_dir`` into a block-FP8 checkpoint at ``output_dir``. Every 2-D linear weight left in bf16 is
    listed in ``modules_to_not_convert`` by its exact module path, as vLLM matches the list by exact path. Raises on a
    family no GPU row has validated, or a source already quantized."""
    reject_in_place_conversion(input_dir, output_dir)
    if "quantization_config" in read_checkpoint_config(input_dir):
        raise ValueError(f"{input_dir} already declares a quantization_config; export from the bf16 checkpoint.")
    plan = Fp8RolloutPlan.for_config(AutoConfig.from_pretrained(input_dir))
    input_shards = checkpoint_shard_files(input_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kept_bf16: set[str] = set()

    with StreamingShardWriter(output_dir, max_shard_size, enabled=True) as writer:
        preflight_resource_warning(
            "export_block_fp8",
            output_dir,
            disk_bytes=sum(os.path.getsize(shard) for shard in input_shards),
            ram_bytes=writer.max_resident_bytes,
        )
        os.makedirs(output_dir, exist_ok=True)
        for key, tensor in iter_checkpoint_tensors(input_dir):
            for name, stored in plan.expand(key, tensor.to(device)):
                if stored.ndim == 2 and name.endswith(".weight") and stored.dtype != torch.float8_e4m3fn:
                    kept_bf16.add(name.removesuffix(".weight"))
                writer.add(name, stored.cpu())
        writer.close_as_hf_checkpoint()

    copy_checkpoint_aux_files(input_dir, output_dir)
    config = read_checkpoint_config(output_dir)
    config["quantization_config"] = block_fp8_quantization_config(sorted(kept_bf16))
    write_config_json(output_dir, config)
    logger.info(f"Wrote the block-FP8 checkpoint to {output_dir}; {len(kept_bf16)} linear modules stay bf16.")
