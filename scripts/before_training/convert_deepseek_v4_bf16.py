#!/usr/bin/env python
"""Convert a DeepSeek-V4 FP8/FP4 hub checkpoint to a plain BF16 checkpoint for training.

The hub checkpoints (``deepseek-ai/DeepSeek-V4-Flash[-Base]``) ship fine-grained FP8 dense
weights (e4m3, 128x128 blocks, ue8m0 scales) with FP4-packed routed experts
(``config.expert_dtype = "fp4"``: two e2m1 nibbles per int8 byte, [1, 32] scale grid). EP
training needs plain BF16 expert tensors (``apply_ep_layers`` rejects non-float expert
storage), so this script routes the checkpoint through transformers' dequantizing loader
(``FineGrainedFP8Config(dequantize=True)`` attaches ``Fp8Dequantize`` to the V4 weight-conversion
chain, verified exact against a manual e2m1 LUT dequant) and re-saves it sharded in BF16.

Disk budget: ~910 GB total, ~330 GB for the FP8/FP4 download cache (HF_HOME) plus ~580 GB for the
BF16 output. RAM: the model is materialized on CPU (~580 GB), so use a high-memory host.

The saved checkpoint is uniform BF16: transformers' ``_keep_in_fp32_modules_strict`` would keep the
HC/norm modules fp32, whose fp32 outputs fail the eager-only bf16 forward on a dtype mismatch, and
those modules upcast to fp32 internally anyway, so bf16 storage is safe.

Usage (inside the training image; point HF_HOME and the output at a large volume):
    HF_HOME=<large-volume>/hf python scripts/before_training/convert_deepseek_v4_bf16.py \\
        --model_id deepseek-ai/DeepSeek-V4-Flash \\
        --output_dir <large-volume>/models/DeepSeek-V4-Flash-BF16
"""

import argparse

from halo.checkpoint.tools.deepseek_v4_dequant import convert_deepseek_v4_checkpoint
from halo.log import configure_cli_logging
from scripts.cli_arguments import add_hub_source_args, add_max_shard_size_arg, add_trust_remote_code_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_hub_source_args(parser, source="The FP8 release to dequantize", default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--output_dir", required=True, help="Destination for the BF16 checkpoint (needs ~580 GB)")
    # This tool loads through from_pretrained, so it exposes the same flag as every sibling that
    # executes a checkpoint's own code. Opt-in, since --model_id defaults to a Hub repo and the hub
    # V4 checkpoints are transformers-native.
    add_trust_remote_code_arg(parser, default=False)
    add_max_shard_size_arg(parser)
    return parser.parse_args()


def main() -> None:
    configure_cli_logging()
    args = parse_args()
    convert_deepseek_v4_checkpoint(
        args.model_id,
        args.output_dir,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
