#!/usr/bin/env python
"""Stream an FP8 Mistral 3/4 checkpoint into its fused-layout BF16 equivalent.

The public FP8 releases (``mistralai/Mistral-Small-4-119B-2603``: ``quant_method: "fp8"``, static
activation scales, no block grid) store each quantized weight as fp8-e4m3 with a co-located
``<name>_scale_inv``: one scalar per 2-D matrix, one scale per expert for the fused 3-D expert banks.
This tool dequantizes shard-by-shard and re-saves sharded BF16 with no full-model RAM footprint;
every other floating tensor is cast to bf16 too, so the output is uniform. The activation scales
(``*_activation_scale``) describe the fp8 forward and are dropped with the ``quantization_config``.

Usage (inside the training image; point HF_HOME and the output at a large volume):
    HF_HOME=<large-volume>/hf python scripts/before_training/convert_mistral4_bf16.py \\
        --model_id mistralai/Mistral-Small-4-119B-2603 \\
        --output_dir <large-volume>/models/Mistral-Small-4-119B-BF16
"""

import argparse

from halo.checkpoint.tools.fp8_dequant import run_dequant_conversion, static_fp8_rules
from halo.log import configure_cli_logging
from scripts.cli_arguments import add_hub_source_args, add_max_shard_size_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_hub_source_args(parser, source="The FP8 release to dequantize")
    parser.add_argument("--output_dir", required=True, help="Destination for the BF16 checkpoint")
    add_max_shard_size_arg(parser)
    return parser.parse_args()


def main() -> None:
    configure_cli_logging()
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
