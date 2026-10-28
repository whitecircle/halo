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
    HF_HOME=<large-volume>/hf python scripts/before_training/convert_glm5_bf16.py \\
        --model_id zai-org/GLM-5.3-Flash \\
        --output_dir <large-volume>/models/GLM-5.3-Flash-BF16
"""

import argparse

from halo.checkpoint.tools.fp8_dequant import block_fp8_rules, run_dequant_conversion
from halo.log import configure_cli_logging
from scripts.cli_arguments import add_hub_source_args, add_max_shard_size_arg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_hub_source_args(parser, source="The FP8 release to dequantize", default="zai-org/GLM-5.3-Flash")
    parser.add_argument("--output_dir", required=True, help="Destination for the BF16 checkpoint")
    add_max_shard_size_arg(parser)
    return parser.parse_args()


def main() -> None:
    configure_cli_logging()
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
