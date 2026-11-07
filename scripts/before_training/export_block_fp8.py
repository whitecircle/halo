"""Write the block-FP8 checkpoint a rollout server boots from, for ``rollout_weight_precision: fp8_block``.

The attention and MLP projections of the text decoder (routed and shared experts included, split per expert) become
float8_e4m3fn weights with fp32 ``weight_scale_inv`` per 128x128 block; routers, linear-attention blocks, embeddings,
the head, norms, vision towers and MTP heads stay bf16. The weight push applies the same quantizer, so the served
tensors equal what a fresh boot of this checkpoint holds. Serve it with the engine's ``--moe-backend triton``.

    python scripts/before_training/export_block_fp8.py --input_dir <bf16 checkpoint> --output_dir <fp8 checkpoint>
"""

import argparse

from halo.checkpoint.tools.block_fp8_export import export_block_fp8_checkpoint
from halo.log import configure_cli_logging
from scripts.cli_arguments import add_max_shard_size_arg


def main() -> None:
    configure_cli_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="The bf16 checkpoint directory, as the run trains it.")
    parser.add_argument("--output_dir", required=True, help="Destination of the block-FP8 checkpoint.")
    add_max_shard_size_arg(parser)
    args = parser.parse_args()
    export_block_fp8_checkpoint(args.input_dir, args.output_dir, max_shard_size=args.max_shard_size)


if __name__ == "__main__":
    main()
