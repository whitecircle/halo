#!/usr/bin/env python
"""Merge the per-node directories of a pipeline-parallel save into one ``from_pretrained``-loadable checkpoint.

A pipeline-parallel save on a non-shared output filesystem (``DIST_OUTPUT_SHARED_FILESYSTEM=0``) leaves
each node's directory with the whole-model index but only the stage parts that node wrote. Copy every
node's directory of the save onto one host, then merge them; a shared-filesystem save needs no merge.

Usage:
    python scripts/after_training/merge_pp_stages.py \\
        --input_dir /copies/node0/checkpoint-500 /copies/node1/checkpoint-500 \\
        --output_dir /path/to/merged_checkpoint
"""

import argparse

from halo.checkpoint.pp_stage_merge import merge_pp_stage_parts
from halo.log import configure_cli_logging


def main():
    configure_cli_logging()
    parser = argparse.ArgumentParser(
        description="Merge the per-node directories of a pipeline-parallel save into one checkpoint."
    )
    parser.add_argument(
        "--input_dir", nargs="+", required=True, help="Each node's directory of the save, one per node"
    )
    parser.add_argument("--output_dir", required=True, help="Path to save the merged checkpoint")
    args = parser.parse_args()
    merge_pp_stage_parts(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
