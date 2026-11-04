#!/usr/bin/env python
"""Merge a pipeline-parallel save's per-node directories on a non-shared output filesystem into one checkpoint.

Each node's directory has the whole-model index but only the stage parts it wrote: copy them onto one host, then merge.
"""

import argparse

from halo.checkpoint.tools.pp_stage_merge import merge_pp_stage_parts
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
