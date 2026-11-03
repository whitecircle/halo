#!/usr/bin/env python
"""Reset a checkpoint's attention sinks to the dtype's most negative value, disabling the sink mechanism.

For an existing checkpoint; a training run's ``reset_sinks: true`` (the default) applies the same reset. Rewrites only
the shards holding a sink and links the rest; ``--in_place`` has no undo, so it must be asked for. No model code is
loaded, so the tool takes no ``--trust_remote_code``."""

import argparse

from halo.checkpoint.tools.sink_reset import reset_sinks
from halo.log import configure_cli_logging
from scripts.cli_arguments import add_hub_source_args


def parse_args():
    parser = argparse.ArgumentParser(description="Reset attention sinks in a safetensors checkpoint")
    add_hub_source_args(parser, source="The checkpoint whose attention sinks are reset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the modified checkpoint. Required unless --in_place or --dry_run is given.",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Rewrite the --model_id directory itself instead of writing a copy. There is no undo, so "
        "it must be asked for explicitly (a local directory only, never a Hub repo id).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print sink values without modifying the checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    configure_cli_logging()
    args = parse_args()
    reset_sinks(
        args.model_id,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        in_place=args.in_place,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
