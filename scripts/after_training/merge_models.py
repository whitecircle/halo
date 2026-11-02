#!/usr/bin/env python
"""Merge several fine-tuned checkpoints of the same architecture into one model.

A self-contained replacement for external merge tools, with no mergekit dependency. Supports the
common weight-space methods:

- ``linear``          — weighted average of the model weights (always normalized by the weight sum).
- ``slerp``           — spherical interpolation between exactly two models (per-tensor), param ``--t``.
- ``task_arithmetic`` — base + Σ wᵢ·(modelᵢ − base): sum the task vectors onto a base model.
- ``ties``            — TIES-merge: trim each task vector to its top-``--density`` magnitudes, elect a
                        sign per parameter, average the sign-agreeing values, add ``--lambda`` × that
                        to the base (arXiv:2306.01708).

Merging streams one tensor at a time across the input checkpoints (each key is loaded from every
model, merged, then written), so peak host memory is every input's copy of the costliest tensor (the
largest, in practice), the method's float32 working set over it, and the output shard being filled
plus one per write thread, never the merged model. Output is HF-sharded safetensors
(``--max_shard_size`` sets the per-file cap). Config, tokenizer and any remote-code modules are
copied from ``--tokenizer_source`` (default: the base model, or the first model), a local directory
like every other source.

Examples:
    # Weighted linear average (weights normalized by their sum)
    python scripts/after_training/merge_models.py --method linear \\
        --models /ckpt/run-a:0.3 /ckpt/run-b:0.7 --output_dir <large-volume>/merged-linear

    # SLERP halfway between two checkpoints
    python scripts/after_training/merge_models.py --method slerp --t 0.5 \\
        --models /ckpt/run-a /ckpt/run-b --output_dir <large-volume>/merged-slerp

    # Task arithmetic / TIES onto a base
    python scripts/after_training/merge_models.py --method ties --density 0.6 --lambda 1.0 \\
        --base_model /ckpt/base --models /ckpt/run-a:0.5 /ckpt/run-b:0.5 --output_dir <large-volume>/merged-ties
"""

import argparse
import sys

from halo.checkpoint.tools.model_merge import KNOB_DEFAULTS, KNOB_DOMAINS, MERGE_METHODS, knob_dest, merge_models
from halo.log import configure_cli_logging
from scripts.cli_arguments import add_dtype_arg, add_max_shard_size_arg, add_trust_remote_code_arg


def main() -> int:
    configure_cli_logging()
    parser = argparse.ArgumentParser(
        description="Merge same-architecture checkpoints (linear / slerp / task_arithmetic / ties).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Local model directories, each optionally 'path:weight'. Not Hub ids — weights are read off disk.",
    )
    parser.add_argument("--output_dir", required=True, help="New directory for the merged checkpoint.")
    parser.add_argument(
        "--method", default="linear", choices=list(MERGE_METHODS), help="Merge method (default: %(default)s)."
    )
    parser.add_argument("--base_model", default=None, help="Base for task_arithmetic / ties.")
    add_dtype_arg(parser)
    # Knob defaults stay None here and resolve in merge_models(), so a knob a method ignores raises;
    # the help text renders the method table's default rather than a copy of it.
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help=f"ties: fraction of deltas kept, {KNOB_DOMAINS['density'][1]} (default {KNOB_DEFAULTS['density']}).",
    )
    parser.add_argument(
        "--lambda",
        dest=knob_dest("lambda"),
        type=float,
        default=None,
        help=f"ties: merged-delta scale, {KNOB_DOMAINS['lambda'][1]} (default {KNOB_DEFAULTS['lambda']}).",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=None,
        help=f"slerp: interpolation factor, {KNOB_DOMAINS['t'][1]} (default {KNOB_DEFAULTS['t']}).",
    )
    parser.add_argument(
        "--tokenizer_source",
        default=None,
        help="Local directory to copy config, tokenizer and remote-code modules from (default: the base model, "
        "else the first model).",
    )
    add_max_shard_size_arg(parser)
    parser.add_argument(
        "--allow_missing_tokenizer",
        action="store_true",
        help="Accept a merged checkpoint whose --tokenizer_source ships no tokenizer files (refused by default: "
        "from_pretrained-based consumers cannot load it).",
    )
    add_trust_remote_code_arg(parser)
    parser.add_argument("--quiet", action="store_true", help="Log only warnings and the final line.")
    args = parser.parse_args()

    merge_models(
        model_specs=args.models,
        output_dir=args.output_dir,
        method=args.method,
        base_model=args.base_model,
        dtype=args.dtype,
        knobs={knob: getattr(args, knob_dest(knob)) for knob in KNOB_DEFAULTS},
        tokenizer_source=args.tokenizer_source,
        max_shard_size=args.max_shard_size,
        allow_missing_tokenizer=args.allow_missing_tokenizer,
        trust_remote_code=args.trust_remote_code,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
