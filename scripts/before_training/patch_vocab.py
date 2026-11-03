"""Add tokens to a model's vocabulary and resize its embeddings, saving a patched checkpoint.

New embeddings are the mean of the existing ones. Only patterns tokenizing to more than one token are added: one
already a single token would gain nothing and shift every id above it. Growth only: a target under ``len(tokenizer)``
cuts into the specials and the model can no longer stop generating. ``--output_dir`` is written fresh."""

import argparse
import json
import logging

from halo.checkpoint.tools.vocab_patch import patch_vocabulary
from halo.log import configure_cli_logging
from scripts.cli_arguments import (
    add_device_map_arg,
    add_dtype_arg,
    add_hub_source_args,
    add_max_shard_size_arg,
    add_trust_remote_code_arg,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Patch vocabulary of a model by adding new tokens")
    # No --revision: the patch threads none. Pin by downloading the source first, then pointing --model_id at it.
    add_hub_source_args(
        parser, source="The model whose vocabulary is patched (e.g. 'unsloth/gpt-oss-120b-BF16')", revision=False
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Directory to save the patched model and tokenizer "
            "(every model*.safetensors/index the completed save did not produce is removed afterwards)"
        ),
    )
    parser.add_argument(
        "--patterns",
        type=str,
        default=None,
        help="JSON string or path to JSON file containing list of patterns to add as tokens",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Optional custom chat template to set on the tokenizer: a .jinja/.jinja2/.j2 file, or the "
        "template string itself (the spelling the training configs' chat_template takes).",
    )
    parser.add_argument(
        "--reset_sinks",
        action="store_true",
        help="Reset all attention sinks to minimum value before saving",
    )
    add_dtype_arg(parser)
    add_device_map_arg(parser)
    add_max_shard_size_arg(parser)
    # --model_id is usually a freshly downloaded third-party repo, so remote code stays opt-in.
    add_trust_remote_code_arg(parser, default=False)
    return parser.parse_args()


def load_patterns(patterns_arg: str | None) -> list[str]:
    if patterns_arg:
        try:
            patterns = json.loads(patterns_arg)
            logger.info(f"Loaded {len(patterns)} patterns from command line")
            return patterns
        except json.JSONDecodeError:
            with open(patterns_arg) as f:
                patterns = json.load(f)
            logger.info(f"Loaded {len(patterns)} patterns from file: {patterns_arg}")
            return patterns

    return []


def main():
    configure_cli_logging()
    args = parse_args()
    patch_vocabulary(
        args.model_id,
        args.output_dir,
        patterns=load_patterns(args.patterns),
        chat_template=args.chat_template,
        reset_sinks=args.reset_sinks,
        dtype=args.dtype,
        device_map=args.device_map,
        max_shard_size=args.max_shard_size,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
